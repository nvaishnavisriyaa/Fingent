"""
Fingent platform facade — the assembly line (§3) wired end to end.

create_agent() is the ONLY path to a saved agent:
   form -> input validator -> LLM compiler -> spec validator -> dependency resolver -> save
        -> factory/register.
Adding a new agent never requires a code change: it is a template (config) + an LLM compilation.
"""
from __future__ import annotations

from .compiler import SpecCompiler
from .dependencies import DependencyResolver
from .middleware import detect_injection
from .runtime import AgentRuntime
from .factory import Factory
from .planner import Planner
from .registry import ToolRegistry
from .schemas import AgentSpec, AgentTemplate, CreateAgentRequest, TemplateParameter
from .store import Store
from .templates import load_catalog
from .validators import validate_inputs


class Fingent:
    def __init__(self, db_path: str = ":memory:") -> None:
        self.store = Store(db_path)
        self.registry = ToolRegistry(self.store)
        self.compiler = SpecCompiler(self.registry)
        self.resolver = DependencyResolver(self.store)
        self.factory = Factory(self.registry, self.store)
        self.planner = Planner(self.factory, self.store)
        self.runtime = AgentRuntime(self.registry, self.store)
        load_catalog(self.store)

    # ----- catalog / form ------------------------------------------------ #
    def templates(self) -> list[AgentTemplate]:
        return self.store.list_templates()

    def form_schema(self, template_name: str) -> dict:
        """The HTML form is a pure rendering of this (params + mandatory free-text field)."""
        tpl = self.store.get_template(template_name)
        if not tpl:
            raise KeyError(template_name)
        return {
            "template": tpl.name, "tier": tpl.tier, "description": tpl.description,
            "parameters": [p.model_dump() for p in tpl.parameters],
            "free_text_field": {
                "name": "additional_requirements", "type": "textarea",
                "label": "Additional / specific requirements",
                "placeholder": "Describe anything special: extra checks, data sources, tone, "
                               "edge cases, external tools to use (e.g. 'also pull recent news "
                               "on the borrower'), etc.",
            },
        }

    # ----- the assembly line -------------------------------------------- #
    def create_agent(self, req: CreateAgentRequest, actor: str = "operator",
                     auto_provision: bool = False) -> dict:
        tpl = self.store.get_template(req.template) if req.template else None

        # step 2: validate STRUCTURED inputs (free-text is NOT validated here)
        input_errors = validate_inputs(tpl, req.answers)
        if input_errors:
            return {"ok": False, "stage": "input_validation", "errors": input_errors}

        # injection scan of the UNTRUSTED free-text field — detected + logged (never
        # executed). The validator is still the boundary that prevents privilege
        # escalation; this gives operators visibility into manipulation attempts.
        injection_signatures = (detect_injection(req.additional_requirements)
                                if req.additional_requirements else [])
        if injection_signatures:
            self.store.audit(req.tenant_id, actor, "freetext_injection_flagged",
                             req.answers.get("name", "?"),
                             {"signatures": injection_signatures})

        # steps 3-4: LLM compiles a candidate, spec validator disposes
        compiled = self.compiler.compile(req, tpl)
        compiler_log = {
            "agent": req.answers.get("name", "?"),
            "used_llm": compiled.used_llm, "attempts": compiled.attempts,
            "ok": compiled.ok, "candidate_specs": compiled.candidate_specs,
            "verdicts": [v.model_dump() for v in compiled.verdicts],
            "message": compiled.message,
            "freetext_injection_signatures": injection_signatures,
        }
        self.store.log_compile(req.tenant_id, req.answers.get("name", "?"), compiler_log)
        if not compiled.ok or compiled.spec is None:
            return {"ok": False, "stage": "compile", "message": compiled.message,
                    "compiler_log": compiler_log,
                    "verdicts": [v.model_dump() for v in compiled.verdicts]}
        spec = compiled.spec

        # step 5: dependency resolution + notification
        dep = self.resolver.check(spec)
        if dep.cycle:
            return {"ok": False, "stage": "dependency", "cycle": dep.cycle,
                    "message": f"dependency cycle: {' -> '.join(dep.cycle)}"}
        if dep.missing_hard and not auto_provision:
            self.store.audit(req.tenant_id, actor, "dependency_notify", spec.name,
                             {"missing_hard": [d.model_dump() for d in dep.missing_hard]})
            return {"ok": False, "stage": "dependency",
                    "needs_prerequisites": True,
                    "missing_hard": [d.model_dump() for d in dep.missing_hard],
                    "missing_soft": [d.model_dump() for d in dep.missing_soft],
                    "creation_order": dep.creation_order,
                    "compiler_log": compiler_log,
                    "spec": spec.model_dump()}

        provisioned = []
        if dep.missing_hard and auto_provision:
            provisioned = self._auto_provision(dep.creation_order[:-1], req.tenant_id, actor)

        # step 6: save  (agent now exists as data)
        self.store.save_spec(spec)
        self.store.audit(req.tenant_id, actor, "create", spec.name,
                         {"template": spec.template, "tools": spec.tools})

        # step 7-8: factory build + register implicitly (spec = source of truth)
        self.factory.build(spec)

        # crystallize a from-scratch spec into a reusable template (§5)
        crystallized = None
        if spec.template is None:
            crystallized = self._crystallize(spec, actor)

        return {"ok": True, "spec": spec.model_dump(),
                "provisioned": provisioned,
                "missing_soft": [d.model_dump() for d in dep.missing_soft],
                "crystallized_template": crystallized,
                "compiler_log": compiler_log,
                "used_llm": compiled.used_llm}

    def _auto_provision(self, order: list[str], tenant_id: str, actor: str) -> list[str]:
        """Create each prerequisite (bottom-up) from its template with defaults."""
        made = []
        for tpl_name in order:
            if self.resolver._satisfied(tenant_id, tpl_name):
                continue
            tpl = self.store.get_template(tpl_name)
            if not tpl:
                continue
            answers = {"name": tpl_name}
            for p in tpl.parameters:
                if p.default is not None:
                    answers[p.name] = p.default
            sub = CreateAgentRequest(template=tpl_name, answers=answers, tenant_id=tenant_id)
            res = self.create_agent(sub, actor=actor, auto_provision=True)
            if res.get("ok"):
                made.append(tpl_name)
                self.store.audit(tenant_id, actor, "auto_provision", tpl_name, "prerequisite")
        return made

    def _crystallize(self, spec: AgentSpec, actor: str) -> str | None:
        """Store a from-scratch spec shape as a new reusable template."""
        tpl_name = f"custom_{spec.name}"
        if self.store.get_template(tpl_name):
            return None
        tpl = AgentTemplate(
            name=tpl_name, tier=spec.tier,
            description=f"Crystallized from from-scratch agent '{spec.name}'.",
            # carry the tool grant forward as both the default and the grantable universe
            # so the crystallized template is genuinely reusable from the form.
            fixed={"base_role": spec.role_prompt.split(chr(10))[0],
                   "required_tools": list(spec.tools)},
            parameters=[TemplateParameter(name="name", type="text",
                                          label="Agent name", required=True)],
            default_depends_on=spec.depends_on,
            default_guardrails=spec.guardrails, grantable_tools=spec.tools,
        )
        self.store.save_template(tpl, tenant_id=spec.security.tenant_id)
        self.store.audit(spec.security.tenant_id, actor, "crystallize_template", tpl_name,
                         {"from_agent": spec.name})
        return tpl_name

    # ----- run ----------------------------------------------------------- #
    def run(self, tenant_id: str, agent_names: list[str], inputs: dict | None = None) -> dict:
        specs = [s for s in self.store.list_specs(tenant_id) if s.name in agent_names]
        if not specs:
            return {"ok": False, "message": "no enabled agents matched"}
        for s in specs:
            self.store.audit(tenant_id, "scheduler", "run", s.name, "")
        return self.planner.run(tenant_id, specs, inputs)

    def run_workflow(self, tenant_id: str, tier: int = 1, inputs: dict | None = None) -> dict:
        specs = [s for s in self.store.list_specs(tenant_id) if s.tier == tier]
        if not specs:
            return {"ok": False, "message": f"no tier-{tier} agents for tenant"}
        return self.planner.run(tenant_id, specs, inputs)

    # ----- admin / MCP --------------------------------------------------- #
    def register_mcp(self, server, actor: str = "admin") -> list[str]:
        return self.registry.register_mcp_server(server, actor)

    # ----- real single-agent execution (playground / deploy / invoke) ---- #
    def run_task(self, tenant_id: str, name: str, user_input,
                 approve_side_effecting: bool = False,
                 approved_tools: list[str] | None = None) -> dict:
        spec = self.store.get_spec(tenant_id, name)
        if spec is None:
            return {"ok": False, "message": f"no deployed agent '{name}' for this tenant"}
        return self.runtime.run(spec, user_input, tenant_id,
                                approve_side_effecting=approve_side_effecting,
                                approved_tools=approved_tools)

    def resolve_review(self, tenant_id: str, run_id: str, decision: str,
                       reviewer: str = "reviewer", note: str = "") -> dict:
        """Approve or reject a run that is waiting on human review. Approval executes any
        held side-effecting action; rejection cancels it. Status is saved to run history."""
        rec = self.store.get_run(tenant_id, run_id)
        if rec is None:
            return {"ok": False, "message": "run not found"}
        rec["reviewer"] = reviewer
        rec["review_note"] = note
        if decision == "approve":
            pending = rec.get("pending_action")
            if pending:
                fn = self.registry.callable(pending["tool"])
                try:
                    out = fn(**(pending.get("args") or {})) if fn else {"error": "no callable"}
                except Exception as e:  # noqa: BLE001
                    out = {"error": str(e)}
                rec.setdefault("steps", []).append({
                    "idx": len(rec.get("steps", [])), "kind": "tool", "tool": pending["tool"],
                    "tool_input": pending.get("args"), "tool_output": out,
                    "note": "executed after human approval", "blocked": False, "latency_ms": 0.0})
                rec["output"] = out
                rec["pending_action"] = None
            rec["status"] = "approved"
        else:
            rec["status"] = "rejected"
        self.store.save_run(rec)
        self.store.audit(tenant_id, reviewer, decision, run_id, {"agent": rec.get("agent")})
        return {"ok": True, "run": rec}

    # ----- agent lifecycle (view / edit / duplicate / delete) ------------ #
    _EDITABLE = ("purpose", "instructions", "input_schema", "output_format",
                 "risk_level", "requires_human_review", "deployed", "role_prompt")

    def get_agent(self, tenant_id: str, name: str) -> dict | None:
        spec = self.store.get_spec(tenant_id, name)
        return spec.model_dump() if spec else None

    def update_agent(self, tenant_id: str, name: str, patch: dict, actor: str = "operator") -> dict:
        spec = self.store.get_spec(tenant_id, name)
        if spec is None:
            return {"ok": False, "message": "agent not found"}
        data = spec.model_dump()
        for k in self._EDITABLE:
            if k in patch and patch[k] is not None:
                data[k] = patch[k]
        if isinstance(patch.get("guardrails"), dict):
            data["guardrails"].update({k: v for k, v in patch["guardrails"].items() if v is not None})
        new = AgentSpec.model_validate(data)
        self.store.save_spec(new)
        self.store.audit(tenant_id, actor, "edit", name, {"fields": list(patch.keys())})
        return {"ok": True, "spec": new.model_dump()}

    def duplicate_agent(self, tenant_id: str, name: str, new_name: str,
                        actor: str = "operator") -> dict:
        spec = self.store.get_spec(tenant_id, name)
        if spec is None:
            return {"ok": False, "message": "agent not found"}
        data = spec.model_dump()
        data["name"] = new_name
        new = AgentSpec.model_validate(data)
        self.store.save_spec(new)
        self.store.audit(tenant_id, actor, "duplicate", new_name, {"from": name})
        return {"ok": True, "spec": new.model_dump()}

    def delete_agent(self, tenant_id: str, name: str, actor: str = "operator") -> dict:
        self.store.delete_spec(tenant_id, name)
        self.store.audit(tenant_id, actor, "delete", name, "")
        return {"ok": True, "deleted": name}

    # ----- analytics / catalog (for the dashboard) ----------------------- #
    def tool_catalog(self, tenant_id: str) -> list[dict]:
        """Every tool the tenant can see, for the Toolkits catalog view."""
        return [
            {"name": d.name, "kind": d.kind.value, "description": d.description,
             "side_effecting": d.side_effecting, "untrusted_output": d.untrusted_output,
             "mcp_server": d.mcp_server}
            for d in self.registry.visible_to(tenant_id)
        ]

    def analytics(self, tenant_id: str, days: int = 7) -> dict:
        """Aggregate run metrics for the monitoring dashboard. Computed from stored
        traces + structured run logs; everything is tenant-scoped (§10)."""
        import time
        now = time.time()
        cutoff = now - days * 86400
        specs = self.store.list_specs(tenant_id)
        traces = self.store.list_traces(tenant_id)
        logs = self.store.get_run_logs(tenant_id)

        total_credits = 0.0
        total_tool_calls = 0
        guardrail_trips = 0
        hitl_pauses = 0
        injection_blocks = 0
        over_time: dict[str, float] = {}
        runs_in_window = 0
        for tr in traces:
            ts = tr.get("ts", now)
            if ts < cutoff:
                continue
            runs_in_window += 1
            m = tr.get("metrics", {}) or {}
            credits = float(m.get("cost_usd", 0) or 0)
            total_credits += credits
            total_tool_calls += sum((m.get("tool_calls") or {}).values())
            guardrail_trips += int(m.get("guardrail_trips", 0) or 0)
            hitl_pauses += int(m.get("hitl_pauses", 0) or 0)
            day = time.strftime("%Y-%m-%d", time.localtime(ts))
            over_time[day] = round(over_time.get(day, 0.0) + credits, 6)

        for l in logs:
            if l.get("event") == "injection_blocked":
                injection_blocks += 1

        agent_calls: dict[str, int] = {}
        agent_runs: dict[str, int] = {}
        for l in logs:
            if l.get("event") == "completed":
                a = l.get("agent", "?")
                agent_calls[a] = agent_calls.get(a, 0) + len(l.get("tools", []))
                agent_runs[a] = agent_runs.get(a, 0) + 1

        names = sorted(set(list(agent_calls) + list(agent_runs)))
        breakdown = [{"agent": a, "tool_calls": agent_calls.get(a, 0),
                      "runs": agent_runs.get(a, 0)} for a in names]
        top = sorted(breakdown, key=lambda x: x["tool_calls"], reverse=True)[:6]
        over = [{"date": d, "credits": c} for d, c in sorted(over_time.items())]

        return {
            "totals": {
                "credits": round(total_credits, 4),
                "tool_calls": total_tool_calls,
                "active_agents": len(specs),
                "runs": runs_in_window,
                "guardrail_trips": guardrail_trips,
                "hitl_pauses": hitl_pauses,
                "injection_blocks": injection_blocks,
            },
            "credits_over_time": over,
            "top_agents": top,
            "agent_breakdown": breakdown,
        }

    def resume_hitl(self, tenant_id: str, trace_id: str, actor: str,
                    decision: str, diff: dict | None = None) -> dict:
        self.store.log_hitl(tenant_id, trace_id, actor, decision, diff or {})
        self.store.audit(tenant_id, actor, decision, trace_id, diff or {})
        return {"ok": True, "decision": decision, "trace_id": trace_id}
