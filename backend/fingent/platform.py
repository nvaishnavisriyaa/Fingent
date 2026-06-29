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
from .planner import Planner
from .registry import ToolRegistry
from .jobs import JobRunner
from . import auth
from .schemas import AgentSpec, AgentTemplate, CreateAgentRequest, TemplateParameter
from .store import Store
from .templates import load_catalog
from .validators import validate_inputs, validate_spec


class Fingent:
    def __init__(self, db_path: str = ":memory:") -> None:
        self.store = Store(db_path)
        # bind the encrypted credential vault to this store (real tool credentials at rest)
        from .vault import vault as _vault
        _vault.attach_store(self.store)
        self.registry = ToolRegistry(self.store)
        self.compiler = SpecCompiler(self.registry)
        self.resolver = DependencyResolver(self.store)
        self.runtime = AgentRuntime(self.registry, self.store)
        self.planner = Planner(self.store, self.runtime)
        load_catalog(self.store)
        # rebuild the in-process MCP tool registry from persisted servers so registered
        # MCP tools survive a process restart (previously they were lost on restart).
        self.registry.load_persisted()
        # bind long-term agent memory to this durable store so recall survives a restart and is
        # tenant-scoped in SQL (Pinecone takes over automatically when PINECONE_API_KEY is set).
        from .memory import get_memory as _get_memory
        _get_memory(self.store)
        self.jobs = JobRunner(self)

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

    def run_supervised(self, tenant_id: str, agent_names: list[str] | None = None,
                       tier: int | None = None, inputs: dict | None = None) -> dict:
        """Supervised multi-agent run where every sub-agent is a REAL agent (LLM runtime)
        and the Synthesis agent writes the final prose. Select sub-agents by name or tier."""
        all_specs = self.store.list_specs(tenant_id)
        if agent_names:
            specs = [s for s in all_specs if s.name in agent_names]
        elif tier is not None:
            specs = [s for s in all_specs if s.tier == tier]
        else:
            specs = all_specs
        if not specs:
            return {"ok": False, "message": "no agents matched for supervised run"}
        return self.planner.run_supervised(tenant_id, specs, inputs)

    # ----- admin / MCP --------------------------------------------------- #
    def register_mcp(self, server, actor: str = "admin", connect: bool | None = None) -> list[str]:
        return self.registry.register_mcp_server(server, actor, connect=connect)

    def refresh_mcp(self, tenant_id: str, name: str, actor: str = "admin") -> dict:
        return self.registry.refresh_mcp_server(tenant_id, name, actor)

    # ----- real single-agent execution (playground / deploy / invoke) ---- #
    def run_task(self, tenant_id: str, name: str, user_input,
                 approve_side_effecting: bool = False,
                 approved_tools: list[str] | None = None,
                 run_id: str | None = None) -> dict:
        spec = self.store.get_spec(tenant_id, name)
        if spec is None:
            return {"ok": False, "message": f"no deployed agent '{name}' for this tenant"}
        return self.runtime.run(spec, user_input, tenant_id,
                                approve_side_effecting=approve_side_effecting,
                                approved_tools=approved_tools, run_id=run_id)

    def submit_run(self, tenant_id: str, name: str, inputs: dict,
                   approve_side_effecting: bool = False,
                   idempotency_key: str | None = None) -> dict:
        """Enqueue a run on the DURABLE worker queue and return its run_id immediately, so the
        HTTP request thread is not held for the agent's tool loop. The run survives a restart
        (orphan recovery), retries transient failures, and can be cancelled."""
        if self.store.get_spec(tenant_id, name) is None:
            return {"ok": False, "message": f"no deployed agent '{name}' for this tenant"}
        run_id = self.jobs.submit(tenant_id, name, inputs or {}, approve_side_effecting,
                                  idempotency_key=idempotency_key)
        job = self.store.get_job(tenant_id, run_id)
        return {"ok": True, "run_id": run_id, "status": (job or {}).get("status", "queued"),
                "poll": f"/api/runs/{run_id}"}

    def cancel_run(self, tenant_id: str, run_id: str) -> dict:
        """Cancel a queued/running job. Queued jobs stop cleanly; a running attempt finishes but
        is not retried."""
        ok = self.jobs.cancel(tenant_id, run_id)
        return {"ok": ok, "run_id": run_id, "status": "cancelled" if ok else "not_cancellable"}

    def job_status(self, tenant_id: str, run_id: str) -> dict | None:
        return self.store.get_job(tenant_id, run_id)

    # ----- deployment lifecycle ------------------------------------------ #
    def deploy_agent(self, tenant_id: str, name: str, actor: str = "operator",
                     label: str = "default") -> dict:
        """Mark an agent deployed and provision a per-agent invocation token. The token
        authorizes calling ONLY this agent, so each deployed agent is an independently
        callable, credentialed endpoint."""
        if self.store.get_spec_any(tenant_id, name) is None:
            return {"ok": False, "message": "agent not found"}
        self.store.set_enabled(tenant_id, name, True)
        token = auth.new_token()
        self.store.create_deploy_token(token, tenant_id, name, label)
        self.store.audit(tenant_id, actor, "deploy", name, {"label": label})
        return {"ok": True, "name": name, "status": "deployed", "token": token,
                "endpoint": f"/api/agents/{name}/invoke"}

    def undeploy_agent(self, tenant_id: str, name: str, actor: str = "operator") -> dict:
        self.store.set_enabled(tenant_id, name, False)
        revoked = self.store.revoke_deploy_tokens(tenant_id, name)
        self.store.audit(tenant_id, actor, "undeploy", name, {"revoked_tokens": revoked})
        return {"ok": True, "name": name, "status": "undeployed", "revoked_tokens": revoked}

    def resolve_review(self, tenant_id: str, run_id: str, decision: str,
                       reviewer: str = "reviewer", note: str = "") -> dict:
        """Approve or reject a run waiting on human review.

        Approval is a GOVERNED RESUME, never a raw tool fire: the run re-enters the one agent
        kernel with the held tool pre-approved, so the action executes through the same
        `_govern_and_invoke` controls (least-privilege, injection scan, PII redaction, audit,
        risk re-score) and the agent then continues to a governed final answer. Rejection
        cancels the held action and records the reviewer's reason.
        """
        rec = self.store.get_run(tenant_id, run_id)
        if rec is None:
            return {"ok": False, "message": "run not found"}
        pending = rec.get("pending_action") or {}

        # ---- reject: cancel the held action, record the reason + counterfactual ---------- #
        if decision != "approve":
            rec["status"] = "rejected"
            rec["reviewer"] = reviewer
            rec["review_note"] = note
            rec["review_decision"] = "rejected"
            rec["pending_action"] = None
            if pending.get("tool"):
                rec.setdefault("steps", []).append({
                    "idx": len(rec.get("steps", [])), "kind": "review", "tool": pending["tool"],
                    "tool_input": pending.get("args"), "blocked": True, "latency_ms": 0.0,
                    "note": (f"REJECTED by {reviewer}: held action '{pending['tool']}' was NOT "
                             f"executed. {note}").strip()})
            self.store.save_run(rec)
            self.store.audit(tenant_id, reviewer, "reject", run_id,
                             {"agent": rec.get("agent"), "note": note,
                              "cancelled_action": pending.get("tool")})
            return {"ok": True, "run": rec}

        # ---- approve a HELD TOOL: resume through the governed kernel ---------------------- #
        spec = self.store.get_spec(tenant_id, rec.get("agent", ""))
        if pending.get("tool") and spec is not None:
            resumed = self.runtime.run(
                spec, rec.get("input") or {}, tenant_id,
                approved_tools=[pending["tool"]], run_id=run_id)   # same record, tool approved
            resumed["reviewer"] = reviewer
            resumed["review_note"] = note
            resumed["review_decision"] = "approved"
            resumed["approved_tool"] = pending["tool"]
            # if the governed resume finished without pausing again, record the human approval;
            # if it paused again (chained side-effect / high risk), keep needs_review (honest)
            if resumed.get("status") == "success":
                resumed["status"] = "approved"
            self.store.save_run(resumed)
            self.store.audit(tenant_id, reviewer, "approve", run_id,
                             {"agent": rec.get("agent"), "approved_tool": pending["tool"],
                              "resumed_status": resumed.get("status")})
            return {"ok": True, "run": resumed, "resumed": True}

        # ---- approve an OUTPUT/RISK review (no held tool): human signs off on the already
        #      governed output. Re-running would just re-pause, so we don't. ---------------- #
        rec["status"] = "approved"
        rec["reviewer"] = reviewer
        rec["review_note"] = note
        rec["review_decision"] = "approved"
        if pending.get("tool") and spec is None:
            rec["review_note"] = (note + " | NOTE: agent spec unavailable; held action not "
                                  "executed.").strip(" |")
        rec["pending_action"] = None
        self.store.save_run(rec)
        self.store.audit(tenant_id, reviewer, "approve", run_id, {"agent": rec.get("agent")})
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

        # Editing the agent's TOOLS (add/remove native or MCP tools) goes through the SAME
        # validator as creation, so least-privilege, the side-effecting-tool approval gate and the
        # injection-check floor all hold. The picked tools can never widen privilege past the
        # tenant's grantable universe.
        stripped: list[str] = []
        if "tools" in patch and isinstance(patch["tools"], list):
            tpl = self.store.get_template(spec.template) if spec.template else None
            candidate = dict(data)
            candidate["tools"] = list(dict.fromkeys(patch["tools"]))
            new_spec, verdict = validate_spec(
                candidate, tpl, self.registry, tenant_id,
                approve_side_effecting=bool(patch.get("approve_side_effecting", False)),
                approved_tools=patch.get("approved_side_effecting_tools", []))
            if new_spec is None:
                return {"ok": False, "message": "; ".join(verdict.errors) or "invalid tool set"}
            data["tools"] = new_spec.tools
            data["security"] = new_spec.security.model_dump()
            data["guardrails"] = new_spec.guardrails.model_dump()
            stripped = verdict.stripped

        new = AgentSpec.model_validate(data)
        self.store.save_spec(new)
        self.store.audit(tenant_id, actor, "edit", name,
                         {"fields": list(patch.keys()), "stripped": stripped})
        return {"ok": True, "spec": new.model_dump(), "stripped": stripped}

    def grantable_tools(self, tenant_id: str, template_name: str | None = None) -> list[dict]:
        """The tool universe an agent of this template (or a from-scratch agent) may be granted —
        native + the tenant's approved MCP/external tools — enriched for the UI tool picker. This
        is what makes connecting MCP tools to an agent discoverable (Dify-style), not a guess."""
        tpl = self.store.get_template(template_name) if template_name else None
        if tpl is not None:
            names = self.registry.effective_grantable(tpl.grantable_tools, tenant_id)
            default = set(tpl.fixed.get("required_tools", []) or [])
        else:
            names = self.registry.grantable_for_tenant(tenant_id)
            default = set()
        out = []
        for n in dict.fromkeys(names):
            d = self.registry.get(n)
            if d is None:
                continue
            out.append({"name": d.name, "kind": d.kind.value, "description": d.description,
                        "side_effecting": d.side_effecting,
                        "untrusted_output": getattr(d, "untrusted_output", False),
                        "mcp_server": getattr(d, "mcp_server", None),
                        "default": d.name in default})
        return out

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
        """Monitoring rollups, computed in SQL over indexed columns + a LIGHT metrics projection
        (never the full trace blobs). Tenant-scoped (§10)."""
        import time
        now = time.time()
        cutoff = now - days * 86400

        usage = self.store.usage_totals(tenant_id, cutoff)          # SQL SUM(cost), SUM(tokens)
        rows = self.store.trace_metrics_since(tenant_id, cutoff)    # light: metrics blob + columns

        total_prompt = total_completion = total_llm_calls = 0
        total_tool_calls = guardrail_trips = hitl_pauses = 0
        cost_estimated = False
        over_time: dict[str, float] = {}
        agent_calls: dict[str, int] = {}
        agent_runs: dict[str, int] = {}
        for r in rows:
            m = r["metrics"] or {}
            total_prompt += int(m.get("prompt_tokens", 0) or 0)
            total_completion += int(m.get("completion_tokens", 0) or 0)
            total_llm_calls += int(m.get("llm_calls", 0) or 0)
            tcalls = sum((m.get("tool_calls") or {}).values())
            total_tool_calls += tcalls
            guardrail_trips += int(m.get("guardrail_trips", 0) or 0)
            hitl_pauses += int(m.get("hitl_pauses", 0) or 0)
            cost_estimated = cost_estimated or bool(m.get("cost_estimated"))
            day = time.strftime("%Y-%m-%d", time.localtime(r["ts"]))
            over_time[day] = round(over_time.get(day, 0.0) + r["cost_usd"], 6)
            a = r["agent"] or "?"
            agent_calls[a] = agent_calls.get(a, 0) + tcalls
            agent_runs[a] = agent_runs.get(a, 0) + 1

        names = sorted(set(list(agent_calls) + list(agent_runs)))
        breakdown = [{"agent": a, "tool_calls": agent_calls.get(a, 0),
                      "runs": agent_runs.get(a, 0)} for a in names]
        top = sorted(breakdown, key=lambda x: x["tool_calls"], reverse=True)[:6]
        over = [{"date": d, "credits": c} for d, c in sorted(over_time.items())]

        return {
            "totals": {
                "credits": round(usage["cost_usd"], 4),
                "tool_calls": total_tool_calls,
                "active_agents": len(self.store.list_specs(tenant_id)),
                "runs": usage["runs"],
                "guardrail_trips": guardrail_trips,
                "hitl_pauses": hitl_pauses,
                "injection_blocks": 0,
                "tokens": usage["tokens"],
                "prompt_tokens": total_prompt,
                "completion_tokens": total_completion,
                "llm_calls": total_llm_calls,
                "cost_estimated": cost_estimated,
                "cost_basis": ("real LLM usage" if total_llm_calls and not cost_estimated
                               else ("estimated" if cost_estimated else "no LLM usage yet")),
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
