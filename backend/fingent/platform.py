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
from .factory import Factory
from .planner import Planner
from .registry import ToolRegistry
from .schemas import AgentSpec, AgentTemplate, CreateAgentRequest
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

        # steps 3-4: LLM compiles a candidate, spec validator disposes
        compiled = self.compiler.compile(req, tpl)
        compiler_log = {
            "agent": req.answers.get("name", "?"),
            "used_llm": compiled.used_llm, "attempts": compiled.attempts,
            "ok": compiled.ok, "candidate_specs": compiled.candidate_specs,
            "verdicts": [v.model_dump() for v in compiled.verdicts],
            "message": compiled.message,
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
            fixed={"base_role": spec.role_prompt.split(chr(10))[0]},
            parameters=[], default_depends_on=spec.depends_on,
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

    def resume_hitl(self, tenant_id: str, trace_id: str, actor: str,
                    decision: str, diff: dict | None = None) -> dict:
        self.store.log_hitl(tenant_id, trace_id, actor, decision, diff or {})
        self.store.audit(tenant_id, actor, decision, trace_id, diff or {})
        return {"ok": True, "decision": decision, "trace_id": trace_id}
