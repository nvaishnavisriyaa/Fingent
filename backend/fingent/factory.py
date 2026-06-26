"""
Factory (§1, §3 step 7) — instantiate ANY agent from its saved spec.

One factory, one runtime. The factory binds real tools from the registry, wires blackboard
reads/writes, and attaches guardrail + security + logging + observability middleware. The live
agent is rebuilt from the spec on every run (spec = single source of truth), so editing the spec
changes the agent with no code change.
"""
from __future__ import annotations

from .blackboard import Blackboard, MemoryAccessDenied
from .middleware import (
    Budget, HumanReviewRequired, ToolDenied, compliance_overseer, redact_pii, scan_untrusted,
)
from .observability import Tracer
from .registry import ToolRegistry
from .schemas import AgentSpec, ToolKind
from .store import Store
from .vault import vault


class AgentNode:
    """A live, runnable agent built from an AgentSpec."""

    def __init__(self, spec: AgentSpec, registry: ToolRegistry, store: Store):
        self.spec = spec
        self.registry = registry
        self.store = store

    def _call_tool(self, tool_name: str, tracer: Tracer, trace_id: str, **kwargs):
        # TOOL GUARDRAIL: only allowed tools may be invoked (§8, §10)
        if tool_name not in self.spec.security.allowed_tools:
            self.store.log_run_step(trace_id, self.spec.security.tenant_id, {
                "agent": self.spec.name, "event": "tool_denied", "tool": tool_name,
                "reason": "not in allow-list"})
            self.store.audit(self.spec.security.tenant_id, self.spec.name,
                             "tool_denied", tool_name, "least-privilege")
            raise ToolDenied(f"agent '{self.spec.name}' may not call '{tool_name}'")

        desc = self.registry.get(tool_name)
        fn = self.registry.callable(tool_name)
        with tracer.start(f"tool:{tool_name}", "tool", tool_kind=desc.kind.value) as span:
            tracer.record_tool(desc.kind.value)
            secrets = vault.resolve_all(desc.secrets_ref) if desc.secrets_ref else {}
            result = fn(**kwargs)
            span.close(secrets_used=list(secrets.keys()), untrusted=bool(desc.untrusted_output))

        # INPUT GUARDRAIL on untrusted tool output: injection-scan as DATA, never instructions
        if desc.untrusted_output or desc.kind in (ToolKind.WEB_SEARCH, ToolKind.MCP,
                                                  ToolKind.EXTERNAL_API):
            hits = scan_untrusted(result, tool_name)
            if hits and self.spec.guardrails.injection_check:
                tracer.metrics["guardrail_trips"] += 1
                self.store.log_run_step(trace_id, self.spec.security.tenant_id, {
                    "agent": self.spec.name, "event": "injection_blocked", "tool": tool_name,
                    "signatures": hits})
                return {"_quarantined": True, "tool": tool_name,
                        "reason": "prompt_injection_detected", "signatures": hits}
        return result

    def run(self, blackboard: Blackboard, tracer: Tracer, trace_id: str,
            inputs: dict | None = None) -> dict:
        spec = self.spec
        tenant = spec.security.tenant_id
        budget = Budget(spec.guardrails)

        with tracer.start(f"agent:{spec.name}", "agent", template=spec.template,
                          tier=spec.tier) as span:
            ctx = {**(inputs or {})}
            for prior in spec.reads:
                try:
                    val = blackboard.read(prior, spec.name, spec.security.memory_read)
                    if val is not None:
                        ctx[prior] = val
                except MemoryAccessDenied as e:
                    self.store.log_run_step(trace_id, tenant, {
                        "agent": spec.name, "event": "memory_denied", "detail": str(e)})

            if spec.guardrails.input_pii_check:
                _, pii = redact_pii(str(ctx))
                if pii:
                    span.attrs["pii_redacted"] = pii

            outputs: dict = {}
            for tool in spec.tools:
                budget.step(tokens=350)
                tracer.add_tokens(350)
                outputs[tool] = self._call_tool(tool, tracer, trace_id,
                                                **self._tool_args(tool, ctx))

            tracer.add_tokens(500)
            result = {"agent": spec.name, "summary": f"{spec.name} completed", "outputs": outputs}

            wkey = spec.writes[0] if spec.writes else f"{spec.name}.out"
            try:
                blackboard.write(wkey, result, spec.name, spec.security.memory_write)
            except MemoryAccessDenied as e:
                self.store.log_run_step(trace_id, tenant, {
                    "agent": spec.name, "event": "memory_write_denied", "detail": str(e)})

            review = compliance_overseer(result)
            span.attrs["compliance"] = review
            if spec.guardrails.output_review_required or spec.requires_human_review \
                    or review["verdict"] == "BLOCK":
                tracer.metrics["hitl_pauses"] += 1
                with tracer.start("hitl:pause", "hitl", verdict=review["verdict"]):
                    pass
                payload = {"agent": spec.name, "trace_id": trace_id,
                           "compliance": review, "result": result}
                self.store.log_run_step(trace_id, tenant, {
                    "agent": spec.name, "event": "hitl_pause", "compliance": review})
                raise HumanReviewRequired(payload)

            self.store.log_run_step(trace_id, tenant, {
                "agent": spec.name, "event": "completed",
                "tools": [{"name": t, "kind": self.registry.get(t).kind.value}
                          for t in spec.tools],
                "compliance": review})
            return result

    @staticmethod
    def _tool_args(tool: str, ctx: dict) -> dict:
        company = ctx.get("company", "Acme Corp")
        person = ctx.get("name", "Jane Doe")
        mapping = {
            "edgar_search": {"query": company}, "news_monitor": {"company": company},
            "enrich_company": {"company": company}, "find_persona": {"company": company},
            "resolve_contact": {"name": person}, "web_search": {"query": company},
            "ofac_screen": {"name": person}, "adverse_media_search": {"name": person},
            "pep_check": {"name": person}, "ocr_extract": {"document": "financials.pdf"},
        }
        return mapping.get(tool, {})


class Factory:
    def __init__(self, registry: ToolRegistry, store: Store):
        self.registry = registry
        self.store = store

    def build(self, spec: AgentSpec) -> AgentNode:
        return AgentNode(spec, self.registry, self.store)
