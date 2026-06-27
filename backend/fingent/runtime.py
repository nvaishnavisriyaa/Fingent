"""
Agent runtime — runs ONE agent on a real task (the core of real execution).

This is a governed tool-use loop. The agent observes the task, decides the NEXT action
(a real Llama call when GROQ_API_KEY is set — mode="llm"; otherwise a transparent
deterministic engine — mode="demo"), invokes the REAL tool function from the registry,
observes the result, and repeats until it produces a final output.

Every decision is governed by the same controls the rest of the platform uses:
  * least-privilege allow-list  — the loop may only call security.allowed_tools;
  * prompt-injection scan        — untrusted tool output is quarantined, never acted on;
  * PII redaction                — hard identifiers stripped from tool output;
  * side-effect HITL gate        — write/send/pay tools pause for human approval;
  * cost/loop budget             — max_steps / tokens / timeout;
  * risk scoring                 — high-risk or flagged runs route to human review.

The result is a persisted RunRecord with an explicit status:
  success | failed | blocked | needs_review | approved | rejected
"""
from __future__ import annotations

import json
import os
import time
import uuid

from .middleware import (
    Budget, GuardrailTrip, compliance_overseer, detect_injection,
    redact_obj, redact_pii, scan_untrusted,
)
from .observability import Tracer
from .registry import ToolRegistry
from .schemas import AgentSpec, RunRecord, RunStep
from .store import Store

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
_UNTRUSTED_KINDS = {"web_search", "mcp", "external_api"}


class AgentRuntime:
    def __init__(self, registry: ToolRegistry, store: Store) -> None:
        self.registry = registry
        self.store = store

    # ------------------------------------------------------------------ #
    def run(self, spec: AgentSpec, user_input, tenant_id: str,
            approve_side_effecting: bool = False,
            approved_tools: list[str] | None = None,
            resume: dict | None = None) -> dict:
        approved = set(approved_tools or [])
        if not isinstance(user_input, dict):
            user_input = {"input": user_input}

        tracer = Tracer(tenant_id)
        run_id = "run_" + uuid.uuid4().hex[:12]
        start = time.monotonic()
        budget = Budget(spec.guardrails)
        steps: list[RunStep] = []
        flags: list[str] = []
        observations: list[dict] = []
        pending = None
        status = "success"
        output = None

        api_key = os.getenv("GROQ_API_KEY", "")
        mode = "llm" if api_key else "demo"
        descriptors = [d for d in (self.registry.get(t) for t in spec.security.allowed_tools) if d]

        # ---- input safety -------------------------------------------------- #
        in_text = json.dumps(user_input, default=str)
        _, in_pii = redact_pii(in_text)
        if in_pii:
            flags.append("input_pii:" + ",".join(in_pii))
        if detect_injection(in_text):
            flags.append("input_injection_attempt")
            steps.append(RunStep(idx=len(steps), kind="guardrail",
                                 note="prompt-injection pattern detected in task input"))

        with tracer.start(f"run:{spec.name}", "agent", agent=spec.name, mode=mode):
            try:
                for _ in range(spec.guardrails.max_steps):
                    budget.step(tokens=300)
                    tracer.add_tokens(300)
                    action = (self._decide_llm(spec, user_input, descriptors, observations, api_key)
                              if mode == "llm"
                              else self._decide_demo(spec, user_input, descriptors, observations))

                    if action.get("finish"):
                        output = action.get("output")
                        steps.append(RunStep(idx=len(steps), kind="output", note="final answer"))
                        break

                    tool = action.get("tool")
                    args = action.get("args") or {}
                    desc = self.registry.get(tool)

                    # least privilege
                    if desc is None or tool not in spec.security.allowed_tools:
                        flags.append(f"unauthorized_tool:{tool}")
                        steps.append(RunStep(idx=len(steps), kind="guardrail", tool=str(tool),
                                             blocked=True, note="tool not in allow-list — blocked"))
                        self.store.audit(tenant_id, spec.name, "tool_denied", str(tool),
                                         "runtime allow-list")
                        status = "blocked"
                        break

                    # side-effect gate -> hold for human approval
                    if desc.side_effecting and not (approve_side_effecting or tool in approved):
                        pending = {"tool": tool, "args": args,
                                   "reason": "side-effecting action requires approval"}
                        flags.append(f"side_effect_pending:{tool}")
                        steps.append(RunStep(idx=len(steps), kind="review", tool=tool,
                                             tool_input=args,
                                             note="side-effecting tool held for human approval"))
                        tracer.metrics["hitl_pauses"] += 1
                        status = "needs_review"
                        break

                    # invoke the REAL tool
                    t0 = time.monotonic()
                    with tracer.start(f"tool:{tool}", "tool", tool_kind=desc.kind.value):
                        tracer.record_tool(desc.kind.value)
                        fn = self.registry.callable(tool)
                        try:
                            result = fn(**args) if fn else {"error": "no callable bound"}
                        except Exception as e:  # noqa: BLE001
                            flags.append(f"tool_error:{tool}")
                            steps.append(RunStep(idx=len(steps), kind="error", tool=tool,
                                                 tool_input=args, note=str(e)))
                            observations.append({"tool": tool, "error": str(e)})
                            continue
                    lat = (time.monotonic() - t0) * 1000

                    # injection scan on untrusted output
                    if desc.untrusted_output or desc.kind.value in _UNTRUSTED_KINDS:
                        hits = scan_untrusted(result, tool)
                        if hits and spec.guardrails.injection_check:
                            tracer.metrics["guardrail_trips"] += 1
                            flags.append(f"injection_blocked:{tool}")
                            result = {"_quarantined": True, "tool": tool, "signatures": hits}
                    # redact PII from tool output before it is recorded/returned
                    if spec.guardrails.input_pii_check:
                        result = redact_obj(result)

                    steps.append(RunStep(idx=len(steps), kind="tool", tool=tool, tool_input=args,
                                         tool_output=result, latency_ms=round(lat, 2)))
                    observations.append({"tool": tool, "output": result})
            except GuardrailTrip as g:
                flags.append("budget_exceeded")
                steps.append(RunStep(idx=len(steps), kind="guardrail", blocked=True, note=str(g)))
                status = "blocked"

            if output is None and status == "success":
                output = self._compose_output(spec, observations)

            # output safety + compliance overseer
            review = compliance_overseer({"output": output, "observations": observations})
            if review["verdict"] == "BLOCK":
                flags.append("compliance_block")
                status = "blocked"
            elif review["flags"] or review["leaked_pii"]:
                flags.append("compliance_annotate")

            if spec.requires_human_review and status == "success":
                flags.append("agent_requires_review")
                status = "needs_review"

        # ---- risk score + final status routing ----------------------------- #
        score = self._risk_score(spec, flags)
        level = "high" if score >= 60 else ("medium" if score >= 30 else "low")
        if level == "high" and status == "success":
            flags.append("high_risk_auto_review")
            status = "needs_review"

        trace = tracer.finalize()
        td = trace.to_dict()
        td["run_id"] = run_id
        td["agent"] = spec.name
        td["executed"] = [spec.name]
        td["status"] = status
        self.store.save_trace(trace.trace_id, tenant_id, td)

        rec = RunRecord(
            id=run_id, tenant_id=tenant_id, agent=spec.name, trace_id=trace.trace_id, mode=mode,
            input=user_input, status=status, steps=steps, output=output,
            risk_score=score, risk_level=level, risk_flags=flags, pending_action=pending,
            duration_ms=round((time.monotonic() - start) * 1000, 2), ts=time.time(),
        ).model_dump()
        self.store.save_run(rec)
        self.store.audit(tenant_id, spec.name, "run", run_id,
                         {"status": status, "risk": level, "mode": mode})
        return rec

    # ----- decision engines -------------------------------------------- #
    def _decide_demo(self, spec, user_input, descriptors, observations) -> dict:
        """Transparent deterministic engine: call each allowed tool once, then finish.
        Clearly labelled mode='demo' — swap GROQ_API_KEY in for a real reasoning loop."""
        called = {o.get("tool") for o in observations}
        for d in descriptors:
            if d.name in called:
                continue
            return {"tool": d.name, "args": self._demo_args(d.name, user_input, observations)}
        return {"finish": True, "output": self._compose_output(spec, observations)}

    def _decide_llm(self, spec, user_input, descriptors, observations, api_key) -> dict:
        import requests
        tools = [{"name": d.name, "description": d.description,
                  "side_effecting": d.side_effecting} for d in descriptors]
        system = (
            f"You are '{spec.name}', an AI agent for financial services.\n"
            f"Purpose: {spec.purpose or spec.role_prompt}\n"
            f"Instructions: {spec.instructions or '(none)'}\n"
            f"You may ONLY use these tools (never invent others): {json.dumps(tools)}\n"
            f"Desired output format: {spec.output_format}.\n"
            "Decide the SINGLE next action. Reply with ONE JSON object and nothing else:\n"
            '  {"tool": "<tool_name>", "args": {...}}  to call a tool, or\n'
            '  {"finish": true, "output": <final answer>}  when the task is complete.'
        )
        user = (f"Task input: {json.dumps(user_input, default=str)}\n"
                f"Observations so far: {json.dumps(observations, default=str)[:3000]}\n"
                "Next action JSON:")
        resp = requests.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json={"model": os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
                  "messages": [{"role": "system", "content": system},
                               {"role": "user", "content": user}],
                  "temperature": 0.1, "response_format": {"type": "json_object"}},
            timeout=30,
        )
        resp.raise_for_status()
        return json.loads(resp.json()["choices"][0]["message"]["content"])

    # ----- helpers ----------------------------------------------------- #
    @staticmethod
    def _demo_args(tool, user_input, observations) -> dict:
        company = user_input.get("company") or "Acme Corp"
        person = user_input.get("name") or user_input.get("person") or "Jane Doe"
        financials = None
        for o in observations:
            out = o.get("output")
            if isinstance(out, dict) and {"revenue", "ebitda"} <= set(out):
                financials = out
        ocr = next((o["output"] for o in observations
                    if o.get("tool") == "ocr_extract" and isinstance(o.get("output"), dict)), None)
        text = ocr.get("text", "") if isinstance(ocr, dict) else ""
        ctx = {o.get("tool"): {"outputs": {o.get("tool"): o.get("output")}}
               for o in observations if o.get("output") is not None}
        return {
            "edgar_search": {"query": company}, "news_monitor": {"company": company},
            "enrich_company": {"company": company}, "find_persona": {"company": company},
            "resolve_contact": {"name": person}, "web_search": {"query": company},
            "ofac_screen": {"name": person}, "adverse_media_search": {"name": person},
            "pep_check": {"name": person}, "ocr_extract": {"document": "financials.pdf"},
            "parse_financials": {"text": text}, "compute_ratios": {"financials": financials},
            "anomaly_detect": {"transactions": user_input.get("transactions", [])},
            "reg_feed_ingest": {}, "compose_summary": {"context": ctx},
            "risk_score": {"ratios": next((o["output"] for o in observations
                           if o.get("tool") == "compute_ratios" and isinstance(o.get("output"), dict)), None),
                           "financials": financials},
            "compliance_check": {"payload": {o.get("tool"): o.get("output") for o in observations}},
            "identity_verify": {"name": person, "id_number": user_input.get("id_number", ""),
                                "document": user_input.get("document", "")},
            "account_lookup": {"account_id": user_input.get("account_id", "")},
        }.get(tool, {})

    @staticmethod
    def _compose_output(spec, observations) -> dict:
        for o in reversed(observations):
            if o.get("tool") == "compose_summary":
                return o.get("output")
        outs = {o["tool"]: o.get("output") for o in observations if "tool" in o}
        return {"summary": f"{spec.name} completed {len(observations)} tool call(s).",
                "outputs": outs}

    @staticmethod
    def _risk_score(spec, flags) -> int:
        score = {"low": 10, "medium": 25, "high": 45}.get(spec.risk_level, 25)
        for f in flags:
            if f.startswith("side_effect"):
                score += 35
            elif f.startswith("unauthorized"):
                score += 40
            elif f == "compliance_block":
                score += 40
            elif "injection" in f:
                score += 30
            elif "pii" in f:
                score += 15
            elif f.startswith("tool_error"):
                score += 10
            elif f == "agent_requires_review":
                score += 20
            elif f == "budget_exceeded":
                score += 25
        return min(100, score)
