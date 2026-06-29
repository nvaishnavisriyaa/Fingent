"""
Observability (§9) — automatic tracing for every run.

One trace_id per run. Every agent step, tool call (incl. web/MCP), and HITL pause is a span.
This is an OpenTelemetry-style in-process tracer; swap `emit()` for an OTel/LangSmith exporter
in production without touching agent code. Agent authors write zero observability code — the
middleware (see middleware.py) opens/closes spans around every node and tool call.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field

# --------------------------------------------------------------------------- #
# Model pricing — USD per 1,000 tokens. Cost is computed from REAL usage returned
# by the provider (see llm.py) priced against this table. Override/extend via the
# FINGENT_MODEL_PRICING env var (JSON: {"model": {"in": <usd_per_1k>, "out": <usd_per_1k>}}).
# A model not in the table is priced with FALLBACK and flagged estimated, never silently faked.
# --------------------------------------------------------------------------- #
_DEFAULT_PRICING: dict[str, dict] = {
    # Groq list prices (USD / 1M tokens) -> per 1k. Update as provider prices change.
    "llama-3.3-70b-versatile": {"in": 0.00059, "out": 0.00079},
    "openai/gpt-oss-120b": {"in": 0.00015, "out": 0.00075},
    "openai/gpt-oss-20b": {"in": 0.00010, "out": 0.00050},
    "llama-3.1-8b-instant": {"in": 0.00005, "out": 0.00008},
}
_FALLBACK_PRICE = {"in": 0.00050, "out": 0.00050}   # used + flagged when a model is unpriced


def _load_pricing() -> dict[str, dict]:
    table = dict(_DEFAULT_PRICING)
    try:
        override = json.loads(os.getenv("FINGENT_MODEL_PRICING", "") or "{}")
        for k, v in override.items():
            table[k] = {"in": float(v.get("in", 0)), "out": float(v.get("out", 0))}
    except Exception:  # noqa: BLE001 - bad override must not break tracing
        pass
    return table


def price_for(model: str) -> tuple[dict, bool]:
    """Return (price_per_1k, is_fallback) for a model id."""
    table = _load_pricing()
    if model in table:
        return table[model], False
    # tolerate provider prefixes / suffixes (e.g. versioned ids)
    for k, v in table.items():
        if model and (model.startswith(k) or k in model):
            return v, False
    return _FALLBACK_PRICE, True


@dataclass
class Span:
    name: str
    kind: str                      # "agent" | "tool" | "hitl" | "guardrail" | "planner"
    start: float
    end: float | None = None
    attrs: dict = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)

    def close(self, **attrs) -> None:
        self.end = time.time()
        self.attrs.update(attrs)

    @property
    def latency_ms(self) -> float:
        return round(((self.end or time.time()) - self.start) * 1000, 2)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "latency_ms": self.latency_ms,
            "attrs": self.attrs,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class Trace:
    trace_id: str
    tenant_id: str
    root: Span
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "trace_id": self.trace_id,
            "tenant_id": self.tenant_id,
            "metrics": self.metrics,
            "spans": self.root.to_dict(),
        }


class Tracer:
    """A tracer scoped to one run. Spans nest via the `current` pointer stack."""

    def __init__(self, tenant_id: str) -> None:
        self.trace_id = "tr_" + uuid.uuid4().hex[:12]
        self.tenant_id = tenant_id
        self.root = Span(name="run", kind="planner", start=time.time())
        self._stack: list[Span] = [self.root]
        self.metrics = {
            "latency_ms": 0.0,
            # token usage — REAL counts captured from each provider response (llm.py)
            "tokens": 0, "prompt_tokens": 0, "completion_tokens": 0,
            "llm_calls": 0,
            # cost priced from real usage against the model price table
            "cost_usd": 0.0,
            "cost_estimated": False,      # True if any cost used the fallback/estimated path
            "usage_source": "none",       # none | actual | estimated
            "model_usage": {},            # per-model {prompt, completion, cost_usd}
            "tool_calls": {}, "guardrail_trips": 0, "retries": 0,
            "errors": 0, "hitl_pauses": 0,
        }

    def span(self, name: str, kind: str, **attrs) -> Span:
        s = Span(name=name, kind=kind, start=time.time(), attrs=attrs)
        self._stack[-1].children.append(s)
        return s

    class _Ctx:
        def __init__(self, tracer: "Tracer", span: Span):
            self.tracer, self.span = tracer, span

        def __enter__(self) -> Span:
            self.tracer._stack.append(self.span)
            return self.span

        def __exit__(self, exc_type, exc, tb):
            if exc is not None:
                self.span.attrs["error"] = repr(exc)
                self.tracer.metrics["errors"] += 1
            if self.span.end is None:
                self.span.close()
            self.tracer._stack.pop()
            return False

    def start(self, name: str, kind: str, **attrs) -> "Tracer._Ctx":
        return Tracer._Ctx(self, self.span(name, kind, **attrs))

    def record_tool(self, kind: str) -> None:
        self.metrics["tool_calls"][kind] = self.metrics["tool_calls"].get(kind, 0) + 1

    def add_usage(self, prompt_tokens: int, completion_tokens: int, model: str,
                  estimated: bool = False) -> None:
        """Record REAL token usage from one LLM call and price it from the model table.

        `prompt_tokens`/`completion_tokens` come straight from the provider response
        (`response.usage`). `estimated=True` means the provider returned no usage and the
        counts were approximated — in that case the cost is flagged estimated so the UI can
        say so instead of presenting a fabricated figure.
        """
        prompt_tokens = int(prompt_tokens or 0)
        completion_tokens = int(completion_tokens or 0)
        total = prompt_tokens + completion_tokens
        price, fallback = price_for(model or "")
        cost = round(prompt_tokens / 1000 * price["in"]
                     + completion_tokens / 1000 * price["out"], 6)

        m = self.metrics
        m["llm_calls"] += 1
        m["tokens"] += total
        m["prompt_tokens"] += prompt_tokens
        m["completion_tokens"] += completion_tokens
        m["cost_usd"] = round(m["cost_usd"] + cost, 6)
        if estimated or fallback:
            m["cost_estimated"] = True
        # usage_source: actual unless we only ever estimated
        if not estimated:
            m["usage_source"] = "actual"
        elif m["usage_source"] == "none":
            m["usage_source"] = "estimated"
        mu = m["model_usage"].setdefault(
            model or "unknown", {"prompt": 0, "completion": 0, "cost_usd": 0.0,
                                 "estimated": False})
        mu["prompt"] += prompt_tokens
        mu["completion"] += completion_tokens
        mu["cost_usd"] = round(mu["cost_usd"] + cost, 6)
        mu["estimated"] = mu["estimated"] or estimated or fallback

    def add_tokens(self, n: int, cost_per_1k: float = 0.0005) -> None:
        """Deprecated shim. Real metrics come from add_usage(); this only exists so any
        legacy caller does not crash. Counts as ESTIMATED completion tokens, never priced
        as if it were real measured usage."""
        self.add_usage(0, int(n or 0), model="", estimated=True)

    def finalize(self) -> Trace:
        self.root.close()
        self.metrics["latency_ms"] = self.root.latency_ms
        return Trace(self.trace_id, self.tenant_id, self.root, self.metrics)
