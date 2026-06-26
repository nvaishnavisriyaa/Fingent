"""
Observability (§9) — automatic tracing for every run.

One trace_id per run. Every agent step, tool call (incl. web/MCP), and HITL pause is a span.
This is an OpenTelemetry-style in-process tracer; swap `emit()` for an OTel/LangSmith exporter
in production without touching agent code. Agent authors write zero observability code — the
middleware (see middleware.py) opens/closes spans around every node and tool call.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field


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
            "latency_ms": 0.0, "tokens": 0, "cost_usd": 0.0,
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

    def add_tokens(self, n: int, cost_per_1k: float = 0.0005) -> None:
        self.metrics["tokens"] += n
        self.metrics["cost_usd"] = round(
            self.metrics["cost_usd"] + n / 1000 * cost_per_1k, 6
        )

    def finalize(self) -> Trace:
        self.root.close()
        self.metrics["latency_ms"] = self.root.latency_ms
        return Trace(self.trace_id, self.tenant_id, self.root, self.metrics)
