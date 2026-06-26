"""
Planner / supervisor (§1) — LangGraph-style supervisor pattern.

The planner emits a structured plan (a DAG of agent calls), a router dispatches to agent nodes,
and all nodes share ONE state object (the blackboard). Ordering is derived from the agents'
declared dependencies (topological sort). HITL pauses surface as an interrupt the caller resumes.
"""
from __future__ import annotations

from .blackboard import Blackboard
from .factory import Factory
from .middleware import GuardrailTrip, HumanReviewRequired, ToolDenied
from .observability import Tracer
from .schemas import AgentSpec
from .store import Store


class Planner:
    def __init__(self, factory: Factory, store: Store):
        self.factory = factory
        self.store = store

    # ----- planning: topological DAG over declared dependencies --------- #
    def plan(self, specs: list[AgentSpec]) -> list[str]:
        by_name = {}
        for s in specs:
            by_name[s.name] = s
            by_name[s.template or s.name] = s  # allow dependency-by-template-name
        order, visited, temp = [], set(), set()

        def visit(name: str):
            if name in visited:
                return
            if name in temp:
                return  # tolerate soft cycles at plan time
            spec = by_name.get(name)
            if spec is None:
                return
            temp.add(name)
            for dep in spec.depends_on:
                if dep.agent in by_name:
                    visit(dep.agent)
            temp.discard(name)
            if spec.name not in visited:
                visited.add(spec.name)
                order.append(spec.name)

        for s in specs:
            visit(s.name)
        return order

    # ----- run the DAG -------------------------------------------------- #
    def run(self, tenant_id: str, specs: list[AgentSpec], inputs: dict | None = None) -> dict:
        tracer = Tracer(tenant_id)
        blackboard = Blackboard(tenant_id=tenant_id)
        order = self.plan(specs)
        by_name = {s.name: s for s in specs}

        executed, hitl_pause, error = [], None, None
        with tracer.start("planner:dispatch", "planner", plan=order):
            for name in order:
                spec = by_name[name]
                node = self.factory.build(spec)            # rebuild from spec every run
                try:
                    node.run(blackboard, tracer, tracer.trace_id, inputs)
                    executed.append(name)
                except HumanReviewRequired as h:
                    hitl_pause = h.payload
                    executed.append(name)
                    break                                   # interrupt: wait for human
                except (GuardrailTrip, ToolDenied) as g:
                    error = {"agent": name, "type": type(g).__name__, "detail": str(g)}
                    tracer.metrics["errors"] += 1
                    break

        trace = tracer.finalize()
        trace_dict = trace.to_dict()
        trace_dict["executed"] = executed
        trace_dict["hitl_pause"] = hitl_pause
        trace_dict["error"] = error
        trace_dict["blackboard"] = blackboard.snapshot()
        self.store.save_trace(trace.trace_id, tenant_id, trace_dict)
        return trace_dict
