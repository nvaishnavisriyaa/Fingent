"""
Planner / supervisor (§1) — a real planning node over ONE governed kernel.

The supervisor does NOT merely topologically sort hardcoded dependencies. Given a GOAL and the
set of available sub-agents, it asks the model to DECOMPOSE the goal into a task graph at runtime:
which sub-agents to invoke, in what order, with a focused subtask for each, and how they depend on
one another. The model PROPOSES the plan; a deterministic validator DISPOSES — every step must
name a real, available agent; dependency edges are reduced to a DAG (forward edges only, so cycles
are impossible by construction); a synthesis/summary agent is forced last; and an unusable plan
falls back to the dependency-derived order. As intermediate results arrive the supervisor can
ADAPT — revising the not-yet-executed tail of the plan (bounded by a replan budget).

Every sub-agent still runs through the SINGLE agent runtime (`AgentRuntime.run_node`) — the same
governed loop and per-tool governance (least-privilege, side-effect HITL, injection scan, PII
redaction) used everywhere else. The planner decides WHAT to run and with WHICH subtask; the
kernel governs HOW each run executes. Inter-agent data still flows only along each agent's
validated memory scope (the blackboard ACL), so dynamic planning never widens least-privilege.

When no model is configured the planner degrades to a deterministic dependency toposort (clearly
flagged `plan_mode="dependency"`), so the platform still runs fully offline and existing behaviour
is preserved.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .blackboard import Blackboard
from .llm import LlmProvider
from .observability import Tracer
from .schemas import AgentSpec
from .store import Store

_MAX_REPLANS = 2   # how many times intermediate results may revise the remaining plan (budget)


# --------------------------------------------------------------------------- #
# Plan data model — a runtime task graph (not the static depends_on of a spec)
# --------------------------------------------------------------------------- #
@dataclass
class PlanStep:
    agent: str
    task: str = ""                              # decomposed subtask for this agent ("" = none)
    depends_on: list = field(default_factory=list)        # indices of earlier steps it needs
    reason: str = ""                            # why this agent / why here

    def to_dict(self) -> dict:
        return {"agent": self.agent, "task": self.task,
                "depends_on": list(self.depends_on), "reason": self.reason}


@dataclass
class PlanGraph:
    steps: list
    mode: str                                   # "llm" | "dependency"
    goal: str = ""
    reasoning: str = ""
    notes: list = field(default_factory=list)


PLAN_SYSTEM_PROMPT = """You are Fingent's multi-agent SUPERVISOR ("the planner") for financial \
services. Decompose the GOAL into an ordered task graph using ONLY the available agents below.

Return a SINGLE JSON object and NOTHING else:
{{"reasoning": "<one or two sentences on your approach>",
  "plan": [
    {{"agent": "<an agent name from the available list>",
      "task": "<the specific, self-contained subtask this agent should accomplish toward the goal>",
      "depends_on": [<indices of EARLIER plan steps whose output this step needs>],
      "reason": "<why this agent, and why at this point>"}}
  ]}}

RULES:
- Use ONLY agent names from the AVAILABLE AGENTS list. NEVER invent an agent or a tool.
- Pick the FEWEST agents that actually achieve the goal — do not invoke irrelevant agents.
- Order the steps so every dependency appears BEFORE the step that needs it (a DAG).
- Write each `task` concretely, for that specific agent, in terms of the goal.
- If a synthesis / summary agent is available, make it the FINAL step so it can compose the answer.

GOAL: {goal}
AVAILABLE AGENTS (name, purpose, tools): {agents}

Return ONLY the JSON object."""


REPLAN_SYSTEM_PROMPT = """You are Fingent's multi-agent SUPERVISOR adapting a RUNNING plan. Given \
the GOAL, what has executed so far (with short result digests), and the agents still queued, decide \
whether to keep the rest of the plan or revise it based on the intermediate results.

Return a SINGLE JSON object and NOTHING else, either:
  {{"keep": true}}                                  # proceed with the queued steps unchanged
or
  {{"keep": false, "revised_remaining": [
      {{"agent": "<available agent name>", "task": "<subtask>", "reason": "<why>"}} ]}}

RULES:
- Use ONLY agent names from the AVAILABLE AGENTS list; never invent one.
- Do NOT re-run an already-completed agent.
- Keep it minimal — only revise if the results so far warrant it.
- Keep any synthesis / summary agent LAST.

GOAL: {goal}
AVAILABLE AGENTS: {agents}
COMPLETED (agent -> status / result digest): {completed}
REMAINING (queued agent names): {remaining}

Return ONLY the JSON object."""


class Planner:
    def __init__(self, store: Store, runtime) -> None:
        self.store = store
        self.runtime = runtime  # AgentRuntime — the one governed kernel every path shares

    # ===================================================================== #
    # Planning
    # ===================================================================== #
    def plan(self, specs: list) -> list:
        """Deterministic dependency toposort over declared depends_on — the offline fallback and
        the validator's reference order. Kept stable so behaviour with no model is unchanged."""
        by_name = self._by_name(specs)
        order, visited, temp = [], set(), set()

        def visit(name: str):
            if name in visited or name in temp:
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

    def plan_graph(self, specs: list, goal: str = "", inputs: dict | None = None) -> PlanGraph:
        """Build the runtime task graph. With a model configured and more than one agent, the LLM
        decomposes the goal into a plan (validated deterministically); otherwise the dependency
        toposort is used. Either way the result is a single PlanGraph the executor runs."""
        provider = LlmProvider()
        if provider.enabled and len(specs) > 1:
            try:
                g = self._llm_plan(provider, specs, goal, inputs or {})
            except Exception:  # noqa: BLE001 — a planning failure must never break the run
                g = None
            if g is not None and g.steps:
                return g
        return self._dependency_graph(specs, goal)

    def _dependency_graph(self, specs: list, goal: str = "") -> PlanGraph:
        order = self.plan(specs)
        by_name = self._by_name(specs)
        index = {name: i for i, name in enumerate(order)}
        steps = []
        for i, name in enumerate(order):
            spec = by_name[name]
            deps = []
            for d in spec.depends_on:
                dep_spec = by_name.get(d.agent)
                if dep_spec and dep_spec.name in index and index[dep_spec.name] < i:
                    deps.append(index[dep_spec.name])
            reason = (spec.purpose or (spec.role_prompt or "").split("\n")[0] or "")[:160]
            steps.append(PlanStep(agent=name, task="", depends_on=sorted(set(deps)), reason=reason))
        return PlanGraph(steps=steps, mode="dependency", goal=goal,
                         reasoning="Derived from declared agent dependencies "
                                   "(no model configured for dynamic planning).")

    def _llm_plan(self, provider, specs: list, goal: str, inputs: dict):
        catalog = [
            {"name": s.name, "purpose": (s.purpose or s.role_prompt or "")[:200],
             "tools": list(s.tools), "tier": s.tier}
            for s in specs
        ]
        goal_text = (goal or inputs.get("goal") or inputs.get("task")
                     or f"Accomplish the objective for these inputs: {json.dumps(inputs, default=str)}")
        messages = [
            {"role": "system", "content": PLAN_SYSTEM_PROMPT.format(
                goal=goal_text, agents=json.dumps(catalog))},
            {"role": "user", "content": "Produce the plan JSON now."},
        ]
        msg = provider.chat(messages, temperature=0.1, response_format={"type": "json_object"})
        raw = json.loads(msg.get("content") or "{}")
        return self._validate_plan(raw, specs, goal_text)

    # ----- the validator (DISPOSES the LLM's proposed plan) -------------- #
    def _validate_plan(self, raw: dict, specs: list, goal: str):
        """Reduce an LLM-proposed plan to a safe, executable DAG: drop steps naming unknown
        agents, drop duplicate agents, force any synthesis agent last, and keep only forward
        dependency edges (so the graph is acyclic by construction). Returns None if nothing
        usable remains (the caller then falls back to the dependency order)."""
        names = {s.name for s in specs}
        raw_steps = raw.get("plan") or raw.get("steps") or []
        if not isinstance(raw_steps, list):
            return None

        notes = []
        seen = set()
        items = []   # (agent, task, reason, [dep_agent_names])
        for i, st in enumerate(raw_steps):
            if not isinstance(st, dict):
                continue
            agent = st.get("agent")
            if agent not in names:
                notes.append(f"dropped step {i}: unknown agent '{agent}'")
                continue
            if agent in seen:
                notes.append(f"dropped step {i}: duplicate agent '{agent}'")
                continue
            seen.add(agent)
            dep_agents = []
            for d in st.get("depends_on") or []:
                if isinstance(d, bool):
                    da = None
                elif isinstance(d, int) and 0 <= d < len(raw_steps):
                    da = (raw_steps[d] or {}).get("agent") if isinstance(raw_steps[d], dict) else None
                elif isinstance(d, str):
                    da = d
                else:
                    da = None
                if da in names:
                    dep_agents.append(da)
            items.append((agent, str(st.get("task") or "")[:600],
                          str(st.get("reason") or "")[:200], dep_agents))

        # STRICT selection: the planner may reorder and decompose tasks, but must NOT drop an agent
        # the caller explicitly required. Append any omitted agent so strict=[aml, credit] always
        # runs BOTH. By default (strict=False) the planner is free to choose a relevant subset of
        # the available agents (the GTM-discovery design).
        if strict:
            planned = {it[0] for it in items}
            for s in specs:
                if s.name not in planned:
                    items.append((s.name, "", "required in this run but omitted by the planner — "
                                  "added so the selection is honored", []))
                    notes.append(f"added omitted required agent '{s.name}'")
                    planned.add(s.name)

        if not items:
            return None

        # force synthesis/summary agents to the end (stable within each group)
        by_name = {s.name: s for s in specs}
        non_synth = [it for it in items if not self._is_synth_name(it[0], by_name)]
        synth = [it for it in items if self._is_synth_name(it[0], by_name)]
        ordered = non_synth + synth

        # finalize indices; keep only BACKWARD edges (forward edges dropped -> DAG)
        pos = {agent: idx for idx, (agent, *_rest) in enumerate(ordered)}
        steps = []
        for idx, (agent, task, reason, dep_agents) in enumerate(ordered):
            deps = sorted({pos[da] for da in dep_agents if pos.get(da, idx) < idx})
            steps.append(PlanStep(agent=agent, task=task, depends_on=deps, reason=reason))

        return PlanGraph(steps=steps, mode="llm", goal=goal,
                         reasoning=str(raw.get("reasoning") or "")[:400], notes=notes)

    # ----- adaptation: revise the not-yet-executed tail ------------------ #
    def _maybe_replan(self, provider, goal: str, specs: list, graph: PlanGraph,
                      idx: int, digests: list, executed: list):
        """Given results so far, ask the model whether to revise the remaining steps. Returns a
        validated new tail (list[PlanStep]) to replace graph.steps[idx+1:], or None to keep it."""
        remaining = graph.steps[idx + 1:]
        names = {s.name for s in specs}
        by_name = {s.name: s for s in specs}
        catalog = [{"name": s.name, "purpose": (s.purpose or s.role_prompt or "")[:160]}
                   for s in specs]
        messages = [
            {"role": "system", "content": REPLAN_SYSTEM_PROMPT.format(
                goal=goal, agents=json.dumps(catalog), completed=json.dumps(digests, default=str),
                remaining=json.dumps([s.agent for s in remaining]))},
            {"role": "user", "content": "Decide now. Return ONLY the JSON object."},
        ]
        try:
            msg = provider.chat(messages, temperature=0.1, response_format={"type": "json_object"})
            raw = json.loads(msg.get("content") or "{}")
        except Exception:  # noqa: BLE001 — adaptation is best-effort, never fatal
            return None
        if raw.get("keep", True) and not raw.get("revised_remaining"):
            return None

        done = set(executed)
        seen = set()
        new_non_synth, new_synth = [], []
        for st in raw.get("revised_remaining") or []:
            if not isinstance(st, dict):
                continue
            agent = st.get("agent")
            if agent not in names or agent in done or agent in seen:
                continue
            seen.add(agent)
            step = PlanStep(agent=agent, task=str(st.get("task") or "")[:600],
                            depends_on=[],
                            reason=str(st.get("reason") or "adapted from results")[:200])
            (new_synth if self._is_synth_name(agent, by_name) else new_non_synth).append(step)
        revised = new_non_synth + new_synth
        if not revised:
            return None
        if [s.agent for s in revised] == [s.agent for s in remaining]:
            return None   # a no-op "revision" — keep the existing tail
        return revised

    # ===================================================================== #
    # Execution — dispatch the plan through the ONE governed kernel
    # ===================================================================== #
    def run(self, tenant_id: str, specs: list, inputs: dict | None = None) -> dict:
        """Dispatch a set of agents as a governed multi-agent run. Thin alias over the one
        supervised kernel — there is no separate deterministic path."""
        return self.run_supervised(tenant_id, specs, inputs)

    def run_supervised(self, tenant_id: str, specs: list, inputs: dict | None = None) -> dict:
        if self.runtime is None:
            raise RuntimeError("Planner.run_supervised requires an AgentRuntime")
        inputs = inputs or {}
        goal = inputs.get("goal") or inputs.get("task") or ""
        tracer = Tracer(tenant_id)
        blackboard = Blackboard(tenant_id=tenant_id)
        by_name = {s.name: s for s in specs}
        provider = LlmProvider()

        # PLAN: the model decomposes the goal into a runtime task graph (or the dependency
        # toposort when no model is configured). This is the real "supervisor that plans".
        graph = self.plan_graph(specs, goal=goal, inputs=inputs)

        executed, agents, hitl_pause, error = [], [], None, None
        final_prose, final_agent = None, None
        digests = []
        replans = 0

        with tracer.start("planner:dispatch", "planner",
                          plan=[s.agent for s in graph.steps], plan_mode=graph.mode,
                          goal=graph.goal or None):
            i = 0
            while i < len(graph.steps):
                step = graph.steps[i]
                spec = by_name.get(step.agent)
                if spec is None:                      # a revised tail named a now-absent agent
                    i += 1
                    continue
                ctx = dict(inputs)
                for prior in spec.reads:              # collaborate via the board (ACL-checked)
                    try:
                        val = blackboard.read(prior, spec.name, spec.security.memory_read)
                    except Exception:                 # noqa: BLE001 (ACL denial)
                        val = None
                    if val is not None:
                        ctx[prior] = val
                        ctx.update(self._hoist(val))
                # one kernel: the sub-agent runs the SAME governed loop, with its decomposed subtask
                res = self.runtime.run_node(spec, ctx, tenant_id, tracer, task=step.task or None)
                agents.append({"agent": step.agent, "status": res["status"],
                               "output": res["output"], "mode": res["mode"],
                               "steps": len(res["steps"]), "flags": res.get("flags", []),
                               "task": step.task, "reason": step.reason})
                executed.append(step.agent)
                wkey = spec.writes[0] if spec.writes else f"{spec.name}.out"
                try:
                    blackboard.write(wkey, res["output"], spec.name, spec.security.memory_write)
                except Exception:                     # noqa: BLE001
                    pass
                digests.append({"agent": step.agent, "status": res["status"],
                                "result": self._digest(res["output"])})
                if self._is_synth(spec):
                    final_prose, final_agent = res["output"], step.agent
                if res["status"] == "needs_review":
                    hitl_pause = res.get("pending_action") or {
                        "agent": step.agent, "reason": "agent requires human review",
                        "flags": res.get("flags", [])}
                    break
                if res["status"] == "blocked":
                    error = {"agent": step.agent, "type": "blocked",
                             "detail": "guardrail/compliance blocked the sub-agent",
                             "flags": res.get("flags", [])}
                    break
                # ADAPT: let the intermediate results revise the remaining (LLM-planned) tail
                if (graph.mode == "llm" and provider.enabled and replans < _MAX_REPLANS
                        and i < len(graph.steps) - 1):
                    revised = self._maybe_replan(provider, graph.goal, specs, graph, i,
                                                 digests, executed)
                    if revised is not None:
                        graph.steps = graph.steps[:i + 1] + revised
                        graph.notes.append(
                            f"replanned after '{step.agent}': "
                            f"remaining -> {[s.agent for s in revised]}")
                        replans += 1
                i += 1

        if final_prose is None and agents:                     # no synthesis spec: last wins
            final_prose, final_agent = agents[-1]["output"], agents[-1]["agent"]

        trace = tracer.finalize()
        trace_dict = trace.to_dict()
        trace_dict.update({"executed": executed, "hitl_pause": hitl_pause, "error": error,
                           "blackboard": blackboard.snapshot(), "agents": agents,
                           "final_prose": final_prose, "final_agent": final_agent,
                           "supervised": True,
                           # the runtime plan, exposed so a reviewer can SEE how the goal was
                           # decomposed and adapted (not just a static dependency sort)
                           "plan": [s.to_dict() for s in graph.steps],
                           "plan_mode": graph.mode, "planner_reasoning": graph.reasoning,
                           "plan_notes": graph.notes, "replans": replans})
        self.store.save_trace(trace.trace_id, tenant_id, trace_dict)
        self.store.audit(tenant_id, "planner", "run_supervised", trace.trace_id,
                         {"executed": executed, "final_agent": final_agent,
                          "plan_mode": graph.mode, "replans": replans})
        return trace_dict

    # ===================================================================== #
    # Helpers
    # ===================================================================== #
    @staticmethod
    def _by_name(specs: list) -> dict:
        by_name = {}
        for s in specs:
            by_name[s.name] = s
            by_name[s.template or s.name] = s   # allow dependency-by-template-name
        return by_name

    @staticmethod
    def _is_synth(spec) -> bool:
        return (spec.name == "synthesis" or (spec.template or "") == "synthesis"
                or "synth" in spec.name.lower())

    @staticmethod
    def _is_synth_name(name: str, by_name: dict) -> bool:
        spec = by_name.get(name)
        if spec is not None:
            return Planner._is_synth(spec)
        return "synth" in (name or "").lower()

    @staticmethod
    def _digest(output):
        """A SHORT, structured digest of a sub-agent's output for the replanner — never the full
        blob. Output is already PII-redacted by the runtime; this just keeps the prompt compact."""
        if isinstance(output, dict):
            d = {}
            for k in ("recommendation", "recommended_next_action", "status", "risk_band",
                      "ofac_hit", "pep", "verdict", "summary", "account"):
                if k in output:
                    d[k] = output[k]
            d["fields"] = list(output.keys())[:8]
            return d
        return str(output)[:200]

    @staticmethod
    def _hoist(upstream) -> dict:
        """Pull well-known fields out of an upstream agent's output so the next agent's tools
        operate on real data produced earlier in the DAG (genuine collaboration)."""
        out = {}
        if not isinstance(upstream, dict):
            return out
        for f in ("company", "name", "ticker"):
            if upstream.get(f):
                out[f] = upstream[f]
        for blob in (upstream.get("outputs") or {}).values():
            if isinstance(blob, dict):
                for f in ("company", "name", "ticker"):
                    if blob.get(f):
                        out[f] = blob[f]
                if {"revenue", "ebitda"} <= set(blob):
                    out["financials"] = blob
        return out
