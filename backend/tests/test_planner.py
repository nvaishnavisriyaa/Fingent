"""
Real PLANNER tests — the supervisor decomposes a goal into a runtime task graph, selects which
sub-agents to invoke (with a focused subtask each), validates the plan deterministically, and
adapts on intermediate results. The deterministic dependency toposort remains the offline
fallback.

conftest neutralizes LLM keys (offline), so to exercise the LLM planning path we monkeypatch the
planner's provider with a fake that returns canned plan JSON. Sub-agent EXECUTION still runs the
offline demo engine (runtime builds its own provider, which stays disabled), so these tests are
fully deterministic.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
from fingent import planner as planner_mod
from fingent.planner import PlanStep, PlanGraph
from fingent.schemas import CreateAgentRequest


@pytest.fixture
def fp():
    return Fingent()


def _mk(fp, template, tenant="gtm"):
    tpl = fp.store.get_template(template)
    answers = {"name": template}
    for p in tpl.parameters:
        if p.default is not None:
            answers[p.name] = p.default
    fp.create_agent(CreateAgentRequest(template=template, answers=answers, tenant_id=tenant),
                    auto_provision=True)


def _gtm_agents(fp, tenant="gtm"):
    for t in ["signal_trigger", "icp_matching", "enrichment_validation",
              "persona_decision_maker", "contact", "synthesis"]:
        _mk(fp, t, tenant)
    return fp.store.list_specs(tenant)


class FakeProvider:
    """Stand-in LLM that returns queued JSON responses; later calls return '{}' (keep/no-op)."""
    enabled = True
    model = "fake-model"
    name = "fake-model @ test"

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        content = self._responses.pop(0) if self._responses else "{}"
        return {"content": content}


# --------------------------------------------------------------------------- #
# 1) Real decomposition: the LLM plan selects a SUBSET of agents, each with a subtask
# --------------------------------------------------------------------------- #
def test_planner_decomposes_goal_into_task_graph_with_llm(fp, monkeypatch):
    specs = _gtm_agents(fp)
    plan_json = json.dumps({
        "reasoning": "Find the trigger, then synthesize an outreach recommendation.",
        "plan": [
            {"agent": "signal_trigger", "task": "Detect funding/CFO signals for Acme Corp",
             "depends_on": [], "reason": "start from a real buying signal"},
            {"agent": "synthesis", "task": "Compose the next action from the signal",
             "depends_on": [0], "reason": "final recommendation"},
        ],
    })
    fake = FakeProvider([plan_json])
    monkeypatch.setattr(planner_mod, "LlmProvider", lambda: fake)

    graph = fp.planner.plan_graph(specs, goal="Find a way into Acme Corp")

    assert graph.mode == "llm"                       # the model planned, not a toposort
    assert [s.agent for s in graph.steps] == ["signal_trigger", "synthesis"]
    # genuine task DECOMPOSITION — each step carries a concrete subtask
    assert all(s.task for s in graph.steps)
    assert "signal" in graph.steps[0].task.lower()
    # the plan is a DAG: every dependency points at an EARLIER step
    for i, s in enumerate(graph.steps):
        assert all(d < i for d in s.depends_on)
    assert fake.calls, "the planner actually called the model to plan"


# --------------------------------------------------------------------------- #
# 2) The validator DISPOSES: unknown agents stripped, duplicates dropped,
#    synthesis forced last, forward edges removed (acyclic by construction)
# --------------------------------------------------------------------------- #
def test_planner_validator_strips_unknown_and_forces_synthesis_last(fp):
    specs = _gtm_agents(fp)
    raw = {
        "reasoning": "messy proposal",
        "plan": [
            {"agent": "synthesis", "task": "synthesize", "depends_on": ["contact"]},
            {"agent": "totally_made_up", "task": "hallucinated", "depends_on": []},
            {"agent": "signal_trigger", "task": "signals", "depends_on": []},
            {"agent": "signal_trigger", "task": "dup", "depends_on": []},     # duplicate
            {"agent": "contact", "task": "contacts", "depends_on": ["signal_trigger"]},
        ],
    }
    graph = fp.planner._validate_plan(raw, specs, "goal")
    agents = [s.agent for s in graph.steps]

    assert "totally_made_up" not in agents            # invented agent stripped
    assert agents.count("signal_trigger") == 1        # duplicate dropped
    assert agents[-1] == "synthesis"                  # synthesis forced last
    # the synthesis step's forward dep on 'contact' is now a valid BACKWARD edge after reorder
    synth = graph.steps[-1]
    assert all(d < len(graph.steps) - 1 for d in synth.depends_on)
    # contact depends on signal_trigger, and that edge is backward (acyclic)
    contact = next(s for s in graph.steps if s.agent == "contact")
    ci = agents.index("contact")
    assert all(d < ci for d in contact.depends_on)


def test_planner_validator_rejects_unusable_plan(fp):
    specs = _gtm_agents(fp)
    # nothing references a real agent -> None so the caller falls back to the dependency order
    assert fp.planner._validate_plan({"plan": [{"agent": "nope"}]}, specs, "g") is None
    assert fp.planner._validate_plan({"plan": []}, specs, "g") is None


# --------------------------------------------------------------------------- #
# 3) Offline fallback: no model -> deterministic dependency order (unchanged behaviour)
# --------------------------------------------------------------------------- #
def test_planner_falls_back_to_dependency_order_without_model(fp):
    # credit_underwriting HARD-depends on document_intelligence (auto-provisioned)
    _mk(fp, "credit_underwriting", tenant="acme")
    specs = [s for s in fp.store.list_specs("acme")
             if s.name in ("credit_underwriting", "document_intelligence")]

    graph = fp.planner.plan_graph(specs, goal="underwrite")     # offline: no LLM key

    assert graph.mode == "dependency"
    order = [s.agent for s in graph.steps]
    assert order.index("document_intelligence") < order.index("credit_underwriting")
    # dependency steps carry no decomposed subtask, so run_node behaves exactly as before
    assert all(s.task == "" for s in graph.steps)


# --------------------------------------------------------------------------- #
# 4) Adaptation: intermediate results revise the not-yet-executed tail
# --------------------------------------------------------------------------- #
def test_planner_adapts_remaining_plan_on_intermediate_results(fp, monkeypatch):
    specs = _gtm_agents(fp)
    plan_json = json.dumps({"reasoning": "minimal start", "plan": [
        {"agent": "signal_trigger", "task": "find signals", "depends_on": []},
        {"agent": "synthesis", "task": "summarize", "depends_on": [0]},
    ]})
    # after the first step's result, the supervisor decides to add an enrichment step
    replan_json = json.dumps({"keep": False, "revised_remaining": [
        {"agent": "icp_matching", "task": "score the surfaced company", "reason": "results warrant scoring"},
        {"agent": "synthesis", "task": "summarize with the score", "reason": "final"},
    ]})
    fake = FakeProvider([plan_json, replan_json])
    monkeypatch.setattr(planner_mod, "LlmProvider", lambda: fake)

    res = fp.planner.run_supervised("gtm", specs, inputs={"company": "Acme Corp",
                                                          "goal": "qualify Acme Corp"})

    assert res["plan_mode"] == "llm"
    assert res["replans"] >= 1                                  # the plan adapted at runtime
    assert "icp_matching" in res["executed"]                   # the added step actually ran
    assert res["executed"][0] == "signal_trigger"
    assert res["final_agent"] == "synthesis"
    # the runtime plan is exposed on the trace so a reviewer can see the decomposition
    assert res["plan"] and all("agent" in p and "task" in p for p in res["plan"])
    assert any("replanned" in n for n in res["plan_notes"])


# --------------------------------------------------------------------------- #
# 5) End-to-end: a planned supervised run dispatches the chosen agents through the
#    one governed kernel and exposes the plan (subset selection, not the full toposort)
# --------------------------------------------------------------------------- #
def test_planned_run_dispatches_only_selected_agents(fp, monkeypatch):
    specs = _gtm_agents(fp)
    plan_json = json.dumps({"reasoning": "two-step", "plan": [
        {"agent": "signal_trigger", "task": "signals for Acme", "depends_on": []},
        {"agent": "synthesis", "task": "recommend next action", "depends_on": [0]},
    ]})
    fake = FakeProvider([plan_json])
    monkeypatch.setattr(planner_mod, "LlmProvider", lambda: fake)

    res = fp.run_supervised("gtm", agent_names=[s.name for s in specs],
                            inputs={"company": "Acme Corp", "goal": "win Acme Corp"})

    # the planner chose a 2-agent subset out of the 6 available — not a full dependency sort
    assert res["executed"] == ["signal_trigger", "synthesis"]
    assert res["supervised"] is True
    for a in res["agents"]:
        assert a["mode"] in ("llm", "rules")                    # ran the real governed kernel
        assert a["status"] in ("success", "needs_review", "blocked")
    assert res["final_agent"] == "synthesis" and res["final_prose"] is not None
