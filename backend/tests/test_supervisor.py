"""
Supervised multi-agent path — every sub-agent runs the REAL agent runtime (run_node),
collaborating via the blackboard, and the Synthesis agent produces the final answer.

Offline/deterministic (conftest neutralizes LLM keys + forces FINGENT_LIVE_DATA=0), so this
exercises the demo engine of the same loop the LLM uses. On a machine with GROQ_API_KEY set,
the identical code path runs mode='llm' (see demo_supervisor.py for a live trace).
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
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
    return fp.create_agent(
        CreateAgentRequest(template=template, answers=answers, tenant_id=tenant),
        auto_provision=True)


def test_supervisor_dispatches_subagents_then_synthesizes(fp):
    for t in ["signal_trigger", "icp_matching", "enrichment_validation",
              "persona_decision_maker", "contact", "synthesis", "planner"]:
        _mk(fp, t)

    res = fp.run_supervised("gtm", tier=1, inputs={"company": "Acme Corp"})

    # ≥2 sub-agents dispatched in dependency order, ending at synthesis (acceptance #4)
    assert res["supervised"] is True
    assert len(res["executed"]) >= 2
    assert "signal_trigger" in res["executed"]
    assert res["final_agent"] == "synthesis"

    # every dispatched agent ran the real runtime loop (has a mode + step count)
    for a in res["agents"]:
        assert a["mode"] in ("llm", "rules")
        assert a["status"] in ("success", "needs_review", "blocked")

    # the supervisor produced a final answer from shared memory, not an empty blob
    assert res["final_prose"] is not None
    assert res["blackboard"]  # sub-agents collaborated on the board


def test_supervisor_run_node_is_the_same_loop_as_run(fp):
    """A single agent driven via run_node yields the same governed result shape as run()."""
    _mk(fp, "aml_sanctions_screening", tenant="acme")
    spec = next(s for s in fp.store.list_specs("acme") if s.name == "aml_sanctions_screening")
    from fingent.observability import Tracer
    res = fp.runtime.run_node(spec, {"name": "Oleg Petrov"}, "acme", Tracer("acme"))
    assert res["agent"] == "aml_sanctions_screening"
    assert "output" in res and res["status"] in ("success", "blocked", "needs_review")
    # the loop actually invoked real tools (recorded as tool steps)
    assert any(getattr(st, "kind", "") == "tool" for st in res["steps"])
