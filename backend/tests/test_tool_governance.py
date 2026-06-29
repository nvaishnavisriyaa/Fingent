"""
Honest tool governance:
  * an unreachable live source ("source":"unavailable") raises a source_degraded flag so it is
    never silently treated as a clean negative (a dangerous false negative in AML/KYC);
  * MCP tools are safe-by-default: an unannotated tool that does not clearly look read-only is
    treated as side-effecting and must pass the HITL gate.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
from fingent.schemas import CreateAgentRequest
from fingent.observability import Tracer
from fingent.mcp_client import infer_side_effecting


@pytest.fixture
def fp():
    f = Fingent()
    f.create_agent(CreateAgentRequest(template="aml_sanctions_screening",
                                      answers={"name": "aml", "lists": ["OFAC"]},
                                      tenant_id="acme"))
    return f


def test_unreachable_source_hard_fails_not_a_clean_negative(fp):
    # simulate the OFAC feed being unreachable (live mode would return source="unavailable")
    fp.registry._callables["ofac_screen"] = lambda **k: {"source": "unavailable", "ofac_hit": None}
    spec = fp.store.get_spec("acme", "aml")
    res = fp.runtime.run_node(spec, {"name": "Oleg Petrov"}, "acme", Tracer("acme"))
    assert any(f == "source_degraded:ofac_screen" for f in res["flags"]), res["flags"]
    # NEW hard-fail policy: the run refuses to answer on an unavailable source — it fails loudly
    # and names the degraded tool, instead of composing a (false-negative) clean pass.
    assert res["status"] == "failed", res["status"]
    assert "hard_fail_no_real_source" in res["flags"], res["flags"]
    out = res["output"]
    assert isinstance(out, dict) and "ofac_screen" in out.get("degraded_tools", []), out


def test_mcp_unannotated_tool_is_side_effecting_by_default():
    # clearly read-only -> safe
    assert infer_side_effecting({"name": "get_quote"}) is False
    assert infer_side_effecting({"name": "search_filings"}) is False
    # clearly write -> side-effecting
    assert infer_side_effecting({"name": "send_email"}) is True
    assert infer_side_effecting({"name": "wire_transfer"}) is True
    # UNKNOWN + UNANNOTATED -> default-deny (treated as side-effecting, must be approved)
    assert infer_side_effecting({"name": "frobnicate"}) is True
    # explicit server annotations always win
    assert infer_side_effecting({"name": "frobnicate", "annotations": {"readOnlyHint": True}}) is False
    assert infer_side_effecting({"name": "get_quote", "annotations": {"destructiveHint": True}}) is True
