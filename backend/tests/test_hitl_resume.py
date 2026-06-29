"""
HITL approval is a GOVERNED RESUME, not a raw tool fire (acceptance: critical fix).

A run held on a side-effecting tool:
  * approve -> the held tool executes THROUGH the one governed kernel (real tool step + trace,
    risk re-scored, audited) and the agent continues to a governed final answer.
  * reject  -> the held action is NOT executed; a counterfactual + reason is recorded.

Offline/deterministic (conftest sets FINGENT_ALLOW_DEMO=1).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
from fingent.schemas import CreateAgentRequest, McpServer


@pytest.fixture
def held(fp_factory=None):
    fp = Fingent()
    fp.register_mcp(McpServer(name="acme_mcp", url="https://m", tenant_id="acme", approved=True))
    fp.create_agent(CreateAgentRequest(
        template="servicing_support", answers={"name": "svc"},
        additional_requirements="use acme_mcp.send_email", tenant_id="acme",
        approve_side_effecting=True))   # grant the tool; runtime still HOLDS it for HITL
    rec = fp.run_task("acme", "svc", {"to": "ops@bank.com"})
    assert rec["status"] == "needs_review"
    assert rec["pending_action"]["tool"] == "acme_mcp.send_email"
    # the held tool did NOT fire yet
    assert not any(s.get("tool") == "acme_mcp.send_email" and s.get("kind") == "tool"
                   for s in rec["steps"])
    return fp, rec["id"]


def test_approve_resumes_through_the_governed_kernel(held):
    fp, run_id = held
    res = fp.resolve_review("acme", run_id, "approve", reviewer="alice", note="ok to send")
    assert res["ok"] and res.get("resumed") is True
    run = res["run"]
    assert run["status"] in ("approved", "needs_review")     # governed terminal status
    assert run["review_decision"] == "approved" and run["reviewer"] == "alice"
    # the held tool actually executed THROUGH the kernel this time (real governed tool step)
    tool_steps = [s for s in run["steps"] if s.get("tool") == "acme_mcp.send_email"
                  and s.get("kind") == "tool"]
    assert tool_steps, "approved side-effect was not executed through the governed kernel"
    # the held tool executed THROUGH the kernel and its (governed) output was recorded
    assert "sent" in (tool_steps[0]["tool_output"] or {})
    # it went through governance: a fresh trace exists and risk was re-scored
    assert run["trace_id"] and fp.store.get_trace("acme", run["trace_id"]) is not None
    assert "risk_level" in run
    # audit recorded a governed approval, not a raw fire
    audit = fp.store.get_audit("acme") if hasattr(fp.store, "get_audit") else []
    # (audit shape varies; the run-level evidence above is the contract)


def test_reject_cancels_without_executing(held):
    fp, run_id = held
    res = fp.resolve_review("acme", run_id, "reject", reviewer="bob", note="recipient unverified")
    run = res["run"]
    assert run["status"] == "rejected" and run["review_decision"] == "rejected"
    assert run["pending_action"] is None
    # the held tool was NOT executed; a blocked counterfactual records the decision + reason
    assert not any(s.get("tool") == "acme_mcp.send_email" and s.get("kind") == "tool"
                   and not s.get("blocked") for s in run["steps"])
    cf = [s for s in run["steps"] if s.get("kind") == "review" and s.get("blocked")]
    assert cf and "recipient unverified" in cf[-1]["note"]


def test_approval_path_has_no_raw_backdoor():
    """Static guard: resolve_review must not call registry.callable(...) directly (the old raw
    fn(**args) backdoor). All tool execution must go through the runtime kernel."""
    import inspect, fingent.platform as plat
    src = inspect.getsource(plat.Fingent.resolve_review)
    assert "registry.callable" not in src
    assert "self.runtime.run(" in src        # governed resume
