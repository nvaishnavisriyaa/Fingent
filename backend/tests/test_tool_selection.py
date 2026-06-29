"""Explicit per-agent tool selection (Dify-style): an operator can pick which tools an agent
gets — including the tenant's approved MCP tools — at create AND edit time, with the validator
still enforcing least-privilege and the side-effecting-tool approval gate."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
from fingent.schemas import CreateAgentRequest, McpServer


@pytest.fixture
def fp():
    f = Fingent()
    # an approved MCP server -> its demo tools (bloomberg_quote read-only, send_email side-effecting)
    f.register_mcp(McpServer(name="acme_bank", url="https://example.com/mcp",
                             tenant_id="acme", approved=True))
    return f


def test_grantable_includes_native_and_mcp(fp):
    g = fp.grantable_tools("acme", "servicing_support")
    names = {t["name"] for t in g}
    kinds = {t["kind"] for t in g}
    assert "mcp" in kinds and "native" in kinds
    assert "acme_bank.bloomberg_quote" in names      # MCP tool is offerable to the agent
    assert "fx_rate" in names                          # template's real native tools present


def test_create_honors_explicit_tool_picks_incl_mcp(fp):
    r = fp.create_agent(CreateAgentRequest(
        template="servicing_support", answers={"name": "svc"}, tenant_id="acme",
        requested_tools=["acme_bank.bloomberg_quote"]))
    assert r["ok"]
    assert "acme_bank.bloomberg_quote" in r["spec"]["tools"]


def test_edit_can_add_and_remove_tools(fp):
    fp.create_agent(CreateAgentRequest(template="servicing_support", answers={"name": "svc"},
                                       tenant_id="acme"))
    # replace the tool set with an explicit selection (incl. an MCP tool)
    u = fp.update_agent("acme", "svc", {"tools": ["bank_lookup", "acme_bank.bloomberg_quote"]})
    assert u["ok"] and set(u["spec"]["tools"]) == {"bank_lookup", "acme_bank.bloomberg_quote"}


def test_side_effecting_tool_needs_approval_on_edit(fp):
    fp.create_agent(CreateAgentRequest(template="servicing_support", answers={"name": "svc"},
                                       tenant_id="acme"))
    # send_email is side-effecting -> stripped unless explicitly approved
    u = fp.update_agent("acme", "svc", {"tools": ["bank_lookup", "acme_bank.send_email"]})
    assert "acme_bank.send_email" not in u["spec"]["tools"]
    assert any("send_email" in s for s in u["stripped"])
    # now approve it -> kept
    u2 = fp.update_agent("acme", "svc", {"tools": ["bank_lookup", "acme_bank.send_email"],
                                         "approved_side_effecting_tools": ["acme_bank.send_email"]})
    assert "acme_bank.send_email" in u2["spec"]["tools"]


def test_out_of_universe_tool_is_stripped_on_edit(fp):
    fp.create_agent(CreateAgentRequest(template="servicing_support", answers={"name": "svc"},
                                       tenant_id="acme"))
    # ofac_screen is not in servicing's grantable universe -> no self-widening
    u = fp.update_agent("acme", "svc", {"tools": ["bank_lookup", "ofac_screen"]})
    assert "ofac_screen" not in u["spec"]["tools"]
