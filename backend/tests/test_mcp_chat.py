"""
Acceptance #3 via the CHAT surface: a user registers an MCP server, its tools are discovered,
and an agent — talking over the streamed chat endpoint — actually INVOKES an MCP tool and uses
the result. Fully in-sandbox: a local mock LLM (scripted tool calls) + a real local MCP server
(real initialize/tools/list/tools/call). No external network, deterministic.

Reuses the real fake-server helpers from test_llm_runtime.py.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
from fingent.schemas import CreateAgentRequest, McpServer
from fingent.chat import chat_sse
from test_llm_runtime import start_llm, start_mcp, tool_call, final


def test_agent_invokes_mcp_tool_over_chat(monkeypatch):
    # 1) register a (real, local) MCP server — tools/list discovers its tools
    mcp_srv, mcp_url = start_mcp()
    fp = Fingent()
    added = fp.register_mcp(McpServer(name="market", url=mcp_url, tenant_id="acme",
                                      approved=True), connect=True)
    assert "market.get_quote" in added                      # discovered + grantable

    # 2) an agent granted that MCP tool
    fp.create_agent(CreateAgentRequest(
        template="servicing_support", answers={"name": "svc"},
        additional_requirements="use market.get_quote", tenant_id="acme"))

    # 3) drive the real LLM chat loop with a local mock model that calls the MCP tool
    llm_srv, url = start_llm([tool_call("market__get_quote", {"ticker": "NVDA"}),
                              final("NVDA is quoted at 31.42 (via the market MCP server).")])
    monkeypatch.setenv("FINGENT_LLM_API_KEY", "test-key")
    monkeypatch.setenv("FINGENT_LLM_BASE_URL", url)
    try:
        events = [json.loads(c[6:].strip())
                  for c in chat_sse(fp, "acme", "svc", "s1", "What's NVDA trading at?")]
    finally:
        llm_srv.shutdown(); mcp_srv.shutdown()

    types = [e["type"] for e in events]
    assert "start" in types and "final" in types and types[-1] == "done"
    # the MCP tool was actually invoked through the chat loop
    calls = [e for e in events if e["type"] == "tool_call" and e["tool"] == "market.get_quote"]
    assert calls, "agent did not invoke the MCP tool over chat"
    results = [e for e in events if e["type"] == "tool_result" and e["tool"] == "market.get_quote"]
    assert results and results[0]["output"]["ticker"] == "NVDA"
    assert results[0]["output"]["source"] == "live:mcp:market"   # real proxied result
    # the streamed prose answer actually used the result
    final_ev = next(e for e in events if e["type"] == "final")
    assert "31.42" in final_ev["text"]
