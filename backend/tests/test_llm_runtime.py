"""
LLM runtime tests — prove the agent is driven by a REAL tool-calling loop by default.

A mock OpenAI-compatible chat-completions server is stood up on localhost and pointed at via
FINGENT_LLM_BASE_URL. The runtime performs genuine multi-step tool calling against it: the model
emits `tool_calls`, the runtime executes the granted tool, feeds the result back, and the model
then produces a final answer. We also prove: an MCP-discovered tool (dotted name) is callable by
the model, least-privilege still blocks an ungranted tool the model asks for, and that with no
model configured the runtime falls back to a clearly-flagged demo engine.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
from fingent.schemas import CreateAgentRequest, McpServer


# --------------------------------------------------------------------------- #
# mock OpenAI-compatible chat-completions server (scripted)
# --------------------------------------------------------------------------- #
def tool_call(name, args):
    return {"role": "assistant", "content": None, "tool_calls": [
        {"id": "call_" + name, "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}


def final(text):
    return {"role": "assistant", "content": text}


def _llm_handler():
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
            i = self.server.calls
            self.server.calls += 1
            msg = self.server.script[min(i, len(self.server.script) - 1)]
            body = json.dumps({"choices": [{"message": msg}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    return H


def start_llm(script):
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _llm_handler())
    srv.script = script
    srv.calls = 0
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# --------------------------------------------------------------------------- #
# mock MCP server (for the LLM-calls-MCP test)
# --------------------------------------------------------------------------- #
def _mcp_handler():
    TOOLS = [{"name": "get_quote", "description": "Get an equity quote.",
              "inputSchema": {"type": "object", "properties": {"ticker": {"type": "string"}}},
              "annotations": {"readOnlyHint": True}}]

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            b = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0) or "{}")
            m = b.get("method")
            if m == "initialize":
                r = {"jsonrpc": "2.0", "id": b["id"], "result": {
                    "protocolVersion": "2025-06-18", "capabilities": {},
                    "serverInfo": {"name": "mkt", "version": "1"}}}
            elif m == "notifications/initialized":
                self.send_response(202); self.end_headers(); return
            elif m == "tools/list":
                r = {"jsonrpc": "2.0", "id": b["id"], "result": {"tools": TOOLS}}
            elif m == "tools/call":
                p = {"ticker": b["params"]["arguments"].get("ticker", "ACME"), "price": 30.5}
                r = {"jsonrpc": "2.0", "id": b["id"], "result": {"structuredContent": p}}
            else:
                r = {"jsonrpc": "2.0", "id": b.get("id"), "error": {"code": -32601, "message": "x"}}
            d = json.dumps(r).encode()
            self.send_response(200); self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(d))); self.end_headers(); self.wfile.write(d)
    return H


def start_mcp():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _mcp_handler())
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}/mcp"


# --------------------------------------------------------------------------- #
@pytest.fixture
def llm_env(monkeypatch):
    """Enable the LLM runtime but keep the COMPILER offline (it reads GROQ_API_KEY), so agent
    creation stays deterministic while execution is driven by the mock model."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("FINGENT_LLM_API_KEY", raising=False)

    def _point(url):
        # enable the LLM only at this point (called just before run_task), so agent creation
        # earlier in the test ran through the offline (deterministic) compiler.
        monkeypatch.setenv("FINGENT_LLM_API_KEY", "test-key")
        monkeypatch.setenv("FINGENT_LLM_BASE_URL", url)
    return _point


def test_llm_is_default_and_drives_multistep_tool_calls(llm_env):
    fp = Fingent()
    fp.create_agent(CreateAgentRequest(
        template="aml_sanctions_screening",
        answers={"name": "aml", "lists": ["OFAC"]}, tenant_id="acme"))
    # model: call ofac_screen, then (after seeing the result) give a final answer
    srv, url = start_llm([tool_call("ofac_screen", {"name": "Jane Smith"}),
                          final("No sanctions hit for Jane Smith. Cleared.")])
    llm_env(url)
    try:
        rec = fp.run_task("acme", "aml", {"name": "Jane Smith"})
    finally:
        srv.shutdown()

    assert rec["mode"] == "llm"                                   # LLM is the engine
    tools = [s["tool"] for s in rec["steps"] if s["kind"] == "tool"]
    assert tools == ["ofac_screen"]                              # the model's chosen tool ran
    assert rec["status"] == "success"
    assert rec["output"] == "No sanctions hit for Jane Smith. Cleared."  # model's final answer
    assert srv.calls == 2                                        # two real model turns


def test_llm_can_call_an_mcp_tool(llm_env):
    mcp_srv, mcp_url = start_mcp()
    fp = Fingent()
    fp.register_mcp(McpServer(name="market", url=mcp_url, tenant_id="acme", approved=True),
                    connect=True)
    fp.create_agent(CreateAgentRequest(
        template="servicing_support", answers={"name": "svc"},
        additional_requirements="use market.get_quote", tenant_id="acme"))
    # the model addresses the MCP tool by its sanitized function name (dot -> __)
    llm_srv, url = start_llm([tool_call("market__get_quote", {"ticker": "NVDA"}),
                              final("NVDA is trading at 30.5.")])
    llm_env(url)
    try:
        rec = fp.run_task("acme", "svc", {"ticker": "NVDA"})
    finally:
        llm_srv.shutdown(); mcp_srv.shutdown()

    quote_steps = [s for s in rec["steps"] if s["tool"] == "market.get_quote"]
    assert quote_steps, "the MCP tool the model requested was not executed"
    out = quote_steps[0]["tool_output"]
    assert out["ticker"] == "NVDA" and out["source"] == "live:mcp:market"
    assert rec["status"] == "success"


def test_least_privilege_blocks_tool_the_model_asks_for(llm_env):
    fp = Fingent()
    fp.create_agent(CreateAgentRequest(
        template="aml_sanctions_screening",
        answers={"name": "aml2", "lists": ["OFAC"]}, tenant_id="acme"))
    # model tries to use a tool that is NOT in the agent's allow-list
    srv, url = start_llm([tool_call("edgar_search", {"query": "secret"}),
                          final("should never get here")])
    llm_env(url)
    try:
        rec = fp.run_task("acme", "aml2", {"name": "Jane Smith"})
    finally:
        srv.shutdown()

    assert rec["status"] == "blocked"
    assert any(s.get("tool") == "edgar_search" and s.get("blocked") for s in rec["steps"])
    assert any(f.startswith("unauthorized_tool:edgar_search") for f in rec["risk_flags"])


def test_rules_mode_is_flagged_when_no_model_configured(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.delenv("FINGENT_LLM_API_KEY", raising=False)
    fp = Fingent()
    fp.create_agent(CreateAgentRequest(
        template="aml_sanctions_screening",
        answers={"name": "aml3", "lists": ["OFAC"]}, tenant_id="acme"))
    rec = fp.run_task("acme", "aml3", {"name": "Jane Smith"})
    assert rec["mode"] == "rules"
    assert "rules_mode" in rec["risk_flags"]
    assert any("deterministic rules engine" in s.get("note", "") for s in rec["steps"])
