"""
Live MCP client tests — prove Fingent speaks the REAL MCP wire protocol.

A minimal in-process MCP server (Streamable HTTP transport, JSON-RPC 2.0) is stood up on an
ephemeral localhost port. The platform connects to it for real (FINGENT_MCP_LIVE=1), performs
`initialize`, enumerates tools via `tools/list`, and invokes one via `tools/call`. No mocks of
the client itself — only a real, spec-shaped server to talk to.

One server variant returns plain `application/json`; another returns `text/event-stream` (SSE),
exercising both Streamable-HTTP response shapes the client must handle.
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
from fingent.mcp_client import McpClient

SESSION_ID = "test-session-123"

# A tiny tool catalog the fake server exposes.
TOOLS = [
    {"name": "get_quote", "description": "Get an equity quote.",
     "inputSchema": {"type": "object", "properties": {"ticker": {"type": "string"}}},
     "annotations": {"readOnlyHint": True}},
    {"name": "send_email", "description": "Send an email.",
     "inputSchema": {"type": "object", "properties": {"to": {"type": "string"}}},
     "annotations": {"readOnlyHint": False, "destructiveHint": True}},
]


def _handle_rpc(msg: dict) -> dict | None:
    """Return a JSON-RPC response for a request, or None for a notification."""
    method = msg.get("method")
    if method == "initialize":
        return {"jsonrpc": "2.0", "id": msg["id"], "result": {
            "protocolVersion": "2025-06-18",
            "capabilities": {"tools": {}},
            "serverInfo": {"name": "fake-mcp", "version": "9.9"}}}
    if method == "notifications/initialized":
        return None
    if method == "tools/list":
        return {"jsonrpc": "2.0", "id": msg["id"], "result": {"tools": TOOLS}}
    if method == "tools/call":
        params = msg.get("params", {})
        name = params.get("name")
        args = params.get("arguments", {})
        if name == "get_quote":
            payload = {"ticker": args.get("ticker", "ACME"), "price": 31.42, "currency": "USD"}
            return {"jsonrpc": "2.0", "id": msg["id"], "result": {
                "content": [{"type": "text", "text": json.dumps(payload)}],
                "structuredContent": payload, "isError": False}}
        if name == "send_email":
            return {"jsonrpc": "2.0", "id": msg["id"], "result": {
                "content": [{"type": "text", "text": "queued"}], "isError": False}}
        return {"jsonrpc": "2.0", "id": msg["id"],
                "error": {"code": -32601, "message": f"unknown tool {name}"}}
    return {"jsonrpc": "2.0", "id": msg.get("id"),
            "error": {"code": -32601, "message": f"unknown method {method}"}}


def _make_handler(use_sse: bool):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # silence
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(length) or "{}")
            resp = _handle_rpc(body)
            if resp is None:  # notification -> 202, no body
                self.send_response(202)
                if body.get("method") == "initialize":
                    self.send_header("Mcp-Session-Id", SESSION_ID)
                self.end_headers()
                return
            if use_sse:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                if body.get("method") == "initialize":
                    self.send_header("Mcp-Session-Id", SESSION_ID)
                self.end_headers()
                self.wfile.write(f"event: message\ndata: {json.dumps(resp)}\n\n".encode())
            else:
                payload = json.dumps(resp).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                if body.get("method") == "initialize":
                    self.send_header("Mcp-Session-Id", SESSION_ID)
                self.end_headers()
                self.wfile.write(payload)
    return Handler


@pytest.fixture(params=[False, True], ids=["json", "sse"])
def mcp_url(request):
    server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(request.param))
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    host, port = server.server_address
    try:
        yield f"http://{host}:{port}/mcp"
    finally:
        server.shutdown()


def test_client_speaks_real_protocol(mcp_url):
    """Raw client: initialize handshake + tools/list + tools/call over the wire."""
    c = McpClient(mcp_url)
    init = c.initialize()
    assert init["serverInfo"]["name"] == "fake-mcp"
    assert c.session_id == SESSION_ID                       # session id captured
    names = [t["name"] for t in c.list_tools()]
    assert names == ["get_quote", "send_email"]
    result = c.call_tool("get_quote", {"ticker": "IBM"})
    assert result["structuredContent"]["ticker"] == "IBM"
    c.close()


def test_registry_discovers_and_proxies_live(mcp_url):
    """Full platform path: register a live server, real tools are discovered, side-effect
    annotation is honoured, and a granted tool actually proxies to the live server."""
    fp = Fingent()
    added = fp.register_mcp(
        McpServer(name="mkt", url=mcp_url, tenant_id="acme", approved=True), connect=True)
    assert set(added) == {"mkt.get_quote", "mkt.send_email"}

    # the read-only tool is not side-effecting; the destructive one is
    assert fp.registry.get("mkt.get_quote").side_effecting is False
    assert fp.registry.get("mkt.send_email").side_effecting is True

    # the bound callable really calls the remote server
    out = fp.registry.callable("mkt.get_quote")(ticker="MSFT")
    assert out["ticker"] == "MSFT" and out["price"] == 31.42
    assert out["_untrusted"] is True and out["source"] == "live:mcp:mkt"

    # persisted connection state reflects the live discovery
    srv = fp.store.get_mcp("acme", "mkt")
    assert srv.connection_status == "connected"
    assert srv.server_info["name"] == "fake-mcp"


def test_mcp_tools_survive_restart(mcp_url):
    """Restart bug fixed: after a process restart (new Fingent on the same DB) the discovered
    MCP tools are rebuilt into the registry instead of vanishing."""
    import tempfile

    db = os.path.join(tempfile.mkdtemp(), "fingent.db")
    fp = Fingent(db)
    fp.register_mcp(McpServer(name="mkt", url=mcp_url, tenant_id="acme", approved=True),
                    connect=True)
    assert "mkt.get_quote" in [t["name"] for t in fp.tool_catalog("acme")]

    # simulate a restart: a brand-new platform instance on the SAME database
    fp2 = Fingent(db)
    tools = [t["name"] for t in fp2.tool_catalog("acme")]
    assert "mkt.get_quote" in tools and "mkt.send_email" in tools
    # and the rebuilt proxy still reaches the live server
    out = fp2.registry.callable("mkt.get_quote")(ticker="GOOG")
    assert out["ticker"] == "GOOG"


def test_bad_url_does_not_break_registration():
    """A live connection failure is recorded as an error, never crashes registration, and
    registers no tools (honest) rather than fake ones."""
    fp = Fingent()
    added = fp.register_mcp(
        McpServer(name="dead", url="http://127.0.0.1:9/none", tenant_id="acme", approved=True),
        connect=True)
    assert added == []
    srv = fp.store.get_mcp("acme", "dead")
    assert srv.connection_status == "error" and srv.connection_error
