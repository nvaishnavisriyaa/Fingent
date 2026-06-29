"""
Real MCP client (Model Context Protocol) — speaks the actual wire protocol.

This replaces the old hardcoded two-tool stub. Given a registered MCP server URL it:
  1. opens a JSON-RPC session over the **Streamable HTTP** transport (the current MCP
     remote transport, spec rev 2025-03-26 / 2025-06-18),
  2. performs the `initialize` handshake and sends `notifications/initialized`,
  3. enumerates the server's real tools via `tools/list` (with cursor pagination),
  4. invokes them for real via `tools/call`.

The transport is synchronous (built on `requests`) to match the rest of the platform, and
handles BOTH response shapes a Streamable-HTTP server may return to a POST:
  * `application/json`   — a single JSON-RPC response, or
  * `text/event-stream`  — an SSE stream that carries the JSON-RPC response message.

Session continuity (`Mcp-Session-Id`) and the `MCP-Protocol-Version` header are honoured.

Tool *output is untrusted* by definition (it crosses a trust boundary), so callers must
injection-scan it — the registry marks every MCP descriptor `untrusted_output=True`.
"""
from __future__ import annotations

import json

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "fingent-mcp-client", "version": "0.1.0"}

# Tool-name substrings that imply a write/send/pay action when the server gives no
# explicit annotation. Used only as a safe-by-default fallback for the HITL gate.
_WRITE_HINTS = (
    "send", "write", "pay", "transfer", "delete", "remove", "create", "update",
    "post", "email", "execute", "order", "cancel", "approve", "issue", "charge",
)


class McpError(Exception):
    """Any failure talking to an MCP server (transport, protocol, or JSON-RPC error)."""


class McpClient:
    """A minimal, synchronous MCP client over Streamable HTTP.

    Usage:
        c = McpClient(url, headers={"Authorization": "Bearer ..."})
        c.initialize()
        tools = c.list_tools()
        result = c.call_tool("name", {"arg": 1})
        c.close()
    """

    def __init__(self, url: str, headers: dict | None = None,
                 timeout: tuple[float, float] = (5.0, 30.0)) -> None:
        import requests
        self.url = url
        self.timeout = timeout
        self.session_id: str | None = None
        self.protocol_version = PROTOCOL_VERSION
        self.server_info: dict = {}
        self.capabilities: dict = {}
        self._id = 0
        self._http = requests.Session()
        self._http.headers.update(headers or {})
        self._initialized = False

    # ------------------------------------------------------------------ #
    # transport
    # ------------------------------------------------------------------ #
    def _headers(self) -> dict:
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.session_id:
            h["Mcp-Session-Id"] = self.session_id
        if self._initialized:
            h["MCP-Protocol-Version"] = self.protocol_version
        return h

    def _send(self, payload: dict, expect_response: bool = True) -> dict | None:
        try:
            resp = self._http.post(self.url, json=payload, headers=self._headers(),
                                   stream=True, timeout=self.timeout)
        except Exception as e:  # noqa: BLE001 — surface transport failures uniformly
            raise McpError(f"transport error contacting {self.url}: {e}") from e

        # the server assigns/echoes a session id on initialize (and may rotate it)
        sid = resp.headers.get("Mcp-Session-Id") or resp.headers.get("mcp-session-id")
        if sid:
            self.session_id = sid

        if resp.status_code >= 400:
            body = resp.text[:300]
            resp.close()
            raise McpError(f"HTTP {resp.status_code} from MCP server: {body}")

        if not expect_response or resp.status_code == 202:
            resp.close()
            return None

        ctype = (resp.headers.get("Content-Type") or "").lower()
        try:
            if "text/event-stream" in ctype:
                return self._read_sse(resp, payload.get("id"))
            return resp.json()
        finally:
            resp.close()

    @staticmethod
    def _read_sse(resp, req_id) -> dict:
        """Parse an SSE stream and return the JSON-RPC message matching ``req_id``."""
        data_lines: list[str] = []
        for raw in resp.iter_lines(decode_unicode=True):
            if raw is None:
                continue
            line = raw.rstrip("\r")
            if line == "":  # blank line dispatches the buffered event
                if data_lines:
                    payload = "\n".join(data_lines)
                    data_lines = []
                    try:
                        msg = json.loads(payload)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(msg, dict) and ("id" not in msg or msg.get("id") == req_id):
                        if msg.get("id") == req_id or "result" in msg or "error" in msg:
                            return msg
                continue
            if line.startswith(":"):  # comment/keep-alive
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].lstrip())
            # other SSE fields (event:, id:, retry:) are not needed here
        raise McpError("SSE stream ended before a matching JSON-RPC response arrived")

    def _rpc(self, method: str, params: dict | None = None) -> dict:
        self._id += 1
        rid = self._id
        msg = {"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}}
        data = self._send(msg, expect_response=True)
        if not isinstance(data, dict):
            raise McpError(f"empty/invalid response for {method}")
        if data.get("error"):
            err = data["error"]
            raise McpError(f"{method} failed: {err.get('code')} {err.get('message')}")
        return data.get("result", {}) or {}

    def _notify(self, method: str, params: dict | None = None) -> None:
        msg = {"jsonrpc": "2.0", "method": method, "params": params or {}}
        self._send(msg, expect_response=False)

    # ------------------------------------------------------------------ #
    # protocol
    # ------------------------------------------------------------------ #
    def initialize(self) -> dict:
        """Perform the MCP handshake. Captures the session id, negotiated protocol
        version, server info and capabilities, then sends `notifications/initialized`."""
        result = self._rpc("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": {},
            "clientInfo": CLIENT_INFO,
        })
        self.protocol_version = result.get("protocolVersion") or PROTOCOL_VERSION
        self.server_info = result.get("serverInfo", {}) or {}
        self.capabilities = result.get("capabilities", {}) or {}
        self._initialized = True
        self._notify("notifications/initialized")
        return result

    def list_tools(self) -> list[dict]:
        """Enumerate every tool the server exposes (follows `nextCursor` pagination)."""
        if not self._initialized:
            self.initialize()
        tools: list[dict] = []
        cursor: str | None = None
        for _ in range(50):  # hard page cap — defensive against a misbehaving server
            params = {"cursor": cursor} if cursor else {}
            result = self._rpc("tools/list", params)
            tools.extend(result.get("tools", []) or [])
            cursor = result.get("nextCursor")
            if not cursor:
                break
        return tools

    def call_tool(self, name: str, arguments: dict | None = None) -> dict:
        """Invoke a tool for real and return the raw MCP `tools/call` result object."""
        if not self._initialized:
            self.initialize()
        return self._rpc("tools/call", {"name": name, "arguments": arguments or {}})

    def close(self) -> None:
        try:
            self._http.close()
        except Exception:  # noqa: BLE001
            pass


# --------------------------------------------------------------------------- #
# helpers used by the registry to map MCP shapes onto Fingent's tool model
# --------------------------------------------------------------------------- #
_READ_HINTS = (
    "get", "list", "read", "search", "fetch", "lookup", "query", "find", "screen", "check",
    "scan", "describe", "status", "view", "quote", "price", "balance", "report", "show",
)


def infer_side_effecting(tool: dict) -> bool:
    """Decide whether a discovered MCP tool writes/sends/pays (and therefore must pass the HITL
    approval gate). Prefer the server's own annotations; otherwise be SAFE-BY-DEFAULT.

    Governance default-deny: an UNANNOTATED tool is treated as side-effecting UNLESS its name
    clearly looks read-only. This closes the gap where a destructive tool with a benign name and
    no annotation would otherwise skip the human-approval gate and fire silently."""
    ann = tool.get("annotations") or {}
    if ann.get("readOnlyHint") is True:
        return False
    if ann.get("destructiveHint") is True:
        return True
    name = (tool.get("name") or "").lower()
    if any(h in name for h in _WRITE_HINTS):
        return True
    if any(h in name for h in _READ_HINTS):
        return False
    # unknown + unannotated -> require approval rather than fire unsupervised
    return True


def normalize_tool(tool: dict) -> dict:
    """Project an MCP `tools/list` entry onto the fields Fingent persists/registers."""
    return {
        "name": tool.get("name") or "",
        "description": tool.get("description") or tool.get("title") or "",
        "side_effecting": infer_side_effecting(tool),
        "input_schema": tool.get("inputSchema") or {},
    }


def result_to_dict(result: dict, server_name: str, tool_name: str) -> dict:
    """Flatten an MCP `tools/call` result into the dict shape the runtime/factory expect.
    Output is always tagged untrusted so the guardrail layer injection-scans it."""
    out: dict = {"_untrusted": True, "source": f"live:mcp:{server_name}", "tool": tool_name}
    if not isinstance(result, dict):
        out["result"] = result
        return out

    # structured content is the preferred, typed channel when the server provides it
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        out.update(structured)
    else:
        parts: list = []
        for block in result.get("content", []) or []:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(block.get("text", ""))
            else:
                parts.append(block)
        text = "\n".join(p for p in parts if isinstance(p, str)).strip()
        if text:
            try:
                parsed = json.loads(text)
                if isinstance(parsed, dict):
                    out.update(parsed)
                else:
                    out["result"] = parsed
            except json.JSONDecodeError:
                out["text"] = text
        elif parts:
            out["content"] = parts

    if result.get("isError"):
        out["isError"] = True
    return out
