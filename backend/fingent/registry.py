"""
Tool Registry (§1, §6) — the single place tools are described, scoped, and bound.

Holds ToolDescriptors for four kinds: NATIVE, WEB_SEARCH, MCP, EXTERNAL_API. Native + web are
global; MCP/external tools are tenant-scoped and only become grantable once their MCP server is
approved. The LLM compiler reads descriptors to decide grants; the factory binds the callable.

MCP tools are discovered for real: when an approved server is registered (and live MCP is on) we
open an MCP session to its URL, run `initialize` + `tools/list`, and bind each discovered tool to
a callable that proxies to the server via `tools/call`. Offline (tests / no reachable server) a
clearly-labelled deterministic demo catalog is used instead.
"""
from __future__ import annotations

import os
import time

from . import mcp_client
from .schemas import McpServer, McpToolSpec, ToolDescriptor, ToolKind
from .store import Store
from . import tools_native as nt


def _mcp_live() -> bool:
    """Whether to open real network connections to MCP servers. Gated on the same flag the
    native tools use (FINGENT_LIVE_DATA), with a dedicated FINGENT_MCP_LIVE override. The test
    suite sets FINGENT_LIVE_DATA=0, so tests stay offline and deterministic (demo catalog)."""
    override = os.getenv("FINGENT_MCP_LIVE")
    if override is not None:
        return override != "0"
    return os.getenv("FINGENT_LIVE_DATA", "1") != "0"


# The deterministic offline catalog — only used when live MCP is disabled (tests / demos with
# no reachable server). Mirrors the original stub so existing behaviour is preserved offline.
_DEMO_TOOLS = [
    McpToolSpec(name="bloomberg_quote", description="Get a live equity quote.",
                side_effecting=False),
    McpToolSpec(name="send_email", description="Send an email on the user's behalf.",
                side_effecting=True),
]


class ToolRegistry:
    def __init__(self, store: Store) -> None:
        self.store = store
        self._descriptors: dict[str, ToolDescriptor] = {}
        self._callables: dict[str, callable] = {}
        self._seed_native()

    def _seed_native(self) -> None:
        native_meta = {
            "edgar_search": "Search SEC EDGAR filings for a company.",
            "news_monitor": "Monitor news/market feeds for FS trigger signals.",
            "enrich_company": "Enrich a company with firmographic + financial-health data.",
            "find_persona": "Find decision-maker personas (CFO/VP Finance/Treasurer...).",
            "resolve_contact": "Resolve email/phone/LinkedIn for a person.",
            "ocr_extract": "OCR a financial document into text + fields.",
            "parse_financials": "Parse statements/tax/financials into structured numbers.",
            "compute_ratios": "Compute financial ratios from parsed financials.",
            "ofac_screen": "Screen a name against OFAC/EU/UN sanctions lists.",
            "adverse_media_search": "Search adverse media for a name.",
            "pep_check": "Check whether a person is a politically-exposed person.",
            "anomaly_detect": "Run anomaly detection over transactions.",
            "reg_feed_ingest": "Ingest a regulatory feed into obligations.",
            "compose_summary": "Compose an actionable account summary + next action from shared memory.",
            "risk_score": "Score credit risk (0-1) from financial ratios.",
            "compliance_check": "Vet a recommendation for sanctions/PII/red-flag issues (overseer).",
            "identity_verify": "Verify a customer identity (KYC) and return a confidence score.",
            "account_lookup": "Look up a servicing account record.",
            "company_financials": "Fetch REAL public-company financials + credit ratios from SEC EDGAR.",
            "verify_entity": "Verify a legal entity against the GLEIF LEI registry (real KYB).",
            "bank_lookup": "Look up a US bank/counterparty in the FDIC BankFind registry (real).",
            "fx_rate": "Get real FX reference rates (ECB/Frankfurter) for currency conversion.",
            "treasury_rates": "Get real US Treasury average interest rates (benchmark cost of funds).",
        }
        for name, desc in native_meta.items():
            self.register(ToolDescriptor(name=name, kind=ToolKind.NATIVE, description=desc),
                          nt.NATIVE_CALLABLES[name])
        self.register(
            ToolDescriptor(name="web_search", kind=ToolKind.WEB_SEARCH,
                           description="Built-in web search. Returns untrusted public content.",
                           untrusted_output=True),
            nt.BUILTIN_CALLABLES["web_search"],
        )

    def register(self, descriptor: ToolDescriptor, fn: callable | None = None) -> None:
        self._descriptors[descriptor.name] = descriptor
        if fn is not None:
            self._callables[descriptor.name] = fn

    # ----- MCP registration --------------------------------------------- #
    def register_mcp_server(self, server: McpServer, actor: str = "admin",
                            connect: bool | None = None) -> list[str]:
        """Admin registers an MCP server. When live (and approved) we CONNECT FOR REAL over the
        MCP wire protocol — `initialize` + `tools/list` — and register the server's actual tools,
        each bound to a callable that proxies to the live server via `tools/call`. Tools become
        grantable only when server.approved is True, and remain tenant-scoped.

        Offline (FINGENT_LIVE_DATA=0 or no reachable server) we register a clearly-labelled
        deterministic demo catalog so the platform still runs without a live server."""
        self.store.save_mcp(server)
        self.store.audit(server.tenant_id, actor, "mcp_register", server.name,
                         {"approved": server.approved, "url": server.url})
        if not server.approved:
            return []

        live = _mcp_live() if connect is None else connect
        if live:
            try:
                discovered, info = self._discover_live(server)
                server.discovered_tools = discovered
                server.server_info = info
                server.connection_status = "connected"
                server.connection_error = ""
                server.last_connected = time.time()
                self.store.save_mcp(server)
                added = self._register_discovered(server, live=True)
                self.store.audit(server.tenant_id, actor, "mcp_connect", server.name,
                                 {"tools": added, "server_info": info})
                return added
            except Exception as e:  # noqa: BLE001 — a bad URL must never break registration
                server.connection_status = "error"
                server.connection_error = str(e)[:300]
                server.discovered_tools = []
                self.store.save_mcp(server)
                self.store.audit(server.tenant_id, actor, "mcp_connect_failed", server.name,
                                 {"error": str(e)[:300]})
                return []

        # offline / demo path
        server.discovered_tools = list(_DEMO_TOOLS)
        server.connection_status = "demo"
        self.store.save_mcp(server)
        added = self._register_discovered(server, live=False)
        self.store.audit(server.tenant_id, actor, "mcp_approve", server.name, {"tools": added})
        return added

    def approve_mcp_server(self, tenant_id: str, name: str, actor: str = "admin") -> list[str]:
        server = self.store.get_mcp(tenant_id, name)
        if not server:
            raise KeyError(f"no MCP server '{name}' for tenant {tenant_id}")
        server.approved = True
        return self.register_mcp_server(server, actor)

    def refresh_mcp_server(self, tenant_id: str, name: str, actor: str = "admin") -> dict:
        """Re-connect to an approved server and re-enumerate its tools (e.g. after the server
        adds/removes tools). Returns the connection result for the admin UI."""
        server = self.store.get_mcp(tenant_id, name)
        if not server:
            raise KeyError(f"no MCP server '{name}' for tenant {tenant_id}")
        added = self.register_mcp_server(server, actor, connect=True)
        server = self.store.get_mcp(tenant_id, name)
        return {"name": name, "status": server.connection_status, "tools": added,
                "error": server.connection_error, "server_info": server.server_info}

    def load_persisted(self) -> None:
        """Rebuild the in-process tool registry from persisted MCP servers at startup.

        Fixes the restart bug: previously, registered MCP tool descriptors lived only in
        memory, so after a process restart the server rows survived but their tools vanished
        from the registry (and any agent granted them silently lost the grant). Here we
        re-register tools from each approved server's cached discovery — binding live proxies
        for servers that were connected, demo callables for demo servers — without needing to
        reconnect on the startup path."""
        for server in self.store.list_all_mcp():
            if not server.approved:
                continue
            if server.discovered_tools:
                self._register_discovered(
                    server, live=(server.connection_status == "connected"))
            elif not _mcp_live():
                # demo server registered before discovery was cached
                server.discovered_tools = list(_DEMO_TOOLS)
                self._register_discovered(server, live=False)

    # ----- MCP internals ------------------------------------------------ #
    def _discover_live(self, server: McpServer) -> tuple[list[McpToolSpec], dict]:
        client = mcp_client.McpClient(server.url, headers=self._resolve_headers(server))
        try:
            init = client.initialize()
            raw = client.list_tools()
        finally:
            client.close()
        tools = [McpToolSpec(**mcp_client.normalize_tool(t)) for t in raw if t.get("name")]
        return tools, init.get("serverInfo", {}) or {}

    def _register_discovered(self, server: McpServer, live: bool) -> list[str]:
        added = []
        for ts in server.discovered_tools:
            full = f"{server.name}.{ts.name}"
            fn = (self._make_proxy(server, ts.name) if live
                  else nt.MCP_CALLABLES.get(ts.name, self._make_missing(full)))
            self.register(
                ToolDescriptor(
                    name=full, kind=ToolKind.MCP, description=ts.description,
                    side_effecting=ts.side_effecting, mcp_server=server.name,
                    tenant_id=server.tenant_id, untrusted_output=True,
                    secrets_ref=server.secrets_ref, parameters=ts.input_schema or {},
                ),
                fn,
            )
            added.append(full)
        return added

    def _make_proxy(self, server: McpServer, tool_name: str):
        """A callable that invokes the real remote tool via `tools/call` at call time. A fresh
        session is opened per call (simple + stateless); secrets are resolved from the vault
        only at call time and never stored in the spec, logs, or trace."""
        url = server.url
        server_name = server.name

        def _proxy(**arguments):
            client = mcp_client.McpClient(url, headers=self._resolve_headers(server))
            try:
                client.initialize()
                result = client.call_tool(tool_name, arguments)
                return mcp_client.result_to_dict(result, server_name, tool_name)
            except mcp_client.McpError as e:
                return {"_untrusted": True, "source": f"mcp:{server_name}",
                        "tool": tool_name, "error": str(e), "isError": True}
            finally:
                client.close()

        _proxy.__name__ = f"mcp_proxy_{server_name}_{tool_name}"
        return _proxy

    @staticmethod
    def _make_missing(full_name: str):
        def _missing(**_):
            return {"_untrusted": True, "source": "mcp", "tool": full_name,
                    "error": "no callable bound for this MCP tool", "isError": True}
        return _missing

    @staticmethod
    def _resolve_headers(server: McpServer) -> dict:
        """Resolve the server's secret refs into HTTP headers at call time.

        Convention per resolved secret value:
          * "Header-Name: value"  -> sets that custom header,
          * "Bearer <token>"      -> used verbatim as Authorization,
          * "<token>"             -> sent as `Authorization: Bearer <token>`.
        """
        from .vault import vault
        headers: dict = {}
        for ref in server.secrets_ref:
            val = vault.resolve(ref, server.tenant_id)
            if not val:
                continue
            if ":" in val and not val.lower().startswith("bearer "):
                name, _, v = val.partition(":")
                headers[name.strip()] = v.strip()
            elif val.lower().startswith("bearer "):
                headers["Authorization"] = val
            else:
                headers["Authorization"] = f"Bearer {val}"
        return headers

    # ----- lookups ------------------------------------------------------ #
    def get(self, name: str) -> ToolDescriptor | None:
        return self._descriptors.get(name)

    def callable(self, name: str) -> callable | None:
        return self._callables.get(name)

    def visible_to(self, tenant_id: str) -> list[ToolDescriptor]:
        """Tools a tenant may see: globals + that tenant's approved MCP/external tools."""
        return [d for d in self._descriptors.values() if d.tenant_id in (None, tenant_id)]

    def grantable_for_tenant(self, tenant_id: str) -> list[str]:
        return [d.name for d in self.visible_to(tenant_id)]

    def tenant_external(self, tenant_id: str) -> list[str]:
        """Approved MCP/external tools scoped to this tenant (added to a template's grantable
        universe so they can be granted via the form free-text, least-privilege intact)."""
        return [d.name for d in self._descriptors.values()
                if d.tenant_id == tenant_id
                and d.kind in (ToolKind.MCP, ToolKind.EXTERNAL_API)]

    def effective_grantable(self, grantable_tools: list[str], tenant_id: str) -> list[str]:
        return list(dict.fromkeys(list(grantable_tools) + self.tenant_external(tenant_id)))

    def is_grantable(self, tenant_id: str, name: str) -> bool:
        d = self._descriptors.get(name)
        return bool(d and d.tenant_id in (None, tenant_id))
