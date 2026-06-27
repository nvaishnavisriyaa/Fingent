"""
Tool Registry (§1, §6) — the single place tools are described, scoped, and bound.

Holds ToolDescriptors for four kinds: NATIVE, WEB_SEARCH, MCP, EXTERNAL_API. Native + web are
global; MCP/external tools are tenant-scoped and only become grantable once their MCP server is
approved. The LLM compiler reads descriptors to decide grants; the factory binds the callable.
"""
from __future__ import annotations

from .schemas import McpServer, ToolDescriptor, ToolKind
from .store import Store
from . import tools_native as nt


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

    def register_mcp_server(self, server: McpServer, actor: str = "admin") -> list[str]:
        """Admin registers an MCP server; we enumerate + add its tools as descriptors.
        Tools become grantable only when server.approved is True. Tenant-scoped."""
        self.store.save_mcp(server)
        self.store.audit(server.tenant_id, actor, "mcp_register", server.name,
                         {"approved": server.approved})
        added = []
        if not server.approved:
            return added
        catalog = {
            "bloomberg_quote": dict(desc="Get a live equity quote.", side=False),
            "send_email": dict(desc="Send an email on the user's behalf.", side=True),
        }
        for tname, meta in catalog.items():
            full = f"{server.name}.{tname}"
            self.register(
                ToolDescriptor(
                    name=full, kind=ToolKind.MCP, description=meta["desc"],
                    side_effecting=meta["side"], mcp_server=server.name,
                    tenant_id=server.tenant_id, untrusted_output=True,
                    secrets_ref=server.secrets_ref,
                ),
                nt.MCP_CALLABLES[tname],
            )
            added.append(full)
        self.store.audit(server.tenant_id, actor, "mcp_approve", server.name, {"tools": added})
        return added

    def approve_mcp_server(self, tenant_id: str, name: str, actor: str = "admin") -> list[str]:
        server = self.store.get_mcp(tenant_id, name)
        if not server:
            raise KeyError(f"no MCP server '{name}' for tenant {tenant_id}")
        server.approved = True
        return self.register_mcp_server(server, actor)

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
