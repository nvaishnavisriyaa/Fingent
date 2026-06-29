"""Native tools must use the CALLING TENANT's vault-configured credentials/endpoints,
falling back to process env. This is what makes the agents configurable per customer."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import fingent.tools_native as t
from fingent.store import Store
from fingent.vault import vault


def _wire_vault():
    os.environ["FINGENT_VAULT_KEY"] = "tool-cred-test"
    vault.attach_store(Store(tempfile.mktemp(suffix=".db")))


def test_account_lookup_uses_tenant_configured_endpoint(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.delenv("ACCOUNT_API_URL", raising=False)   # NOT in env
    _wire_vault()
    # tenant 'acme' configures their own core-banking endpoint via the vault
    vault.put("ACCOUNT_API_URL", "https://acme.internal/accounts", tenant_id="acme")
    seen = {}

    def fake_get(url, params=None, **k):
        seen["url"] = url
        class R:
            ok = True
            def json(self): return {"account_id": params["account_id"], "balance_usd": 5}
        return R()
    monkeypatch.setattr(t, "_get", fake_get)

    tok = t.set_current_tenant("acme")
    try:
        r = t.account_lookup("ACC-9")
    finally:
        t.reset_current_tenant(tok)
    assert seen["url"] == "https://acme.internal/accounts"
    assert r["source"] == "live:account_api"


def test_other_tenant_does_not_see_acme_endpoint(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.delenv("ACCOUNT_API_URL", raising=False)
    _wire_vault()
    vault.put("ACCOUNT_API_URL", "https://acme.internal/accounts", tenant_id="acme")
    # globex has configured nothing -> tool stays honest 'unavailable', no cross-tenant leak
    tok = t.set_current_tenant("globex")
    try:
        r = t.account_lookup("ACC-9")
    finally:
        t.reset_current_tenant(tok)
    assert r["source"] == "unavailable"


def test_env_fallback_when_no_tenant_credential(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "1")
    monkeypatch.setenv("HUNTER_API_KEY", "env-hunter-key")
    _wire_vault()
    # no tenant credential set; _secret falls back to env
    tok = t.set_current_tenant("acme")
    try:
        assert t._secret("HUNTER_API_KEY") == "env-hunter-key"
    finally:
        t.reset_current_tenant(tok)


def test_resolve_all_is_tenant_scoped():
    _wire_vault()
    vault.put("acme:h", "v1", tenant_id="acme")
    assert vault.resolve_all(["acme:h"], "acme") == {"acme:h": "v1"}
    assert vault.resolve_all(["acme:h"], "globex") == {}


def test_tool_credential_catalog_covers_keyed_tools():
    from fingent.tools_native import TOOL_CREDENTIALS, tool_credentials
    # every native tool that reads a credential/endpoint must be discoverable in the catalog
    for tool in ("find_persona", "resolve_contact", "pep_check", "ocr_extract",
                 "account_lookup", "identity_verify", "web_search"):
        reqs = tool_credentials(tool)
        assert reqs, f"{tool} missing from credential catalog"
        assert all("ref" in r and "label" in r and "required" in r for r in reqs)
