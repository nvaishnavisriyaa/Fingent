"""Credential vault: real encryption at rest, tenant isolation, no plaintext leak."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fingent.store import Store
from fingent.vault import Vault


def _vault():
    os.environ["FINGENT_VAULT_KEY"] = "unit-test-master-key"
    db = tempfile.mktemp(suffix=".db")
    v = Vault()
    v.attach_store(Store(db))
    return v, db


def test_credential_is_encrypted_at_rest():
    v, db = _vault()
    v.put("acme:crm_token", "PLAINTEXT-SECRET-1234", tenant_id="acme", actor="admin")
    raw = open(db, "rb").read()
    assert b"PLAINTEXT-SECRET-1234" not in raw            # never on disk in clear
    assert v.resolve("acme:crm_token", "acme") == "PLAINTEXT-SECRET-1234"


def test_tenant_isolation():
    v, _ = _vault()
    v.put("x:token", "acme-only", tenant_id="acme")
    assert v.resolve("x:token", "acme") == "acme-only"
    assert v.resolve("x:token", "globex") is None         # cross-tenant read blocked


def test_list_never_returns_values():
    v, _ = _vault()
    v.put("acme:k", "secret", tenant_id="acme", actor="admin")
    meta = v.list("acme")
    assert meta and meta[0]["ref"] == "acme:k"
    assert "secret" not in str(meta) and "value" not in meta[0]


def test_env_fallback_and_delete():
    v, _ = _vault()
    os.environ["MY_ENV_KEY"] = "env-value"
    assert v.resolve("MY_ENV_KEY", "acme") == "env-value"  # process secret via env
    v.put("acme:tmp", "v", tenant_id="acme")
    v.delete("acme:tmp", "acme")
    assert v.resolve("acme:tmp", "acme") is None


def test_decrypt_fails_closed_on_bad_key():
    # a credential encrypted under one key must not decrypt under another (returns None, no crash)
    os.environ["FINGENT_VAULT_KEY"] = "key-A"
    db = tempfile.mktemp(suffix=".db")
    st = Store(db)
    v1 = Vault(); v1.attach_store(st)
    v1.put("acme:k", "topsecret", tenant_id="acme")
    # simulate a different master key (env wins over persisted) on a fresh vault, same store
    os.environ["FINGENT_VAULT_KEY"] = "key-B"
    v2 = Vault(); v2.attach_store(st)
    assert v2.resolve("acme:k", "acme") is None
