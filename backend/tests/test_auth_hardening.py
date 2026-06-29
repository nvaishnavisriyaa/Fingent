"""Auth hardening: hashing, plaintext rejection, session TTL, lockout, secure posture."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import fingent.auth as auth
from fingent.store import Store
from fastapi.testclient import TestClient


def test_hash_roundtrip_and_plaintext_rejection():
    h = auth.hash_password("pw")
    assert auth.password_is_hashed(h)
    assert auth.verify_password("pw", h, allow_plaintext=False)
    assert not auth.verify_password("nope", h, allow_plaintext=False)
    # a plaintext-stored password must NOT authenticate in secure mode
    assert not auth.verify_password("pw", "pw", allow_plaintext=False)
    assert auth.verify_password("pw", "pw", allow_plaintext=True)   # dev only


def test_session_expiry():
    st = Store(tempfile.mktemp(suffix=".db"))
    st.create_session("t_live", "acme", "u", ["admin"], ttl_seconds=100)
    assert st.get_session("t_live")["username"] == "u"
    st.create_session("t_expired", "acme", "u", ["admin"], ttl_seconds=-1)
    assert st.get_session("t_expired") is None           # expired -> rejected
    assert st.get_session("t_expired") is None           # and purged
    st.create_session("t_forever", "acme", "u", ["admin"])
    assert st.get_session("t_forever") is not None       # no ttl -> never expires


def test_signup_creates_hashed_user_and_session(monkeypatch):
    import importlib
    monkeypatch.setenv("FINGENT_DB", ":memory:")
    import fingent.app as app
    app = importlib.reload(app)
    c = TestClient(app.app)
    r = c.post("/api/signup", json={"username": "keerthi", "password": "pass123"})
    assert r.status_code == 200, r.text
    tok = r.json()["token"]
    me = c.get("/api/me", headers={"Authorization": f"Bearer {tok}"}).json()
    assert me["principal"] == "keerthi" and me["tenant"].startswith("user_keerthi_")
    stored = app.fp.store.get_user("keerthi")
    assert auth.password_is_hashed(stored["password"])
    assert c.post("/api/signup", json={"username": "keerthi", "password": "pass123"}).status_code == 409


def test_signup_users_share_templates_but_not_agents(monkeypatch):
    import importlib
    monkeypatch.setenv("FINGENT_DB", ":memory:")
    import fingent.app as app
    app = importlib.reload(app)
    c = TestClient(app.app)
    alice = c.post("/api/signup", json={"username": "alice", "password": "pass123"}).json()
    bob = c.post("/api/signup", json={"username": "bob", "password": "pass123"}).json()
    assert alice["tenant"] != bob["tenant"]
    ha = {"Authorization": f"Bearer {alice['token']}"}
    hb = {"Authorization": f"Bearer {bob['token']}"}

    alice_templates = {t["name"] for t in c.get("/api/templates", headers=ha).json()}
    bob_templates = {t["name"] for t in c.get("/api/templates", headers=hb).json()}
    assert alice_templates == bob_templates and "fraud_anomaly" in alice_templates

    r = c.post("/api/agents", headers=ha,
               json={"template": "fraud_anomaly", "answers": {"name": "alice_fraud"}})
    assert r.status_code == 200, r.text
    alice_agents = {s["name"] for s in c.get("/api/agents", headers=ha).json()}
    bob_agents = {s["name"] for s in c.get("/api/agents", headers=hb).json()}
    assert "alice_fraud" in alice_agents
    assert "alice_fraud" not in bob_agents


def test_login_lockout_helpers():
    import fingent.app as app
    app._clear_login_failures("bob")
    for _ in range(app._LOGIN_MAX_ATTEMPTS):
        assert app._login_locked("bob") == 0.0
        app._record_login_failure("bob")
    assert app._login_locked("bob") > 0                  # locked after max failures
    app._clear_login_failures("bob")
    assert app._login_locked("bob") == 0.0               # cleared on success


def test_secure_posture_accepts_hardened_config(monkeypatch):
    import fingent.app as app
    monkeypatch.setattr(app, "_SECURE", True)
    monkeypatch.setattr(app, "_AUTH_REQUIRED", True)
    monkeypatch.setattr(app, "_ALLOW_HEADER_TENANT", False)
    monkeypatch.setenv("FINGENT_VAULT_KEY", "a-long-random-production-key")
    monkeypatch.setattr(app, "_USERS",
                        {"alice": {"password": auth.hash_password("x"),
                                   "tenant": "acme", "roles": ["admin"]}})
    app._audit_security_posture()   # must not raise


def test_secure_posture_rejects_plaintext_and_default(monkeypatch):
    import fingent.app as app
    monkeypatch.setattr(app, "_SECURE", True)
    monkeypatch.setattr(app, "_AUTH_REQUIRED", True)
    monkeypatch.setattr(app, "_ALLOW_HEADER_TENANT", False)
    monkeypatch.setattr(app, "_USERS",
                        {"admin": {"password": "admin", "tenant": "acme", "roles": ["admin"]}})
    with pytest.raises(RuntimeError):
        app._audit_security_posture()
