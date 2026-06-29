"""Tenant isolation is REAL, not cosmetic.

- Open dev mode (default): the X-Tenant header selects the tenant and data is fully store-scoped,
  so tenant A cannot see tenant B's agents or runs.
- Auth mode: the tenant comes from the session and the X-Tenant header is IGNORED (unspoofable).
"""
import importlib
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient


def _fresh_app(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("FINGENT_DB", ":memory:")
    import fingent.app as app_mod
    return importlib.reload(app_mod)


def test_open_dev_mode_isolates_by_header(monkeypatch):
    app_mod = _fresh_app(monkeypatch)  # no auth -> header honored by default
    c = TestClient(app_mod.app)
    # create an agent for tenant acme
    r = c.post("/api/agents", headers={"X-Tenant": "acme"},
               json={"template": "fraud_anomaly", "answers": {"name": "acme_fraud"}})
    assert r.json().get("ok"), r.text
    acme = {s["name"] for s in c.get("/api/agents", headers={"X-Tenant": "acme"}).json()}
    globex = {s["name"] for s in c.get("/api/agents", headers={"X-Tenant": "globex"}).json()}
    assert "acme_fraud" in acme
    assert "acme_fraud" not in globex, "tenant globex must NOT see acme's agent"
    assert globex == set(), "a different tenant starts empty (isolated)"


def test_auth_mode_ignores_header_uses_session(monkeypatch):
    users = '{"alice": {"password": "pw", "tenant": "acme", "roles": ["admin"]},' \
            ' "bob": {"password": "pw", "tenant": "globex", "roles": ["admin"]}}'
    app_mod = _fresh_app(monkeypatch, FINGENT_AUTH="1", FINGENT_USERS=users)
    c = TestClient(app_mod.app)
    tok_a = c.post("/api/login", json={"username": "alice", "password": "pw"}).json()["token"]
    tok_b = c.post("/api/login", json={"username": "bob", "password": "pw"}).json()["token"]
    # alice creates an agent; she sends a spoofed X-Tenant: globex header that MUST be ignored
    c.post("/api/agents", headers={"Authorization": f"Bearer {tok_a}", "X-Tenant": "globex"},
           json={"template": "fraud_anomaly", "answers": {"name": "alice_fraud"}})
    # bob (real tenant globex) must NOT see alice's agent despite alice's spoof attempt
    bob_sees = {s["name"] for s in c.get(
        "/api/agents", headers={"Authorization": f"Bearer {tok_b}"}).json()}
    alice_sees = {s["name"] for s in c.get(
        "/api/agents", headers={"Authorization": f"Bearer {tok_a}"}).json()}
    assert "alice_fraud" in alice_sees
    assert "alice_fraud" not in bob_sees, "session tenant must win over a spoofed header"
