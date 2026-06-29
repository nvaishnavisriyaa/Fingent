"""Auth hardening: hashing, plaintext rejection, session TTL, lockout, secure posture."""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
import fingent.auth as auth
from fingent.store import Store


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
