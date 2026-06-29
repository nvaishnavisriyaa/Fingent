"""
Auth primitives — password hashing + role-based access control (RBAC).

Passwords are stored hashed (PBKDF2-HMAC-SHA256). RBAC maps roles -> permissions; endpoints
require a permission rather than a specific user. Sessions and per-agent deploy tokens are
persisted by the store (so they survive a restart) — this module only holds the pure logic.

This is deliberately self-contained stdlib code; for production, federate login to a real IdP
(OIDC/SAML) and have it mint the bearer token this layer already understands.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

# role -> granted permissions
ROLE_PERMISSIONS: dict[str, set[str]] = {
    "viewer":   {"read"},
    "reviewer": {"read", "review"},
    "operator": {"read", "review", "write", "deploy", "invoke"},
    "admin":    {"read", "review", "write", "deploy", "invoke", "admin"},
}

_ITERATIONS = 120_000


def hash_password(password: str, salt: str | None = None) -> str:
    salt = salt or secrets.token_hex(8)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS).hex()
    return f"pbkdf2${salt}${dk}"


def password_is_hashed(stored: str) -> bool:
    return isinstance(stored, str) and stored.startswith("pbkdf2$")


def verify_password(password: str, stored: str, allow_plaintext: bool = True) -> bool:
    """Verify against a pbkdf2 hash. A plaintext stored value is accepted ONLY when
    allow_plaintext is True (dev). In secure mode the caller passes allow_plaintext=False so a
    non-hashed stored password can never authenticate."""
    if password_is_hashed(stored):
        try:
            _, salt, dk = stored.split("$", 2)
        except ValueError:
            return False
        cand = hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), _ITERATIONS).hex()
        return hmac.compare_digest(cand, dk)
    if not allow_plaintext:
        return False
    return hmac.compare_digest(password, stored)


def permissions_for(roles: list[str]) -> set[str]:
    perms: set[str] = set()
    for r in roles or []:
        perms |= ROLE_PERMISSIONS.get(r, set())
    return perms


def new_token() -> str:
    return secrets.token_urlsafe(24)
