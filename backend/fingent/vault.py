"""
Secrets vault - encrypted-at-rest credential store for real external-tool credentials.

Credentials are NEVER stored in specs, logs, the compiler round-trip, or returned by any list
endpoint. Tools (and MCP servers) declare `secrets_ref` names; the registry asks the vault to
resolve them into live values at call time only.

Storage & encryption:
  * Values are encrypted with Fernet (AES-128-CBC + HMAC) before they touch the database, and
    only ever decrypted in-process at the moment a tool call needs them.
  * The encryption key comes from FINGENT_VAULT_KEY if set (recommended: inject from a real
    KMS / cloud secret manager). If it is not set, a random key is generated once and persisted
    in the store so credentials survive restarts on the same database. Setting FINGENT_VAULT_KEY
    (so the key lives outside the DB) is the production posture.
  * Process secrets from the environment (e.g. GROQ_API_KEY) are resolved from the environment
    directly and are never written to the credential table.

Tenant isolation: persisted credentials are keyed by (tenant_id, ref); resolve() takes the
tenant so Tenant A can never read Tenant B's credentials.
"""
from __future__ import annotations

import base64
import hashlib
import os


def _derive_fernet(key_material: str):
    from cryptography.fernet import Fernet
    digest = hashlib.sha256(key_material.encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


class Vault:
    def __init__(self, store=None) -> None:
        self.store = store
        self._fernet = None
        if store is not None:
            self._fernet = self._init_cipher()

    def attach_store(self, store) -> None:
        """Bind the vault to a persistent store and initialize the cipher."""
        self.store = store
        self._fernet = self._init_cipher()

    def _init_cipher(self):
        key_material = os.getenv("FINGENT_VAULT_KEY")
        if not key_material and self.store is not None:
            key_material = self.store.vault_meta_get("vault_key")
            if not key_material:
                from cryptography.fernet import Fernet
                key_material = Fernet.generate_key().decode()
                self.store.vault_meta_set("vault_key", key_material)
        return _derive_fernet(key_material or "fingent-ephemeral-key")

    # ----- credential management (encrypted) ----------------------------- #
    def put(self, ref: str, value: str, tenant_id: str = "*", actor: str = "") -> None:
        """Encrypt and persist a tenant-scoped credential value."""
        if self.store is None or self._fernet is None:
            raise RuntimeError("vault has no store attached")
        ciphertext = self._fernet.encrypt(value.encode()).decode()
        self.store.save_credential(tenant_id, ref, ciphertext, actor)

    def resolve(self, ref: str, tenant_id: str | None = None) -> str | None:
        """Return the live secret value, or None. Order: tenant credential -> environment."""
        if self.store is not None and self._fernet is not None and tenant_id:
            ct = self.store.get_credential_ciphertext(tenant_id, ref)
            if ct:
                try:
                    return self._fernet.decrypt(ct.encode()).decode()
                except Exception:  # noqa: BLE001 - bad/rotated key: fail closed, never crash
                    return None
        env = os.getenv(ref)
        return env or None

    def resolve_all(self, refs: list[str], tenant_id: str | None = None) -> dict:
        """Resolve a list of refs into {ref: value} for those that exist (telemetry + MCP
        header assembly). Values for missing refs are simply omitted."""
        out: dict = {}
        for ref in (refs or []):
            val = self.resolve(ref, tenant_id)
            if val:
                out[ref] = val
        return out

    def list(self, tenant_id: str) -> list[dict]:
        """Metadata only - values are never returned."""
        return self.store.list_credentials(tenant_id) if self.store is not None else []

    def delete(self, ref: str, tenant_id: str) -> None:
        if self.store is not None:
            self.store.delete_credential(tenant_id, ref)

    def has(self, ref: str, tenant_id: str | None = None) -> bool:
        return bool(self.resolve(ref, tenant_id))


vault = Vault()
