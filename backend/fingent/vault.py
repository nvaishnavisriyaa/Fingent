"""
Secrets vault (§10).

Credentials are *never* stored in specs, logs, or the free-text round-trip. Tools declare
`secrets_ref` names; the factory asks the vault to resolve them at call time only. This is a
demo in-memory vault; in production back it with a real KMS / Supabase vault / cloud secret store.
"""
from __future__ import annotations

import os


class Vault:
    def __init__(self) -> None:
        # demo seed; real secrets come from env / KMS, keyed by tenant where relevant
        self._store: dict[str, str] = {
            "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
            "acme:bloomberg_api_key": "sk-demo-bloomberg-xxxx",
            "acme:experian_api_key": "sk-demo-experian-xxxx",
        }

    def put(self, ref: str, value: str) -> None:
        self._store[ref] = value

    def resolve(self, ref: str) -> str | None:
        return self._store.get(ref)

    def resolve_all(self, refs: list[str]) -> dict[str, str]:
        return {r: self._store[r] for r in refs if r in self._store}

    def has(self, ref: str) -> bool:
        return bool(self._store.get(ref))


vault = Vault()
