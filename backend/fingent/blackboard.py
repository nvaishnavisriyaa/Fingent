"""
Shared memory — the Blackboard (§1).

Namespaced, versioned, deduplicated key/value store that every agent node shares within a run.
Access is mediated by SecurityPolicy prefixes (least-privilege memory ACL, §10): an agent may
only READ keys under one of its `memory_read` prefixes and WRITE under `memory_write` prefixes.
Everything is scoped by tenant_id so Tenant A can never see Tenant B's memory.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field


class MemoryAccessDenied(Exception):
    pass


@dataclass
class Version:
    value: object
    ts: float
    writer: str
    digest: str


@dataclass
class Blackboard:
    tenant_id: str
    # key -> list of versions (latest last)
    _data: dict[str, list[Version]] = field(default_factory=dict)
    # dedup is PER KEY: writing the same payload to a different key is NOT a duplicate.
    _seen_digests: dict[str, set[str]] = field(default_factory=dict)

    @staticmethod
    def _digest(value: object) -> str:
        return hashlib.sha256(
            json.dumps(value, sort_keys=True, default=str).encode()
        ).hexdigest()

    @staticmethod
    def _prefix_ok(key: str, prefixes: list[str]) -> bool:
        return any(key == p or key.startswith(p) for p in prefixes)

    def write(self, key: str, value: object, writer: str, allowed_write: list[str]) -> bool:
        if not self._prefix_ok(key, allowed_write):
            raise MemoryAccessDenied(
                f"agent '{writer}' may not write '{key}' (allowed: {allowed_write})"
            )
        digest = self._digest(value)
        # dedup PER KEY: an identical payload already at THIS key is a no-op write, but the
        # same value under a different key is legitimate and is kept.
        seen = self._seen_digests.setdefault(key, set())
        if digest in seen:
            return False
        seen.add(digest)
        self._data.setdefault(key, []).append(
            Version(value=value, ts=time.time(), writer=writer, digest=digest)
        )
        return True

    def read(self, key: str, reader: str, allowed_read: list[str]):
        if not self._prefix_ok(key, allowed_read):
            raise MemoryAccessDenied(
                f"agent '{reader}' may not read '{key}' (allowed: {allowed_read})"
            )
        versions = self._data.get(key)
        return versions[-1].value if versions else None

    def history(self, key: str) -> list[Version]:
        return list(self._data.get(key, []))

    def snapshot(self) -> dict[str, object]:
        return {k: v[-1].value for k, v in self._data.items()}

    def keys(self) -> list[str]:
        return list(self._data.keys())
