"""
Persistence (§13) — a tenant-scoped store for specs, templates, MCP registry, run logs, the
immutable audit trail, HITL decisions, compiler logs, traces, runs, auth sessions, deploy tokens
and encrypted credentials.

Production posture (this module):
  * Pluggable backend — **SQLite** (default; a real FILE, never `:memory:`, in the served app)
    or **Postgres** when `DATABASE_URL`/`FINGENT_DATABASE_URL` is set (`postgres://…`). The SQL is
    written once with `?` placeholders + an `_upsert()` helper; the Postgres adapter translates
    placeholders and `REPLACE` → `INSERT … ON CONFLICT`. SQLite is the tested path; Postgres is the
    documented production swap (needs `psycopg`).
  * SQLite concurrency — WAL journal + busy_timeout so readers don't block writers; a re-entrant
    lock serializes writes (FastAPI threadpool + the JobRunner share one connection).
  * Real, ordered, recorded MIGRATIONS (a `schema_migrations` table) instead of ad-hoc ALTERs.
  * INDEXES on every hot tenant/time/status column, and PROMOTED queryable columns (runs.status,
    runs.risk_level, traces.cost_usd/tokens, …) so analytics aggregates in **SQL**, not by loading
    every full trace blob into Python.

Tenant isolation (§10): every read takes a tenant_id and filters on it.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
import uuid
from typing import Any

from .schemas import AgentSpec, AgentTemplate, McpServer


def _now() -> float:
    return time.time()


# --------------------------------------------------------------------------- #
# Backend connections — a uniform interface over SQLite and Postgres
# --------------------------------------------------------------------------- #
class _LockedCursor:
    def __init__(self, rows) -> None:
        self._rows = list(rows)
        self._idx = 0

    def fetchone(self):
        if self._idx < len(self._rows):
            row = self._rows[self._idx]
            self._idx += 1
            return row
        return None

    def fetchall(self):
        rows = self._rows[self._idx:]
        self._idx = len(self._rows)
        return rows


class _SqliteConn:
    """Serializes access to one shared SQLite connection through a re-entrant lock."""

    def __init__(self, path: str) -> None:
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        if path != ":memory:":                 # durable file DB: enable WAL for read/write concurrency
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            except Exception:                  # noqa: BLE001
                pass
        self._conn.execute("PRAGMA busy_timeout=5000")
        self._conn.execute("PRAGMA foreign_keys=ON")

    dialect = "sqlite"
    placeholder = "?"

    def execute(self, sql: str, params: tuple = ()):
        with self._lock:
            cur = self._conn.execute(sql, params)
            rows = cur.fetchall()
            return _LockedCursor(rows)

    def commit(self):
        with self._lock:
            self._conn.commit()


class _PgConn:
    """Postgres adapter (psycopg 3). Same interface as _SqliteConn; translates `?` → `%s` and
    returns dict rows. The tested path is SQLite; this is the production swap."""

    dialect = "postgres"
    placeholder = "%s"

    def __init__(self, url: str) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(
                "DATABASE_URL points at Postgres but `psycopg` is not installed. "
                "Add `psycopg[binary]>=3.1` to requirements and reinstall.") from e
        self._psycopg = psycopg
        self._dict_row = dict_row
        self._lock = threading.RLock()
        self._conn = psycopg.connect(url, autocommit=False)

    def execute(self, sql: str, params: tuple = ()):
        sql = sql.replace("?", "%s")
        with self._lock:
            with self._conn.cursor(row_factory=self._dict_row) as cur:
                cur.execute(sql, params)
                rows = cur.fetchall() if cur.description else []
            return _LockedCursor(rows)

    def commit(self):
        with self._lock:
            self._conn.commit()


def _connect(path: str):
    if path.startswith(("postgres://", "postgresql://")):
        return _PgConn(path)
    return _SqliteConn(path)


# --------------------------------------------------------------------------- #
# Migrations — ordered, recorded, idempotent (Alembic is the production swap)
# --------------------------------------------------------------------------- #
_MIGRATIONS: list[tuple[int, str, list[str]]] = [
    (1, "base_schema", [
        """CREATE TABLE IF NOT EXISTS specs(
            tenant_id TEXT, name TEXT, enabled INTEGER DEFAULT 1,
            json TEXT, updated REAL, PRIMARY KEY(tenant_id, name))""",
        """CREATE TABLE IF NOT EXISTS templates(
            name TEXT PRIMARY KEY, tenant_id TEXT, json TEXT, updated REAL)""",
        """CREATE TABLE IF NOT EXISTS mcp_servers(
            tenant_id TEXT, name TEXT, json TEXT, PRIMARY KEY(tenant_id, name))""",
        """CREATE TABLE IF NOT EXISTS run_logs(
            id TEXT PRIMARY KEY, trace_id TEXT, tenant_id TEXT, ts REAL, json TEXT)""",
        """CREATE TABLE IF NOT EXISTS audit(
            id TEXT PRIMARY KEY, ts REAL, tenant_id TEXT, actor TEXT,
            action TEXT, target TEXT, detail TEXT)""",
        """CREATE TABLE IF NOT EXISTS hitl(
            id TEXT PRIMARY KEY, ts REAL, tenant_id TEXT, trace_id TEXT,
            actor TEXT, decision TEXT, diff TEXT)""",
        """CREATE TABLE IF NOT EXISTS compiler_logs(
            id TEXT PRIMARY KEY, ts REAL, tenant_id TEXT, agent TEXT, json TEXT)""",
        """CREATE TABLE IF NOT EXISTS traces(
            trace_id TEXT PRIMARY KEY, tenant_id TEXT, ts REAL, agent TEXT, status TEXT,
            cost_usd REAL DEFAULT 0, tokens INTEGER DEFAULT 0, metrics TEXT, json TEXT)""",
        """CREATE TABLE IF NOT EXISTS runs(
            id TEXT PRIMARY KEY, tenant_id TEXT, agent TEXT, status TEXT,
            risk_level TEXT, mode TEXT, ts REAL, json TEXT)""",
        """CREATE TABLE IF NOT EXISTS sessions(
            token TEXT PRIMARY KEY, tenant_id TEXT, username TEXT, roles TEXT,
            created REAL, expires REAL)""",
        """CREATE TABLE IF NOT EXISTS deploy_tokens(
            token TEXT PRIMARY KEY, tenant_id TEXT, agent TEXT, label TEXT, created REAL)""",
        """CREATE TABLE IF NOT EXISTS credentials(
            tenant_id TEXT, ref TEXT, ciphertext TEXT, created REAL, created_by TEXT,
            PRIMARY KEY(tenant_id, ref))""",
        """CREATE TABLE IF NOT EXISTS vault_meta(k TEXT PRIMARY KEY, v TEXT)""",
    ]),
    (2, "indexes", [
        "CREATE INDEX IF NOT EXISTS ix_runs_tenant_ts ON runs(tenant_id, ts)",
        "CREATE INDEX IF NOT EXISTS ix_runs_tenant_status ON runs(tenant_id, status)",
        "CREATE INDEX IF NOT EXISTS ix_runs_tenant_agent ON runs(tenant_id, agent)",
        "CREATE INDEX IF NOT EXISTS ix_traces_tenant_ts ON traces(tenant_id, ts)",
        "CREATE INDEX IF NOT EXISTS ix_audit_tenant_ts ON audit(tenant_id, ts)",
        "CREATE INDEX IF NOT EXISTS ix_runlogs_tenant_trace ON run_logs(tenant_id, trace_id)",
        "CREATE INDEX IF NOT EXISTS ix_complogs_tenant_ts ON compiler_logs(tenant_id, ts)",
        "CREATE INDEX IF NOT EXISTS ix_specs_tenant ON specs(tenant_id)",
        "CREATE INDEX IF NOT EXISTS ix_sessions_expires ON sessions(expires)",
    ]),
    # legacy upgrades for file DBs created before columns were promoted (tolerated if present)
    (3, "promote_columns_legacy", [
        "ALTER TABLE runs ADD COLUMN risk_level TEXT",
        "ALTER TABLE runs ADD COLUMN mode TEXT",
        "ALTER TABLE traces ADD COLUMN agent TEXT",
        "ALTER TABLE traces ADD COLUMN status TEXT",
        "ALTER TABLE traces ADD COLUMN cost_usd REAL DEFAULT 0",
        "ALTER TABLE traces ADD COLUMN tokens INTEGER DEFAULT 0",
        "ALTER TABLE traces ADD COLUMN metrics TEXT",
        "ALTER TABLE sessions ADD COLUMN expires REAL",
    ]),
    (4, "jobs_queue", [
        """CREATE TABLE IF NOT EXISTS jobs(
            id TEXT PRIMARY KEY, tenant_id TEXT, agent TEXT, payload TEXT,
            idempotency_key TEXT, status TEXT, attempts INTEGER DEFAULT 0,
            max_attempts INTEGER DEFAULT 3, available_at REAL, lease_until REAL,
            worker TEXT, last_error TEXT, created REAL, updated REAL)""",
        "CREATE INDEX IF NOT EXISTS ix_jobs_claim ON jobs(status, available_at)",
        "CREATE INDEX IF NOT EXISTS ix_jobs_tenant ON jobs(tenant_id)",
        "CREATE INDEX IF NOT EXISTS ix_jobs_idem ON jobs(tenant_id, idempotency_key)",
    ]),
    # Long-term agent memory (vector store) — DURABLE + tenant-scoped. Each row is one memory
    # with its embedding vector, the embedder that produced it (so recall never mixes vectors
    # from different models), and the namespace (tenant_id, agent). Replaces the in-process dict.
    (5, "memories", [
        """CREATE TABLE IF NOT EXISTS memories(
            id TEXT PRIMARY KEY, tenant_id TEXT, agent TEXT, text TEXT, meta TEXT,
            embedder TEXT, dim INTEGER, vec TEXT, ts REAL)""",
        "CREATE INDEX IF NOT EXISTS ix_memories_ns ON memories(tenant_id, agent)",
    ]),
    (6, "local_users", [
        """CREATE TABLE IF NOT EXISTS users(
            username TEXT PRIMARY KEY, password TEXT, tenant_id TEXT, roles TEXT, created REAL)""",
    ]),
]


class Store:
    def __init__(self, path: str | None = None) -> None:
        path = path or os.getenv("FINGENT_DB") or ":memory:"
        self.db = _connect(path)
        self.dialect = self.db.dialect
        self._ph = self.db.placeholder
        self._migrate()

    # ----- migration runner ---------------------------------------------- #
    def _migrate(self) -> None:
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS schema_migrations("
            "version INTEGER PRIMARY KEY, name TEXT, applied REAL)")
        self.db.commit()
        applied = {r["version"] for r in
                   self.db.execute("SELECT version FROM schema_migrations").fetchall()}
        for version, name, statements in _MIGRATIONS:
            if version in applied:
                continue
            for stmt in statements:
                try:
                    self.db.execute(stmt)
                except Exception as e:  # noqa: BLE001 — tolerate "duplicate column" on legacy DBs
                    if "duplicate column" not in str(e).lower() and "already exists" not in str(e).lower():
                        raise
            self.db.execute(
                "INSERT INTO schema_migrations(version,name,applied) VALUES(?,?,?)",
                (version, name, _now()))
            self.db.commit()

    # ----- dialect-aware upsert ------------------------------------------ #
    def _upsert(self, table: str, data: dict, conflict: tuple) -> None:
        cols = list(data)
        ph = ",".join([self._ph] * len(cols))
        collist = ",".join(cols)
        if self.dialect == "postgres":
            updates = ",".join(f"{c}=EXCLUDED.{c}" for c in cols if c not in conflict)
            sql = (f"INSERT INTO {table}({collist}) VALUES({ph}) "
                   f"ON CONFLICT ({','.join(conflict)}) DO UPDATE SET {updates}")
        else:
            sql = f"REPLACE INTO {table}({collist}) VALUES({ph})"
        self.db.execute(sql, tuple(data[c] for c in cols))
        self.db.commit()

    # ----- specs ---------------------------------------------------------- #
    def save_spec(self, spec: AgentSpec) -> None:
        self._upsert("specs", {"tenant_id": spec.security.tenant_id, "name": spec.name,
                               "enabled": 1, "json": spec.model_dump_json(), "updated": _now()},
                     ("tenant_id", "name"))

    def get_spec(self, tenant_id: str, name: str) -> AgentSpec | None:
        r = self.db.execute(
            "SELECT json FROM specs WHERE tenant_id=? AND name=? AND enabled=1",
            (tenant_id, name)).fetchone()
        return AgentSpec.model_validate_json(r["json"]) if r else None

    def get_spec_any(self, tenant_id: str, name: str) -> AgentSpec | None:
        r = self.db.execute(
            "SELECT json FROM specs WHERE tenant_id=? AND name=?", (tenant_id, name)).fetchone()
        return AgentSpec.model_validate_json(r["json"]) if r else None

    def is_enabled(self, tenant_id: str, name: str) -> bool:
        r = self.db.execute("SELECT enabled FROM specs WHERE tenant_id=? AND name=?",
                            (tenant_id, name)).fetchone()
        return bool(r and r["enabled"])

    def list_specs(self, tenant_id: str) -> list[AgentSpec]:
        rows = self.db.execute(
            "SELECT json FROM specs WHERE tenant_id=? AND enabled=1 ORDER BY name",
            (tenant_id,)).fetchall()
        return [AgentSpec.model_validate_json(r["json"]) for r in rows]

    def set_enabled(self, tenant_id: str, name: str, enabled: bool) -> None:
        self.db.execute("UPDATE specs SET enabled=? WHERE tenant_id=? AND name=?",
                        (1 if enabled else 0, tenant_id, name))
        self.db.commit()

    def delete_spec(self, tenant_id: str, name: str) -> None:
        self.db.execute("DELETE FROM specs WHERE tenant_id=? AND name=?", (tenant_id, name))
        self.db.commit()

    # ----- templates ------------------------------------------------------ #
    def save_template(self, tpl: AgentTemplate, tenant_id: str = "*") -> None:
        self._upsert("templates", {"name": tpl.name, "tenant_id": tenant_id,
                                   "json": tpl.model_dump_json(), "updated": _now()}, ("name",))

    def get_template(self, name: str) -> AgentTemplate | None:
        r = self.db.execute("SELECT json FROM templates WHERE name=?", (name,)).fetchone()
        return AgentTemplate.model_validate_json(r["json"]) if r else None

    def list_templates(self) -> list[AgentTemplate]:
        rows = self.db.execute("SELECT json FROM templates ORDER BY name").fetchall()
        return [AgentTemplate.model_validate_json(r["json"]) for r in rows]

    # ----- MCP ------------------------------------------------------------ #
    def save_mcp(self, server: McpServer) -> None:
        self._upsert("mcp_servers", {"tenant_id": server.tenant_id, "name": server.name,
                                    "json": server.model_dump_json()}, ("tenant_id", "name"))

    def get_mcp(self, tenant_id: str, name: str) -> McpServer | None:
        r = self.db.execute("SELECT json FROM mcp_servers WHERE tenant_id=? AND name=?",
                            (tenant_id, name)).fetchone()
        return McpServer.model_validate_json(r["json"]) if r else None

    def list_mcp(self, tenant_id: str) -> list[McpServer]:
        rows = self.db.execute("SELECT json FROM mcp_servers WHERE tenant_id=?",
                              (tenant_id,)).fetchall()
        return [McpServer.model_validate_json(r["json"]) for r in rows]

    def list_all_mcp(self) -> list[McpServer]:
        rows = self.db.execute("SELECT json FROM mcp_servers").fetchall()
        return [McpServer.model_validate_json(r["json"]) for r in rows]

    # ----- sessions (persistent auth) ------------------------------------ #
    def create_session(self, token: str, tenant_id: str, username: str,
                       roles: list[str], ttl_seconds: float | None = None) -> None:
        now = _now()
        self._upsert("sessions", {"token": token, "tenant_id": tenant_id, "username": username,
                                 "roles": json.dumps(roles), "created": now,
                                 "expires": (now + ttl_seconds) if ttl_seconds else None},
                     ("token",))

    def get_session(self, token: str) -> dict | None:
        if not token:
            return None
        r = self.db.execute(
            "SELECT tenant_id,username,roles,expires FROM sessions WHERE token=?",
            (token,)).fetchone()
        if not r:
            return None
        exp = r["expires"]
        if exp is not None and _now() > exp:
            self.delete_session(token)
            return None
        return {"tenant_id": r["tenant_id"], "username": r["username"],
                "roles": json.loads(r["roles"] or "[]")}

    def delete_session(self, token: str) -> None:
        self.db.execute("DELETE FROM sessions WHERE token=?", (token,))
        self.db.commit()

    # ----- local users ---------------------------------------------------- #
    def create_user(self, username: str, password_hash: str, tenant_id: str,
                    roles: list[str]) -> None:
        self._upsert("users", {"username": username, "password": password_hash,
                              "tenant_id": tenant_id, "roles": json.dumps(roles),
                              "created": _now()}, ("username",))

    def get_user(self, username: str) -> dict | None:
        r = self.db.execute(
            "SELECT username,password,tenant_id,roles FROM users WHERE username=?",
            (username,)).fetchone()
        if not r:
            return None
        return {"username": r["username"], "password": r["password"],
                "tenant": r["tenant_id"], "roles": json.loads(r["roles"] or "[]")}

    # ----- deploy tokens -------------------------------------------------- #
    def create_deploy_token(self, token: str, tenant_id: str, agent: str, label: str = "") -> None:
        self._upsert("deploy_tokens", {"token": token, "tenant_id": tenant_id, "agent": agent,
                                      "label": label, "created": _now()}, ("token",))

    def get_deploy_token(self, token: str) -> dict | None:
        if not token:
            return None
        r = self.db.execute(
            "SELECT tenant_id,agent FROM deploy_tokens WHERE token=?", (token,)).fetchone()
        return {"tenant_id": r["tenant_id"], "agent": r["agent"]} if r else None

    def revoke_deploy_tokens(self, tenant_id: str, agent: str) -> int:
        rows = self.db.execute(
            "SELECT COUNT(*) AS n FROM deploy_tokens WHERE tenant_id=? AND agent=?",
            (tenant_id, agent)).fetchone()
        self.db.execute("DELETE FROM deploy_tokens WHERE tenant_id=? AND agent=?",
                        (tenant_id, agent))
        self.db.commit()
        return int(rows["n"]) if rows else 0

    def list_deploy_tokens(self, tenant_id: str, agent: str | None = None) -> list[dict]:
        if agent:
            rows = self.db.execute(
                "SELECT token,agent,label,created FROM deploy_tokens WHERE tenant_id=? AND agent=?",
                (tenant_id, agent)).fetchall()
        else:
            rows = self.db.execute(
                "SELECT token,agent,label,created FROM deploy_tokens WHERE tenant_id=?",
                (tenant_id,)).fetchall()
        return [dict(r) for r in rows]

    # ----- credentials ---------------------------------------------------- #
    def save_credential(self, tenant_id: str, ref: str, ciphertext: str,
                        created_by: str = "") -> None:
        self._upsert("credentials", {"tenant_id": tenant_id, "ref": ref, "ciphertext": ciphertext,
                                    "created": _now(), "created_by": created_by},
                     ("tenant_id", "ref"))

    def get_credential_ciphertext(self, tenant_id: str, ref: str) -> str | None:
        r = self.db.execute("SELECT ciphertext FROM credentials WHERE tenant_id=? AND ref=?",
                            (tenant_id, ref)).fetchone()
        return r["ciphertext"] if r else None

    def list_credentials(self, tenant_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT ref,created,created_by FROM credentials WHERE tenant_id=? ORDER BY ref",
            (tenant_id,)).fetchall()
        return [dict(r) for r in rows]

    def delete_credential(self, tenant_id: str, ref: str) -> None:
        self.db.execute("DELETE FROM credentials WHERE tenant_id=? AND ref=?", (tenant_id, ref))
        self.db.commit()

    def vault_meta_get(self, k: str) -> str | None:
        r = self.db.execute("SELECT v FROM vault_meta WHERE k=?", (k,)).fetchone()
        return r["v"] if r else None

    def vault_meta_set(self, k: str, v: str) -> None:
        self._upsert("vault_meta", {"k": k, "v": v}, ("k",))

    # ----- long-term memory (durable vector store, tenant-scoped) --------- #
    def add_memory(self, tenant_id: str, agent: str, mid: str, text: str, meta: dict,
                   embedder: str, dim: int, vec: list) -> None:
        self._upsert("memories", {
            "id": mid, "tenant_id": tenant_id, "agent": agent, "text": text,
            "meta": json.dumps(meta or {}, default=str), "embedder": embedder, "dim": int(dim),
            "vec": json.dumps(vec), "ts": _now()}, ("id",))

    def list_memories(self, tenant_id: str, agent: str) -> list[dict]:
        """Return every stored memory for a (tenant, agent) namespace. Tenant isolation is a SQL
        WHERE clause, not a Python dict key — Tenant A can never read Tenant B's memories."""
        rows = self.db.execute(
            "SELECT id, text, meta, embedder, dim, vec, ts FROM memories "
            "WHERE tenant_id=? AND agent=? ORDER BY ts", (tenant_id, agent)).fetchall()
        out = []
        for r in rows:
            try:
                vec = json.loads(r["vec"]) if r["vec"] else []
            except Exception:  # noqa: BLE001
                vec = []
            try:
                meta = json.loads(r["meta"]) if r["meta"] else {}
            except Exception:  # noqa: BLE001
                meta = {}
            out.append({"id": r["id"], "text": r["text"], "meta": meta,
                        "embedder": r["embedder"], "dim": r["dim"], "vec": vec, "ts": r["ts"]})
        return out

    # ----- run logs / audit / hitl / compiler ---------------------------- #
    def log_run_step(self, trace_id: str, tenant_id: str, payload: dict) -> None:
        self.db.execute("INSERT INTO run_logs(id,trace_id,tenant_id,ts,json) VALUES(?,?,?,?,?)",
                        (uuid.uuid4().hex, trace_id, tenant_id, _now(),
                         json.dumps(payload, default=str)))
        self.db.commit()

    def get_run_logs(self, tenant_id: str, trace_id: str | None = None) -> list[dict]:
        if trace_id:
            rows = self.db.execute(
                "SELECT json FROM run_logs WHERE tenant_id=? AND trace_id=? ORDER BY ts",
                (tenant_id, trace_id)).fetchall()
        else:
            rows = self.db.execute(
                "SELECT json FROM run_logs WHERE tenant_id=? ORDER BY ts", (tenant_id,)).fetchall()
        return [json.loads(r["json"]) for r in rows]

    def audit(self, tenant_id: str, actor: str, action: str, target: str, detail: Any = "") -> None:
        self.db.execute(
            "INSERT INTO audit(id,ts,tenant_id,actor,action,target,detail) VALUES(?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, _now(), tenant_id, actor, action, target,
             json.dumps(detail, default=str) if not isinstance(detail, str) else detail))
        self.db.commit()

    def get_audit(self, tenant_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT ts,actor,action,target,detail FROM audit WHERE tenant_id=? ORDER BY ts",
            (tenant_id,)).fetchall()
        return [dict(r) for r in rows]

    def log_hitl(self, tenant_id: str, trace_id: str, actor: str, decision: str,
                 diff: Any = "") -> None:
        self.db.execute(
            "INSERT INTO hitl(id,ts,tenant_id,trace_id,actor,decision,diff) VALUES(?,?,?,?,?,?,?)",
            (uuid.uuid4().hex, _now(), tenant_id, trace_id, actor, decision,
             json.dumps(diff, default=str)))
        self.db.commit()

    def log_compile(self, tenant_id: str, agent: str, payload: dict) -> None:
        self.db.execute("INSERT INTO compiler_logs(id,ts,tenant_id,agent,json) VALUES(?,?,?,?,?)",
                        (uuid.uuid4().hex, _now(), tenant_id, agent,
                         json.dumps(payload, default=str)))
        self.db.commit()

    def get_compile_logs(self, tenant_id: str) -> list[dict]:
        rows = self.db.execute(
            "SELECT ts,agent,json FROM compiler_logs WHERE tenant_id=? ORDER BY ts",
            (tenant_id,)).fetchall()
        return [{"ts": r["ts"], "agent": r["agent"], **json.loads(r["json"])} for r in rows]

    # ----- traces --------------------------------------------------------- #
    def save_trace(self, trace_id: str, tenant_id: str, payload: dict) -> None:
        m = payload.get("metrics", {}) or {}
        self._upsert("traces", {
            "trace_id": trace_id, "tenant_id": tenant_id, "ts": _now(),
            "agent": payload.get("agent", ""), "status": payload.get("status", ""),
            "cost_usd": float(m.get("cost_usd", 0) or 0), "tokens": int(m.get("tokens", 0) or 0),
            "metrics": json.dumps(m, default=str), "json": json.dumps(payload, default=str),
        }, ("trace_id",))

    def get_trace(self, tenant_id: str, trace_id: str) -> dict | None:
        r = self.db.execute("SELECT json FROM traces WHERE tenant_id=? AND trace_id=?",
                            (tenant_id, trace_id)).fetchone()
        return json.loads(r["json"]) if r else None

    def list_traces(self, tenant_id: str) -> list[dict]:
        rows = self.db.execute("SELECT ts,json FROM traces WHERE tenant_id=? ORDER BY ts DESC",
                              (tenant_id,)).fetchall()
        out = []
        for r in rows:
            d = json.loads(r["json"])
            d.setdefault("ts", r["ts"])
            out.append(d)
        return out

    # ----- runs ----------------------------------------------------------- #
    def save_run(self, record: dict) -> None:
        self._upsert("runs", {
            "id": record["id"], "tenant_id": record["tenant_id"],
            "agent": record.get("agent", "?"), "status": record.get("status", "success"),
            "risk_level": record.get("risk_level", "low"), "mode": record.get("mode", ""),
            "ts": record.get("ts", _now()), "json": json.dumps(record, default=str),
        }, ("id",))

    def get_run(self, tenant_id: str, run_id: str) -> dict | None:
        r = self.db.execute("SELECT json FROM runs WHERE tenant_id=? AND id=?",
                            (tenant_id, run_id)).fetchone()
        return json.loads(r["json"]) if r else None

    def list_runs(self, tenant_id: str, agent: str | None = None,
                  status: str | None = None, limit: int = 200) -> list[dict]:
        q = "SELECT json FROM runs WHERE tenant_id=?"
        args: list[Any] = [tenant_id]
        if agent:
            q += " AND agent=?"; args.append(agent)
        if status:
            q += " AND status=?"; args.append(status)
        q += " ORDER BY ts DESC LIMIT ?"; args.append(limit)
        rows = self.db.execute(q, tuple(args)).fetchall()
        return [json.loads(r["json"]) for r in rows]

    # ----- analytics: aggregate in SQL over indexed columns (not Python) --- #
    def count_runs_by_status(self, tenant_id: str, since: float = 0.0) -> dict:
        rows = self.db.execute(
            "SELECT status, COUNT(*) AS n FROM runs WHERE tenant_id=? AND ts>=? GROUP BY status",
            (tenant_id, since)).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    def usage_totals(self, tenant_id: str, since: float = 0.0) -> dict:
        r = self.db.execute(
            "SELECT COUNT(*) AS runs, COALESCE(SUM(cost_usd),0) AS cost, "
            "COALESCE(SUM(tokens),0) AS tokens FROM traces WHERE tenant_id=? AND ts>=?",
            (tenant_id, since)).fetchone()
        return {"runs": int(r["runs"] or 0), "cost_usd": round(float(r["cost"] or 0), 6),
                "tokens": int(r["tokens"] or 0)}

    def trace_metrics_since(self, tenant_id: str, since: float = 0.0) -> list[dict]:
        """Light projection for analytics: small metrics blob + indexed columns only — never the
        full trace (spans/blackboard) blob."""
        rows = self.db.execute(
            "SELECT ts, agent, status, cost_usd, tokens, metrics FROM traces "
            "WHERE tenant_id=? AND ts>=? ORDER BY ts", (tenant_id, since)).fetchall()
        out = []
        for r in rows:
            m = {}
            try:
                m = json.loads(r["metrics"]) if r["metrics"] else {}
            except Exception:  # noqa: BLE001
                m = {}
            out.append({"ts": r["ts"], "agent": r["agent"], "status": r["status"],
                        "cost_usd": float(r["cost_usd"] or 0), "tokens": int(r["tokens"] or 0),
                        "metrics": m})
        return out

    # ----- durable job queue (state machine, cross-process-safe claims) ---- #
    def find_job_by_key(self, tenant_id: str, key: str) -> dict | None:
        if not key:
            return None
        r = self.db.execute(
            "SELECT * FROM jobs WHERE tenant_id=? AND idempotency_key=?", (tenant_id, key)).fetchone()
        return dict(r) if r else None

    def enqueue_job(self, job_id: str, tenant_id: str, agent: str, payload: str,
                    max_attempts: int = 3, idempotency_key: str | None = None) -> None:
        now = _now()
        self._upsert("jobs", {
            "id": job_id, "tenant_id": tenant_id, "agent": agent, "payload": payload,
            "idempotency_key": idempotency_key, "status": "queued", "attempts": 0,
            "max_attempts": max_attempts, "available_at": now, "lease_until": None,
            "worker": None, "last_error": None, "created": now, "updated": now}, ("id",))

    def claim_job(self, worker_id: str, lease_seconds: float, now: float | None = None) -> dict | None:
        """Atomically claim the next runnable job: a queued job whose backoff has elapsed, OR a
        'running' job whose worker lease has EXPIRED (orphan recovery). One UPDATE statement makes
        the claim atomic — safe across threads AND processes sharing the DB. Increments attempts."""
        now = now or _now()
        tag = f"{worker_id}:{uuid.uuid4().hex[:8]}"
        lease = now + lease_seconds
        self.db.execute(
            "UPDATE jobs SET status='running', worker=?, lease_until=?, attempts=attempts+1, "
            "updated=? WHERE id=(SELECT id FROM jobs WHERE status IN ('queued','running') "
            "AND available_at<=? AND (lease_until IS NULL OR lease_until<=?) "
            "ORDER BY available_at LIMIT 1)",
            (tag, lease, now, now, now))
        self.db.commit()
        r = self.db.execute("SELECT * FROM jobs WHERE worker=?", (tag,)).fetchone()
        return dict(r) if r else None

    def finish_job(self, job_id: str, status: str, last_error: str | None = None) -> bool:
        """Terminal transition (succeeded/dead). Only the worker currently holding the job (status
        'running') may finish it, so a concurrent cancel is not overwritten."""
        self.db.execute(
            "UPDATE jobs SET status=?, lease_until=NULL, last_error=?, updated=? "
            "WHERE id=? AND status='running'", (status, last_error, _now(), job_id))
        self.db.commit()
        r = self.db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        return bool(r and r["status"] == status)

    def requeue_job(self, job_id: str, available_at: float, last_error: str | None = None) -> None:
        """Transient failure -> back to 'queued' with backoff (only if still 'running')."""
        self.db.execute(
            "UPDATE jobs SET status='queued', lease_until=NULL, worker=NULL, available_at=?, "
            "last_error=?, updated=? WHERE id=? AND status='running'",
            (available_at, last_error, _now(), job_id))
        self.db.commit()

    def cancel_job(self, tenant_id: str, job_id: str) -> bool:
        self.db.execute(
            "UPDATE jobs SET status='cancelled', lease_until=NULL, updated=? "
            "WHERE tenant_id=? AND id=? AND status IN ('queued','running')",
            (_now(), tenant_id, job_id))
        self.db.commit()
        r = self.db.execute("SELECT status FROM jobs WHERE tenant_id=? AND id=?",
                            (tenant_id, job_id)).fetchone()
        return bool(r and r["status"] == "cancelled")

    def recover_orphans(self) -> int:
        """Startup recovery: requeue jobs left 'running' with an EXPIRED lease (the worker/process
        that held them died). Returns how many were recovered."""
        now = _now()
        rows = self.db.execute(
            "SELECT COUNT(*) AS n FROM jobs WHERE status='running' AND lease_until IS NOT NULL "
            "AND lease_until<=?", (now,)).fetchone()
        self.db.execute(
            "UPDATE jobs SET status='queued', lease_until=NULL, worker=NULL, updated=? "
            "WHERE status='running' AND lease_until IS NOT NULL AND lease_until<=?", (now, now))
        self.db.commit()
        return int(rows["n"]) if rows else 0

    def get_job(self, tenant_id: str, job_id: str) -> dict | None:
        r = self.db.execute("SELECT * FROM jobs WHERE tenant_id=? AND id=?",
                            (tenant_id, job_id)).fetchone()
        return dict(r) if r else None

    def list_jobs(self, tenant_id: str, status: str | None = None, limit: int = 200) -> list[dict]:
        if status:
            rows = self.db.execute(
                "SELECT * FROM jobs WHERE tenant_id=? AND status=? ORDER BY created DESC LIMIT ?",
                (tenant_id, status, limit)).fetchall()
        else:
            rows = self.db.execute(
                "SELECT * FROM jobs WHERE tenant_id=? ORDER BY created DESC LIMIT ?",
                (tenant_id, limit)).fetchall()
        return [dict(r) for r in rows]
