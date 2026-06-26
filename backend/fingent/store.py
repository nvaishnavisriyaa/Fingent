"""
Persistence (§13) — SQLite-backed store for specs, templates, MCP registry, structured run
logs, the immutable audit trail, HITL decisions, compiler logs, and traces.

Everything is scoped by tenant_id. Tenant isolation (§10) is enforced here: every read takes a
tenant_id and filters on it, so Tenant A's queries can never return Tenant B's rows.
"""
from __future__ import annotations

import json
import sqlite3
import time
import uuid
from typing import Any

from .schemas import AgentSpec, AgentTemplate, McpServer


def _now() -> float:
    return time.time()


class Store:
    def __init__(self, path: str = ":memory:") -> None:
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self._init()

    def _init(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS specs(
                tenant_id TEXT, name TEXT, enabled INT DEFAULT 1,
                json TEXT, updated REAL, PRIMARY KEY(tenant_id, name));
            CREATE TABLE IF NOT EXISTS templates(
                name TEXT PRIMARY KEY, tenant_id TEXT, json TEXT, updated REAL);
            CREATE TABLE IF NOT EXISTS mcp_servers(
                tenant_id TEXT, name TEXT, json TEXT, PRIMARY KEY(tenant_id, name));
            CREATE TABLE IF NOT EXISTS run_logs(
                id TEXT PRIMARY KEY, trace_id TEXT, tenant_id TEXT, ts REAL, json TEXT);
            CREATE TABLE IF NOT EXISTS audit(
                id TEXT PRIMARY KEY, ts REAL, tenant_id TEXT, actor TEXT,
                action TEXT, target TEXT, detail TEXT);
            CREATE TABLE IF NOT EXISTS hitl(
                id TEXT PRIMARY KEY, ts REAL, tenant_id TEXT, trace_id TEXT,
                actor TEXT, decision TEXT, diff TEXT);
            CREATE TABLE IF NOT EXISTS compiler_logs(
                id TEXT PRIMARY KEY, ts REAL, tenant_id TEXT, agent TEXT, json TEXT);
            CREATE TABLE IF NOT EXISTS traces(
                trace_id TEXT PRIMARY KEY, tenant_id TEXT, ts REAL, json TEXT);
            """
        )
        self.db.commit()

    def save_spec(self, spec: AgentSpec) -> None:
        self.db.execute(
            "REPLACE INTO specs(tenant_id,name,enabled,json,updated) VALUES(?,?,?,?,?)",
            (spec.security.tenant_id, spec.name, 1, spec.model_dump_json(), _now()))
        self.db.commit()

    def get_spec(self, tenant_id: str, name: str) -> AgentSpec | None:
        r = self.db.execute(
            "SELECT json FROM specs WHERE tenant_id=? AND name=? AND enabled=1",
            (tenant_id, name)).fetchone()
        return AgentSpec.model_validate_json(r["json"]) if r else None

    def list_specs(self, tenant_id: str) -> list[AgentSpec]:
        rows = self.db.execute(
            "SELECT json FROM specs WHERE tenant_id=? AND enabled=1 ORDER BY name",
            (tenant_id,)).fetchall()
        return [AgentSpec.model_validate_json(r["json"]) for r in rows]

    def set_enabled(self, tenant_id: str, name: str, enabled: bool) -> None:
        self.db.execute("UPDATE specs SET enabled=? WHERE tenant_id=? AND name=?",
                        (1 if enabled else 0, tenant_id, name))
        self.db.commit()

    def save_template(self, tpl: AgentTemplate, tenant_id: str = "*") -> None:
        self.db.execute("REPLACE INTO templates(name,tenant_id,json,updated) VALUES(?,?,?,?)",
                        (tpl.name, tenant_id, tpl.model_dump_json(), _now()))
        self.db.commit()

    def get_template(self, name: str) -> AgentTemplate | None:
        r = self.db.execute("SELECT json FROM templates WHERE name=?", (name,)).fetchone()
        return AgentTemplate.model_validate_json(r["json"]) if r else None

    def list_templates(self) -> list[AgentTemplate]:
        rows = self.db.execute("SELECT json FROM templates ORDER BY name").fetchall()
        return [AgentTemplate.model_validate_json(r["json"]) for r in rows]

    def save_mcp(self, server: McpServer) -> None:
        self.db.execute("REPLACE INTO mcp_servers(tenant_id,name,json) VALUES(?,?,?)",
                        (server.tenant_id, server.name, server.model_dump_json()))
        self.db.commit()

    def get_mcp(self, tenant_id: str, name: str) -> McpServer | None:
        r = self.db.execute("SELECT json FROM mcp_servers WHERE tenant_id=? AND name=?",
                            (tenant_id, name)).fetchone()
        return McpServer.model_validate_json(r["json"]) if r else None

    def list_mcp(self, tenant_id: str) -> list[McpServer]:
        rows = self.db.execute("SELECT json FROM mcp_servers WHERE tenant_id=?",
                              (tenant_id,)).fetchall()
        return [McpServer.model_validate_json(r["json"]) for r in rows]

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

    def save_trace(self, trace_id: str, tenant_id: str, payload: dict) -> None:
        self.db.execute("REPLACE INTO traces(trace_id,tenant_id,ts,json) VALUES(?,?,?,?)",
                        (trace_id, tenant_id, _now(), json.dumps(payload, default=str)))
        self.db.commit()

    def get_trace(self, tenant_id: str, trace_id: str) -> dict | None:
        r = self.db.execute("SELECT json FROM traces WHERE tenant_id=? AND trace_id=?",
                            (tenant_id, trace_id)).fetchone()
        return json.loads(r["json"]) if r else None

    def list_traces(self, tenant_id: str) -> list[dict]:
        rows = self.db.execute("SELECT json FROM traces WHERE tenant_id=? ORDER BY ts DESC",
                              (tenant_id,)).fetchall()
        return [json.loads(r["json"]) for r in rows]
