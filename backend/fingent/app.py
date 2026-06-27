"""
FastAPI backend (§13) — exposes the whole assembly line + observability + admin over HTTP, and
serves the Next-style single-page frontend. A single Fingent() instance is the runtime.

Tenant is taken from the X-Tenant header (demo stand-in for real auth), enforcing §10 isolation.
"""
from __future__ import annotations

import os
from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .platform import Fingent
from .schemas import CreateAgentRequest, McpServer

app = FastAPI(title="Fingent", version="0.1.0")
fp = Fingent(os.getenv("FINGENT_DB", ":memory:"))

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")


# Optional hardening: map bearer tokens -> tenant via env (JSON: {"token":"tenant"}).
# When configured, the Authorization header is authoritative and a spoofed X-Tenant
# cannot cross tenants. When unset, we fall back to the demo X-Tenant header.
import json as _json
import uuid as _uuid

# Static service tokens (machine-to-machine), optional. {"token": "tenant"}
_TENANT_TOKENS: dict[str, str] = _json.loads(os.getenv("FINGENT_TENANT_TOKENS", "{}"))
# Interactive login: enable with FINGENT_AUTH=1. Users from FINGENT_USERS
# (JSON {"user": {"password": "...", "tenant": "..."}}) or a default demo user.
_AUTH_ON: bool = os.getenv("FINGENT_AUTH") == "1"
_USERS: dict = _json.loads(os.getenv(
    "FINGENT_USERS", '{"admin": {"password": "admin", "tenant": "acme"}}'))
_SESSIONS: dict[str, str] = {}   # session token -> tenant
_AUTH_REQUIRED = _AUTH_ON or bool(_TENANT_TOKENS)


def tenant(x_tenant: str | None, authorization: str | None = None) -> str:
    """Resolve the tenant. When auth is enabled the bearer token is authoritative
    (a spoofed X-Tenant cannot cross tenants); otherwise fall back to the demo header."""
    if _AUTH_REQUIRED:
        token = (authorization or "").removeprefix("Bearer ").strip()
        mapped = _TENANT_TOKENS.get(token) or _SESSIONS.get(token)
        if not mapped:
            raise HTTPException(401, "authentication required")
        return mapped
    return x_tenant or "acme"


@app.get("/api/config")
def auth_config():
    """Public: tells the frontend whether to show a login screen."""
    return {"auth_required": _AUTH_REQUIRED}


class LoginBody(BaseModel):
    username: str
    password: str


@app.post("/api/login")
def login(body: LoginBody):
    u = _USERS.get(body.username)
    if not u or u.get("password") != body.password:
        raise HTTPException(401, "invalid username or password")
    token = _uuid.uuid4().hex
    _SESSIONS[token] = u.get("tenant", "acme")
    return {"token": token, "tenant": _SESSIONS[token], "username": body.username}


@app.post("/api/logout")
def logout(authorization: str | None = Header(default=None)):
    token = (authorization or "").removeprefix("Bearer ").strip()
    _SESSIONS.pop(token, None)
    return {"ok": True}


@app.get("/api/me")
def me(authorization: str | None = Header(default=None)):
    token = (authorization or "").removeprefix("Bearer ").strip()
    t = _SESSIONS.get(token) or _TENANT_TOKENS.get(token)
    if not t:
        raise HTTPException(401, "not authenticated")
    return {"tenant": t}


# ----- catalog / form ---------------------------------------------------- #
@app.get("/api/templates")
def templates():
    return [t.model_dump() for t in fp.templates()]


@app.get("/api/templates/{name}/form")
def form_schema(name: str):
    try:
        return fp.form_schema(name)
    except KeyError:
        raise HTTPException(404, f"no template '{name}'")


# ----- create agent (the assembly line) ---------------------------------- #
class CreateBody(BaseModel):
    template: str | None = None
    answers: dict = {}
    additional_requirements: str = ""
    approve_side_effecting: bool = False
    approved_side_effecting_tools: list[str] = []
    auto_provision: bool = False


@app.post("/api/agents")
def create_agent(body: CreateBody, x_tenant: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)):
    req = CreateAgentRequest(
        template=body.template, answers=body.answers,
        additional_requirements=body.additional_requirements,
        tenant_id=tenant(x_tenant, authorization),
        approve_side_effecting=body.approve_side_effecting,
        approved_side_effecting_tools=body.approved_side_effecting_tools,
    )
    return fp.create_agent(req, auto_provision=body.auto_provision)


@app.get("/api/agents")
def list_agents(x_tenant: str | None = Header(default=None),
                authorization: str | None = Header(default=None)):
    return [s.model_dump() for s in fp.store.list_specs(tenant(x_tenant, authorization))]


# ----- deploy: each saved agent is a governed, callable endpoint ---------- #
class InvokeBody(BaseModel):
    inputs: dict = {}
    approve_side_effecting: bool = False


@app.post("/api/agents/{name}/invoke")
def invoke_agent(name: str, body: InvokeBody, x_tenant: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)):
    """Run a deployed agent on a real task. Returns a full RunRecord (status, steps, tool
    calls, output, risk). Same governed runtime as the playground."""
    t = tenant(x_tenant, authorization)
    rec = fp.run_task(t, name, body.inputs, approve_side_effecting=body.approve_side_effecting)
    if isinstance(rec, dict) and rec.get("ok") is False:
        raise HTTPException(404, rec.get("message", "agent not found"))
    return rec


# ----- agent lifecycle (view / edit / duplicate / delete) ---------------- #
@app.get("/api/agents/{name}")
def get_agent(name: str, x_tenant: str | None = Header(default=None),
              authorization: str | None = Header(default=None)):
    a = fp.get_agent(tenant(x_tenant, authorization), name)
    if a is None:
        raise HTTPException(404, "agent not found")
    return a


@app.put("/api/agents/{name}")
def update_agent(name: str, patch: dict, x_tenant: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)):
    r = fp.update_agent(tenant(x_tenant, authorization), name, patch)
    if not r.get("ok"):
        raise HTTPException(404, r.get("message", "agent not found"))
    return r


class DuplicateBody(BaseModel):
    new_name: str


@app.post("/api/agents/{name}/duplicate")
def duplicate_agent(name: str, body: DuplicateBody, x_tenant: str | None = Header(default=None),
                    authorization: str | None = Header(default=None)):
    r = fp.duplicate_agent(tenant(x_tenant, authorization), name, body.new_name)
    if not r.get("ok"):
        raise HTTPException(404, r.get("message", "agent not found"))
    return r


@app.delete("/api/agents/{name}")
def delete_agent(name: str, x_tenant: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)):
    return fp.delete_agent(tenant(x_tenant, authorization), name)


# ----- runs + human review queue ---------------------------------------- #
@app.get("/api/runs")
def list_runs(agent: str | None = None, status: str | None = None,
              x_tenant: str | None = Header(default=None),
              authorization: str | None = Header(default=None)):
    return fp.store.list_runs(tenant(x_tenant, authorization), agent=agent, status=status)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, x_tenant: str | None = Header(default=None),
            authorization: str | None = Header(default=None)):
    r = fp.store.get_run(tenant(x_tenant, authorization), run_id)
    if r is None:
        raise HTTPException(404, "run not found")
    return r


@app.get("/api/reviews")
def list_reviews(x_tenant: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)):
    return fp.store.list_runs(tenant(x_tenant, authorization), status="needs_review")


class ReviewBody(BaseModel):
    decision: str
    note: str = ""


@app.post("/api/reviews/{run_id}")
def resolve_review(run_id: str, body: ReviewBody, x_tenant: str | None = Header(default=None),
                   authorization: str | None = Header(default=None)):
    r = fp.resolve_review(tenant(x_tenant, authorization), run_id, body.decision, note=body.note)
    if not r.get("ok"):
        raise HTTPException(404, r.get("message", "run not found"))
    return r


class EnabledBody(BaseModel):
    enabled: bool = True


@app.post("/api/agents/{name}/enabled")
def set_agent_enabled(name: str, body: EnabledBody, x_tenant: str | None = Header(default=None),
                      authorization: str | None = Header(default=None)):
    t = tenant(x_tenant, authorization)
    fp.store.set_enabled(t, name, body.enabled)
    fp.store.audit(t, "operator", "deploy" if body.enabled else "undeploy", name, "")
    return {"name": name, "enabled": body.enabled}


# ----- run / traces ------------------------------------------------------ #
class RunBody(BaseModel):
    agents: list[str] = []
    tier: int | None = None
    inputs: dict = {}


@app.post("/api/run")
def run(body: RunBody, x_tenant: str | None = Header(default=None),
        authorization: str | None = Header(default=None)):
    t = tenant(x_tenant, authorization)
    if body.tier is not None:
        return fp.run_workflow(t, tier=body.tier, inputs=body.inputs)
    return fp.run(t, body.agents, inputs=body.inputs)


@app.get("/api/traces")
def traces(x_tenant: str | None = Header(default=None),
           authorization: str | None = Header(default=None)):
    return fp.store.list_traces(tenant(x_tenant, authorization))


@app.get("/api/traces/{trace_id}")
def trace(trace_id: str, x_tenant: str | None = Header(default=None),
          authorization: str | None = Header(default=None)):
    t = tenant(x_tenant, authorization)
    tr = fp.store.get_trace(t, trace_id)
    if not tr:
        raise HTTPException(404, "no trace")
    tr["run_logs"] = fp.store.get_run_logs(t, trace_id)
    return tr


# ----- audit / compiler logs --------------------------------------------- #
@app.get("/api/audit")
def audit(x_tenant: str | None = Header(default=None),
          authorization: str | None = Header(default=None)):
    return fp.store.get_audit(tenant(x_tenant, authorization))


@app.get("/api/compiler-logs")
def compiler_logs(x_tenant: str | None = Header(default=None),
                  authorization: str | None = Header(default=None)):
    return fp.store.get_compile_logs(tenant(x_tenant, authorization))


# ----- analytics / tool catalog ------------------------------------------ #
@app.get("/api/analytics")
def analytics(days: int = 7, x_tenant: str | None = Header(default=None),
              authorization: str | None = Header(default=None)):
    return fp.analytics(tenant(x_tenant, authorization), days=days)


@app.get("/api/tools")
def tools(x_tenant: str | None = Header(default=None),
          authorization: str | None = Header(default=None)):
    return fp.tool_catalog(tenant(x_tenant, authorization))


# ----- MCP admin --------------------------------------------------------- #
class McpBody(BaseModel):
    name: str
    url: str
    approved: bool = True
    secrets_ref: list[str] = []


@app.post("/api/mcp")
def register_mcp(body: McpBody, x_tenant: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)):
    server = McpServer(name=body.name, url=body.url,
                       tenant_id=tenant(x_tenant, authorization),
                       approved=body.approved, secrets_ref=body.secrets_ref)
    added = fp.register_mcp(server)
    return {"registered": body.name, "tools_added": added}


@app.get("/api/mcp")
def list_mcp(x_tenant: str | None = Header(default=None),
             authorization: str | None = Header(default=None)):
    return [m.model_dump() for m in fp.store.list_mcp(tenant(x_tenant, authorization))]


# ----- HITL -------------------------------------------------------------- #
class HitlBody(BaseModel):
    trace_id: str
    decision: str
    diff: dict = {}


@app.post("/api/hitl")
def hitl(body: HitlBody, x_tenant: str | None = Header(default=None),
         authorization: str | None = Header(default=None)):
    return fp.resume_hitl(tenant(x_tenant, authorization), body.trace_id, "reviewer",
                          body.decision, body.diff)


# ----- frontend ---------------------------------------------------------- #
if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
