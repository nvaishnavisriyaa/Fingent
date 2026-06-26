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


def tenant(x_tenant: str | None) -> str:
    return x_tenant or "acme"


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
    auto_provision: bool = False


@app.post("/api/agents")
def create_agent(body: CreateBody, x_tenant: str | None = Header(default=None)):
    req = CreateAgentRequest(
        template=body.template, answers=body.answers,
        additional_requirements=body.additional_requirements,
        tenant_id=tenant(x_tenant), approve_side_effecting=body.approve_side_effecting,
    )
    return fp.create_agent(req, auto_provision=body.auto_provision)


@app.get("/api/agents")
def list_agents(x_tenant: str | None = Header(default=None)):
    return [s.model_dump() for s in fp.store.list_specs(tenant(x_tenant))]


# ----- run / traces ------------------------------------------------------ #
class RunBody(BaseModel):
    agents: list[str] = []
    tier: int | None = None
    inputs: dict = {}


@app.post("/api/run")
def run(body: RunBody, x_tenant: str | None = Header(default=None)):
    t = tenant(x_tenant)
    if body.tier is not None:
        return fp.run_workflow(t, tier=body.tier, inputs=body.inputs)
    return fp.run(t, body.agents, inputs=body.inputs)


@app.get("/api/traces")
def traces(x_tenant: str | None = Header(default=None)):
    return fp.store.list_traces(tenant(x_tenant))


@app.get("/api/traces/{trace_id}")
def trace(trace_id: str, x_tenant: str | None = Header(default=None)):
    tr = fp.store.get_trace(tenant(x_tenant), trace_id)
    if not tr:
        raise HTTPException(404, "no trace")
    tr["run_logs"] = fp.store.get_run_logs(tenant(x_tenant), trace_id)
    return tr


# ----- audit / compiler logs --------------------------------------------- #
@app.get("/api/audit")
def audit(x_tenant: str | None = Header(default=None)):
    return fp.store.get_audit(tenant(x_tenant))


@app.get("/api/compiler-logs")
def compiler_logs(x_tenant: str | None = Header(default=None)):
    return fp.store.get_compile_logs(tenant(x_tenant))


# ----- MCP admin --------------------------------------------------------- #
class McpBody(BaseModel):
    name: str
    url: str
    approved: bool = True
    secrets_ref: list[str] = []


@app.post("/api/mcp")
def register_mcp(body: McpBody, x_tenant: str | None = Header(default=None)):
    server = McpServer(name=body.name, url=body.url, tenant_id=tenant(x_tenant),
                       approved=body.approved, secrets_ref=body.secrets_ref)
    added = fp.register_mcp(server)
    return {"registered": body.name, "tools_added": added}


@app.get("/api/mcp")
def list_mcp(x_tenant: str | None = Header(default=None)):
    return [m.model_dump() for m in fp.store.list_mcp(tenant(x_tenant))]


# ----- HITL -------------------------------------------------------------- #
class HitlBody(BaseModel):
    trace_id: str
    decision: str
    diff: dict = {}


@app.post("/api/hitl")
def hitl(body: HitlBody, x_tenant: str | None = Header(default=None)):
    return fp.resume_hitl(tenant(x_tenant), body.trace_id, "reviewer",
                          body.decision, body.diff)


# ----- frontend ---------------------------------------------------------- #
if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
