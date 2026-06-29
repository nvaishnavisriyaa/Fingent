"""
FastAPI backend (§13) — the assembly line + observability + admin over HTTP, and serves the SPA.

Deployment hardening:
  * Auth sessions are PERSISTED in the store (survive a restart); passwords are pbkdf2-hashed.
  * RBAC — every endpoint requires a PERMISSION (read/review/write/deploy/invoke/admin) derived
    from the caller's roles (viewer/reviewer/operator/admin).
  * Deployment is an explicit lifecycle: POST /deploy provisions a PER-AGENT invocation token
    (callable for that one agent only); /undeploy revokes it and disables the endpoint.
  * Invocation can run async on the background worker pool (?wait=false) so a long agent run does
    not tie up an HTTP worker.
  * Tenant isolation is no longer trust-by-default: the X-Tenant header is honoured only when
    FINGENT_ALLOW_HEADER_TENANT=1; otherwise the tenant comes from the authenticated session /
    service token (or a single fixed default tenant in open dev mode).
"""
from __future__ import annotations

import json as _json
import os
import time as _time

from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import auth
from .platform import Fingent
from .schemas import CreateAgentRequest, McpServer


@asynccontextmanager
async def _lifespan(_app: "FastAPI"):
    # Durable-queue recovery: requeue any runs orphaned by a previous process, then start workers.
    try:
        fp.jobs.start()
    except Exception:  # noqa: BLE001 — never block startup on the worker pool
        pass
    yield


app = FastAPI(title="Fingent", version="0.2.0", lifespan=_lifespan)
# Durable by default: a real file DB (or Postgres via DATABASE_URL). Never :memory: in the
# served app — that would lose every agent, run and audit record on restart.
_DEFAULT_DB = os.path.join(os.path.dirname(__file__), "..", "fingent.db")
_DB = os.getenv("FINGENT_DB") or os.getenv("DATABASE_URL") or _DEFAULT_DB
fp = Fingent(_DB)

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "frontend")

# ---- config ------------------------------------------------------------- #
# Service tokens (machine-to-machine). Value may be a tenant string, or {tenant, roles}.
_TENANT_TOKENS: dict = _json.loads(os.getenv("FINGENT_TENANT_TOKENS", "{}"))
_AUTH_ON: bool = os.getenv("FINGENT_AUTH") == "1"
# Users: {username: {password, tenant, roles}}. Password may be plaintext (dev) or a pbkdf2 hash.
_USERS: dict = _json.loads(os.getenv(
    "FINGENT_USERS", '{"admin": {"password": "admin", "tenant": "acme", "roles": ["admin"]}}'))
_AUTH_REQUIRED = _AUTH_ON or bool(_TENANT_TOKENS)
# Tenant resolution:
#   * auth ON  -> tenant comes from the authenticated session/service token (UNSPOOFABLE); the
#                 X-Tenant header is ignored. This is the production posture.
#   * auth OFF -> open dev mode. The X-Tenant header is the ONLY tenant signal, so honor it by
#                 default. All data is fully tenant-scoped in the store, so demo tenants
#                 (acme/globex/initech) get REAL data isolation. "Spoofable" is moot here because
#                 there is no auth to spoof — turn auth on for an unspoofable boundary.
# An explicit FINGENT_ALLOW_HEADER_TENANT always wins (e.g. force-ignore the header even in dev).
_DEFAULT_TENANT = os.getenv("FINGENT_DEFAULT_TENANT", "acme")
_ALLOW_HEADER_TENANT = (os.getenv("FINGENT_ALLOW_HEADER_TENANT", "0" if _AUTH_REQUIRED else "1")
                        == "1")

# ---- security posture --------------------------------------------------- #
_SECURE = os.getenv("FINGENT_REQUIRE_SECURE") == "1"          # production hardening switch
_SESSION_TTL = float(os.getenv("FINGENT_SESSION_TTL_HOURS", "8")) * 3600.0
_LOGIN_MAX_ATTEMPTS = int(os.getenv("FINGENT_LOGIN_MAX_ATTEMPTS", "5"))
_LOGIN_LOCKOUT_SECONDS = float(os.getenv("FINGENT_LOGIN_LOCKOUT_SECONDS", "300"))

import logging as _logging
import threading as _threading

_log = _logging.getLogger("fingent.security")
_login_fails: dict = {}            # username -> [fail_count, locked_until_ts]
_login_lock = _threading.Lock()


def _uses_default_admin() -> bool:
    u = _USERS.get("admin")
    return bool(u) and str(u.get("password", "")) == "admin"


def _audit_security_posture() -> None:
    """Fail fast in secure mode on an insecure config; otherwise log a loud warning so an
    operator can never deploy Fingent open or with default credentials by accident."""
    problems = []
    if _uses_default_admin():
        problems.append("the default admin/admin account is still active")
    for name, u in _USERS.items():
        if not auth.password_is_hashed(str(u.get("password", ""))):
            problems.append(f"user '{name}' has a non-hashed (plaintext) password")
    if not _AUTH_REQUIRED:
        problems.append("authentication is DISABLED (open dev mode: every caller is admin)")
    if _ALLOW_HEADER_TENANT and _AUTH_REQUIRED:
        problems.append("X-Tenant header is trusted even though auth is on — it overrides the "
                        "session tenant (spoofing possible). Unset FINGENT_ALLOW_HEADER_TENANT.")
    if not os.getenv("FINGENT_VAULT_KEY"):
        problems.append("FINGENT_VAULT_KEY is not set (the credential-encryption key lives in "
                        "the database instead of a KMS/secret manager)")

    if _SECURE and problems:
        raise RuntimeError(
            "FINGENT_REQUIRE_SECURE=1 but the configuration is insecure: "
            + "; ".join(problems)
            + ". Set FINGENT_AUTH=1, provide hashed passwords via FINGENT_USERS "
              "(python -m fingent.hashpw <password>), and remove the default admin.")
    if problems:
        _log.warning("SECURITY: Fingent is running in an INSECURE posture - %s. "
                     "Do NOT use this configuration in production. "
                     "See FINGENT_AUTH / FINGENT_REQUIRE_SECURE / FINGENT_USERS.",
                     "; ".join(problems))


_audit_security_posture()


# ---- auth context + RBAC ------------------------------------------------ #
class _Ctx:
    def __init__(self, tenant: str, roles: list[str], principal: str):
        self.tenant = tenant
        self.roles = roles
        self.perms = auth.permissions_for(roles)
        self.principal = principal


def _bearer(authorization: str | None) -> str:
    return (authorization or "").removeprefix("Bearer ").strip()


def _service_token_ctx(token: str) -> _Ctx | None:
    val = _TENANT_TOKENS.get(token)
    if val is None:
        return None
    if isinstance(val, dict):
        return _Ctx(val.get("tenant", _DEFAULT_TENANT), val.get("roles", ["operator"]),
                    f"service:{token[:6]}")
    return _Ctx(val, ["operator"], f"service:{token[:6]}")


def _context(authorization: str | None, x_tenant: str | None) -> _Ctx:
    """Resolve the caller into a tenant + roles. With auth enabled the bearer token (session or
    service token) is authoritative. In open dev mode everyone is an admin of the default tenant
    (or the X-Tenant header's tenant, only if explicitly allowed)."""
    if _AUTH_REQUIRED:
        token = _bearer(authorization)
        sess = fp.store.get_session(token)
        if sess:
            return _Ctx(sess["tenant_id"], sess["roles"], sess["username"])
        svc = _service_token_ctx(token)
        if svc:
            return svc
        raise HTTPException(401, "authentication required")
    tenant = x_tenant if (_ALLOW_HEADER_TENANT and x_tenant) else _DEFAULT_TENANT
    return _Ctx(tenant, ["admin"], "dev")


def _authz(authorization: str | None, x_tenant: str | None, perm: str) -> _Ctx:
    ctx = _context(authorization, x_tenant)
    if perm not in ctx.perms:
        raise HTTPException(403, f"forbidden: '{perm}' permission required")
    return ctx


# ---- auth endpoints ----------------------------------------------------- #
@app.get("/api/config")
def auth_config():
    from .llm import LlmProvider
    from .runtime import _demo_allowed
    prov = LlmProvider()
    return {"auth_required": _AUTH_REQUIRED,
            "model_configured": prov.enabled,
            "model_name": prov.name if prov.enabled else None,
            # Whether the X-Tenant header actually switches tenant. When false (the safe
            # default) the tenant is fixed by the session/default, so the UI must NOT offer a
            # free tenant switcher that silently does nothing.
            "allow_header_tenant": _ALLOW_HEADER_TENANT,
            "default_tenant": _DEFAULT_TENANT,
            "demo_allowed": _demo_allowed()}


@app.get("/healthz")
def healthz():
    """Liveness/readiness probe for containers and load balancers. Unauthenticated and cheap:
    confirms the process is up and the store answers a trivial query."""
    db_ok = True
    try:
        fp.store.list_specs(_DEFAULT_TENANT)
    except Exception:  # noqa: BLE001
        db_ok = False
    return {"status": "ok" if db_ok else "degraded", "db": db_ok,
            "version": app.version, "auth_required": _AUTH_REQUIRED}


class LoginBody(BaseModel):
    username: str
    password: str


def _login_locked(username: str) -> float:
    """Return seconds remaining on a lockout for this username, or 0 if not locked."""
    with _login_lock:
        rec = _login_fails.get(username)
        if not rec:
            return 0.0
        remaining = rec[1] - _time.time()
        return remaining if remaining > 0 else 0.0


def _record_login_failure(username: str) -> None:
    with _login_lock:
        rec = _login_fails.get(username, [0, 0.0])
        rec[0] += 1
        if rec[0] >= _LOGIN_MAX_ATTEMPTS:
            rec[1] = _time.time() + _LOGIN_LOCKOUT_SECONDS
            rec[0] = 0   # reset counter; lockout window now governs
        _login_fails[username] = rec


def _clear_login_failures(username: str) -> None:
    with _login_lock:
        _login_fails.pop(username, None)


@app.post("/api/login")
def login(body: LoginBody):
    locked = _login_locked(body.username)
    if locked > 0:
        raise HTTPException(429, f"too many attempts; locked for {int(locked)}s")
    u = _USERS.get(body.username)
    ok = bool(u) and auth.verify_password(
        body.password, str(u.get("password", "")), allow_plaintext=not _SECURE)
    if not ok:
        _record_login_failure(body.username)
        fp.store.audit(u.get("tenant", _DEFAULT_TENANT) if u else _DEFAULT_TENANT,
                       body.username, "login_failed", body.username, {})
        raise HTTPException(401, "invalid username or password")
    _clear_login_failures(body.username)
    token = auth.new_token()
    roles = u.get("roles", ["operator"])
    tenant = u.get("tenant", _DEFAULT_TENANT)
    fp.store.create_session(token, tenant, body.username, roles, ttl_seconds=_SESSION_TTL)
    fp.store.audit(tenant, body.username, "login", body.username, {"roles": roles})
    return {"token": token, "tenant": tenant, "username": body.username, "roles": roles}


@app.post("/api/logout")
def logout(authorization: str | None = Header(default=None)):
    fp.store.delete_session(_bearer(authorization))
    return {"ok": True}


@app.get("/api/me")
def me(authorization: str | None = Header(default=None),
       x_tenant: str | None = Header(default=None)):
    ctx = _context(authorization, x_tenant)
    return {"tenant": ctx.tenant, "roles": ctx.roles, "principal": ctx.principal}


# ---- catalog / form ----------------------------------------------------- #
@app.get("/api/templates")
def templates(authorization: str | None = Header(default=None),
              x_tenant: str | None = Header(default=None)):
    _authz(authorization, x_tenant, "read")
    return [t.model_dump() for t in fp.templates()]


@app.get("/api/templates/{name}/form")
def form_schema(name: str, authorization: str | None = Header(default=None),
                x_tenant: str | None = Header(default=None)):
    _authz(authorization, x_tenant, "read")
    try:
        return fp.form_schema(name)
    except KeyError:
        raise HTTPException(404, f"no template '{name}'")


# ---- create agent ------------------------------------------------------- #
class CreateBody(BaseModel):
    template: str | None = None
    answers: dict = {}
    additional_requirements: str = ""
    approve_side_effecting: bool = False
    approved_side_effecting_tools: list[str] = []
    requested_tools: list[str] = []
    auto_provision: bool = False


@app.post("/api/agents")
def create_agent(body: CreateBody, x_tenant: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "write")
    req = CreateAgentRequest(
        template=body.template, answers=body.answers,
        additional_requirements=body.additional_requirements, tenant_id=ctx.tenant,
        approve_side_effecting=body.approve_side_effecting,
        approved_side_effecting_tools=body.approved_side_effecting_tools,
        requested_tools=body.requested_tools,
    )
    return fp.create_agent(req, auto_provision=body.auto_provision)


@app.get("/api/grantable")
def grantable(template: str | None = None, x_tenant: str | None = Header(default=None),
              authorization: str | None = Header(default=None)):
    """The tool universe an agent may be granted (native + this tenant's approved MCP tools),
    enriched for the UI tool picker. Powers Dify-style explicit tool selection on create/edit."""
    ctx = _authz(authorization, x_tenant, "read")
    return fp.grantable_tools(ctx.tenant, template)


@app.get("/api/agents")
def list_agents(x_tenant: str | None = Header(default=None),
                authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "read")
    return [s.model_dump() for s in fp.store.list_specs(ctx.tenant)]


# ---- deploy lifecycle --------------------------------------------------- #
class DeployBody(BaseModel):
    label: str = "default"


@app.post("/api/agents/{name}/deploy")
def deploy_agent(name: str, body: DeployBody = DeployBody(),
                 x_tenant: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "deploy")
    r = fp.deploy_agent(ctx.tenant, name, actor=ctx.principal, label=body.label)
    if not r.get("ok"):
        raise HTTPException(404, r.get("message", "agent not found"))
    return r


@app.post("/api/agents/{name}/undeploy")
def undeploy_agent(name: str, x_tenant: str | None = Header(default=None),
                   authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "deploy")
    return fp.undeploy_agent(ctx.tenant, name, actor=ctx.principal)


@app.post("/api/agents/{name}/enabled")
def set_agent_enabled(name: str, body: dict, x_tenant: str | None = Header(default=None),
                      authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "deploy")
    enabled = bool(body.get("enabled", True))
    if enabled:
        return fp.deploy_agent(ctx.tenant, name, actor=ctx.principal)
    return fp.undeploy_agent(ctx.tenant, name, actor=ctx.principal)


# ---- invoke (sync or async) -------------------------------------------- #
class InvokeBody(BaseModel):
    inputs: dict = {}
    approve_side_effecting: bool = False


def _invoke_ctx(name: str, authorization: str | None, x_tenant: str | None) -> _Ctx:
    """A per-agent deploy token authorizes invoking ONLY that agent; otherwise fall back to a
    normal authenticated caller that holds the 'invoke' permission."""
    token = _bearer(authorization)
    dt = fp.store.get_deploy_token(token)
    if dt:
        if dt["agent"] != name:
            raise HTTPException(403, "this deploy token is for a different agent")
        return _Ctx(dt["tenant_id"], ["operator"], f"deploy-token:{name}")
    return _authz(authorization, x_tenant, "invoke")


@app.post("/api/agents/{name}/invoke")
def invoke_agent(name: str, body: InvokeBody, wait: bool = True,
                 x_tenant: str | None = Header(default=None),
                 authorization: str | None = Header(default=None),
                 idempotency_key: str | None = Header(default=None)):
    """Run a deployed agent. By default runs synchronously and returns the full RunRecord; pass
    ?wait=false to enqueue on the worker pool and return a run_id immediately (poll /api/runs)."""
    ctx = _invoke_ctx(name, authorization, x_tenant)
    if not fp.store.is_enabled(ctx.tenant, name):
        if fp.store.get_spec_any(ctx.tenant, name) is not None:
            raise HTTPException(409, f"agent '{name}' is not deployed")
        raise HTTPException(404, "agent not found")
    if wait:
        rec = fp.run_task(ctx.tenant, name, body.inputs,
                          approve_side_effecting=body.approve_side_effecting)
        if isinstance(rec, dict) and rec.get("ok") is False:
            raise HTTPException(404, rec.get("message", "agent not found"))
        return rec
    return fp.submit_run(ctx.tenant, name, body.inputs,
                         approve_side_effecting=body.approve_side_effecting,
                         idempotency_key=idempotency_key)


class ChatBody(BaseModel):
    message: str
    session_id: str = "default"
    approve_side_effecting: bool = False


@app.post("/api/agents/{name}/chat")
def chat_agent(name: str, body: ChatBody,
               x_tenant: str | None = Header(default=None),
               authorization: str | None = Header(default=None)):
    """Streamed conversation with an agent (SSE). The agent plans, calls real tools (with the
    full governed trace) and streams a prose answer. Session history is the agent's working
    memory, so it recalls earlier turns. Events: start|token|tool_call|tool_result|status|
    final|error|done."""
    ctx = _invoke_ctx(name, authorization, x_tenant)
    from .chat import chat_sse
    gen = chat_sse(fp, ctx.tenant, name, body.session_id, body.message,
                   approve_side_effecting=body.approve_side_effecting)
    return StreamingResponse(gen, media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no", "Connection": "keep-alive"})


# ---- agent lifecycle ---------------------------------------------------- #
@app.get("/api/agents/{name}")
def get_agent(name: str, x_tenant: str | None = Header(default=None),
              authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "read")
    a = fp.get_agent(ctx.tenant, name)
    if a is None:
        raise HTTPException(404, "agent not found")
    return a


@app.put("/api/agents/{name}")
def update_agent(name: str, patch: dict, x_tenant: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "write")
    r = fp.update_agent(ctx.tenant, name, patch)
    if not r.get("ok"):
        raise HTTPException(404, r.get("message", "agent not found"))
    return r


class DuplicateBody(BaseModel):
    new_name: str


@app.post("/api/agents/{name}/duplicate")
def duplicate_agent(name: str, body: DuplicateBody, x_tenant: str | None = Header(default=None),
                    authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "write")
    r = fp.duplicate_agent(ctx.tenant, name, body.new_name)
    if not r.get("ok"):
        raise HTTPException(404, r.get("message", "agent not found"))
    return r


@app.delete("/api/agents/{name}")
def delete_agent(name: str, x_tenant: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "write")
    return fp.delete_agent(ctx.tenant, name)


# ---- runs + review queue ----------------------------------------------- #
@app.get("/api/runs")
def list_runs(agent: str | None = None, status: str | None = None,
              x_tenant: str | None = Header(default=None),
              authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "read")
    return fp.store.list_runs(ctx.tenant, agent=agent, status=status)


@app.get("/api/runs/{run_id}")
def get_run(run_id: str, x_tenant: str | None = Header(default=None),
            authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "read")
    r = fp.store.get_run(ctx.tenant, run_id)
    if r is None:
        raise HTTPException(404, "run not found")
    return r


@app.post("/api/runs/{run_id}/cancel")
def cancel_run(run_id: str, x_tenant: str | None = Header(default=None),
               authorization: str | None = Header(default=None)):
    """Cancel a queued/running async run. Queued jobs stop cleanly; a running attempt finishes
    its current step but is not retried."""
    ctx = _authz(authorization, x_tenant, "invoke")
    return fp.cancel_run(ctx.tenant, run_id)


@app.get("/api/reviews")
def list_reviews(x_tenant: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "read")
    return fp.store.list_runs(ctx.tenant, status="needs_review")


class ReviewBody(BaseModel):
    decision: str
    note: str = ""


@app.post("/api/reviews/{run_id}")
def resolve_review(run_id: str, body: ReviewBody, x_tenant: str | None = Header(default=None),
                   authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "review")
    r = fp.resolve_review(ctx.tenant, run_id, body.decision, reviewer=ctx.principal, note=body.note)
    if not r.get("ok"):
        raise HTTPException(404, r.get("message", "run not found"))
    return r


# ---- multi-agent run / traces ------------------------------------------ #
class RunBody(BaseModel):
    agents: list[str] = []
    tier: int | None = None
    inputs: dict = {}
    supervised: bool = False   # True -> sub-agents run the real LLM runtime + Synthesis prose


@app.post("/api/run")
def run(body: RunBody, x_tenant: str | None = Header(default=None),
        authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "invoke")
    if body.supervised:
        return fp.run_supervised(ctx.tenant, agent_names=body.agents or None,
                                 tier=body.tier, inputs=body.inputs)
    if body.tier is not None:
        return fp.run_workflow(ctx.tenant, tier=body.tier, inputs=body.inputs)
    return fp.run(ctx.tenant, body.agents, inputs=body.inputs)


@app.get("/api/traces")
def traces(x_tenant: str | None = Header(default=None),
           authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "read")
    return fp.store.list_traces(ctx.tenant)


@app.get("/api/traces/{trace_id}")
def trace(trace_id: str, x_tenant: str | None = Header(default=None),
          authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "read")
    tr = fp.store.get_trace(ctx.tenant, trace_id)
    if not tr:
        raise HTTPException(404, "no trace")
    tr["run_logs"] = fp.store.get_run_logs(ctx.tenant, trace_id)
    return tr


# ---- audit / compiler logs --------------------------------------------- #
@app.get("/api/audit")
def audit(x_tenant: str | None = Header(default=None),
          authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "read")
    return fp.store.get_audit(ctx.tenant)


@app.get("/api/compiler-logs")
def compiler_logs(x_tenant: str | None = Header(default=None),
                  authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "read")
    return fp.store.get_compile_logs(ctx.tenant)


# ---- analytics / tool catalog ------------------------------------------ #
@app.get("/api/analytics")
def analytics(days: int = 7, x_tenant: str | None = Header(default=None),
              authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "read")
    return fp.analytics(ctx.tenant, days=days)


@app.get("/api/tools")
def tools(x_tenant: str | None = Header(default=None),
          authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "read")
    return fp.tool_catalog(ctx.tenant)


# ---- Credentials (encrypted tool/secret store) -------------------------- #
class CredentialBody(BaseModel):
    ref: str
    value: str


@app.get("/api/credentials")
def list_credentials(x_tenant: str | None = Header(default=None),
                     authorization: str | None = Header(default=None)):
    """List credential refs + metadata for the tenant. Secret VALUES are never returned."""
    ctx = _authz(authorization, x_tenant, "read")
    return fp.store.list_credentials(ctx.tenant)


@app.get("/api/credentials/requirements")
def credential_requirements(x_tenant: str | None = Header(default=None),
                            authorization: str | None = Header(default=None)):
    """Per-tool credential requirements + whether each is configured for THIS tenant.
    Reports a status only ("vault" / "env" / "missing") - never any secret value."""
    ctx = _authz(authorization, x_tenant, "read")
    import os as _os
    from .tools_native import TOOL_CREDENTIALS

    def _status(ref: str) -> str:
        if fp.store.get_credential_ciphertext(ctx.tenant, ref):
            return "vault"
        if _os.getenv(ref):
            return "env"
        return "missing"

    out = []
    for tool, reqs in TOOL_CREDENTIALS.items():
        items = []
        for r in reqs:
            st = _status(r["ref"])
            items.append({**r, "status": st, "configured": st != "missing"})
        out.append({"tool": tool,
                    "configured": all(i["configured"] for i in items if i["required"]),
                    "requirements": items})
    return out


@app.post("/api/credentials")
def put_credential(body: CredentialBody, x_tenant: str | None = Header(default=None),
                   authorization: str | None = Header(default=None)):
    """Store (encrypt) a tenant-scoped credential. Requires admin. The value is encrypted at
    rest and never written to the audit detail or any log."""
    ctx = _authz(authorization, x_tenant, "admin")
    if not body.ref or not body.value:
        raise HTTPException(400, "ref and value are required")
    from .vault import vault
    vault.put(body.ref, body.value, tenant_id=ctx.tenant, actor=ctx.principal)
    fp.store.audit(ctx.tenant, ctx.principal, "credential_set", body.ref,
                   {"bytes": len(body.value)})  # length only, never the value
    return {"saved": body.ref}


@app.delete("/api/credentials/{ref}")
def delete_credential(ref: str, x_tenant: str | None = Header(default=None),
                      authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "admin")
    from .vault import vault
    vault.delete(ref, ctx.tenant)
    fp.store.audit(ctx.tenant, ctx.principal, "credential_delete", ref, {})
    return {"deleted": ref}


# ---- MCP admin ---------------------------------------------------------- #
class McpBody(BaseModel):
    name: str
    url: str
    approved: bool = True
    secrets_ref: list[str] = []


@app.post("/api/mcp")
def register_mcp(body: McpBody, x_tenant: str | None = Header(default=None),
                 authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "write")
    server = McpServer(name=body.name, url=body.url, tenant_id=ctx.tenant,
                       approved=body.approved, secrets_ref=body.secrets_ref)
    added = fp.register_mcp(server)
    return {"registered": body.name, "tools_added": added}


@app.get("/api/mcp")
def list_mcp(x_tenant: str | None = Header(default=None),
             authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "read")
    return [m.model_dump() for m in fp.store.list_mcp(ctx.tenant)]


@app.post("/api/mcp/{name}/refresh")
def refresh_mcp(name: str, x_tenant: str | None = Header(default=None),
                authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "write")
    try:
        return fp.refresh_mcp(ctx.tenant, name)
    except KeyError:
        raise HTTPException(404, f"no MCP server '{name}'")


# ---- HITL --------------------------------------------------------------- #
class HitlBody(BaseModel):
    trace_id: str
    decision: str
    diff: dict = {}


@app.post("/api/hitl")
def hitl(body: HitlBody, x_tenant: str | None = Header(default=None),
         authorization: str | None = Header(default=None)):
    ctx = _authz(authorization, x_tenant, "review")
    return fp.resume_hitl(ctx.tenant, body.trace_id, ctx.principal, body.decision, body.diff)


# ---- frontend -----------
if os.path.isdir(FRONTEND_DIR):
    @app.get("/")
    def index():
        return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))

    @app.get("/chat")
    def chat_ui():
        return FileResponse(os.path.join(FRONTEND_DIR, "chat.html"))

    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
