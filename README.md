# Fingent — a reusable agentic platform for financial services

Fingent is a **platform**, not a workflow. A financial-services customer can **create, configure,
deploy, observe, and govern AI agents** without ever changing code. There is exactly **one runtime
and one factory**; every agent — Tier 1 or Tier 2 — rides the same assembly line. The only thing
that differs between agents is the spec going in.

> **Adding a new agent = a template (config) + an LLM compilation. No code change, ever.**

---

## The assembly line (the only path to a saved agent)

```
HTML form (template params + mandatory free-text "extras" field)
        │ answers + free_text
        ▼
Input Validator → LLM SPEC COMPILER → Spec Validator → Dependency Resolver → Save → Factory → Planner
 (structured)      (proposes)          (disposes)        (notify / auto-provision)
```

The path from form → spec is **not** a deterministic merge. An LLM **compiles** a *candidate*
spec from the template + form answers + the free-text field; a deterministic validator then
**approves or rejects** it. **The LLM proposes; the validator disposes.** This is what makes
free-text customization safe — privilege escalation via free text is impossible by construction.

## Quick start

```bash
./run.sh                 # installs deps, serves API + UI on http://localhost:8000
# optional, to use Llama instead of the offline deterministic compiler:
export GROQ_API_KEY=sk-...        # Groq, OpenAI-compatible
export GROQ_MODEL=llama-3.3-70b-versatile
```

Open http://localhost:8000 — browse the catalog, fill a form, watch the LLM compile an agent,
auto-provision prerequisites, run it, and inspect the trace.

Run the acceptance tests:

```bash
cd backend && python -m pytest -q     # 14 tests, all offline
```

## Architecture (`backend/fingent/`)

| File | Responsibility |
|---|---|
| `schemas.py` | Typed Pydantic contracts (AgentSpec, AgentTemplate, ToolDescriptor, policies…). An *agent is data*. |
| `registry.py` | Tool Registry: NATIVE · WEB_SEARCH · MCP · EXTERNAL_API. Tenant-scoped; MCP needs admin approval. |
| `blackboard.py` | Shared memory: namespaced · versioned · deduplicated, with per-prefix read/write ACLs. |
| `compiler.py` | **LLM spec compiler** (Groq/Llama). Constrained JSON; free-text treated as behavior, not instructions. Falls back to a deterministic compiler offline. |
| `validators.py` | Input validator + **spec validator (the disposer)** — the security boundary. |
| `dependencies.py` | Dependency resolver: notification, recursive topological auto-provision, cycle detection. |
| `factory.py` | Builds *any* live agent from its spec; wraps every node in guardrail/security/logging/observability middleware. |
| `planner.py` | LangGraph-style supervisor: emits a DAG, dispatches to nodes, shares one blackboard, HITL interrupt. |
| `middleware.py` | Guardrails: PII, prompt-injection, compliance overseer, least-privilege tool ACL, cost/loop, HITL gate. |
| `observability.py` | Tracing: one trace_id per run; spans for every agent/tool/HITL; per-run metrics. |
| `store.py` | SQLite persistence: specs, templates, MCP registry, run logs, **immutable audit trail**, HITL + compiler logs. |
| `templates.py` | The catalog (Tier 1 GTM + Tier 2 FS) as **config only**. |
| `app.py` | FastAPI surface + serves the frontend. |
| `../frontend/index.html` | Single-page UI: catalog, auto-generated form, dependency modal, MCP admin, trace inspector, compiler log, audit. |

## Security model (§10)

- **Least privilege**: an agent may call only `security.allowed_tools` and read/write only its
  declared blackboard prefixes.
- **The LLM compiler is fenced**: it can *propose* but the validator *approves*. It can never grant
  a tool outside the grantable universe, widen memory scope, disable review, or act on injection in
  the free-text field.
- **Tenant isolation**: specs, memory, logs, runs, and MCP servers are scoped to `tenant_id`.
- **Secrets** resolve from a vault by `secrets_ref`, never stored in specs, logs, or the free-text
  round-trip.
- **Side-effecting tools** require explicit grant-time approval and route through the HITL gate.
- **Audit everything**: create/run/approve/MCP-registration, append-only.

## Tech stack

LangGraph-style supervisor · FastAPI · Pydantic · Groq (Llama) for the compiler · built-in web
search + MCP client · SQLite (swap for Postgres/Supabase) · OpenTelemetry-style tracing · vanilla
SPA frontend (drop-in replaceable with Next.js).
