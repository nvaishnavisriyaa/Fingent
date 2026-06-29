# Fingent — build, supervise, secure, observe & deploy AI agents for financial services

Fingent helps financial-services teams **create, run, supervise, secure, observe and deploy
AI agents that can safely use business tools**. It is a *platform*, not a single workflow:
every agent — whether picked from the catalog, customized, or built from scratch by **Fin** —
is compiled into a validated, least-privilege spec, executed by one governed runtime, and
recorded as a first-class run you can inspect, review, and re-run.

> **The LLM proposes; a deterministic validator disposes.** Agents are *data*. Execution is
> *governed*: every tool call is checked, every risky action can pause for a human, and every
> run is traced and persisted.

---

## The product journey (end to end)

```
Create / configure agent  →  give it tools (least-privilege)  →  run it on a real task
   →  inspect the execution trace  →  human review if risky  →  saved run + metrics
   →  deploy & invoke over HTTP
```

Every page in the app supports this one journey:

| Page | What it does |
|---|---|
| **Home (Fin)** | Describe an agent in plain English; Fin builds it from scratch (validator-governed). |
| **Catalog** | Pre-built FS agents (KYC, AML, credit, fraud, compliance) + the GTM discovery workflow. Configure → compile. |
| **Agents** | View / edit / duplicate / delete agents; see config, tools, guardrails, endpoint. |
| **Playground** | Run any deployed agent on real JSON input and watch every governed step. |
| **Runs** | Every persisted run with status, risk, mode, steps and full trace. |
| **Review queue** | Runs paused for human approval — inspect reasoning + risk, then approve / reject. |
| **Monitoring** | Live analytics + run outcomes from real saved data. |
| **Traces / Compiler / Audit** | Span trees, the proposer/disposer security trail, append-only audit. |
| **Guardrails / Toolkits / MCP** | The security model, the tool catalog (real vs demo), and tenant-scoped MCP servers. |

---

## How agent creation works

An agent is a typed, validated **spec** (`AgentSpec`) — not just a name. It carries:
purpose, instructions, expected inputs, allowed tools, output format, **risk level**,
human-approval setting, guardrail policy, memory scope, and deployment status.

Three ways to create one, all through the same assembly line:

```
form answers + free-text  →  Input Validator  →  LLM Spec Compiler (proposes)
   →  Spec Validator (disposes)  →  Dependency Resolver  →  Save  →  Deploy
```

- **From a template** — pick from the catalog, fill configurable params (ICP, personas, watchlists, thresholds), compile.
- **Customize** — the mandatory free-text field is treated as *desired behavior*, never as instructions to the compiler. Ask for an extra tool and it's granted *only* if it's inside the template's grantable universe; anything outside is stripped.
- **From scratch (Fin)** — *describe the agent in plain English and Fin builds it*: it selects
  least-privilege tools, writes the purpose + operating instructions, infers the expected
  inputs, output format, risk level and human-review setting, compiles a full validated spec,
  and crystallizes it into a reusable template. Uses the configured LLM when available; with
  no model it falls back to a real intent-based heuristic builder (not just keyword matching).

You can then **view, edit (purpose/instructions/risk/output/HITL), duplicate, and delete** agents.

---

## How agent execution works (real, not faked)

Running an agent (Playground or `POST /api/agents/{name}/invoke`) starts a **governed tool-use
loop** in `runtime.py`:

1. The agent observes the task input.
2. It decides the **next action** by **default with a real LLM using native tool calling** —
   the model is given each allowed tool's JSON schema (native *and* MCP) and picks the tool +
   arguments at each step (`mode: "llm"`). The provider is OpenAI-compatible and configurable
   (Groq by default; point `FINGENT_LLM_BASE_URL` at any gateway). With no model configured the
   agent still runs for real in a **deterministic rules engine** (`mode: "rules"`): it executes the
   agent's real tools on live data and composes the decision strictly from their outputs (nothing
   simulated). A model upgrades the same agent to adaptive multi-step reasoning.
3. It invokes the **real tool function** from the registry (every tool in the trace was
   actually executed by code).
4. It observes the result, repeats, and produces a final output.

Each run is persisted as a **RunRecord** with an explicit status and a full trace:

```
status ∈ { success | failed | blocked | needs_review | approved | rejected }
```

A run shows: input, agent, mode, status, every tool call (with inputs/outputs), guardrail
events, errors, **risk score + flags**, whether human approval is required, duration, and a
link to its span trace.

---

## How human-in-the-loop works

HITL is part of execution, not a static page. A run enters **`needs_review`** when:

- it wants to use a **side-effecting** tool (send / write / pay) without pre-approval,
- the agent is configured to **require human review**,
- the **risk score is high**, or
- the **compliance overseer** flags the output.

Pending runs appear in the **Review queue**. A reviewer opens the run, inspects the reasoning
summary, tool calls, inputs/outputs and risk flags, then **approves or rejects**:

- **Approve** → status `approved`; any *held side-effecting action is executed now*.
- **Reject** → status `rejected`; the action is cancelled.

The decision (and reviewer note) is saved to run history and the audit trail.

---

## How observability works

Built in, from real saved data — no static mock metrics:

- **Runs** table: status, risk, mode, steps, timestamp, per-agent history.
- **Monitoring**: AI credits, tool calls, active agents, guardrail trips, run outcomes by
  status, review-queue size, top agents, agent-wise usage, usage over time.
- **Traces**: one trace per run; a nested span tree (run → tool → guardrail/review) with
  latency, tokens, cost and tool-call counts.
- **Compiler log**: what the LLM proposed vs what the validator stripped.
- **Audit trail**: append-only create / run / review / deploy / MCP events.

---

## How agentic security & guardrails work

Security affects whether an agent **continues, needs review, or is blocked** — not just UI text.
Every blocked or flagged action has a reason.

- **Least privilege** — the runtime may call only `security.allowed_tools`; an out-of-list call
  is **blocked** and audited.
- **Side-effect HITL gate** — write/send/pay tools never fire unsupervised; they pause for approval.
- **Prompt-injection defense** — untrusted tool output (web/MCP/docs) and the free-text field are
  scanned; matches are **quarantined as data**, never acted on.
- **PII redaction** — SSNs, cards, emails, phones, **IBANs, account numbers, DOB and passport/tax
  IDs** (the identifiers the KYC tools parse) are redacted from tool output before it is recorded
  or returned.
- **Risk scoring** — each run gets a 0–100 score and low/medium/high level from its flags; high
  risk auto-routes to review.
- **Compliance overseer** — output is checked for sanctions hits, leaked identifiers and red-flag
  language; can block.
- **Tenant isolation** — specs, memory, runs, logs and MCP servers are tenant-scoped; an optional
  bearer token makes the tenant unspoofable.
- **Demo vs real connectors** — every tool is labelled `native | web_search | mcp`; mocks are
  clearly marked and swappable.

---

## Tech stack

LangGraph-style supervisor · **FastAPI** · **Pydantic** · **Groq (Llama)** for the compiler and
runtime reasoning · built-in web search + MCP client · **SQLite** (swap for Postgres/Supabase) ·
OpenTelemetry-style tracing · single-file vanilla SPA frontend (drop-in replaceable with Next.js).

### Backend layout (`backend/fingent/`)

| File | Responsibility |
|---|---|
| `schemas.py` | Typed contracts: AgentSpec (+ config), RunRecord, RunStep, policies. |
| `compiler.py` | LLM spec compiler (proposes) with a deterministic offline fallback. |
| `validators.py` | Input validator + **spec validator** (disposes) — the security boundary. |
| `runtime.py` | **Real agent runtime**: governed tool-use loop, risk scoring, run records. |
| `planner.py` | Multi-agent supervisor: an LLM **planning node** that decomposes a goal into a runtime task graph (which sub-agents to call + a per-agent subtask), validates it deterministically into a DAG, **adapts** the remaining plan on intermediate results, and dispatches every sub-agent through the ONE governed runtime kernel (`runtime.run_node`). Falls back to a dependency toposort with no model — no separate execution/enforcement path. |
| `middleware.py` | Guardrails: PII, injection, least-privilege, compliance, budget, HITL. |
| `registry.py` | Tool registry (NATIVE / WEB_SEARCH / MCP / EXTERNAL_API), tenant-scoped. |
| `tools_native.py` | Tool implementations (deterministic mocks + live SEC EDGAR + compose_summary). |
| `blackboard.py` | Shared memory: namespaced, versioned, deduplicated, ACL'd. |
| `memory.py` | Long-term memory: **semantic recall** over real embeddings (OpenAI-compatible API or local sentence-transformers; a clearly-flagged lexical-hashing fallback offline), **durable** and tenant-scoped in the store — or Pinecone. |
| `observability.py` | Tracing: one trace per run; spans + metrics. |
| `store.py` | Pluggable persistence: **SQLite** (durable file default, WAL) or **Postgres** via `DATABASE_URL`. Versioned recorded migrations, indexes on hot tenant/time/status columns, promoted queryable columns, and **SQL-side analytics** (no full-blob scans). Tenant-scoped. |
| `vault.py` | Secrets resolved by ref at call time, never stored in specs/logs. |
| `app.py` | FastAPI surface + auth + serves the frontend. |
| `../frontend/index.html` | The single-page app (all views above). |

---

## Running the project

**Docker (recommended — one command, batteries included):**

```bash
cp .env.example .env          # optional: add a GROQ_API_KEY for LLM reasoning
docker compose up --build     # SQLite, durable via a named volume
# or, with Postgres:
docker compose --profile postgres up --build
```

Open **http://localhost:8000**. The container ships `tesseract` + `poppler` (KYC OCR), runs as a
non-root user, persists data on a volume, and exposes a `/healthz` probe.

**Windows (no Docker):** double-click **`run.bat`**, then open **http://localhost:8000**.

**Any OS (no Docker):**

```bash
cd backend
python -m pip install -r requirements.txt
export FINGENT_DB=fingent.db          # persist agents/runs (omit for in-memory)
python -m uvicorn fingent.app:app --port 8000
```

Open **http://localhost:8000** (not the raw HTML file — it must be served so `/api/*` resolves).

Run the test suite (also runs in CI on every push — see `.github/workflows/ci.yml`):

```bash
cd backend && python -m pytest -q      # ~150 tests; offline & deterministic
```

### Environment variables

| Variable | Effect |
|---|---|
| `FINGENT_LLM_API_KEY` / `GROQ_API_KEY` | API key for the reasoning model. When set, the runtime uses the real LLM tool-calling loop (`mode: "llm"`); without it the agent still runs its real tools deterministically and composes a decision from their outputs (`mode: "rules"`, nothing simulated). |
| `FINGENT_LLM_BASE_URL` | OpenAI-compatible base URL for the model gateway (default Groq `https://api.groq.com/openai/v1`). Point at OpenAI / Azure OpenAI / self-hosted vLLM, etc. |
| `FINGENT_LLM_MODEL` / `GROQ_MODEL` | Model id (default `llama-3.3-70b-versatile`). |
| `FINGENT_DB` | SQLite path; defaults to `:memory:` (set a file to persist). |
| `FINGENT_EMBED_API_KEY` | Enables **real semantic** long-term memory via an OpenAI-compatible `/embeddings` API (with `FINGENT_EMBED_BASE_URL`, `FINGENT_EMBED_MODEL`). Without it (and without sentence-transformers) memory uses a clearly-flagged lexical fallback. |
| `FINGENT_EMBED_BACKEND` | `sentence-transformers` to embed locally (no API key), or `hashing` to force the lexical fallback. |
| `PINECONE_API_KEY` | Use Pinecone as the durable vector memory backend (else the platform store persists vectors). |
| `FINGENT_LIVE_DATA` | **Live tools are ON by default.** Set `0` to force the offline deterministic fallback (used by the test suite). |
| `TAVILY_API_KEY` | Live web search (`web_search`). Without it, `web_search` falls back to keyless news headlines. |
| `OPENSANCTIONS_API_KEY` | Live PEP screening (`pep_check`) via OpenSanctions. |
| `PEOPLE_DATA_API_KEY` | Live decision-maker discovery (`find_persona`) via People Data Labs. |
| `HUNTER_API_KEY` | Live contact resolution (`resolve_contact`) via Hunter.io. |
| `FINGENT_AUTH=1` | Require login. Users from `FINGENT_USERS` (default `admin`/`admin` → tenant `acme`); token is authoritative over the `X-Tenant` header. |
| `FINGENT_TENANT_TOKENS` | JSON `{ "token": "tenant" }` for machine-to-machine access. |

---

## Try the main flow

1. **Create** — Home → describe *"Screen new customers against sanctions and PEP lists and flag hits"* → **Build with Fin**. (Or Catalog → `aml_sanctions_screening` → Configure → Compile.)
2. **Run** — Playground → pick the agent → input `{ "name": "Oleg Petrov" }` → **Run**. Watch `ofac_screen`, `pep_check`, `adverse_media_search` actually execute. A sanctions hit → status **blocked**.
3. **Review** — give an agent a side-effecting tool (e.g. `acme_mcp.send_email`) → run it → it enters **needs_review** → open **Review queue** → **Approve** (the held email fires) or **Reject**.
4. **Observe** — **Runs** shows the persisted run with status/risk; **Monitoring** shows outcomes; open the **trace** for the span tree.
5. **Deploy** — open the agent → copy the `curl` and invoke it over HTTP; the same runtime, guardrails, review and tracing apply.

GTM discovery demo: create the Tier-1 agents (signal_trigger → … → synthesis) and run the
**tier-1 workflow** from *My agents* — the planner orchestrates the 7-step discovery and
`synthesis` composes a real next-action recommendation from shared memory.

---

## What is real vs demo

**Tools are REAL by default** — each calls a live data source and tags its output with a
`source` field (`live:<provider>` vs `mock`). Set `FINGENT_LIVE_DATA=0` to force offline mode.

| Tool | Live source | Key needed |
|---|---|---|
| `edgar_search` | SEC EDGAR full-text search | no |
| `company_financials` | **SEC EDGAR XBRL company-facts** — real, period-aligned 10-K revenue / net income / assets / equity + credit ratios (powers underwriting) | no |
| `verify_entity` | **GLEIF LEI registry** — real legal-entity verification (KYB): legal name, LEI, status, jurisdiction | no |
| `bank_lookup` | **FDIC BankFind Suite** — real US bank/counterparty profile (assets, deposits, charter, status) | no |
| `fx_rate` | **Frankfurter / ECB** — real FX reference rates | no |
| `treasury_rates` | **US Treasury Fiscal Data** — real average interest rates (benchmark cost of funds) | no |
| `ofac_screen` | US Treasury **OFAC SDN** (cached) + **real entity resolution**: normalized-token + edit-distance fuzzy matching with scored, classified candidates (exact/strong/partial), not a substring boolean | no |
| `news_monitor` | Google News RSS | no |
| `adverse_media_search` | Google News RSS + **real adverse-media NLP**: per-headline risk-category classification (financial-crime/sanctions/corruption/legal/regulatory/terrorism), negation handling, aggregate 0–100 risk score | no |
| `ocr_extract` (KYC doc intelligence) | **Real document extraction**: PDF text via pdfplumber/pypdf, scanned images via **Tesseract OCR** (multi-page scanned PDFs rendered with **poppler**), plus **table extraction** and **form key-value** parsing (account/IBAN, DOB, tax id, beneficial-owner tables) and regex fields. Groq vision model as a fallback for image URLs | no (tesseract + poppler for scans) |
| `reg_feed_ingest` | US **Federal Register** API | no |
| `enrich_company` | **Real public-company firmographics from SEC EDGAR** company-facts (revenue / net income / assets from the latest 10-K XBRL, free, no key); Clearbit autocomplete for domain/logo; `ENRICH_API_URL` for private-company providers | no |
| `web_search` | Tavily (→ news fallback) | `TAVILY_API_KEY` |
| `pep_check` | OpenSanctions (live); offline runs the **same entity-resolution engine** against a PEP fixture (scored), never a flat boolean | `OPENSANCTIONS_API_KEY` |
| `find_persona` | People Data Labs (live); without a key returns **target decision-maker titles** (heuristic) — no fabricated individuals | `PEOPLE_DATA_API_KEY` |
| `resolve_contact` | Hunter.io (live); without a key returns **ranked email-pattern candidates with confidence** (real heuristic, clearly labelled) | `HUNTER_API_KEY` |
| `identity_verify` (KYC) | KYC provider at `KYC_API_URL` (else real structural checks) | `KYC_API_URL` (+ `KYC_API_KEY`) |
| `account_lookup` (servicing) | core-banking / CRM endpoint | `ACCOUNT_API_URL` |
| `parse_financials`, `compute_ratios`, `risk_score`, `compliance_check`, `anomaly_detect` | **pure computation (always real)** | no |

- **Also real:** spec compilation + validation, the governed runtime loop, **actual tool
  invocation**, least-privilege/injection/PII/HITL guardrails, risk scoring, persisted runs with
  status, review workflow that affects execution, tracing, audit, tenant isolation, auth, and the
  deployable HTTP endpoint.
- **Fallback / demo:** when a live call fails, times out, or a key is missing, the tool returns a
  clearly-labelled deterministic sample so the platform still runs fully offline (and tests stay
  deterministic). The runtime is `mode: "rules"` (real tools, deterministic decision) until
  `GROQ_API_KEY` is set, then `mode: "llm"` (adaptive reasoning). `ocr_extract` now does **real** local extraction (PDF text + Tesseract image OCR); only a no-document offline call returns a labelled sample.
- **MCP is real:** registering an approved server opens a live MCP session to its URL
  (Streamable HTTP, JSON-RPC `initialize` + `tools/list`), registers the server's actual
  tools, and proxies calls to it via `tools/call`. Discovery is cached so registered MCP
  tools survive a restart; offline (no key / unreachable) a labelled demo catalog is used.

## Known limitations

- `find_persona` / `resolve_contact` / `pep_check` / `web_search` need a (free-tier) API key to
  go live; without it they 