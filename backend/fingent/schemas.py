"""
Fingent — typed, validated contracts (the data model).

Everything in the platform flows through these Pydantic models. An *agent is data*:
a saved AgentSpec is the single source of truth, rebuilt into a live agent on each run.
"""
from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# External tools
# --------------------------------------------------------------------------- #
class ToolKind(str, Enum):
    NATIVE = "native"            # platform-built (edgar_search, ofac_screen, ...)
    WEB_SEARCH = "web_search"    # built-in web search
    MCP = "mcp"                  # tool exposed by a registered MCP server
    EXTERNAL_API = "external_api"


class ToolDescriptor(BaseModel):
    name: str
    kind: ToolKind
    description: str                       # the LLM compiler reads this to decide grants
    side_effecting: bool = False           # writes/sends/pays -> needs explicit approval
    mcp_server: str | None = None          # for kind=MCP: which server it came from
    secrets_ref: list[str] = Field(default_factory=list)  # injected from vault at call time
    tenant_id: str | None = None           # None = global (e.g. native, web_search)
    untrusted_output: bool = False         # output must be treated as data, not instructions


class McpServer(BaseModel):
    name: str
    url: str
    tenant_id: str                         # MCP servers are tenant-scoped
    approved: bool = False                  # admin must approve before tools can be granted
    secrets_ref: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Dependencies / policies
# --------------------------------------------------------------------------- #
class DependencyType(str, Enum):
    HARD = "hard"
    SOFT = "soft"


class Dependency(BaseModel):
    agent: str
    type: DependencyType
    reason: str


class GuardrailPolicy(BaseModel):
    input_pii_check: bool = True
    injection_check: bool = True           # MUST be on for any agent with web/MCP tools
    output_review_required: bool = False   # route output through compliance overseer
    max_steps: int = 12
    max_tokens: int = 100_000
    timeout_seconds: int = 120


class SecurityPolicy(BaseModel):
    allowed_tools: list[str]               # least privilege: agent may call ONLY these
    memory_read: list[str]                 # blackboard key-prefixes it may READ
    memory_write: list[str]                # blackboard key-prefixes it may WRITE
    tenant_id: str
    secrets_ref: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------- #
# Spec
# --------------------------------------------------------------------------- #
class AgentSpec(BaseModel):
    name: str
    template: str | None = None            # which template minted it (None = from scratch)
    tier: int
    role_prompt: str
    tools: list[str] = Field(default_factory=list)   # resolved against the tool registry
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    depends_on: list[Dependency] = Field(default_factory=list)
    guardrails: GuardrailPolicy = Field(default_factory=GuardrailPolicy)
    security: SecurityPolicy
    requires_human_review: bool = False
    # ---- first-class agent configuration (editable by the operator) ---------- #
    purpose: str = ""                      # one-line business purpose
    instructions: str = ""                 # operating instructions for the runtime
    input_schema: dict = Field(default_factory=dict)   # expected inputs {field: hint}
    output_format: str = "summary"         # summary | json | recommendation
    risk_level: str = "medium"             # low | medium | high (operator-declared)
    deployed: bool = True                  # exposed as a callable endpoint


# --------------------------------------------------------------------------- #
# Runs (first-class, persisted, with status)
# --------------------------------------------------------------------------- #
class RunStep(BaseModel):
    idx: int
    kind: str                              # think | tool | guardrail | review | output | error
    tool: str | None = None
    tool_input: dict | None = None
    tool_output: Any | None = None
    note: str = ""
    blocked: bool = False
    latency_ms: float = 0.0


class RunRecord(BaseModel):
    id: str
    tenant_id: str
    agent: str
    trace_id: str
    mode: str                              # "llm" (real model) | "demo" (deterministic engine)
    input: dict = Field(default_factory=dict)
    status: str = "success"                # success|failed|blocked|needs_review|approved|rejected
    steps: list[RunStep] = Field(default_factory=list)
    output: Any | None = None
    risk_score: int = 0                    # 0-100
    risk_level: str = "low"                # low | medium | high
    risk_flags: list[str] = Field(default_factory=list)
    pending_action: dict | None = None     # side-effecting tool held for approval
    reviewer: str | None = None
    review_note: str = ""
    duration_ms: float = 0.0
    ts: float = 0.0


# --------------------------------------------------------------------------- #
# Template + form
# --------------------------------------------------------------------------- #
class TemplateParameter(BaseModel):
    name: str
    type: str                              # text | number | boolean | select | multi_select
    label: str                             # shown in the HTML form
    options: list[str] | None = None
    default: Any | None = None
    required: bool = True
    min: float | None = None
    max: float | None = None


class AgentTemplate(BaseModel):
    name: str
    tier: int
    description: str
    fixed: dict = Field(default_factory=dict)          # base_role, required_tools, ...
    parameters: list[TemplateParameter] = Field(default_factory=list)
    default_depends_on: list[Dependency] = Field(default_factory=list)
    default_guardrails: GuardrailPolicy = Field(default_factory=GuardrailPolicy)
    grantable_tools: list[str] = Field(default_factory=list)  # universe the LLM may pick from


# --------------------------------------------------------------------------- #
# Creation request (off the HTML form)
# --------------------------------------------------------------------------- #
class CreateAgentRequest(BaseModel):
    template: str | None = None            # None -> build from scratch
    answers: dict = Field(default_factory=dict)   # structured form fields
    additional_requirements: str = ""      # the free-text last field (UNTRUSTED input)
    tenant_id: str
    approve_side_effecting: bool = False    # blanket operator approval for ALL write/pay tools
    approved_side_effecting_tools: list[str] = Field(default_factory=list)  # per-tool approval


# --------------------------------------------------------------------------- #
# Dependency resolution result
# --------------------------------------------------------------------------- #
class DependencyCheck(BaseModel):
    ok: bool
    missing_hard: list[Dependency] = Field(default_factory=list)
    missing_soft: list[Dependency] = Field(default_factory=list)
    creation_order: list[str] = Field(default_factory=list)   # topo-sorted chain to provision
    cycle: list[str] | None = None


# --------------------------------------------------------------------------- #
# Compiler / validator audit records
# --------------------------------------------------------------------------- #
class ValidationVerdict(BaseModel):
    ok: bool
    errors: list[str] = Field(default_factory=list)
    stripped: list[str] = Field(default_factory=list)         # what was removed + why
    warnings: list[str] = Field(default_factory=list)


class CompileResult(BaseModel):
    ok: bool
    spec: AgentSpec | None = None
    attempts: int = 0
    candidate_specs: list[dict] = Field(default_factory=list)  # raw LLM proposals
    verdicts: list[ValidationVerdict] = Field(default_factory=list)
    message: str = ""
    used_llm: bool = False
