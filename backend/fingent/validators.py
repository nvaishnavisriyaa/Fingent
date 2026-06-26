"""
Validators (§3).

1. INPUT VALIDATOR — checks structured form answers against template.parameters BEFORE the LLM is
   ever called. Malformed structured input is rejected per-field here. The free-text field is NOT
   validated here (natural language) — it is handled safely by the compiler + spec validator.

2. SPEC VALIDATOR (the *disposer*) — the security boundary. The LLM *proposes* a candidate spec;
   this deterministic validator *disposes*: every tool exists + is tenant-approved; no tool outside
   the grantable universe (no self-widening); side-effecting tools need explicit approval (else
   stripped); memory prefixes stay within scope; injection_check forced ON with any web/MCP tool;
   guardrails only get stricter; the whole thing re-parses as AgentSpec. Privilege escalation via
   free text is impossible by construction.
"""
from __future__ import annotations

from .registry import ToolRegistry
from .schemas import AgentSpec, AgentTemplate, ToolKind, ValidationVerdict


def validate_inputs(template: AgentTemplate | None, answers: dict) -> list[str]:
    """Returns a list of per-field error strings (empty = valid)."""
    errors: list[str] = []
    if template is None:
        return errors
    for p in template.parameters:
        present = p.name in answers and answers[p.name] not in (None, "", [])
        if p.required and not present:
            errors.append(f"'{p.label}' ({p.name}) is required")
            continue
        if not present:
            continue
        v = answers[p.name]
        if p.type == "number":
            try:
                num = float(v)
            except (TypeError, ValueError):
                errors.append(f"'{p.label}' must be a number")
                continue
            if p.min is not None and num < p.min:
                errors.append(f"'{p.label}' must be >= {p.min}")
            if p.max is not None and num > p.max:
                errors.append(f"'{p.label}' must be <= {p.max}")
        elif p.type == "boolean":
            if not isinstance(v, bool):
                errors.append(f"'{p.label}' must be true/false")
        elif p.type == "select":
            if p.options and v not in p.options:
                errors.append(f"'{p.label}' must be one of {p.options}")
        elif p.type == "multi_select":
            if not isinstance(v, list):
                errors.append(f"'{p.label}' must be a list")
            elif p.options and any(x not in p.options for x in v):
                errors.append(f"'{p.label}' has values outside {p.options}")
    return errors


def validate_spec(candidate: dict, template: AgentTemplate | None, registry: ToolRegistry,
                  tenant_id: str, approve_side_effecting: bool = False):
    errors: list[str] = []
    stripped: list[str] = []
    warnings: list[str] = []

    if template is not None:
        grantable = set(registry.effective_grantable(template.grantable_tools, tenant_id))
    else:
        grantable = set(registry.grantable_for_tenant(tenant_id))

    candidate.setdefault("security", {})
    candidate["security"]["tenant_id"] = tenant_id

    requested_tools = list(candidate.get("tools", []))
    kept_tools: list[str] = []
    has_untrusted_tool = False

    for t in requested_tools:
        desc = registry.get(t)
        if desc is None:
            stripped.append(f"{t}: not in registry (LLM may have invented it)")
            continue
        if not registry.is_grantable(tenant_id, t):
            stripped.append(f"{t}: not approved for tenant '{tenant_id}'")
            continue
        if t not in grantable:
            stripped.append(f"{t}: outside grantable universe (no self-widening)")
            continue
        if desc.side_effecting and not approve_side_effecting:
            stripped.append(f"{t}: side-effecting tool requires explicit grant-time approval")
            continue
        if desc.untrusted_output or desc.kind in (ToolKind.WEB_SEARCH, ToolKind.MCP,
                                                  ToolKind.EXTERNAL_API):
            has_untrusted_tool = True
        kept_tools.append(t)

    candidate["tools"] = kept_tools
    candidate["security"]["allowed_tools"] = kept_tools

    allowed_prefixes = None
    if template is not None:
        allowed_prefixes = set(template.fixed.get("memory_prefixes", []))
    if allowed_prefixes:
        for field in ("reads", "writes"):
            cleaned = []
            for k in candidate.get(field, []):
                if any(k == p or k.startswith(p) for p in allowed_prefixes):
                    cleaned.append(k)
                else:
                    stripped.append(f"memory {field} '{k}': outside template memory scope")
            candidate[field] = cleaned
    candidate["security"]["memory_read"] = candidate.get("reads", [])
    candidate["security"]["memory_write"] = candidate.get("writes", [])

    candidate.setdefault("guardrails", {})
    if has_untrusted_tool:
        candidate["guardrails"]["injection_check"] = True
        warnings.append("injection_check forced ON (agent has web/MCP/external tools)")

    if template is not None:
        td = template.default_guardrails
        if td.output_review_required and not candidate["guardrails"].get("output_review_required"):
            candidate["guardrails"]["output_review_required"] = True
            warnings.append("output_review_required floored ON per template policy")
        if td.input_pii_check:
            candidate["guardrails"]["input_pii_check"] = True
        for cap in ("max_steps", "max_tokens", "timeout_seconds"):
            req = candidate["guardrails"].get(cap)
            ceiling = getattr(td, cap)
            if req is None or req > ceiling:
                candidate["guardrails"][cap] = ceiling

    try:
        spec = AgentSpec.model_validate(candidate)
    except Exception as e:  # noqa: BLE001
        errors.append(f"spec failed AgentSpec validation: {e}")
        return None, ValidationVerdict(ok=False, errors=errors,
                                       stripped=stripped, warnings=warnings)

    return spec, ValidationVerdict(ok=len(errors) == 0, errors=errors,
                                   stripped=stripped, warnings=warnings)
