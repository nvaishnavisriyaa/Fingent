"""
Cross-cutting middleware (§8, §9, §10, §11) — wraps EVERY node so agent authors write zero
guardrail / observability / security / logging code.

  * Input guardrail:  PII detect+redact; prompt-injection detection on all untrusted input
                      (web, MCP, external, documents, AND the customer free-text field).
  * Output guardrail: compliance overseer (when output_review_required) — regulatory red-flags,
                      leaked PII, unsupported claims; can block or annotate, then HITL.
  * Tool guardrail:   least privilege — only security.allowed_tools may be invoked; anything else
                      is denied + logged.
  * Cost/loop:        enforce max_steps / max_tokens / timeout_seconds.
  * HITL gate:        pauses (LangGraph interrupt-style) when human review is required.
"""
from __future__ import annotations

import re

_PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b|\b\d{16}\b"),
    # phone: require separators / + prefix so bare integers (e.g. revenue 62000000) don't match
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
}

_INJECTION_SIGNATURES = [
    r"ignore (all|any|previous) instructions",
    r"disregard (the )?(above|previous|system)",
    r"grant (yourself|me) ",
    r"you are now",
    r"bypass (the )?(review|guardrail|policy)",
    r"exfiltrat|leak the|send the .* to ",
    r"(email|wire|transfer) .* to \S+@\S+",
    r"system prompt",
]


class GuardrailTrip(Exception):
    def __init__(self, kind: str, detail: str):
        self.kind, self.detail = kind, detail
        super().__init__(f"{kind}: {detail}")


class ToolDenied(Exception):
    pass


class HumanReviewRequired(Exception):
    """Raised to pause the run at the HITL gate (LangGraph interrupt analogue)."""
    def __init__(self, payload: dict):
        self.payload = payload
        super().__init__("human review required")


def redact_pii(text: str):
    found = []
    out = text
    for kind, pat in _PII_PATTERNS.items():
        if pat.search(out):
            found.append(kind)
            out = pat.sub(f"[REDACTED_{kind.upper()}]", out)
    return out, found


def detect_injection(text: str) -> list[str]:
    hits = []
    low = (text or "").lower()
    for sig in _INJECTION_SIGNATURES:
        if re.search(sig, low):
            hits.append(sig)
    return hits


def _flatten(obj) -> str:
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return " ".join(_flatten(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return " ".join(_flatten(v) for v in obj)
    return str(obj)


def scan_untrusted(payload, label: str) -> list[str]:
    """Flatten a tool result and injection-scan any text in it."""
    return detect_injection(_flatten(payload))


_RED_FLAGS = ["guaranteed return", "no risk", "insider tip", "evade tax",
              "ofac_hit': true", "launder"]


def compliance_overseer(output) -> dict:
    text = _flatten(output)
    low = text.lower()
    flags = [f for f in _RED_FLAGS if f in low]
    _, pii = redact_pii(text)
    leaked = [p for p in pii if p in ("ssn", "credit_card")]  # block only hard identifiers
    blocked = bool(leaked) or "guaranteed return" in low or "no risk" in low
    return {"flags": flags, "leaked_pii": pii, "blocked": blocked,
            "verdict": "BLOCK" if blocked else ("ANNOTATE" if (flags or pii) else "PASS")}


class Budget:
    def __init__(self, guardrails):
        self.max_steps = guardrails.max_steps
        self.max_tokens = guardrails.max_tokens
        self.timeout_seconds = guardrails.timeout_seconds
        self.steps = 0
        self.tokens = 0

    def step(self, tokens: int = 0):
        self.steps += 1
        self.tokens += tokens
        if self.steps > self.max_steps:
            raise GuardrailTrip("cost_loop", f"max_steps {self.max_steps} exceeded")
        if self.tokens > self.max_tokens:
            raise GuardrailTrip("cost_loop", f"max_tokens {self.max_tokens} exceeded")
