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
import time

_PII_PATTERNS = {
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[ -]){3}\d{4}\b|\b\d{16}\b"),
    # phone: require separators / + prefix so bare integers (e.g. revenue 62000000) don't match
    "phone": re.compile(r"\b(?:\+?\d{1,3}[-.\s])?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"),
    # IBAN: intrinsically specific (2 letters + 2 check digits + 10-30 alnum) — low false-positive.
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b"),
    # Account number: ONLY when label-introduced AND the value contains a digit, so legitimate
    # words ("account active") and bare financial figures are never redacted.
    "account": re.compile(
        r"\b(?:account|acct|a/c)\s*(?:no\.?|number|num|#)?\s*[:#]?\s*"
        r"((?=[A-Za-z0-9-]*\d)[A-Za-z0-9][A-Za-z0-9-]{5,})\b", re.I),
    # Date of birth: only when DOB-labelled, so ordinary dates (filing/statement dates) survive.
    "dob": re.compile(
        r"\b(?:dob|d\.o\.b\.?|date of birth|born)\b\s*[:#]?\s*"
        r"(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})", re.I),
    # Passport / national id: label-introduced alnum identifier.
    "passport": re.compile(
        r"\b(?:passport|national\s*id|nin|tax\s*id|ein)\s*(?:no\.?|number|#)?\s*[:#]?\s*"
        r"([A-Z]{0,3}\d[A-Z0-9-]{4,})\b", re.I),
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


def redact_obj(obj):
    """Recursively redact PII from any nested structure, returning a clean copy.
    Used to scrub agent results BEFORE they are persisted to the blackboard / returned."""
    if isinstance(obj, str):
        return redact_pii(obj)[0]
    if isinstance(obj, dict):
        return {k: redact_obj(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [redact_obj(v) for v in obj]
    return obj


def _find_true_flag(obj, keys: set) -> bool:
    """Recursively look for a boolean True under any key in `keys` (e.g. ofac_hit).
    Structural — does NOT rely on flattening, so it survives JSON shape changes."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys and v is True:
                return True
            if _find_true_flag(v, keys):
                return True
    elif isinstance(obj, (list, tuple)):
        return any(_find_true_flag(v, keys) for v in obj)
    return False


_RED_FLAGS = ["guaranteed return", "no risk", "insider tip", "evade tax", "launder"]
_SANCTION_KEYS = {"ofac_hit", "sanctions_hit", "pep", "watchlist_hit"}


def compliance_overseer(output) -> dict:
    text = _flatten(output)
    low = text.lower()
    flags = [f for f in _RED_FLAGS if f in low]
    sanctions_hit = _find_true_flag(output, _SANCTION_KEYS)   # structural, reliable
    if sanctions_hit:
        flags.append("sanctions/PEP/watchlist hit")
    _, pii = redact_pii(text)
    leaked = [p for p in pii if p in ("ssn", "credit_card")]  # block only hard identifiers
    blocked = bool(leaked) or sanctions_hit or "guaranteed return" in low or "no risk" in low
    return {"flags": flags, "leaked_pii": pii, "blocked": blocked,
            "verdict": "BLOCK" if blocked else ("ANNOTATE" if (flags or pii) else "PASS")}


class Budget:
    def __init__(self, guardrails):
        self.max_steps = guardrails.max_steps
        self.max_tokens = guardrails.max_tokens
        self.timeout_seconds = guardrails.timeout_seconds
        self.steps = 0
        self.tokens = 0
        self._start = time.monotonic()

    def step(self, tokens: int = 0):
        self.steps += 1
        self.tokens += tokens
        if self.steps > self.max_steps:
            raise GuardrailTrip("cost_loop", f"max_steps {self.max_steps} exceeded")
        if self.tokens > self.max_tokens:
            raise GuardrailTrip("cost_loop", f"max_tokens {self.max_tokens} exceeded")
        if time.monotonic() - self._start > self.timeout_seconds:
            raise GuardrailTrip("cost_loop",
                                f"timeout_seconds {self.timeout_seconds} exceeded")
