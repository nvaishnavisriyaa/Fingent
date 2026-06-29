"""
Token/cost metrics are HONEST (no fabricated constants).

- A run with a model that reports usage records REAL prompt/completion tokens, priced from the
  model table (observability.price_for).
- A run with no usage is flagged ESTIMATED, never priced as if measured.
- Demo mode (no LLM) reports ZERO tokens/cost — not a fake per-step constant.
- analytics() surfaces cost_estimated / cost_basis so the dashboard can tell the truth.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
from fingent.schemas import CreateAgentRequest
from fingent.observability import price_for, Tracer
import fingent.runtime as rt


class _UsageProvider:
    """Fake model that reports real usage like a provider would."""
    enabled = True
    name = "metered"
    model = "llama-3.3-70b-versatile"

    def __init__(self):
        self.last_usage = {}

    def chat(self, messages, tools=None, tool_choice="auto", **k):
        already = any(m.get("role") == "tool" for m in messages)
        if tools and not already:
            fn = tools[0]["function"]["name"]
            self.last_usage = {"prompt_tokens": 1000, "completion_tokens": 200,
                               "total_tokens": 1200}
            return {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "c1", "function": {"name": fn, "arguments": "{}"}}]}
        self.last_usage = {"prompt_tokens": 1500, "completion_tokens": 300,
                           "total_tokens": 1800}
        return {"role": "assistant", "content": "Done."}

    def usage_split(self):
        u = self.last_usage
        return (u.get("prompt_tokens", 0), u.get("completion_tokens", 0), bool(u.get("_estimated")))


def _agent(fp):
    fp.create_agent(CreateAgentRequest(template="aml_sanctions_screening",
                                       answers={"name": "aml", "lists": ["OFAC"]},
                                       tenant_id="acme"))


def test_pricing_table_flags_unknown_models_estimated():
    price, fallback = price_for("llama-3.3-70b-versatile")
    assert fallback is False and price["in"] > 0
    _, fb2 = price_for("some-unlisted-model-x")
    assert fb2 is True            # unknown model -> fallback price, flagged


def test_real_usage_is_recorded_and_priced(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "0")
    monkeypatch.setattr(rt, "LlmProvider", _UsageProvider)
    fp = Fingent()
    _agent(fp)
    rec = fp.run_task("acme", "aml", {"name": "Jane Smith"})
    tr = fp.store.get_trace("acme", rec["trace_id"])
    m = tr["metrics"]
    # two model turns: (1000+200) + (1500+300) = 3000 real tokens, NOT a 300/step constant
    assert m["tokens"] == 3000
    assert m["prompt_tokens"] == 2500 and m["completion_tokens"] == 500
    assert m["llm_calls"] == 2
    assert m["usage_source"] == "actual" and m["cost_estimated"] is False
    # cost = priced from the real split against the llama table
    p = price_for("llama-3.3-70b-versatile")[0]
    expected = round(2500/1000*p["in"] + 500/1000*p["out"], 6)
    assert abs(m["cost_usd"] - expected) < 1e-6
    assert m["cost_usd"] > 0


def test_rules_mode_reports_zero_not_fabricated(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "0")
    for k in ("GROQ_API_KEY", "FINGENT_LLM_API_KEY"):
        monkeypatch.setenv(k, "")
    fp = Fingent()
    _agent(fp)
    rec = fp.run_task("acme", "aml", {"name": "Jane Smith"})
    assert rec["mode"] == "rules"
    m = fp.store.get_trace("acme", rec["trace_id"])["metrics"]
    assert m["tokens"] == 0 and m["cost_usd"] == 0.0   # no LLM -> no fabricated tokens/cost


def test_analytics_exposes_cost_basis(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "0")
    monkeypatch.setattr(rt, "LlmProvider", _UsageProvider)
    fp = Fingent()
    _agent(fp)
    fp.run_task("acme", "aml", {"name": "Jane Smith"})
    a = fp.analytics("acme")
    assert "cost_estimated" in a["totals"] and "cost_basis" in a["totals"]
    assert a["totals"]["tokens"] == 3000
    assert a["totals"]["cost_basis"] == "real LLM usage"
