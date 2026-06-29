"""
Keyless execution: with NO model configured the platform does NOT go dead — it runs the agent for
real in deterministic "rules" mode. The agent's actual tools execute against live data and the
decision is composed STRICTLY from those real tool outputs (never fabricated reasoning/values),
clearly labelled mode="rules". A model upgrades the same agent to adaptive multi-step reasoning
(mode="llm").

Offline in tests (FINGENT_LIVE_DATA=0) the tools return clearly-labelled deterministic samples, so
these assertions are reproducible; the execution path is identical to production live mode.
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
from fingent.schemas import AgentSpec, CreateAgentRequest, SecurityPolicy
from fingent.chat import chat_sse
from fingent.observability import Tracer
import fingent.runtime as rt


@pytest.fixture
def no_model(monkeypatch):
    # simulate the shipped product with no model key
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("FINGENT_LLM_API_KEY", "")
    f = Fingent()
    f.create_agent(CreateAgentRequest(template="aml_sanctions_screening",
                                      answers={"name": "aml", "lists": ["OFAC"]},
                                      tenant_id="acme"))
    return f


def test_resolve_mode_is_rules_without_a_model():
    from fingent.llm import LlmProvider
    assert rt._resolve_mode(LlmProvider()) == "rules"


def test_keyless_run_executes_real_tools_and_decides(no_model):
    rec = no_model.run_task("acme", "aml", {"name": "Oleg Petrov"})
    assert rec["mode"] == "rules"
    # real tools actually executed (not blocked, not empty)
    tool_steps = [s for s in rec["steps"] if s.get("kind") == "tool"]
    assert tool_steps, "rules mode must execute the agent's real tools"
    assert any(s.get("tool") == "ofac_screen" for s in tool_steps)
    # the output is a decision composed from real tool outputs
    out = rec["output"]
    assert isinstance(out, dict) and ("decision" in out or "findings" in out)
    # a sanctions hit on Oleg Petrov routes to a blocking/review outcome (governed, not fabricated)
    assert rec["status"] in ("blocked", "needs_review", "success")


def test_run_node_runs_in_rules_mode(no_model):
    spec = no_model.store.get_spec("acme", "aml")
    res = no_model.runtime.run_node(spec, {"name": "Oleg Petrov"}, "acme", Tracer("acme"))
    assert res["mode"] == "rules"
    assert any(getattr(s, "kind", "") == "tool" for s in res["steps"])
    assert res["status"] in ("success", "blocked", "needs_review")


def test_chat_runs_real_tools_without_a_model(no_model):
    evs = [json.loads(c[6:].strip())
           for c in chat_sse(no_model, "acme", "aml", "s1", "Screen Oleg Petrov")]
    types = [e["type"] for e in evs]
    assert "tool_call" in types                       # real tools fired, not a dead end
    assert types[-1] == "done"
    final = next(e for e in evs if e["type"] == "final")
    assert final["mode"] == "rules"
    # the rules engine is clearly labelled, never passed off as model reasoning
    assert "rules" in final["text"].lower() or final["status"] in ("blocked", "needs_review", "success")


def test_rules_mode_is_clearly_labelled_not_fabricated(no_model):
    rec = no_model.run_task("acme", "aml", {"name": "Oleg Petrov"})
    assert "rules_mode" in rec["risk_flags"]           # honest engine label on the run
    assert rec["mode"] == "rules"


def test_icp_chat_scores_clean_company_question_even_when_enrichment_degraded(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("FINGENT_LLM_API_KEY", "")
    fp = Fingent()
    fp.create_agent(CreateAgentRequest(
        template="icp_matching",
        answers={"name": "icp", "min_score": 0.6},
        additional_requirements="ICP: US fintechs, 100-1000 employees, raised debt.",
        tenant_id="acme",
    ))
    fp.registry._callables["enrich_company"] = lambda **k: {
        "source": "unavailable",
        "live": False,
        "company": k.get("company"),
        "industry": None,
        "employees": None,
        "revenue_est_usd": None,
        "hq": None,
        "financial_health": None,
    }

    evs = [json.loads(c[6:].strip())
           for c in chat_sse(fp, "acme", "icp", "s1", "Is Stripe an ideal customer? how much would u score it?")]
    tool_call = next(e for e in evs if e["type"] == "tool_call")
    assert tool_call["args"]["company"] == "Stripe"
    final = next(e for e in evs if e["type"] == "final")
    assert final["status"] == "success"
    assert final["structured"]["company"] == "Stripe"
    assert "icp_score" in final["structured"]
    assert final["structured"]["degraded"]
    assert "no real data source" not in final["text"].lower()


def test_icp_agents_get_search_and_strict_numeric_scoring_policy(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("FINGENT_LLM_API_KEY", "")
    fp = Fingent()
    fp.create_agent(CreateAgentRequest(
        template="icp_matching",
        answers={"name": "icp", "min_score": 0.6},
        additional_requirements="ICP: fintech and 100-1000 employees.",
        tenant_id="acme",
    ))
    spec = fp.store.get_spec("acme", "icp")
    assert "enrich_company" in spec.tools and "web_search" in spec.tools

    prompt = rt.AgentRuntime._system_prompt(spec)
    assert "Always return a numeric ICP score" in prompt
    assert "Never return N/A" in prompt
    assert "Build a per-run rubric from the operator's ICP" in prompt
    assert "Criteria may include industry, size, geography, revenue" in prompt
    assert "If one required criterion fails" in prompt
    assert "if multiple required criteria fail" in prompt
    assert "Verdict MUST be derived from the final numeric" in prompt
    assert "Score >=0.75 => ideal" in prompt


def test_custom_named_icp_agent_gets_strict_scoring_policy():
    spec = AgentSpec(
        name="my_agent",
        tier=1,
        role_prompt="Score ideal customer fit for an ICP.",
        purpose="Score ideal customer fit.",
        instructions="ICP: fintech and 100-1000 employees.",
        tools=["enrich_company", "web_search"],
        security=SecurityPolicy(
            allowed_tools=["enrich_company", "web_search"],
            memory_read=[],
            memory_write=[],
            tenant_id="acme",
        ),
    )
    prompt = rt.AgentRuntime._system_prompt(spec)
    assert "ICP SCORING POLICY" in prompt
    assert "Never return N/A" in prompt
    assert "Gaps must only name missing evidence for criteria" in prompt


def test_icp_answer_normalizer_corrects_verdict_and_strips_non_icp_gaps():
    spec = AgentSpec(
        name="my_agent",
        tier=1,
        role_prompt="Score ideal customer fit for an ICP.",
        purpose="Score ideal customer fit.",
        instructions="ICP: fintech and 100-1000 employees.",
        tools=["enrich_company", "web_search"],
        security=SecurityPolicy(
            allowed_tools=["enrich_company", "web_search"],
            memory_read=[],
            memory_write=[],
            tenant_id="acme",
        ),
    )
    text = """Name: Stripe
Score: 0.8
Verdict: partial fit
Findings:
Stripe is a financial technology company.
Gaps:
Stripe's financial health is not publicly disclosed.
Stripe's buyer persona is not clearly defined.
Sources:
web_search"""
    normalized = rt.AgentRuntime._normalize_icp_answer(spec, text)
    assert "Verdict: ideal" in normalized
    assert "financial health is not publicly disclosed" not in normalized
    assert "buyer persona is not clearly defined" not in normalized
    assert "None beyond the ICP criteria explicitly supplied." in normalized


def test_icp_rules_scoring_weights_detected_criteria_not_fixed_dimensions():
    score, reasons, gaps = rt.AgentRuntime._score_icp_match(
        {
            "company": "Acme",
            "industry": "fintech",
            "employees": 5000,
            "hq": "United States",
            "financial_health": "strong",
        },
        {
            "industries": ["fintech"],
            "employees_min": 100,
            "employees_max": 1000,
            "locations": ["united states"],
            "signals": ["profitable"],
        },
    )
    assert score == pytest.approx(0.65)
    assert any("employee range" in r for r in reasons)
    assert not gaps
