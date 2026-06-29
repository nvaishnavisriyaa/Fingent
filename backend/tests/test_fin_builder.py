"""
"Describe an agent and Fin builds it" — from-scratch agent creation.

Covers both engines:
  * offline heuristic — no model configured; Fin still builds a COMPLETE spec (tools, purpose,
    instructions, expected inputs, output format, risk level, human-review) by inferring intent.
  * LLM engine — provider-agnostic; pointed at a mock OpenAI-compatible server. Proves the
    validator still disposes (an out-of-scope tool the model proposes is stripped).
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
from fingent.schemas import CreateAgentRequest


def _fin(fp, prompt, name="fin_agent", tenant="acme", **kw):
    return fp.create_agent(CreateAgentRequest(
        template=None, answers={"name": name}, additional_requirements=prompt,
        tenant_id=tenant, **kw))


# --------------------------------------------------------------------------- #
# offline heuristic engine
# --------------------------------------------------------------------------- #
def test_fin_offline_builds_a_complete_sanctions_agent():
    fp = Fingent()
    r = _fin(fp, "Screen new customers against OFAC sanctions and PEP lists and flag any "
                 "hits with a recommendation for review.")
    assert r["ok"] and r["used_llm"] is False
    spec = r["spec"]
    # right least-privilege tools were chosen from intent
    assert "ofac_screen" in spec["tools"] and "pep_check" in spec["tools"]
    # the spec is COMPLETE, not a stub
    assert spec["purpose"] and spec["purpose"] != "Custom financial-services agent."
    assert spec["instructions"]
    assert spec["output_format"] == "recommendation"     # "recommendation"/"flag" language
    assert spec["risk_level"] == "high"                  # sanctions/PEP -> high
    assert spec["requires_human_review"] is True         # high-risk + "review"
    assert "name" in spec["input_schema"]                # person-name input inferred
    # it was crystallized into a reusable template
    assert r["crystallized_template"]


def test_fin_offline_informational_agent_is_lower_risk():
    fp = Fingent()
    r = _fin(fp, "Summarize recent news and SEC filings for a company.", name="news_agent")
    assert r["ok"]
    spec = r["spec"]
    assert "web_search" in spec["tools"] or "news_monitor" in spec["tools"]
    assert "edgar_search" in spec["tools"]
    assert spec["output_format"] == "summary"
    assert spec["risk_level"] == "medium"                # purely informational
    assert "company" in spec["input_schema"]


# --------------------------------------------------------------------------- #
# LLM engine (provider-agnostic) + validator still disposes
# --------------------------------------------------------------------------- #
def _spec_server(spec_json):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            _ = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
            body = json.dumps({"choices": [{"message": {
                "role": "assistant", "content": json.dumps(spec_json)}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_fin_llm_builds_and_validator_strips_out_of_scope_tool(monkeypatch):
    # the "model" proposes two legitimate tools plus an out-of-scope one it must not get
    proposed = {
        "name": "esg_agent", "template": None, "tier": 2,
        "purpose": "Screen ESG and reputational risk for an issuer.",
        "instructions": "Search adverse media and the web, then summarize red flags.",
        "input_schema": {"company": "Issuer name"},
        "output_format": "recommendation", "risk_level": "high",
        "role_prompt": "You are an ESG risk screening agent.",
        "tools": ["adverse_media_search", "web_search", "wire_transfer"],  # wire_transfer is bogus
        "reads": [], "writes": [], "depends_on": [], "guardrails": {},
        "requires_human_review": True,
    }
    srv, url = _spec_server(proposed)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("FINGENT_LLM_API_KEY", "test-key")
    monkeypatch.setenv("FINGENT_LLM_BASE_URL", url)
    try:
        fp = Fingent()
        r = _fin(fp, "Build an ESG risk screener using adverse media and web search.",
                 name="esg_agent")
    finally:
        srv.shutdown()

    assert r["ok"] and r["used_llm"] is True
    spec = r["spec"]
    assert "adverse_media_search" in spec["tools"] and "web_search" in spec["tools"]
    assert "wire_transfer" not in spec["tools"]          # validator disposed of it
    strips = [s for v in r["compiler_log"]["verdicts"] for s in v["stripped"]]
    assert any("wire_transfer" in s for s in strips)
    # the LLM-built spec is complete
    assert spec["purpose"] and spec["output_format"] == "recommendation"
    assert spec["risk_level"] == "high"


def test_fin_compiler_is_provider_agnostic(monkeypatch):
    """The compiler honours FINGENT_LLM_BASE_URL (not just Groq): if it reaches our mock
    server it gets used (used_llm True)."""
    proposed = {"name": "x", "tools": ["web_search"], "purpose": "p",
                "output_format": "summary", "risk_level": "low"}
    srv, url = _spec_server(proposed)
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("FINGENT_LLM_API_KEY", "k")
    monkeypatch.setenv("FINGENT_LLM_BASE_URL", url)
    try:
        fp = Fingent()
        r = _fin(fp, "search the web", name="searcher")
    finally:
        srv.shutdown()
    assert r["ok"] and r["used_llm"] is True
    assert "web_search" in r["spec"]["tools"]
