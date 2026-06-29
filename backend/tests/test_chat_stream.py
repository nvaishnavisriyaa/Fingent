"""
Streaming chat (acceptance #2, #5) + session working memory (#6, session part).
Offline/deterministic: conftest neutralizes LLM keys, so this exercises the demo engine of the
same streaming loop the LLM uses live (mode='llm'). Drives the real SSE generator `chat_sse`
(what the /api/agents/{name}/chat StreamingResponse wraps).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
from fingent.schemas import CreateAgentRequest
from fingent.chat import chat_sse, get_history, clear_session


def _mk(fp):
    tpl = fp.store.get_template("aml_sanctions_screening")
    answers = {"name": "aml"}
    for p in tpl.parameters:
        if p.default is not None:
            answers[p.name] = p.default
    fp.create_agent(CreateAgentRequest(template="aml_sanctions_screening",
                                       answers=answers, tenant_id="acme"))


def _drain(fp, session, text):
    evs = []
    for chunk in chat_sse(fp, "acme", "aml", session, text):
        assert chunk.startswith("data: ") and chunk.endswith("\n\n")
        evs.append(json.loads(chunk[6:].strip()))
    return evs


@pytest.fixture
def fp():
    f = Fingent()
    _mk(f)
    clear_session("acme", "aml", "s1")
    clear_session("acme", "aml", "mem")
    return f


def test_chat_streams_tool_trace_and_prose(fp):
    evs = _drain(fp, "s1", "Screen Oleg Petrov for sanctions")
    types = [e["type"] for e in evs]
    # a real run: tool actually called, results observed, prose answer produced, stream closed
    assert "start" in types and "tool_call" in types and "tool_result" in types
    assert "token" in types          # the prose answer streamed token-by-token
    assert "final" in types and types[-1] == "done"
    final = next(e for e in evs if e["type"] == "final")
    assert isinstance(final["text"], str) and final["text"].strip()   # conversational surface
    assert "structured" in final                                      # structured payload secondary
    assert final["run_id"].startswith("run_")
    called = [e["tool"] for e in evs if e["type"] == "tool_call"]
    assert any(t in ("ofac_screen", "pep_check", "adverse_media_search") for t in called)
    # the streamed turn is persisted as a real RunRecord (shows in Runs/Monitoring)
    assert fp.store.get_run("acme", final["run_id"]) is not None


def test_session_memory_recall_within_session(fp):
    _drain(fp, "mem", "The subject of interest is Oleg Petrov")
    hist = get_history("acme", "aml", "mem")
    assert any(m["role"] == "user" and "Oleg Petrov" in m["content"] for m in hist)
    assert any(m["role"] == "assistant" for m in hist)
    # second turn: the runtime is handed the prior turn as working memory (recall)
    captured = {}
    orig = fp.runtime.run_stream
    def spy(spec, history, user_text, tenant, **kw):
        captured["history"] = list(history)
        return orig(spec, history, user_text, tenant, **kw)
    fp.runtime.run_stream = spy
    _drain(fp, "mem", "Is that subject sanctioned?")
    assert any("Oleg Petrov" in (m.get("content") or "") for m in captured["history"]), \
        "second turn did not receive the first turn as context"
