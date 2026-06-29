"""
Long-term memory (acceptance #6, cross-session): a fact learned in one chat session is recalled
in a DIFFERENT session of the same agent. Offline -> local vector backend (the Pinecone backend
activates when PINECONE_API_KEY is set; same interface, see demo_memory.py).
"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fingent import Fingent
from fingent.schemas import CreateAgentRequest
from fingent.chat import chat_sse, clear_session
from fingent.memory import get_memory, reset_memory_for_tests, LocalVectorMemory


@pytest.fixture
def fp():
    reset_memory_for_tests()
    f = Fingent()
    tpl = f.store.get_template("aml_sanctions_screening")
    answers = {"name": "aml"}
    for p in tpl.parameters:
        if p.default is not None:
            answers[p.name] = p.default
    f.create_agent(CreateAgentRequest(template="aml_sanctions_screening",
                                      answers=answers, tenant_id="acme"))
    return f


def _events(fp, session, text):
    return [json.loads(c[6:].strip()) for c in chat_sse(fp, "acme", "aml", session, text)]


def test_backend_is_local_without_key():
    reset_memory_for_tests()
    assert isinstance(get_memory(), LocalVectorMemory)
    assert get_memory().backend == "local"


def test_fact_from_one_session_recalled_in_another(fp):
    # Session A: establish a memory by chatting
    _events(fp, "sessionA", "Please screen Oleg Petrov for OFAC sanctions and PEP status.")
    # the turn was written to long-term memory
    assert get_memory().recall("acme", "aml", "Oleg Petrov", k=1)

    # Session B: a brand-new session (no shared working memory) recalls the earlier fact
    clear_session("acme", "aml", "sessionB")
    evs = _events(fp, "sessionB", "What did we find about Oleg Petrov earlier?")
    mem_events = [e for e in evs if e["type"] == "memory"]
    assert mem_events, "no long-term memory was recalled into the new session"
    recalled_text = " ".join(r["text"] for e in mem_events for r in e["recalled"])
    assert "Oleg Petrov" in recalled_text


def test_recalled_memory_is_injected_into_the_agent_context(fp):
    _events(fp, "s1", "Note: Acme Corp was flagged for adverse media in 2024.")
    clear_session("acme", "aml", "s2")
    captured = {}
    orig = fp.runtime.run_stream
    def spy(spec, history, user_text, tenant, **kw):
        captured["history"] = list(history)
        return orig(spec, history, user_text, tenant, **kw)
    fp.runtime.run_stream = spy
    _events(fp, "s2", "Any adverse media on Acme Corp?")
    sys_notes = [m for m in captured["history"] if m.get("role") == "system"]
    assert sys_notes and any("Acme Corp" in m["content"] for m in sys_notes), \
        "recalled memory was not injected as agent context"
