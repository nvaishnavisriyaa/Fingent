"""
Chat layer — one streamed conversation per agent.

Holds short-term WORKING MEMORY (per-session message history) so an agent recalls earlier turns
within a session, and turns the runtime's event stream into Server-Sent-Events for the UI. The
final answer is conversational prose; the structured tool payload rides along in the `final`
event as a secondary artifact (never the primary surface).

Long-term cross-session memory (vector store / Pinecone) plugs in here later via
`MemoryStore.recall()` — this module is the single seam for it.
"""
from __future__ import annotations

import json
import threading

from .memory import get_memory

# (tenant, agent, session_id) -> list[{role, content}]
_SESSIONS: dict[tuple, list[dict]] = {}
_LOCK = threading.Lock()
_MAX_TURNS = 24  # keep the last N messages in working memory


def get_history(tenant: str, agent: str, session_id: str) -> list[dict]:
    with _LOCK:
        return list(_SESSIONS.get((tenant, agent, session_id), []))


def _append(tenant: str, agent: str, session_id: str, msgs: list[dict]) -> None:
    with _LOCK:
        key = (tenant, agent, session_id)
        hist = _SESSIONS.setdefault(key, [])
        hist.extend(msgs)
        if len(hist) > _MAX_TURNS:
            del hist[: len(hist) - _MAX_TURNS]


def clear_session(tenant: str, agent: str, session_id: str) -> None:
    with _LOCK:
        _SESSIONS.pop((tenant, agent, session_id), None)


def _sse(event: dict) -> str:
    return f"data: {json.dumps(event, default=str)}\n\n"


def chat_sse(fp, tenant: str, agent_name: str, session_id: str, user_text: str,
             approve_side_effecting: bool = False):
    """Yield SSE strings for one chat turn, persisting working memory across turns."""
    spec = fp.store.get_spec(tenant, agent_name)
    if spec is None:
        yield _sse({"type": "error", "text": f"agent '{agent_name}' not found"})
        yield _sse({"type": "done"})
        return

    history = get_history(tenant, agent_name, session_id)

    # LONG-TERM MEMORY (cross-session): recall relevant past memories for this agent and
    # inject them as a system note so the agent can use them even in a brand-new session.
    # Passing the platform store makes memory DURABLE (survives restart) and tenant-scoped in SQL.
    mem = get_memory(fp.store)
    recalled = mem.recall(tenant, agent_name, user_text, k=3)
    run_history = list(history)
    if recalled:
        note = "Relevant long-term memory from earlier sessions (use if helpful):\n" + \
               "\n".join(f"- {r['text']}" for r in recalled)
        run_history = [{"role": "system", "content": note}] + run_history
        yield _sse({"type": "memory", "recalled": [
            {"text": r["text"], "score": r["score"]} for r in recalled]})

    final_text = None
    try:
        for ev in fp.runtime.run_stream(spec, run_history, user_text, tenant,
                                        approve_side_effecting=approve_side_effecting):
            if ev.get("type") == "final":
                final_text = ev.get("text")
            yield _sse(ev)
    except Exception as e:  # noqa: BLE001
        yield _sse({"type": "error", "text": f"chat failed: {e}"})
        yield _sse({"type": "done"})
        return

    # commit this turn to working memory so the next turn recalls it
    turn = [{"role": "user", "content": user_text}]
    if final_text:
        turn.append({"role": "assistant", "content": final_text})
    _append(tenant, agent_name, session_id, turn)

    # persist a durable long-term memory of this turn for future sessions
    if final_text:
        try:
            mem.add(tenant, agent_name,
                    f"User asked: {user_text}\nAgent answered: {final_text}",
                    {"session": session_id})
        except Exception:  # noqa: BLE001 — memory write must never break the chat
            pass
