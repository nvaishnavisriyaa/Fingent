"""
DEMO (acceptance #2, #5, and #6-session): chat with an agent in natural language.

The agent PLANS, calls >=1 REAL tool (with the full governed trace), and streams back a
conversational PROSE answer (structured payload secondary). A second turn shows session
working-memory recall. Run on a machine with internet + GROQ_API_KEY set:

    cd backend && python demo_chat.py

With a key -> mode='llm' (real native tool-calling, token-streamed). Without -> mode='demo'
(deterministic) so the stream/trace is still visible. This drives the same `chat_sse` generator
the /api/agents/{name}/chat SSE endpoint wraps; the web UI at /chat renders these same events.
"""
from fingent import Fingent
from fingent.schemas import CreateAgentRequest
from fingent.chat import chat_sse, get_history
import json


def turn(fp, session, text):
    print(f"\n\033[1m>>> you:\033[0m {text}\n")
    answer = ""
    for chunk in chat_sse(fp, "acme", "aml", session, text):
        ev = json.loads(chunk[6:].strip())
        t = ev["type"]
        if t == "start":
            print(f"[start] {ev['agent']} · mode={ev['mode']} · tools={ev['allowed_tools']}")
        elif t == "tool_call":
            print(f"  \033[36m[tool_call]\033[0m {ev['tool']}({json.dumps(ev['args'])})")
        elif t == "tool_result":
            print(f"  \033[32m[tool_result]\033[0m {ev['tool']} -> {json.dumps(ev['output'])[:180]}")
        elif t == "token":
            print(ev["text"], end="", flush=True); answer += ev["text"]
        elif t == "status":
            print(f"\n  \033[33m[status]\033[0m {ev['text']}")
        elif t == "final":
            print(f"\n\n[final] status={ev['status']} risk={ev['risk_level']} run={ev['run_id']}")
            print(f"[structured payload — secondary] {json.dumps(ev['structured'])[:200]}...")
    return answer


def main():
    fp = Fingent()
    tpl = fp.store.get_template("aml_sanctions_screening")
    answers = {"name": "aml"}
    for p in tpl.parameters:
        if p.default is not None:
            answers[p.name] = p.default
    fp.create_agent(CreateAgentRequest(template="aml_sanctions_screening",
                                       answers=answers, tenant_id="acme"))

    turn(fp, "demo", "Screen Oleg Petrov against sanctions and PEP lists and tell me if we can onboard him.")
    turn(fp, "demo", "What was the name of the person I just asked you to screen?")  # session recall

    hist = get_history("acme", "aml", "demo")
    print("\n\n=== session working memory (turns retained) ===")
    for m in hist:
        print(f"  {m['role']:<9} {str(m['content'])[:90]}")


if __name__ == "__main__":
    main()
