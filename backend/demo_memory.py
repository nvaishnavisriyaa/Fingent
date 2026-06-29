"""
DEMO (acceptance #6): long-term, cross-session memory.

A fact learned while chatting in session A is RECALLED in a brand-new session B of the same
agent. Run:

    cd backend && python demo_memory.py

Backend: set PINECONE_API_KEY (+ optional PINECONE_INDEX/REGION) in .env for the durable
Pinecone vector store; otherwise Fingent uses its built-in local vector store. Either way the
recall behaviour is identical — this script prints which backend is active.
"""
import json
from fingent import Fingent
from fingent.schemas import CreateAgentRequest
from fingent.chat import chat_sse
from fingent.memory import get_memory


def run_turn(fp, session, text):
    print(f"\n[{session}] >>> {text}")
    for chunk in chat_sse(fp, "acme", "aml", session, text):
        ev = json.loads(chunk[6:].strip())
        if ev["type"] == "memory":
            print("   \033[35m[recalled long-term memory]\033[0m")
            for r in ev["recalled"]:
                print(f"     · ({r['score']}) {r['text'][:90]}")
        elif ev["type"] == "token":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "final":
            print(f"\n   [final · run {ev['run_id']} · mode {ev['mode']}]")


def main():
    fp = Fingent()
    print(f"Long-term memory backend: {get_memory().backend}")
    tpl = fp.store.get_template("aml_sanctions_screening")
    answers = {"name": "aml"}
    for p in tpl.parameters:
        if p.default is not None:
            answers[p.name] = p.default
    fp.create_agent(CreateAgentRequest(template="aml_sanctions_screening",
                                       answers=answers, tenant_id="acme"))

    # Session A — establish facts
    run_turn(fp, "sessionA", "Screen Oleg Petrov against OFAC and PEP lists.")
    run_turn(fp, "sessionA", "Also note Acme Corp had adverse media in 2024.")

    # Session B — a NEW session; the agent should recall the earlier facts
    print("\n--- new session (no shared working memory) ---")
    run_turn(fp, "sessionB", "What do we already know about Oleg Petrov and Acme Corp?")

    print("\n\nDirect recall check:")
    for q in ["Oleg Petrov sanctions", "Acme Corp adverse media"]:
        hits = get_memory().recall("acme", "aml", q, k=1)
        print(f"  query '{q}' -> {hits[0]['text'][:80] if hits else '(none)'}")


if __name__ == "__main__":
    main()
