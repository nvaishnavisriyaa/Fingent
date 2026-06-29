"""
DEMO (acceptance #4): the Planner supervisor dispatches to >=2 REAL sub-agents in sequence,
each running the real LLM tool-use loop, then the Synthesis agent writes the final prose.

Run on a machine with internet + GROQ_API_KEY set (the project .env is loaded automatically):

    cd backend && python demo_supervisor.py

With a key you get mode='llm' (real native tool-calling). With no key it transparently runs
mode='demo' (deterministic) so the orchestration is still visible. Either way it prints the
dispatch order, each sub-agent's mode/status/steps, and the final synthesized answer.
"""
import json
from fingent import Fingent
from fingent.schemas import CreateAgentRequest


def main():
    fp = Fingent()
    tenant = "gtm"
    # Create the GTM tier-1 agents from templates (specs only — no per-agent code).
    for t in ["signal_trigger", "icp_matching", "enrichment_validation",
              "persona_decision_maker", "contact", "synthesis", "planner"]:
        tpl = fp.store.get_template(t)
        answers = {"name": t}
        for p in tpl.parameters:
            if p.default is not None:
                answers[p.name] = p.default
        fp.create_agent(CreateAgentRequest(template=t, answers=answers, tenant_id=tenant),
                        auto_provision=True)

    res = fp.run_supervised(tenant, tier=1, inputs={"company": "Stripe"})

    print("\n=== SUPERVISED RUN ===")
    print("dispatch order :", res["executed"])
    print("final agent    :", res["final_agent"])
    print("\n--- per sub-agent (each is a real agent loop) ---")
    for a in res["agents"]:
        print(f"  {a['agent']:<24} mode={a['mode']:<5} status={a['status']:<12} steps={a['steps']}")
    print("\n--- FINAL SYNTHESIZED ANSWER (prose) ---")
    fp_out = res["final_prose"]
    print(fp_out if isinstance(fp_out, str) else json.dumps(fp_out, indent=2)[:2000])
    print("\n(>=2 sub-agents in sequence + synthesis =", 
          len(res["executed"]) >= 2 and res["final_agent"] == "synthesis", ")")


if __name__ == "__main__":
    main()
