"""In LLM mode the runtime must FORCE the agent to call a tool (gather real data) before it
can answer from the model's memory."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fingent import Fingent
from fingent.schemas import CreateAgentRequest
import fingent.runtime as rt


class _FakeProvider:
    enabled = True
    name = "fake-model"
    model = "fake"
    seen = []

    def chat(self, messages, tools=None, tool_choice="auto", **k):
        _FakeProvider.seen.append(tool_choice)
        already = any(m.get("role") == "tool" for m in messages)
        if tools and not already:
            fn = tools[0]["function"]["name"]          # call the first granted tool
            return {"role": "assistant", "content": None,
                    "tool_calls": [{"id": "c1", "function": {"name": fn, "arguments": "{}"}}]}
        return {"role": "assistant", "content": "Final answer grounded in the tool result."}


def test_runtime_forces_tool_use_in_llm_mode(monkeypatch):
    monkeypatch.setenv("FINGENT_LIVE_DATA", "0")     # tools return labeled demo data offline
    _FakeProvider.seen = []
    monkeypatch.setattr(rt, "LlmProvider", _FakeProvider)

    fp = Fingent()
    built = fp.create_agent(CreateAgentRequest(
        additional_requirements="Screen the person against OFAC sanctions and PEP lists.",
        answers={"name": "aml_x"}, tenant_id="acme"))
    assert built["ok"], built
    assert built["spec"]["tools"], "agent should have gathered real tools"

    rec = fp.run_task("acme", "aml_x", {"name": "Oleg Petrov"})
    assert rec["mode"] == "llm"
    # the FIRST model turn must have been forced to call a tool
    assert _FakeProvider.seen and _FakeProvider.seen[0] == "required"
    # and a real tool actually executed (recorded as a tool step)
    assert any(getattr(s, "kind", s.get("kind") if isinstance(s, dict) else None) == "tool"
               for s in rec["steps"])


def test_agent_without_tools_does_not_force(monkeypatch):
    # an agent with no granted tools must not be forced (no tools to call)
    _FakeProvider.seen = []
    monkeypatch.setattr(rt, "LlmProvider", _FakeProvider)
    fp = Fingent()
    from fingent.schemas import AgentSpec, SecurityPolicy
    spec = AgentSpec(name="bare", tier=1, role_prompt="r", tools=[],
                     security=SecurityPolicy(allowed_tools=[], memory_read=[], memory_write=[], tenant_id="acme"))
    fp.store.save_spec(spec)
    rec = fp.run_task("acme", "bare", {"q": "hi"})
    assert all(c == "auto" for c in _FakeProvider.seen)   # never forced when no tools
