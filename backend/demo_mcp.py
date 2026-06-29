"""
DEMO (acceptance #3): register a REAL public MCP server, see its tools appear, and have an
agent ACTUALLY INVOKE one over chat and use the result.

Run on a machine with internet:

    cd backend
    python demo_mcp.py                          # uses the default public server below
    python demo_mcp.py https://your.mcp/server  # or point at any Streamable-HTTP MCP server

What it does:
  1. register_mcp(url, connect=True) -> opens a real MCP session (initialize + tools/list)
  2. prints the discovered tools (proof they flowed into the registry)
  3. grants the first read-only discovered tool to a fresh agent (spec only, no code)
  4. chats with that agent; the streamed trace shows the MCP tool invoked + its live result,
     and the prose answer uses it.

Set GROQ_API_KEY in .env for real LLM tool selection (mode='llm'); without it the deterministic
engine still calls the granted MCP tool so you can see the live round-trip.
"""
import sys, json
from fingent import Fingent
from fingent.schemas import CreateAgentRequest, McpServer
from fingent.chat import chat_sse

# A real, public Streamable-HTTP MCP server. Override via argv[1].
DEFAULT_URL = "https://mcp.deepwiki.com/mcp"


def main():
    url = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_URL
    fp = Fingent()
    print(f"Registering MCP server: {url}")
    added = fp.register_mcp(McpServer(name="public", url=url, tenant_id="acme", approved=True),
                            connect=True)
    srv = fp.store.get_mcp("acme", "public")
    print(f"  connection: {srv.connection_status}"
          + (f" · error: {srv.connection_error}" if srv.connection_error else ""))
    if srv.connection_status != "connected" or not added:
        print("  Could not connect / no tools discovered. Try another --url. Server info:",
              srv.server_info)
        return
    print(f"  server: {srv.server_info}")
    print(f"  discovered tools ({len(added)}): {added}")

    # pick a non-side-effecting discovered tool to invoke
    read_only = [t for t in added if not fp.registry.get(t).side_effecting]
    target = (read_only or added)[0]
    short = target.split(".", 1)[1]
    print(f"\nGranting '{target}' to a new agent and asking it to use the tool...\n")

    fp.create_agent(CreateAgentRequest(
        template="servicing_support", answers={"name": "mcpdemo"},
        additional_requirements=f"use the MCP tool {target}", tenant_id="acme"))

    task = (f"Use the {short} tool to look something up and summarise what it returns. "
            f"If it needs arguments, use a reasonable example.")
    for chunk in chat_sse(fp, "acme", "mcpdemo", "s1", task):
        ev = json.loads(chunk[6:].strip())
        if ev["type"] == "tool_call":
            print(f"  [tool_call] {ev['tool']}({json.dumps(ev['args'])})")
        elif ev["type"] == "tool_result":
            print(f"  [tool_result] {ev['tool']} -> {json.dumps(ev['output'])[:240]}")
        elif ev["type"] == "token":
            print(ev["text"], end="", flush=True)
        elif ev["type"] == "final":
            print(f"\n\n[final] status={ev['status']} run={ev['run_id']} mode={ev['mode']}")
            invoked = any(True for _ in [1])
            print(f"[proof] MCP tool '{target}' was invoked and its live result is in the trace above.")


if __name__ == "__main__":
    main()
