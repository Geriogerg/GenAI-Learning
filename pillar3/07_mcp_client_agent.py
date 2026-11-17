"""
Step 7b - an agent that gets its tools from an MCP server.

The agent knows nothing about the registry: it starts the server, asks it which tools exist
(list_tools), converts their schemas to the Groq format, and forwards every tool call to
the server (call_tool). Swap the server for another one (GitHub, Postgres, filesystem... there
are hundreds) and the same agent can use those tools instead.

Usage: python 07_mcp_client_agent.py            (starts 07_mcp_server.py by itself)
"""
import asyncio
import json
import sys

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from agent_utils import ROOT, ask_user_approval, assistant_message, chat

SERVER = StdioServerParameters(command=sys.executable, args=[str(ROOT / "07_mcp_server.py")])

SYSTEM_PROMPT = """You are an MLOps assistant. Use the tools to answer, never invent numbers.
Only queue retraining jobs when the user explicitly asks for it."""


def to_groq_schema(mcp_tool):
    return {"type": "function", "function": {
        "name": mcp_tool.name,
        "description": mcp_tool.description or "",
        "parameters": mcp_tool.input_schema,
    }}


def result_to_text(result):
    text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
    return f"Error: {text}" if result.is_error else text


async def run_agent(session, tools, destructive, task, max_steps=8):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": task}]
    for step in range(1, max_steps + 1):
        msg = chat(messages, tools=tools)
        if not msg.tool_calls:
            return msg.content
        messages.append(assistant_message(msg))
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments or "{}")
            print(f"  [{step}] {call.function.name}({args})")
            # the server told us which tools are destructive, the client decides what to do with it
            if call.function.name in destructive and not ask_user_approval(call.function.name, args):
                text = "The user refused this action."
            else:
                text = result_to_text(await session.call_tool(call.function.name, args))
            print(f"       -> {text[:200]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": text})
    return "Step limit reached."


async def main():
    async with stdio_client(SERVER) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            listed = await session.list_tools()
            tools = [to_groq_schema(t) for t in listed.tools]
            destructive = {t.name for t in listed.tools if t.annotations and t.annotations.destructive_hint}

            print(f"Connected to MCP server, {len(tools)} tools:")
            for t in listed.tools:
                flag = " (destructive)" if t.name in destructive else ""
                print(f"  - {t.name}{flag}: {(t.description or '').splitlines()[0]}")

            resources = await session.list_resources()
            print(f"Resources: {[str(r.uri) for r in resources.resources]}\n")

            print("Ask about the models (quit to exit).")
            while True:
                task = input("\nYou: ").strip()
                if task.lower() in ("quit", "exit", "q"):
                    break
                if task:
                    print(f"\nAgent: {await run_agent(session, tools, destructive, task)}")


if __name__ == "__main__":
    asyncio.run(main())

# Using the same server from VS Code (agent mode of Copilot Chat): create .vscode/mcp.json
#   {
#     "servers": {
#       "ml-registry": {
#         "type": "stdio",
#         "command": "C:\\Users\\GREgo\\miniconda3\\envs\\first-env\\python.exe",
#         "args": ["D:\\pythoncode\\pillar3\\07_mcp_server.py"]
#       }
#     }
#   }
# Then in the chat, ask "which models are drifting?" and it will call your tools.
# The README has the config for Claude Desktop / Claude Code too.
