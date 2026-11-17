"""
Step 7a - an MCP server exposing the model registry tools.

MCP (Model Context Protocol) is a standard way to package tools so any MCP client can use
them: your own agent (07_mcp_client_agent.py), VS Code / Copilot agent mode, Claude Desktop,
Claude Code, Cursor... You write the tools once, every client can plug them in.

Before MCP, each framework had its own tool format (we already have 3 in this folder:
our Tool class, LangChain's StructuredTool, and Groq's JSON schema).

This server talks over stdio: the client starts it as a subprocess and they exchange JSON
messages on stdin/stdout. So never print() in here, it would corrupt the protocol.

You don't run this file yourself, the client starts it. To check it starts, run it: it will
wait silently for input (Ctrl+C to stop).
Note: written for the mcp 2.x SDK (pip install "mcp>=2"), where FastMCP became MCPServer.
"""
import functools

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

import registry_tools as reg

mcp = MCPServer(
    name="ml-registry",
    instructions="Tools to query the CV team's model registry, drift reports and retraining costs, "
                 "and to queue retraining jobs.",
)

# The SDK builds the JSON schema from type hints and docstrings, exactly like our Tool class.
# Annotations are hints for the CLIENT (e.g. ask the user before destructive actions).
READ_ONLY = ToolAnnotations(read_only_hint=True)
DESTRUCTIVE = ToolAnnotations(destructive_hint=True)


def expose(fn, annotations):
    """
    Register fn as a tool. In mcp 2.x an unexpected exception reaches the client as a bare
    "Error executing tool X" (so server internals don't leak). Our errors are written for the
    model ("unknown model, available: ..."), so we re-raise them as ToolError, whose message
    is passed through. Same lesson as step 2.
    """
    @functools.wraps(fn)  # keeps the signature + docstring the SDK reads to build the schema
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            raise ToolError(str(e)) from e

    mcp.tool(annotations=annotations)(wrapper)


for fn in (reg.list_models, reg.get_model, reg.get_drift_report, reg.estimate_retraining_cost, reg.calculator):
    expose(fn, READ_ONLY)
expose(reg.flag_for_retraining, DESTRUCTIVE)


# MCP servers can also expose resources (read-only documents the client can load as context).
@mcp.resource("registry://drift-policy")
def drift_policy() -> str:
    """The team's drift thresholds and retraining rules."""
    return (f"PSI > {reg.PSI_ALERT}: drift. {reg.PSI_WARNING}-{reg.PSI_ALERT}: warning. "
            f"Max {reg.MAX_RETRAININGS_PER_QUARTER} retrainings per model per quarter without lead approval.")


if __name__ == "__main__":
    mcp.run(transport="stdio")
