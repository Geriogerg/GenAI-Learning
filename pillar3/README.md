# Pillar 3 - Agents & tools

From "the model calls a function" (pillar 1) to agents that work on a goal over many steps:
the loop, tool design, agentic RAG, planning, memory, LangGraph, MCP, evaluation, and a data
analysis agent as the mini project. Still $0 (Groq free tier + local models).

## Setup

```
cd d:\pythoncode\pillar3
conda activate first-env
pip install -r requirements.txt
python data/make_sales_data.py
```

- `.env` with `GROQ_API_KEY` in `d:\pythoncode` is picked up automatically.
- Step 3 imports the pillar 2 retriever: keep `pillar2` next to `pillar3`, with `06_ingest.py` already run.
- Default model: `openai/gpt-oss-120b`. Step 1 `--parallel` and step 8 `--models` also use
  `qwen/qwen3.6-27b`. Override with `GROQ_MODEL` / `GROQ_PARALLEL_MODEL` in `.env`.
- Agents make many calls. If you hit rate limits, the code waits and retries; if it keeps
  happening, wait a minute between scripts.

## Layout

```
pillar3/
  agent_utils.py        Tool wrapper (schema from type hints, validation, errors) + Agent loop
  registry_tools.py     fake MLOps platform: registry, drift, costs, retraining queue
  01..08_*.py           one concept per script, in order
  09_project_*.py       mini project: data analysis agent
  data/
    make_sales_data.py  generates sales.csv (with planted anomalies)
  outputs/              agent reports, charts, eval results
  memory_db/            created by step 5 (long-term memory)
```

## Steps

| # | Script | What you learn |
|---|---|---|
| 1 | `01_agent_loop.py` | ReAct loop by hand: think / act / observe, step limit, parallel tool calls |
| 2 | `02_tool_design.py` | Why tool names, schemas and error messages decide if an agent recovers |
| 3 | `03_agentic_rag.py` | Retriever as a tool, multi-hop questions, classic vs agentic RAG |
| 4 | `04_planning.py` | ReAct vs plan-and-execute on a long task |
| 5 | `05_memory.py` | Conversation summarization + long-term memory in Chroma |
| 6 | `06_langgraph_agent.py` | Graph, state, checkpointer, human approval with `interrupt()` |
| 7 | `07_mcp_server.py` + `07_mcp_client_agent.py` | Package tools as an MCP server, use them from any client |
| 8 | `08_evaluating_agents.py` | Test suite on answers AND trajectories, model comparison |
| 9 | `09_project_data_agent.py` | Agent that writes and runs pandas code, makes charts, writes a report |

## Using your MCP server elsewhere

Same server, other clients (adapt the paths to your machine):

- **VS Code** (Copilot agent mode): `.vscode/mcp.json`, example at the bottom of `07_mcp_client_agent.py`
- **Claude Desktop**: in `claude_desktop_config.json`
  ```json
  {"mcpServers": {"ml-registry": {"command": "C:\\Users\\GREgo\\miniconda3\\envs\\first-env\\python.exe",
                                  "args": ["D:\\pythoncode\\pillar3\\07_mcp_server.py"]}}}
  ```
- **Claude Code**: `claude mcp add ml-registry -- python D:\pythoncode\pillar3\07_mcp_server.py`

## Bonus: try a coding agent

Now is a good time. Claude Code, Codex and similar tools are the same ReAct loop as step 1,
with tools like read_file / edit_file / run_command, a planning step, approvals for risky
commands, and a lot of tool-design polish. Ideas to try it on this folder:

1. "Add a `/compare <a> <b>` command to the pillar 2 paper assistant." Then read the diff.
2. "Add a replanner to 04_planning.py." Compare with how you would have done it.
3. "Make 09 keep variables between run_python calls (like a notebook)." Harder than it
   looks, a good test of whether you can review agent-written code.

Watch the loop while it works: which files it reads first, when it runs code to check
itself, when it asks for approval. You'll recognize every piece from this pillar.

## Common issues

- `tool_use_failed` (400): the model wrote a malformed tool call. `chat()` retries, it's random.
- Step 3 `ModuleNotFoundError: rag_utils`: the pillar2 folder isn't next to pillar3.
- Step 7 hangs or crashes at start: run `python 07_mcp_server.py` alone to see the error
  (it should wait silently; Ctrl+C to stop). Requires `mcp>=2`.

---

## Spoiler: what's planted in sales.csv

Try step 9 before reading this.

1. **North, March:** camera_module stockout from March 4 to 24 (units drop ~95%), which
   explains most of North's March revenue dip. Other North products are normal.
2. **lighting_bar returns:** from May 6, the return rate jumps from ~3% to ~18% in every
   region (defective batch).
3. **West** grows ~4% per month in units. Revenue is noisier because of the expensive
   sensor_kit, so it's harder to see on revenue than on units.
4. **Data quality:** 40 missing unit_price (so 40 missing revenue), 10 duplicated rows,
   1 row with negative units.
5. Weekend retail sales are half of weekdays (normal seasonality, not an anomaly).
