"""
Step 2 - tool design.

The model only knows a tool through 3 things: its name, its description, its parameters.
And when something goes wrong, it only knows what the error message tells it.
So tool design is prompt engineering for actions, and it decides whether an agent
recovers from its mistakes or spirals.

Part 1 shows validation and error messages on their own (no LLM call).
Part 2 runs the same tasks with a badly designed toolset and a well designed one.
"""
from agent_utils import Agent, Tool
from registry_tools import MODELS, SYSTEM_PROMPT, TOOLS, estimate_retraining_cost, get_model

# ---------------------------------------------------------------------------
# Part 1 - what the agent gets back when it makes a mistake
# ---------------------------------------------------------------------------

good_get_model = Tool(get_model)
good_cost = Tool(estimate_retraining_cost)

print("Mistakes the model typically makes, and the error it gets back:\n")
for tool, raw in [
    (good_get_model, '{"model_name": "YOLO v8"}'),                      # name close but not exact
    (good_get_model, '{"model": "yolov8"}'),                            # wrong argument name
    (good_cost, '{"model_name": "yolov8", "n_samples": "ten thousand"}'),  # wrong type
    (good_cost, '{"model_name": "yolov8", "n_samples": -5}'),           # violates gt=0
    (good_cost, '{"model_name": "yolov8", "n_samples": "10000"}'),       # string number: pydantic coerces it
]:
    print(f"  {tool.name}({raw})\n    -> {tool(raw)}\n")

# Each error says what's wrong AND how to fix it (list of valid models, expected field...).
# "Error: KeyError: 'YOLO v8'" would say neither.

# ---------------------------------------------------------------------------
# Part 2 - same tasks, bad tools vs good tools
# ---------------------------------------------------------------------------


def get_data(name: str):
    """Gets data."""
    return MODELS[name]  # case sensitive, crashes with a bare KeyError


def cost(m: str, n: str):
    """Computes cost."""
    return MODELS[m]["params_M"] * int(n) / 150_000 * 2.5 + int(n) * 0.04


BAD_TOOLS = [Tool(get_data), Tool(cost)]
GOOD_TOOLS = [t for t in TOOLS if t.name in ("get_model", "estimate_retraining_cost", "calculator")]

TASKS = [
    "What is the latency of YOLOv8, and is it under our 50 ms budget?",
    "How much would it cost to retrain EfficientNet on 8k new images? Detail GPU vs labeling.",
]

summary = []
for label, tools in [("bad tools", BAD_TOOLS), ("good tools", GOOD_TOOLS)]:
    agent = Agent(tools, SYSTEM_PROMPT, max_steps=6)
    for task in TASKS:
        print("=" * 80)
        print(f"[{label}] {task}")
        result = agent.run(task)
        print(f"\nAnswer: {result.answer}\n")
        summary.append((label, task[:45], result.steps, result.tool_errors, result.tokens))

print("=" * 80)
print(f"{'toolset':<12}{'task':<48}{'steps':>6}{'errors':>8}{'tokens':>8}")
for row in summary:
    print(f"{row[0]:<12}{row[1]:<48}{row[2]:>6}{row[3]:>8}{row[4]:>8}")

# What usually happens with the bad tools:
#  - "YOLOv8" -> KeyError -> the model guesses "yolov8", "YOLO", "yolo_v8"... burning steps
#  - cost() returns one number, so the model can't split GPU vs labeling and either says so
#    or (worse) invents a split. Good tools return structured, labeled results.
#
# Checklist for your own tools:
#  1. names that say what they do (verb_noun), docstrings that say WHEN to use them
#  2. typed parameters with descriptions and constraints (Literal for fixed choices)
#  3. errors that explain how to fix the call
#  4. small outputs: return what's needed, not the whole database row dump
#  5. few tools. Past ~15-20 tools, models start picking the wrong one.
