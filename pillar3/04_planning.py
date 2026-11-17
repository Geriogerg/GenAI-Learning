"""
Step 4 - planning: ReAct vs plan-and-execute.

ReAct decides one step at a time. Great for short tasks, but on long ones it tends to
forget part of the request, or stop early once it has "enough".

Plan-and-execute splits the work in 3 roles:
  planner     -> writes the list of steps upfront (one LLM call, JSON)
  executor    -> a small ReAct agent runs each step, with the results of previous steps
  synthesizer -> writes the final answer from all step results

Same task, both approaches, compare the answers and the cost.
"""
import json

from agent_utils import Agent, chat
from registry_tools import SYSTEM_PROMPT, TOOLS

TASK = (
    "Prepare a retraining briefing for the team lead. For every production model with drift or a "
    "drift warning: its PSI, its owner, the estimated cost of retraining on 12,000 new images, and "
    "whether it can still be retrained this quarter without an exception. End with the total cost "
    "and a recommended order of priority. Don't queue anything, it's only a briefing."
)

# read-only tools for this one, it's a briefing
TOOLS = [t for t in TOOLS if not t.requires_approval]
TOOL_LIST = "\n".join(f"- {t.name}: {t.description.splitlines()[0]}" for t in TOOLS)

# ---------------------------------------------------------------------------
# A) ReAct
# ---------------------------------------------------------------------------

print("=" * 80)
print("A) ReAct")
print("=" * 80)
react = Agent(TOOLS, SYSTEM_PROMPT, max_steps=12)
react_result = react.run(TASK)
print(f"\n{react_result.answer}")

# ---------------------------------------------------------------------------
# B) Plan-and-execute
# ---------------------------------------------------------------------------

PLANNER_PROMPT = f"""You plan tasks for an MLOps agent. Available tools:
{TOOL_LIST}

Write a short plan: 3 to 7 steps, each one a concrete instruction that can be done with the tools.
Steps run in order and each one sees the results of the previous ones.
Return JSON: {{"steps": ["...", "..."]}}"""

SYNTH_PROMPT = """You write the final answer of a task from the results of each step.
Use only the information in the step results. Be structured and include the numbers."""

print("\n" + "=" * 80)
print("B) Plan-and-execute")
print("=" * 80)

msg = chat([{"role": "system", "content": PLANNER_PROMPT}, {"role": "user", "content": TASK}], json_mode=True)
plan = json.loads(msg.content)["steps"]
print("Plan:")
for i, s in enumerate(plan, 1):
    print(f"  {i}. {s}")

executor = Agent(TOOLS, SYSTEM_PROMPT, max_steps=5)
results, steps_total, tokens_total = [], 0, 0
for i, step in enumerate(plan, 1):
    print(f"\n--- executing step {i}: {step}")
    done = "\n".join(f"Step {j}: {s}\nResult: {r}" for j, (s, r) in enumerate(results, 1)) or "nothing yet"
    r = executor.run(f"Overall task: {TASK}\n\nAlready done:\n{done}\n\nYour current step: {step}\n"
                     "Do only this step and report its result concisely.")
    results.append((step, r.answer))
    steps_total += r.steps
    tokens_total += r.tokens

step_results = "\n\n".join(f"Step {i}: {s}\nResult: {r}" for i, (s, r) in enumerate(results, 1))
final = chat([{"role": "system", "content": SYNTH_PROMPT},
              {"role": "user", "content": f"Task: {TASK}\n\n{step_results}"}])
print(f"\n{final.content}")

print("\n" + "=" * 80)
print(f"ReAct:            {react_result.steps} LLM steps, {react_result.tokens} tokens")
print(f"Plan-and-execute: {steps_total} executor steps + planner + synthesizer, "
      f"{tokens_total} executor tokens (planner/synth not counted)")

# What to compare:
#  - completeness: does each version cover efficientnet (warning) and not only the 2 drifting
#    models? Does it catch that unet_coating has no retraining left this quarter?
#  - cost: plan-and-execute is usually more expensive here, the plan helps on long tasks but
#    every step re-sends context.
#  - the plan is a real artifact: you can show it to a human, edit it, approve it before
#    execution. That's often the main reason to use it in companies.
# Exercise: add a "replanner" that looks at the results after each step and may change the
# remaining steps (e.g. if a step failed).
