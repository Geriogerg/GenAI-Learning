"""
Step 1 - the agent loop, from scratch.

Same loop as pillar1_functions.py, but written as an agent:
  - the model works on a GOAL and decides each next step from what it has seen (ReAct:
    think -> act -> observe -> think ...)
  - a step limit, so a confused agent can't loop forever on your free-tier quota
  - a trace of every action, because you can't debug an agent you can't see

Everything here is explicit on purpose. agent_utils.Agent is the same loop, packaged.

Usage:
    python 01_agent_loop.py
    python 01_agent_loop.py --parallel     # model that can call several tools in one step
"""
import argparse
import json

from agent_utils import MODEL, PARALLEL_MODEL, TOKENS, assistant_message, chat
from registry_tools import SYSTEM_PROMPT, TOOLS

parser = argparse.ArgumentParser()
parser.add_argument("--parallel", action="store_true", help=f"use {PARALLEL_MODEL} with parallel tool calls")
parser.add_argument("--max-steps", type=int, default=8)
args = parser.parse_args()

model = PARALLEL_MODEL if args.parallel else MODEL
tools_by_name = {t.name: t for t in TOOLS}
schemas = [t.schema() for t in TOOLS]

# the model never sees our Python code, only this. Worth printing once:
print("What the model knows about one tool:")
print(json.dumps(schemas[3], indent=2))
print()


def run_agent(goal):
    print("=" * 80)
    print(f"GOAL: {goal}   (model: {model})")
    print("=" * 80)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": goal}]
    tokens_before = TOKENS["total"]

    for step in range(1, args.max_steps + 1):
        msg = chat(messages, tools=schemas, model=model, parallel_tool_calls=True if args.parallel else None)

        # THINK - reasoning models expose their reasoning, others sometimes write a sentence
        thinking = getattr(msg, "reasoning", None) or msg.content
        if thinking:
            print(f"\n[step {step}] think: {' '.join(thinking.split())[:250]}")

        # no tool call -> the model considers it's done
        if not msg.tool_calls:
            print(f"\nFINAL ANSWER ({step} steps, {TOKENS['total'] - tokens_before} tokens):\n{msg.content}\n")
            return

        messages.append(assistant_message(msg))
        if len(msg.tool_calls) > 1:
            print(f"[step {step}] {len(msg.tool_calls)} tool calls in parallel")

        for call in msg.tool_calls:
            # ACT
            tool = tools_by_name[call.function.name]
            print(f"[step {step}] act:     {call.function.name}({call.function.arguments})")
            if tool.requires_approval:
                result = "Error: this action requires human approval, which isn't available in this script."
            else:
                result = tool(call.function.arguments)
            # OBSERVE - the result goes back into the conversation
            print(f"[step {step}] observe: {result[:200]}")
            messages.append({"role": "tool", "tool_call_id": call.id, "content": result})

    print(f"\nStopped: step limit ({args.max_steps}) reached without an answer.\n")


# 1 step: a single lookup
run_agent("Who owns the YOLOv8 model?")

# several dependent steps: the second call depends on the first result
run_agent("Which production models have drift right now, and how much would it cost to retrain "
          "each of them on 10,000 new images?")

# the agent must notice a constraint by itself (unet_coating's retraining budget)
run_agent("Can we retrain all the drifting models this quarter without asking anyone?")

# Things to look at:
#  - question 2: count the steps. With --parallel, the cost estimates for both models
#    should come in a single step. Fewer round trips = faster and cheaper.
#  - question 3: does it check retrainings_this_quarter? If it answers without checking,
#    that's a prompt / tool description problem, which is exactly what step 2 is about.
#  - try --max-steps 2 to see what happens when an agent runs out of budget.
