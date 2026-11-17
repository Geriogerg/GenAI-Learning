"""
Step 8 - evaluating agents.

For RAG we checked the answer. For an agent the PATH matters too: an agent that gets the right
answer in 12 steps with 3 tool errors, or that queued a retraining nobody asked for, is not
a good agent. So each test task checks:

  - the final answer contains the key facts
  - the tools that had to be used were used (it didn't answer from memory)
  - the tools that must NOT be used weren't (no side effects on a read-only question)
  - steps, tokens, tool errors -> cost and reliability

Run the suite on 2 models (or 2 prompts, 2 toolsets...) and compare, like any model eval.

Usage:
    python 08_evaluating_agents.py                     # default model
    python 08_evaluating_agents.py --models openai/gpt-oss-120b qwen/qwen3.6-27b
    python 08_evaluating_agents.py --repeat 3          # agents are stochastic, run several times
"""
import argparse
import json
import time

import registry_tools as reg
from agent_utils import MODEL, OUTPUT_DIR, Agent

TASKS = [
    {
        "task": "Who owns unet_coating?",
        "expect_in_answer": ["julien"],
        "must_call": ["get_model"],
        "must_not_call": ["flag_for_retraining"],
        "max_steps": 3,
    },
    {
        "task": "Which production models have drift (not just a warning)?",
        "expect_in_answer": ["yolov8", "unet_coating"],
        "must_call": ["get_drift_report"],
        "must_not_call": ["flag_for_retraining"],
        "max_steps": 3,
    },
    {
        "task": "How much would it cost to retrain yolov8 and unet_coating on 10,000 images each, in total?",
        "expect_in_answer": ["810"],  # 403 + 407.75 = 810.75
        "must_call": ["estimate_retraining_cost"],
        "must_not_call": ["flag_for_retraining"],
        "max_steps": 5,
    },
    {
        "task": "Can unet_coating be retrained again this quarter without the team lead?",
        "expect_in_answer": ["no"],
        "must_call": ["get_model"],
        "must_not_call": ["flag_for_retraining"],
        "max_steps": 4,
    },
    {
        "task": "Queue yolov8 for retraining with high priority, it's drifting.",
        "expect_in_answer": ["queue"],
        "must_call": ["flag_for_retraining"],
        "must_not_call": [],
        "max_steps": 4,
    },
    {
        "task": "Is ViT Large fast enough for our 50 ms latency budget? By how many ms is it over or under?",
        "expect_in_answer": ["70"],
        "must_call": ["get_model"],
        "must_not_call": ["flag_for_retraining"],
        "max_steps": 4,
    },
]


def grade(case, result):
    called = {t["tool"] for t in result.trace}
    answer = result.answer.lower()
    checks = {
        "answer": all(s in answer for s in case["expect_in_answer"]),
        "used_tools": all(t in called for t in case["must_call"]),
        "no_side_effects": not any(t in called for t in case["must_not_call"]),
        "within_steps": result.steps <= case["max_steps"],
    }
    return checks


def evaluate(model, repeat):
    rows = []
    agent = Agent(reg.TOOLS, reg.SYSTEM_PROMPT, model=model, max_steps=8, verbose=False,
                  approve=lambda name, args: True)  # auto-approve in tests, we check WHAT it called
    for case in TASKS:
        for _ in range(repeat):
            reg.RETRAINING_QUEUE.clear()
            result = agent.run(case["task"])
            checks = grade(case, result)
            rows.append({"model": model, "task": case["task"], "passed": all(checks.values()), **checks,
                         "steps": result.steps, "tokens": result.tokens, "tool_errors": result.tool_errors,
                         "answer": result.answer, "tools": [t["tool"] for t in result.trace]})
            mark = "PASS" if all(checks.values()) else "FAIL " + ",".join(k for k, v in checks.items() if not v)
            print(f"  {mark:<28} {result.steps} steps {result.tokens:>6} tok  {case['task'][:60]}")
            time.sleep(1)
    return rows


parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=[MODEL])
parser.add_argument("--repeat", type=int, default=1)
args = parser.parse_args()

all_rows = []
for model in args.models:
    print(f"\n=== {model} ===")
    all_rows += evaluate(model, args.repeat)

print(f"\n{'model':<28}{'pass rate':>10}{'avg steps':>11}{'avg tokens':>12}{'tool errors':>13}")
for model in args.models:
    rows = [r for r in all_rows if r["model"] == model]
    n = len(rows)
    print(f"{model:<28}{sum(r['passed'] for r in rows) / n:>10.0%}{sum(r['steps'] for r in rows) / n:>11.1f}"
          f"{sum(r['tokens'] for r in rows) / n:>12.0f}{sum(r['tool_errors'] for r in rows):>13}")

OUTPUT_DIR.mkdir(exist_ok=True)
out = OUTPUT_DIR / "agent_eval.jsonl"
with out.open("w", encoding="utf-8") as f:
    for r in all_rows:
        f.write(json.dumps(r) + "\n")
print(f"\nDetails (answers, tool sequences) in {out}")

# Notes:
#  - string matching on answers is crude (the "no" check passes on "not"...). Fine for numbers
#    and names; for open answers use an LLM judge like in pillar 2 step 9.
#  - --repeat 3 matters: the same agent can pass, then fail on the same task. A pass rate over
#    several runs is more honest than one green run.
#  - in production you'd also log every trace (LangSmith, Langfuse, Arize Phoenix, or just
#    jsonl files like here) and turn real failures into new test cases.
