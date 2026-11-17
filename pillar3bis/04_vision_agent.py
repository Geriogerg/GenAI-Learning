"""
Step 4 - a vision agent: an LLM orchestrating CV specialists.

The pattern that works in practice: the LLM doesn't look at pixels itself, it DECIDES which
specialist to call (quality check, anomaly detection, YOLO, CLIP, VLM on a crop), reads their
results, and reasons over them. Each tool does what it's good at:
    classic CV  -> fast, precise, measurable (where, how much)
    VLM         -> open-ended understanding (what is it, does it matter)
    LLM agent   -> planning, combining, explaining

The agent itself is the pillar 3 Agent (text model). Images only go to the VLM through the
ask_vlm tool, because tool results can only be text.

Usage:
    python 04_vision_agent.py              # runs the demo tasks
    python 04_vision_agent.py --chat       # ask your own questions
"""
import argparse

from vision_tools import TOOLS
from vision_utils import Agent, load_labels

SYSTEM_PROMPT = """You are a visual quality-control assistant for a line producing brushed metal plates
(a good plate has 4 corner screws and a clean, uniform surface).

Tools: image quality check, anomaly detection against a golden sample, cropping, a vision-language
model (VLM), YOLO and CLIP. How to inspect a plate:
1. check_image_quality. If not usable, say so and don't conclude on defects.
2. find_anomalies to locate suspicious regions.
3. For each region (max 3), crop_region then ask_vlm on the crop to identify it.
4. Conclude: ACCEPT or REJECT, defect type, location (bbox), and how sure you are, with the evidence.
Don't use YOLO or CLIP for defects, they are not made for it. Be concise."""

parser = argparse.ArgumentParser()
parser.add_argument("--chat", action="store_true")
args = parser.parse_args()

agent = Agent(TOOLS, SYSTEM_PROMPT, max_steps=12)

if args.chat:
    print("Ask about the plates (quit to exit). E.g. 'inspect plate_012.jpg'")
    while (q := input("\nYou: ").strip()).lower() not in ("quit", "exit", "q"):
        print(f"\nAgent: {agent.run(q).answer}")
    raise SystemExit

labels = load_labels()


def pick(defect, subtle=False, issue=None):
    return next(f for f, x in sorted(labels.items())
                if x["defect"] == defect and x["subtle"] == subtle and x["imaging_issue"] == issue)


tasks = [
    f"Inspect {pick('scratch', subtle=True)}. Accept or reject?",
    f"Inspect {pick('dent', issue='blur')}. Accept or reject?",
    f"Inspect {pick('ok')}. Accept or reject?",
]

for task in tasks:
    file = task.split()[1].rstrip(".")
    print("\n" + "=" * 80)
    print(f"TASK: {task}   (ground truth: {labels[file]['defect']}"
          f"{', subtle' if labels[file]['subtle'] else ''}{', ' + labels[file]['imaging_issue'] if labels[file]['imaging_issue'] else ''})")
    print("=" * 80)
    result = agent.run(task)
    print(f"\n{result.answer}\n\n({result.steps} steps, {result.tokens} tokens)")

# What to look at in the traces:
#  - task 1 (subtle scratch, which the VLM alone often missed in step 2): find_anomalies points
#    to the region, the crop makes the scratch big enough for the VLM to name it.
#  - task 2 (blurred image): a good agent stops at the quality check and asks for a new image
#    instead of guessing. If it concludes anyway, tighten the system prompt.
#  - task 3 (good plate): the shortest path, quality ok + no anomaly = accept, no VLM call.
#
# Try --chat with: "which images in the batch have quality problems?" -> watch how many tool
# calls it makes. Looping over 64 images one call at a time is exactly the kind of job that
# should be plain code, not an agent. That's the starting point of step 6.
