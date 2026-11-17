"""
Pillar 3bis mini project - an inspection pipeline with an agent for the hard cases.

Step 4 showed that an agent looping over 64 images is slow and expensive. Real systems are
mostly a WORKFLOW (plain code, cheap, predictable) with AI at the points that need judgment:

  stage 1  code    quality check + anomaly detection on every image      (free, ~10 ms/image)
                     unusable image        -> RETAKE
                     usable, no anomaly    -> ACCEPT
                     anomaly found         -> stage 2
  stage 2  VLM     zoomed crop of the anomaly -> defect type + confidence (1 call/image)
                     clear defect          -> REJECT
                     "none" / low conf     -> stage 3
  stage 3  agent   investigates with all tools -> ACCEPT / REJECT / HUMAN_REVIEW
  report           markdown report with annotated images + scores vs ground truth

Usage:
    python 06_project_inspection_agent.py                  # full batch
    python 06_project_inspection_agent.py --limit 20       # first 20 images (fewer API calls)
    python 06_project_inspection_agent.py --no-agent       # stages 1-2 only
"""
import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime

import cv_tools as cv
from vision_tools import TOOLS
from vision_utils import OUTPUT_DIR, PLATES_DIR, REFERENCE, TOKENS, Agent, load_labels, vlm

parser = argparse.ArgumentParser()
parser.add_argument("--limit", type=int, default=0)
parser.add_argument("--no-agent", action="store_true")
parser.add_argument("--min-confidence", type=float, default=0.6)
args = parser.parse_args()

RUN_DIR = OUTPUT_DIR / f"inspection_{datetime.now():%Y%m%d_%H%M%S}"
RUN_DIR.mkdir(parents=True)

CROP_PROMPT = """This is a zoomed-in crop of a brushed metal plate from a production line, around a region
that an anomaly detector flagged. Normal surface: fine horizontal brushing marks, uniform gray.
Normal screw: light disc with a cross slot. Identify what is in the crop.
Return JSON: {"defect_type": "scratch" | "dent" | "stain" | "missing_screw" | "none" | "unclear",
              "confidence": 0-1, "description": "one short sentence"}"""

AGENT_PROMPT = """You are a senior quality inspector reviewing a plate that the automatic pipeline could not
decide on. You get the pipeline's findings. Use the tools to investigate (look at other crops,
ask the VLM differently, compare with the reference), then decide.
Rules: REJECT only with visual evidence of a defect. If the evidence stays ambiguous, choose HUMAN_REVIEW,
a human will look at it. Keep the investigation short (a few tool calls).
End your answer with exactly one line:
DECISION: ACCEPT|REJECT|HUMAN_REVIEW | TYPE: scratch|dent|stain|missing_screw|none | REASON: <one sentence>"""

agent = Agent(TOOLS, AGENT_PROMPT, max_steps=8, verbose=False)
stats = Counter()


def inspect(name):
    path = PLATES_DIR / name
    case = {"image": name, "stage": 1}

    # ---- stage 1: deterministic checks
    quality = cv.check_image_quality(path)
    if not quality["usable"]:
        return {**case, "decision": "RETAKE", "type": "-", "reason": f"image {', '.join(quality['problems'])}"}

    anomalies = cv.find_anomalies(path)
    if not anomalies["anomaly_found"]:
        return {**case, "decision": "ACCEPT", "type": "none", "reason": "no difference with the golden sample"}

    # ---- stage 2: VLM on the crop of the main region
    region = anomalies["regions"][0]
    crop = cv.crop_region(path, *region["bbox"])
    answer = vlm(crop["crop_path"], CROP_PROMPT, json_mode=True, max_tokens=600)
    stats["vlm_calls"] += 1
    case.update(stage=2, bbox=region["bbox"], n_regions=len(anomalies["regions"]), vlm=answer)

    dtype = str(answer.get("defect_type", "unclear"))
    conf = float(answer.get("confidence", 0) or 0)
    if dtype in ("scratch", "dent", "stain", "missing_screw") and conf >= args.min_confidence:
        return {**case, "decision": "REJECT", "type": dtype, "reason": answer.get("description", "")}

    # ---- stage 3: the agent takes over
    if args.no_agent:
        return {**case, "decision": "HUMAN_REVIEW", "type": dtype, "reason": "VLM unsure, agent disabled"}

    stats["agent_runs"] += 1
    findings = (f"Image: {name}\nImage quality: {quality}\nAnomaly regions: {anomalies['regions']}\n"
                f"VLM on the crop of the first region: {answer}")
    result = agent.run(f"Pipeline findings:\n{findings}\n\nInvestigate and decide.")
    m = re.search(r"DECISION:\s*(ACCEPT|REJECT|HUMAN_REVIEW)\s*\|\s*TYPE:\s*(\w+)\s*\|\s*REASON:\s*(.+)", result.answer)
    if not m:
        return {**case, "stage": 3, "decision": "HUMAN_REVIEW", "type": "unclear",
                "reason": "agent answer could not be parsed", "agent_steps": result.steps}
    return {**case, "stage": 3, "decision": m.group(1), "type": m.group(2), "reason": m.group(3).strip(),
            "agent_steps": result.steps, "agent_tools": [t["tool"] for t in result.trace]}


# ---------------------------------------------------------------------------
# run
# ---------------------------------------------------------------------------

images = sorted(p.name for p in PLATES_DIR.glob("*.jpg") if p.name != REFERENCE.name)
if args.limit:
    images = images[:args.limit]

start, tokens_before = time.time(), TOKENS["total"]
results = []
for name in images:
    r = inspect(name)
    results.append(r)
    print(f"  {name}  stage {r['stage']}  {r['decision']:<13} {r['type']:<14} {r['reason'][:60]}")
    if r["decision"] == "REJECT" and r.get("bbox"):
        cv.draw_boxes(PLATES_DIR / name, [r["bbox"]], RUN_DIR / name, labels=[r["type"]])

elapsed = time.time() - start
(RUN_DIR / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

# ---------------------------------------------------------------------------
# scoring against the ground truth (possible because the data is synthetic)
# ---------------------------------------------------------------------------

labels = load_labels()


def verdict(r):
    lab = labels[r["image"]]
    if lab["imaging_issue"]:
        return "correct" if r["decision"] == "RETAKE" else "wrong"
    if r["decision"] == "HUMAN_REVIEW":
        return "review"
    if lab["defect"] == "ok":
        return "correct" if r["decision"] == "ACCEPT" else "false reject"
    if r["decision"] == "ACCEPT":
        return "ESCAPE"  # the worst case: defective part shipped
    return "correct" if r["type"] == lab["defect"] else "rejected, wrong type"


scores = Counter(verdict(r) for r in results)
decisions = Counter(r["decision"] for r in results)
by_stage = Counter(r["stage"] for r in results)

# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

lines = [
    f"# Inspection report - {datetime.now():%Y-%m-%d %H:%M}",
    "",
    f"**{len(results)} plates** inspected in {elapsed:.0f} s: "
    + ", ".join(f"{v} {k}" for k, v in decisions.most_common()),
    "",
    f"Decided at stage 1 (code only): {by_stage[1]}, stage 2 (VLM): {by_stage[2]}, stage 3 (agent): {by_stage[3]}. "
    f"VLM calls: {stats['vlm_calls']}, agent runs: {stats['agent_runs']}, tokens: {TOKENS['total'] - tokens_before}.",
    "",
    "## Score vs ground truth",
    "",
    "| outcome | count |",
    "|---|---|",
    *[f"| {k} | {v} |" for k, v in scores.most_common()],
    "",
    "## Rejected plates",
    "",
]
for r in results:
    if r["decision"] == "REJECT":
        lines += [f"### {r['image']} - {r['type']}", "", f"{r['reason']}", "", f"![{r['image']}]({r['image']})", ""]

lines += ["## Needs a human", ""]
lines += [f"- **{r['image']}** ({r['decision']}): {r['reason']}" for r in results
          if r["decision"] in ("HUMAN_REVIEW", "RETAKE")] or ["- nothing"]
lines += ["", "## All results", "", "| image | stage | decision | type | truth | reason |", "|---|---|---|---|---|---|"]
for r in results:
    lab = labels[r["image"]]
    truth = lab["defect"] + (f" ({lab['imaging_issue']})" if lab["imaging_issue"] else "") + (" subtle" if lab["subtle"] else "")
    lines.append(f"| {r['image']} | {r['stage']} | {r['decision']} | {r['type']} | {truth} | {r['reason'][:80]} |")

(RUN_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")

print("\n" + "=" * 80)
print(f"Decisions: {dict(decisions)}")
print(f"Vs ground truth: {dict(scores)}")
print(f"Stages: {dict(by_stage)} | VLM calls {stats['vlm_calls']} | agent runs {stats['agent_runs']} | "
      f"{TOKENS['total'] - tokens_before} tokens | {elapsed:.0f} s")
print(f"Report: {RUN_DIR / 'report.md'}")

vlm_only = OUTPUT_DIR / "vlm_eval.json"
if vlm_only.exists():
    preds = json.loads(vlm_only.read_text(encoding="utf-8")).get("zero-shot", {})
    common = [f for f in preds if f in {r["image"] for r in results}]
    if common:
        acc = sum(preds[f] == labels[f]["defect"] for f in common) / len(common)
        print(f"For comparison, VLM alone (step 2, zero-shot) on the same {len(common)} images: {acc:.0%} correct")

# Ideas to go further:
#  - log every HUMAN_REVIEW decision with the human's final answer: that's your next labeled dataset
#  - expose run_pipeline as an MCP tool (step 7) so you can ask "inspect today's batch" from VS Code
#  - replace find_anomalies with a learned anomaly model (anomalib / PatchCore) on real images
#  - track cost per plate: here most plates cost 0 API calls, which is the point of the design
