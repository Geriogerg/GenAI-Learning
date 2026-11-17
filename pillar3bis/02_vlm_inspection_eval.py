"""
Step 2 - how good is the VLM as a defect classifier, really?

Same approach as your CV models: a labeled test set, a metric, a confusion matrix.
We compare two prompting strategies on the synthetic plates:
    zero-shot       just the image + instructions
    with reference  golden sample + image (2 images per call)

Costs ~1 API call per image per strategy. The default 24 images x 2 strategies = 48 calls,
which takes a few minutes on the free tier (rate limits). --all for the full 64.

Usage:
    python 02_vlm_inspection_eval.py
    python 02_vlm_inspection_eval.py --all
"""
import argparse
import json
import random
import time
from collections import Counter, defaultdict

from vision_utils import DEFECT_CLASSES, OUTPUT_DIR, PLATES_DIR, REFERENCE, load_labels, vlm

parser = argparse.ArgumentParser()
parser.add_argument("--all", action="store_true")
parser.add_argument("--n", type=int, default=24)
args = parser.parse_args()

PROMPT = """You inspect brushed metal plates on a production line. A good plate has 4 screws (one in each
corner) and a clean, uniform surface. Possible defects: scratch, dent, stain, missing_screw.
Look at the whole surface carefully, defects can be small.
Return JSON: {"defect_type": "ok" | "scratch" | "dent" | "stain" | "missing_screw", "confidence": 0-1}"""

REF_PREFIX = ("The FIRST image is a known-good reference plate. The SECOND image is the plate to inspect, "
              "same camera and position. Compare them.\n\n")

labels = load_labels()
files = sorted(labels)
if not args.all:
    random.seed(0)
    files = sorted(random.sample(files, args.n))  # same subset every run, so results are comparable


def predict(file, with_reference):
    images = [REFERENCE, PLATES_DIR / file] if with_reference else [PLATES_DIR / file]
    out = vlm(images, (REF_PREFIX if with_reference else "") + PROMPT, json_mode=True, max_tokens=800)
    pred = str(out.get("defect_type", "error")).lower().strip()
    return pred if pred in DEFECT_CLASSES else "invalid"


results = defaultdict(dict)
for strategy, with_ref in [("zero-shot", False), ("with reference", True)]:
    print(f"\n=== {strategy} ===")
    for f in files:
        pred = predict(f, with_ref)
        truth = labels[f]["defect"]
        results[strategy][f] = pred
        tags = " ".join(t for t in ["subtle" if labels[f]["subtle"] else "", labels[f]["imaging_issue"] or ""] if t)
        print(f"  {f}  truth={truth:<14} pred={pred:<14} {'ok ' if pred == truth else 'X  '} {tags}")
        time.sleep(0.5)


def report(strategy):
    preds = results[strategy]
    n = len(preds)
    acc = sum(preds[f] == labels[f]["defect"] for f in preds) / n
    # in inspection the costly mistake is a defective part marked "ok" (escape)
    defective = [f for f in preds if labels[f]["defect"] != "ok"]
    escapes = sum(preds[f] == "ok" for f in defective)
    good = [f for f in preds if labels[f]["defect"] == "ok"]
    false_rejects = sum(preds[f] != "ok" for f in good)
    print(f"\n--- {strategy}: accuracy {acc:.0%} | escapes (defect called ok) {escapes}/{len(defective)} "
          f"| false rejects {false_rejects}/{len(good)}")

    # confusion matrix, rows = truth
    cols = DEFECT_CLASSES + ["invalid"]
    print("truth \\ pred".ljust(15) + "".join(f"{c[:8]:>9}" for c in cols))
    for t in DEFECT_CLASSES:
        row = Counter(preds[f] for f in preds if labels[f]["defect"] == t)
        if row:
            print(f"{t:<15}" + "".join(f"{row.get(c, 0):>9}" for c in cols))

    hard = [f for f in preds if labels[f]["subtle"] or labels[f]["imaging_issue"]]
    if hard:
        print(f"accuracy on hard cases (subtle or bad image): "
              f"{sum(preds[f] == labels[f]['defect'] for f in hard)}/{len(hard)}")


for strategy in results:
    report(strategy)

OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "vlm_eval.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
print(f"\nPredictions saved to {OUTPUT_DIR / 'vlm_eval.json'} (step 6 compares against them)")

# Typical picture (varies with the model):
#  - big, colorful or semantic defects (stain, missing screw) are found reliably
#  - thin or low-contrast ones (subtle scratches, small dents) escape
#  - the reference image helps on some classes and can add false rejects on others
#  - on blurred / dark images it still answers confidently instead of saying "can't tell"
# -> a VLM alone is a great generalist, not a precise inspector. Step 3 looks at the
#    opposite trade-off with classic CV, steps 4 and 6 combine both.
