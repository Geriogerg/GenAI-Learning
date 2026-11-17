"""
Step 1 - open-vocabulary detection: detect things by NAME, without training.

A classic YOLO only knows its training classes (80 COCO classes, you saw it find nothing on the
plates in 3bis). Open-vocabulary detectors (Grounding DINO, OWLv2, YOLO-World, YOLOE...) align
image regions with text embeddings (CLIP-style), so the classes are just strings you pass at
run time. YOLOE-26 runs in real time, even on CPU at small sizes.

This script:
  1. text prompts on a normal photo (sanity check)
  2. text prompts on the plates, scored against the ground truth, with 2 prompt vocabularies
  3. visual prompts: "find things that look like THIS box" from one example image

No API calls. First run downloads yoloe-26s-seg.pt (~30 MB) and a text encoder (~240 MB).
"""
import numpy as np
from ultralytics import YOLOE
from ultralytics.utils import ASSETS

from p6_utils import OUTPUT_DIR, PLATES_DIR, boxes_overlap, load_plate_labels

OUTPUT_DIR.mkdir(exist_ok=True)
model = YOLOE("yoloe-26s-seg.pt")

# --- 1. sanity check on a normal photo ------------------------------------------
model.set_classes(["person", "bus", "tie", "backpack"])
r = model.predict(ASSETS / "bus.jpg", verbose=False)[0]
print("bus.jpg:", [(r.names[int(c)], round(float(p), 2)) for c, p in zip(r.boxes.cls, r.boxes.conf)])
r.save(filename=str(OUTPUT_DIR / "yoloe_bus.jpg"))
# 'tie' and 'backpack' aren't special: any word works, as long as the text encoder knows it.

# --- 2. the plates, scored -----------------------------------------------------
labels = load_plate_labels()
clean = {f: x for f, x in labels.items() if not x["imaging_issue"]}

# Same defects, two ways of naming them. The words you choose ARE the model's "training".
VOCABULARIES = {
    "defect names": {"scratch": "scratch", "dent": "dent", "stain": "stain", "missing_screw": "empty screw hole"},
    "visual words": {"scratch": "thin bright line", "dent": "small dark round spot",
                     "stain": "brown blotch", "missing_screw": "black hole"},
}
CONF = 0.10  # open-vocabulary scores on unusual images are low, the threshold matters a lot


def evaluate(vocab):
    prompt_to_class = {v: k for k, v in vocab.items()}
    model.set_classes(list(vocab.values()))
    located = right_class = false_alarm = 0
    for f, lab in sorted(clean.items()):
        r = model.predict(PLATES_DIR / f, conf=CONF, verbose=False)[0]
        dets = [(prompt_to_class[r.names[int(c)]], b.tolist()) for c, b in zip(r.boxes.cls, r.boxes.xyxy)]
        if lab["defect"] == "ok":
            false_alarm += bool(dets)
            continue
        hits = [cls for cls, box in dets if boxes_overlap(box, lab["bbox"])]
        located += bool(hits)
        right_class += lab["defect"] in hits
    n_def = sum(x["defect"] != "ok" for x in clean.values())
    n_ok = len(clean) - n_def
    print(f"  {'/'.join(vocab.values())}")
    print(f"     defect located {located}/{n_def} | with the right name {right_class}/{n_def} | "
          f"false alarm on good plates {false_alarm}/{n_ok}")


print(f"\nPlates, {len(clean)} clean images, conf >= {CONF}")
for name, vocab in VOCABULARIES.items():
    print(f"\n[{name}]")
    evaluate(vocab)

# --- 3. visual prompt: one example box -> find similar things elsewhere -----------
example = next(f for f, x in sorted(clean.items()) if x["defect"] == "stain" and not x["subtle"])
targets = [f for f, x in sorted(clean.items()) if x["defect"] == "stain" and f != example][:4] + \
          [f for f, x in sorted(clean.items()) if x["defect"] == "ok"][:4]
visual_prompts = {"bboxes": np.array([labels[example]["bbox"]], dtype=float), "cls": np.array([0])}
print(f"\nVisual prompt: the stain box of {example} as the only example")
for f in targets:
    r = model.predict(PLATES_DIR / f, refer_image=str(PLATES_DIR / example), visual_prompts=visual_prompts,
                      conf=0.01, verbose=False)[0]
    best = float(r.boxes.conf.max()) if len(r.boxes) else 0.0
    hit = any(boxes_overlap(b.tolist(), labels[f]["bbox"]) for b in r.boxes.xyxy) if labels[f]["bbox"] else False
    print(f"  {f} (truth {labels[f]['defect']:<6}) best score {best:.3f}" + ("  <- on the stain" if hit else ""))
# If the stains score clearly above the good plates, one example is enough to build a detector
# for a NEW defect type in minutes. If not, it tells you this defect needs real training data.

# What to take away:
#  - zero training, decent localization on several defect types: impressive, and a great way to
#    bootstrap a labeling job (pre-annotate, a human corrects, then train a normal detector)
#  - very sensitive to the words and the threshold: "prompt engineering" for detectors
#  - confidences are low and poorly calibrated on images far from web photos (brushed metal...).
#    For production on a fixed defect list, a YOLO fine-tuned on a few hundred labeled images
#    usually wins. Open vocabulary shines when the list changes or labels don't exist yet.
