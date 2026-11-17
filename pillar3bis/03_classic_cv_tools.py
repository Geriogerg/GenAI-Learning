"""
Step 3 - the specialists: classic CV models as tools (no LLM, no API call).

Before giving tools to an agent, check what each one is good at on its own:
    check_image_quality  -> is the image usable at all?
    find_anomalies       -> difference with a golden sample: where is something unusual?
    detect_objects       -> YOLO on COCO: great on everyday objects, blind to your defects
    clip_classify        -> zero-shot labels from text, coarse but needs no training

At the end, the same evaluation as step 2, for the reference-difference method.
"""
from ultralytics.utils import ASSETS

import cv_tools as cv
from vision_utils import OUTPUT_DIR, PLATES_DIR, load_labels

labels = load_labels()
OUTPUT_DIR.mkdir(exist_ok=True)


def first(defect, subtle=False, issue=None):
    return next(f for f, x in sorted(labels.items())
                if x["defect"] == defect and x["subtle"] == subtle and x["imaging_issue"] == issue)


# --- image quality ------------------------------------------------------------
print("=" * 80 + "\nImage quality\n")
for f in [first("ok"), first("dent", issue="blur"), first("ok", issue="dark"), first("ok", issue="overexposed")]:
    print(f"  {cv.check_image_quality(PLATES_DIR / f)}")

# --- anomalies vs golden sample ----------------------------------------------
print("\n" + "=" * 80 + "\nAnomalies (difference with reference_good.jpg)\n")
for f in [first("ok"), first("scratch", subtle=True), first("stain"), first("missing_screw")]:
    res = cv.find_anomalies(PLATES_DIR / f)
    print(f"  {f} (truth: {labels[f]['defect']}, box {labels[f]['bbox']})")
    print(f"     -> found={res['anomaly_found']} regions={res['regions'][:2]}")
    if res["regions"]:
        cv.draw_boxes(PLATES_DIR / f, [r["bbox"] for r in res["regions"]], OUTPUT_DIR / f"anomaly_{f}")
print(f"  (images with the boxes drawn: {OUTPUT_DIR}/anomaly_*.jpg)")

# --- YOLO ---------------------------------------------------------------------
print("\n" + "=" * 80 + "\nYOLO (COCO classes)\n")
bus = cv.detect_objects(ASSETS / "bus.jpg")
print(f"  bus.jpg: {[(o['label'], o['confidence']) for o in bus['objects']]}")
print(f"  {first('scratch')}: {cv.detect_objects(PLATES_DIR / first('scratch'))['objects']}")
# -> nothing on the plate. A detector only knows its training classes. For defects you'd
#    fine-tune YOLO on labeled defect images (pillar 4 territory), or use a reference/anomaly
#    method like above, or open-vocabulary detectors (Grounding DINO, YOLO-World, OWLv2).

# --- CLIP zero-shot -----------------------------------------------------------
print("\n" + "=" * 80 + "\nCLIP zero-shot\n")
clip_labels = ["a clean metal plate", "a scratched metal plate", "a dented metal plate",
               "a metal plate with a brown stain", "a metal plate with a missing screw"]
for f in [first("ok"), first("scratch"), first("stain"), first("missing_screw")]:
    scores = cv.clip_classify(PLATES_DIR / f, clip_labels)["scores"]
    best = next(iter(scores))
    print(f"  {f} truth={labels[f]['defect']:<14} CLIP top: {best!r} ({scores[best]:.2f})")
# CLIP sees the global look of the image. Small local defects barely move its embedding,
# so expect it to be mostly wrong here. Where it shines: search and coarse sorting (step 5).

# --- evaluation of the reference method --------------------------------------
print("\n" + "=" * 80 + "\nReference-difference method on all 64 plates\n")


def overlaps(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


rows = []
for f, lab in sorted(labels.items()):
    q = cv.check_image_quality(PLATES_DIR / f)
    a = cv.find_anomalies(PLATES_DIR / f)
    located = lab["bbox"] is not None and any(overlaps(r["bbox"], lab["bbox"]) for r in a["regions"])
    correct = located if lab["defect"] != "ok" else not a["anomaly_found"]
    rows.append((f, lab, q["usable"], correct))

for name, subset in [("good images", [r for r in rows if r[2]]), ("bad images", [r for r in rows if not r[2]])]:
    if subset:
        print(f"  {name:<12} {sum(r[3] for r in subset)}/{len(subset)} correct "
              f"(defect located, or no false alarm on good plates)")
flagged = sum(not r[2] for r in rows)
real = sum(bool(r[1]["imaging_issue"]) for r in rows)
print(f"  quality check flagged {flagged} images, {real} actually have an imaging issue")

# Takeaways:
#  - on usable images, the golden-sample difference finds everything, subtle defects included,
#    in milliseconds, for free. But it only says WHERE, not WHAT.
#  - on bad images it falls apart -> always gate on image quality first.
#  - honest caveat: our plates are perfectly aligned and lit. Real lines need image
#    registration (cv2.findTransformECC), and lighting / part variation makes this much
#    harder. That's where learned anomaly detection (PatchCore, from the anomalib library)
#    comes in, a good pillar 4 topic.
#  - VLM (step 2) = knows WHAT but misses small things. Reference diff = finds WHERE, no idea
#    WHAT. Steps 4 and 6 combine them: CV finds the region, the VLM names it on a zoomed crop.
