"""
Step 2 - Segment Anything (SAM): pixel-precise masks from a click or a box.

SAM doesn't know classes. You give it a PROMPT (a point, a box) and it returns the mask of
"the thing there", on any kind of image, without training. In industry it's used to:
  - measure: a box says "there's a stain", a mask says "the stain covers 37 mm2"
  - label 10x faster: click once, get a polygon (that's how most segmentation datasets are made now)
  - chain with a detector ("grounded segmentation"): detector finds boxes -> SAM draws the masks

Models (CPU-friendly): MobileSAM (40 MB) and SAM 2 tiny (78 MB), via ultralytics.
SAM 3 adds text prompts ("scratch") but is 3.4 GB, gated and GPU-only in practice (see README).

Pipeline on the plates: golden-sample anomaly regions (pillar 3bis) -> boxes -> SAM masks ->
area in mm2 -> severity. No API calls.
"""
import sys
import time

import cv2
import numpy as np
from ultralytics import SAM
from ultralytics.utils import ASSETS

from p6_utils import OUTPUT_DIR, PILLAR3BIS, PLATES_DIR, load_plate_labels

sys.path.insert(0, str(PILLAR3BIS))
import cv_tools  # noqa: E402  (pillar 3bis: find_anomalies)

OUTPUT_DIR.mkdir(exist_ok=True)
MM_PER_PIXEL = 200 / 320  # our plates are 200 mm wide and fill 320 px: 0.625 mm per pixel

# --- 1. prompts on a normal photo ------------------------------------------------
for weights in ["mobile_sam.pt", "sam2_t.pt"]:
    model = SAM(weights)
    start = time.perf_counter()
    r = model.predict(ASSETS / "bus.jpg", points=[[400, 400]], labels=[1], verbose=False)[0]  # one click on the bus
    print(f"{weights:<14} one click on the bus -> mask of {int(r.masks.data[0].sum()):,} px "
          f"in {time.perf_counter() - start:.1f}s")
r.save(filename=str(OUTPUT_DIR / "sam_bus_click.jpg"))

# --- 2. grounded segmentation on the plates -------------------------------------
sam = SAM("sam2_t.pt")
labels = load_plate_labels()
cases = [f for f, x in sorted(labels.items()) if x["defect"] != "ok" and not x["imaging_issue"]][:10]


def severity(area_mm2):
    # thresholds would come from the quality team; these split our plates into 3 sensible groups
    return "high" if area_mm2 > 500 else "medium" if area_mm2 > 100 else "low"


print(f"\n{'plate':<15}{'truth':<15}{'box px':>8}{'mask px':>9}{'area mm2':>10}  severity")
for f in cases:
    regions = cv_tools.find_anomalies(PLATES_DIR / f)["regions"]  # WHERE (classic CV, pillar 3bis)
    if not regions:
        print(f"{f:<15}{labels[f]['defect']:<15}  no anomaly found")
        continue
    box = regions[0]["bbox"]
    r = sam.predict(PLATES_DIR / f, bboxes=[box], verbose=False)[0]  # the exact SHAPE (SAM)
    mask = r.masks.data[0].cpu().numpy().astype(bool)
    box_px = (box[2] - box[0]) * (box[3] - box[1])
    area = mask.sum() * MM_PER_PIXEL ** 2
    print(f"{f:<15}{labels[f]['defect'] + (' (subtle)' if labels[f]['subtle'] else ''):<15}"
          f"{box_px:>8}{int(mask.sum()):>9}{area:>10.1f}  {severity(area)}")

    img = cv2.imread(str(PLATES_DIR / f))
    overlay = img.copy()
    overlay[mask] = (0, 0, 255)
    img = cv2.addWeighted(overlay, 0.45, img, 0.55, 0)
    cv2.rectangle(img, tuple(box[:2]), tuple(box[2:]), (0, 255, 0), 1)
    cv2.putText(img, f"{area:.0f} mm2", (box[0], max(12, box[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
    cv2.imwrite(str(OUTPUT_DIR / f"sam_{f}"), img)

print(f"\nOverlays saved to {OUTPUT_DIR}/sam_plate_*.jpg (green = box prompt, red = SAM mask)")

# What to look at in the overlays:
#  - scratches: the box is mostly empty metal, the mask follows the line -> area 5-10x smaller
#    than the box. Measuring with boxes would overestimate every elongated defect.
#  - stains / dents: soft edges, SAM picks a boundary. Check if you agree with it: "where does a
#    stain end" is a labeling-policy question, not a model question.
#  - area is not visibility: the subtle stain (faint) can be as large as a strong one. Severity
#    rules usually need both size and contrast.
#  - on real images (texture, reflections, low contrast) SAM sometimes grabs the whole part or a
#    neighbor instead. Fixes: tighter box, a point prompt at the center, or a negative point
#    (label 0) on what you DON'T want.
#
# SAM 3 (text prompts, "segment every scratch") on Colab: see README. It's the same pipeline
# without the detector, at the cost of a GPU.
