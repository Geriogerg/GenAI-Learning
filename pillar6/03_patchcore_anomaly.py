"""
Step 3 - learned anomaly detection with PatchCore (from scratch, ~100 lines, CPU).

In 3bis, the golden-sample difference found every defect, but only because our plates are
perfectly aligned. Shift the part by a few pixels (loose fixture, vibration) and the difference
lights up everywhere. PatchCore (Roth et al., 2022) is the standard fix, and still one of the
strongest baselines on MVTec AD:

  1. take a CNN pretrained on ImageNet, keep mid-level feature maps: one feature vector per
     image patch, describing local texture and shape
  2. run it on GOOD images only and store all patch features in a "memory bank"
     (subsampled with a greedy coreset to keep it small)
  3. for a new image, each patch's anomaly score = distance to its nearest neighbor in the bank.
     A patch unlike anything seen on good parts = anomaly. Max over patches = image score.

No defect labels needed: you train with good parts, which every factory has plenty of.
The threshold is also set on good parts only.

Usage:
    python 03_patchcore_anomaly.py                    # wide_resnet50_2, as in the paper
    python 03_patchcore_anomaly.py --random-weights   # ablation: same network, untrained
First run downloads the ImageNet weights (~130 MB). No API calls. ~2-4 min on CPU.
"""
import argparse
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision

from p6_utils import OUTPUT_DIR, PILLAR3BIS, PLATES_DIR, boxes_overlap, load_plate_labels, plate_generator

sys.path.insert(0, str(PILLAR3BIS))
import cv_tools  # noqa: E402  (golden-sample method from 3bis, for comparison)

parser = argparse.ArgumentParser()
parser.add_argument("--random-weights", action="store_true")
parser.add_argument("--n-train", type=int, default=30)
parser.add_argument("--coreset", type=float, default=0.1)
args = parser.parse_args()
torch.manual_seed(0)
rng = np.random.default_rng(0)

# ---------------------------------------------------------------------------
# feature extractor
# ---------------------------------------------------------------------------

weights = None if args.random_weights else torchvision.models.Wide_ResNet50_2_Weights.IMAGENET1K_V1
backbone = torchvision.models.wide_resnet50_2(weights=weights).eval()
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


@torch.no_grad()
def patch_features(img_bgr):
    """(H/8 * W/8, 1536) features: layer2 + upsampled layer3, smoothed over 3x3 neighborhoods."""
    x = torch.from_numpy(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()[None] / 255
    x = (x - MEAN) / STD
    x = backbone.maxpool(backbone.relu(backbone.bn1(backbone.conv1(x))))
    l2 = backbone.layer2(backbone.layer1(x))
    l3 = backbone.layer3(l2)
    feats = torch.cat([F.avg_pool2d(l2, 3, 1, 1),
                       F.interpolate(F.avg_pool2d(l3, 3, 1, 1), size=l2.shape[-2:], mode="bilinear")], 1)
    h, w = feats.shape[-2:]
    return feats[0].permute(1, 2, 0).reshape(-1, feats.shape[1]), (h, w)


def greedy_coreset(feats, ratio):
    """Pick points that cover the feature space: each new point = the one farthest from those
    already picked. Keeps rare-but-normal patches that random sampling would drop."""
    n = max(1, int(len(feats) * ratio))
    proj = feats @ torch.randn(feats.shape[1], 128) / 128 ** 0.5   # random projection: faster distances
    picked = [int(torch.randint(len(feats), (1,)))]
    dist = torch.cdist(proj, proj[picked]).squeeze(1)
    for _ in range(n - 1):
        idx = int(dist.argmax())
        picked.append(idx)
        dist = torch.minimum(dist, torch.cdist(proj, proj[idx:idx + 1]).squeeze(1))
    return feats[picked]


def anomaly_map(img_bgr, bank):
    feats, (h, w) = patch_features(img_bgr)
    scores = torch.cdist(feats, bank).min(dim=1).values.reshape(1, 1, h, w)
    amap = F.interpolate(scores, size=img_bgr.shape[:2], mode="bilinear")[0, 0].numpy()
    return cv2.GaussianBlur(amap, (0, 0), 4)


# ---------------------------------------------------------------------------
# data: good plates with realistic variation, and a "misaligned" copy of the test set
# ---------------------------------------------------------------------------

def jitter(img, max_shift=8, brightness=10, seed=None):
    """A loose fixture: the part moves by a few pixels, lighting changes a bit."""
    r = np.random.default_rng(seed)
    dx, dy = r.integers(-max_shift, max_shift + 1, 2)
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    out = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
    return np.clip(out.astype(np.int16) + r.integers(-brightness, brightness + 1), 0, 255).astype(np.uint8), (dx, dy)


mp = plate_generator()
mp.rng = np.random.default_rng(123)
train = [jitter(mp.make_plate("ok")[0], seed=i)[0] for i in range(args.n_train)]
calib = [jitter(mp.make_plate("ok")[0], seed=1000 + i)[0] for i in range(10)]  # held-out good plates

labels = load_plate_labels()
clean = {f: x for f, x in sorted(labels.items()) if not x["imaging_issue"]}
aligned = {f: cv2.imread(str(PLATES_DIR / f)) for f in clean}
shifted, shifted_boxes = {}, {}
(OUTPUT_DIR / "shifted").mkdir(parents=True, exist_ok=True)
for i, f in enumerate(clean):
    shifted[f], (dx, dy) = jitter(aligned[f], seed=5000 + i)
    b = clean[f]["bbox"]
    shifted_boxes[f] = [b[0] + dx, b[1] + dy, b[2] + dx, b[3] + dy] if b else None
    cv2.imwrite(str(OUTPUT_DIR / "shifted" / f), shifted[f])

# ---------------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------------

start = time.time()
all_feats = torch.cat([patch_features(img)[0] for img in train])
bank = greedy_coreset(all_feats, args.coreset)
print(f"backbone: {'RANDOM weights' if args.random_weights else 'wide_resnet50_2 ImageNet'} | "
      f"{len(all_feats):,} patch features from {len(train)} good plates -> memory bank of {len(bank):,} "
      f"({time.time() - start:.0f}s)")

calib_scores = [anomaly_map(img, bank).max() for img in calib]
threshold = max(calib_scores) * 1.05   # just above the worst GOOD plate: no defect sample needed
print(f"threshold from {len(calib)} held-out good plates: {threshold:.2f} (their scores: "
      f"{min(calib_scores):.2f}-{max(calib_scores):.2f})")

# ---------------------------------------------------------------------------
# evaluate both methods on aligned and shifted plates
# ---------------------------------------------------------------------------


def auroc(scores, is_defect):
    pos = [s for s, d in zip(scores, is_defect) if d]
    neg = [s for s, d in zip(scores, is_defect) if not d]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))


def evaluate_patchcore(images, boxes):
    scores, located, false_alarms = [], 0, 0
    for f, img in images.items():
        amap = anomaly_map(img, bank)
        scores.append(float(amap.max()))
        y, x = np.unravel_index(amap.argmax(), amap.shape)
        if boxes[f] is None:
            false_alarms += amap.max() > threshold
        elif amap.max() > threshold and boxes_overlap([x - 6, y - 6, x + 6, y + 6], boxes[f]):
            located += 1
    is_def = [boxes[f] is not None for f in images]
    return auroc(scores, is_def), located, false_alarms


def evaluate_golden(folder, boxes):
    located, false_alarms = 0, 0
    for f in boxes:
        regions = cv_tools.find_anomalies(folder / f)["regions"]
        if boxes[f] is None:
            false_alarms += bool(regions)
        elif any(boxes_overlap(r["bbox"], boxes[f]) for r in regions):
            located += 1
    return located, false_alarms


aligned_boxes = {f: clean[f]["bbox"] for f in clean}
n_def = sum(b is not None for b in aligned_boxes.values())
n_ok = len(clean) - n_def
print(f"\n{'':<34}{'aligned plates':>22}{'shifted plates':>22}")
g_a, g_s = evaluate_golden(PLATES_DIR, aligned_boxes), evaluate_golden(OUTPUT_DIR / "shifted", shifted_boxes)
print(f"{'golden-sample diff (3bis)':<34}{f'{g_a[0]}/{n_def} found, {g_a[1]}/{n_ok} FA':>22}"
      f"{f'{g_s[0]}/{n_def} found, {g_s[1]}/{n_ok} FA':>22}")
p_a, p_s = evaluate_patchcore(aligned, aligned_boxes), evaluate_patchcore(shifted, shifted_boxes)
print(f"{'PatchCore':<34}{f'{p_a[1]}/{n_def} found, {p_a[2]}/{n_ok} FA':>22}"
      f"{f'{p_s[1]}/{n_def} found, {p_s[2]}/{n_ok} FA':>22}")
print(f"{'PatchCore image AUROC':<34}{p_a[0]:>22.3f}{p_s[0]:>22.3f}")
print("(found = defect detected AND located; FA = false alarms on good plates)")

# heatmaps for a few shifted plates
for f in [f for f in clean if clean[f]["defect"] != "ok"][:4]:
    amap = anomaly_map(shifted[f], bank)
    heat = cv2.applyColorMap(np.clip(amap / threshold * 180, 0, 255).astype(np.uint8), cv2.COLORMAP_JET)
    cv2.imwrite(str(OUTPUT_DIR / f"patchcore_{f}"), np.hstack([shifted[f], cv2.addWeighted(shifted[f], 0.5, heat, 0.5, 0)]))
print(f"\nHeatmaps: {OUTPUT_DIR}/patchcore_plate_*.jpg")

# What to look at:
#  - golden-sample diff, shifted plates: false alarms on (nearly) every good plate, because every
#    screw edge "moved". That part doesn't depend on the backbone (19/19 in my test run).
#  - PatchCore doesn't care WHERE a patch is, only whether it looks like something seen on good
#    plates. With ImageNet weights it should keep most defects found and few false alarms on the
#    shifted set: check how close to the aligned column it stays.
#  - --random-weights: same network, no pretraining. In my run: AUROC ~0.75, 17/35 found. Compare
#    with your pretrained run: the gap IS the value of pretrained features, the thing that makes
#    a 30-image "training" possible. That's the foundation-model idea in miniature.
#  - production version: the anomalib library (PatchCore, EfficientAD, ...), and DINOv2/v3 features
#    are a strong alternative backbone. Validate on real images: MVTec AD is the standard benchmark.
