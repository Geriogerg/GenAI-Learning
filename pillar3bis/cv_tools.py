"""
The "specialists": classic computer vision functions the agents can call.

    check_image_quality   blur / exposure checks (OpenCV, a few ms)
    find_anomalies        difference with a known-good reference image (golden sample)
    crop_region           cut and enlarge a region, e.g. to show it to the VLM
    detect_objects        YOLO pretrained on COCO (80 everyday classes)
    clip_classify         zero-shot classification with CLIP (any labels, no training)

They're plain functions returning dicts, so the same code is used by the scripts, wrapped
as agent tools, and exposed through MCP in step 7.

Thresholds were tuned on data/plates (03_classic_cv_tools.py shows how). On real images,
expect to re-tune them, that's normal.
"""
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from vision_utils import OUTPUT_DIR, REFERENCE

BLUR_MIN_SHARPNESS = 20.0      # variance of the Laplacian, below = blurry
DARK_MAX_BRIGHTNESS = 60.0     # mean gray level
BRIGHT_MIN_BRIGHTNESS = 200.0
CLIPPED_MAX_RATIO = 0.25       # share of pixels at 255 -> overexposed

ANOMALY_Z = 3.0                # pixel threshold on the normalized difference (in std units)
ANOMALY_MIN_STRENGTH = 7.5     # a region must peak above this, sensor noise peaks around 5-7
ANOMALY_MIN_AREA = 25          # pixels, smaller blobs are noise


def _read(path):
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"cannot read image '{path}'")
    return img


# ---------------------------------------------------------------------------
# image quality
# ---------------------------------------------------------------------------

def check_image_quality(image_path: str):
    """Check if an image is usable for inspection: sharpness (blur), brightness, overexposure.
    Run it first: a bad image makes every other result unreliable."""
    gray = cv2.cvtColor(_read(image_path), cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    clipped = float((gray >= 250).mean())

    problems = []
    if sharpness < BLUR_MIN_SHARPNESS:
        problems.append("blurry")
    if brightness < DARK_MAX_BRIGHTNESS:
        problems.append("too dark")
    if brightness > BRIGHT_MIN_BRIGHTNESS or clipped > CLIPPED_MAX_RATIO:
        problems.append("overexposed")
    return {"image": Path(image_path).name, "usable": not problems, "problems": problems,
            "sharpness": round(sharpness, 1), "brightness": round(brightness, 1),
            "clipped_ratio": round(clipped, 3)}


# ---------------------------------------------------------------------------
# reference-based anomaly detection
# ---------------------------------------------------------------------------

def _normalized(img):
    """Per-channel standardization (removes global brightness / contrast differences) + light blur."""
    f = img.astype(np.float32)
    f = (f - f.mean(axis=(0, 1))) / (f.std(axis=(0, 1)) + 1e-6)
    return cv2.GaussianBlur(f, (0, 0), 1.5)


def find_anomalies(image_path: str, reference_path: str = ""):
    """Compare an image with a known-good reference of the same part (same camera position)
    and return the regions that differ: bounding boxes [x1, y1, x2, y2], area, strength.
    Very sensitive to small defects, but it doesn't say WHAT the defect is."""
    img = _read(image_path)
    ref = _read(reference_path or REFERENCE)
    if img.shape != ref.shape:
        ref = cv2.resize(ref, (img.shape[1], img.shape[0]))

    # max over the 3 channels: catches brightness defects (scratch, dent) AND color ones (stain)
    diff = np.abs(_normalized(img) - _normalized(ref)).max(axis=2)
    # the difference itself is noisy, express it in "how unusual for this image" units
    z = (diff - np.median(diff)) / (diff.std() + 1e-6)
    mask = (z > ANOMALY_Z).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    mask = cv2.dilate(mask, np.ones((5, 5), np.uint8))  # merge pieces of the same defect

    n, _, stats, _ = cv2.connectedComponentsWithStats(mask)
    regions = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < ANOMALY_MIN_AREA:
            continue
        strength = float(z[y:y + h, x:x + w].max())
        if strength < ANOMALY_MIN_STRENGTH:
            continue
        regions.append({"bbox": [int(x), int(y), int(x + w), int(y + h)], "area": int(area),
                        "strength": round(strength, 1)})
    regions.sort(key=lambda r: r["area"] * r["strength"], reverse=True)
    return {"image": Path(image_path).name, "anomaly_found": bool(regions), "regions": regions[:5],
            "max_strength": round(float(z.max()), 1)}


# ---------------------------------------------------------------------------
# crops
# ---------------------------------------------------------------------------

def crop_region(image_path: str, x1: int, y1: int, x2: int, y2: int, margin: int = 20):
    """Cut a region of the image (with a margin), enlarge it to 256 px and save it.
    Returns the path of the crop, which can then be passed to ask_vlm to look closer."""
    img = _read(image_path)
    h, w = img.shape[:2]
    x1, y1 = max(0, x1 - margin), max(0, y1 - margin)
    x2, y2 = min(w, x2 + margin), min(h, y2 + margin)
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"empty region after clipping to the image size ({w}x{h})")
    crop = img[y1:y2, x1:x2]
    scale = 256 / max(crop.shape[:2])
    crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    out_dir = OUTPUT_DIR / "crops"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{Path(image_path).stem}_{x1}_{y1}_{x2}_{y2}.jpg"
    cv2.imwrite(str(out), crop)
    return {"crop_path": str(out), "region": [x1, y1, x2, y2]}


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _yolo():
    from ultralytics import YOLO
    return YOLO("yolo11n.pt")  # 5MB, downloaded on first use, runs fine on CPU


def detect_objects(image_path: str, min_confidence: float = 0.3):
    """Detect everyday objects (people, vehicles, tools, animals... the 80 COCO classes) with YOLO.
    Returns label, confidence and bbox for each object. Knows nothing about industrial defects."""
    result = _yolo().predict(str(image_path), conf=min_confidence, verbose=False)[0]
    objects = [
        {"label": result.names[int(c)], "confidence": round(float(p), 2), "bbox": [int(v) for v in b]}
        for b, c, p in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist())
    ]
    return {"image": Path(image_path).name, "count": len(objects), "objects": objects}


# ---------------------------------------------------------------------------
# CLIP
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def clip_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("clip-ViT-B-32")  # ~600MB, downloaded once


def clip_classify(image_path: str, labels: list[str]):
    """Zero-shot classification: score how well the image matches each text label, e.g.
    ["a clean metal plate", "a scratched metal plate"]. Returns probabilities."""
    from PIL import Image
    model = clip_model()
    img_emb = model.encode([Image.open(image_path).convert("RGB")], normalize_embeddings=True)
    txt_emb = model.encode([f"a photo of {label}" for label in labels], normalize_embeddings=True)
    logits = 100 * (img_emb @ txt_emb.T)[0]  # CLIP's usual temperature
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    ranked = sorted(zip(labels, probs.tolist()), key=lambda x: x[1], reverse=True)
    return {"image": Path(image_path).name, "scores": {label: round(p, 3) for label, p in ranked}}


def draw_boxes(image_path, boxes, out_path, color=(0, 0, 255), labels=None):
    """Save a copy of the image with boxes drawn on it (for reports)."""
    img = _read(image_path)
    for i, b in enumerate(boxes):
        cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), color, 2)
        if labels:
            cv2.putText(img, labels[i], (b[0], max(12, b[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
    cv2.imwrite(str(out_path), img)
    return str(out_path)
