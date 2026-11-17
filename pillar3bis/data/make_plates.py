"""
Generates a small synthetic inspection dataset: photos of brushed-metal plates with 4 screws,
taken by a fixed camera on a production line.

    python data/make_plates.py

Output:
    data/plates/plate_000.jpg ... plate_063.jpg
    data/plates/reference_good.jpg   a known-good plate (golden sample)
    data/plates/labels.json          ground truth: defect type, bounding box, imaging issue

Why synthetic: real defect datasets (MVTec AD, NEU...) are great but need a download and
have licenses to check. Here we know the exact ground truth, so every method in this
pillar (VLM, classic CV, agent) can be scored on the same images. Swap in real images
later, the code doesn't care.

Defects: scratch, dent, stain, missing_screw (+ "ok"). Some scratches and stains are
deliberately subtle, and some images have imaging problems (blur, too dark, overexposed)
on top, like a dirty lens or a lighting failure.
"""
import json
from pathlib import Path

import cv2
import numpy as np

SIZE = 320
SCREWS = [(42, 42), (278, 42), (42, 278), (278, 278)]
OUT = Path(__file__).parent / "plates"

rng = np.random.default_rng(7)


def base_plate():
    """Same plate texture for every image (same part model, same camera)."""
    tex_rng = np.random.default_rng(0)
    streaks = cv2.blur(tex_rng.normal(0, 1, (SIZE, SIZE)).astype(np.float32), (81, 1))  # brushed metal
    gray = 150 + streaks * 40 + np.linspace(-10, 10, SIZE)[:, None]  # light falloff top -> bottom
    gray[:10, :] = gray[-10:, :] = gray[:, :10] = gray[:, -10:] = 95   # dark edge of the plate
    return gray.astype(np.float32)


BASE = base_plate()


def draw_screws(img, missing=None):
    for i, (x, y) in enumerate(SCREWS):
        if i == missing:
            cv2.circle(img, (x, y), 13, (115, 115, 118), 2)  # empty thread ring
            cv2.circle(img, (x, y), 10, (35, 35, 38), -1)    # hole
            continue
        cv2.circle(img, (x, y), 14, (200, 200, 205), -1)
        cv2.circle(img, (x, y), 14, (120, 120, 125), 1)
        cv2.line(img, (x - 8, y), (x + 8, y), (80, 80, 85), 2)
        cv2.line(img, (x, y - 8), (x, y + 8), (80, 80, 85), 2)


def random_spot(margin=70):
    while True:
        x, y = rng.integers(margin, SIZE - margin, 2)
        if all(abs(x - sx) > 40 or abs(y - sy) > 40 for sx, sy in SCREWS):
            return int(x), int(y)


def gaussian_blob(cx, cy, sx, sy):
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    return np.exp(-(((xx - cx) / sx) ** 2 + ((yy - cy) / sy) ** 2) / 2).astype(np.float32)


def make_plate(defect, subtle=False):
    img = np.dstack([BASE, BASE, BASE + 4])  # slightly cold metal tint
    bbox = None

    if defect == "scratch":
        cx, cy = random_spot()
        length = rng.uniform(25, 40) if subtle else rng.uniform(60, 140)
        angle = rng.uniform(0, np.pi)
        dx, dy = np.cos(angle) * length / 2, np.sin(angle) * length / 2
        p1, p2 = (int(cx - dx), int(cy - dy)), (int(cx + dx), int(cy + dy))
        layer = np.zeros((SIZE, SIZE), np.float32)
        cv2.line(layer, p1, p2, 1.0, 1 if subtle else 2, cv2.LINE_AA)
        img += (layer * (22 if subtle else 55) * rng.choice([-1, 1]))[..., None]
        bbox = [min(p1[0], p2[0]) - 3, min(p1[1], p2[1]) - 3, max(p1[0], p2[0]) + 3, max(p1[1], p2[1]) + 3]

    elif defect == "dent":
        cx, cy = random_spot()
        r = rng.uniform(9, 18)
        img -= (gaussian_blob(cx, cy, r, r * 0.8) * 45)[..., None]                    # shadow
        img += (gaussian_blob(cx - r * 0.5, cy - r * 0.5, r * 0.4, r * 0.3) * 35)[..., None]  # highlight
        bbox = [int(cx - 2 * r), int(cy - 2 * r), int(cx + 2 * r), int(cy + 2 * r)]

    elif defect == "stain":
        cx, cy = random_spot()
        mask = sum(gaussian_blob(cx + rng.uniform(-12, 12), cy + rng.uniform(-12, 12),
                                 rng.uniform(8, 16), rng.uniform(8, 16)) for _ in range(3))
        mask = np.clip(mask, 0, 1) * (0.22 if subtle else 0.5)
        color = np.array([45, 75, 120], np.float32)  # brownish, in BGR (opencv order)
        img = img * (1 - mask[..., None]) + color * mask[..., None]
        bbox = [cx - 32, cy - 32, cx + 32, cy + 32]

    img = np.clip(img, 0, 255).astype(np.uint8)
    missing = int(rng.integers(0, 4)) if defect == "missing_screw" else None
    draw_screws(img, missing)
    if missing is not None:
        x, y = SCREWS[missing]
        bbox = [x - 18, y - 18, x + 18, y + 18]

    # camera: sensor noise + small exposure variation, on the whole image
    img = img.astype(np.float32) + rng.normal(0, 3, img.shape[:2]).astype(np.float32)[..., None] + rng.uniform(-6, 6)
    return np.clip(img, 0, 255).astype(np.uint8), bbox


def degrade(img, issue):
    if issue == "blur":
        return cv2.GaussianBlur(img, (0, 0), 3)
    if issue == "dark":
        return (img * 0.3).astype(np.uint8)
    if issue == "overexposed":
        return np.clip(img.astype(np.float32) * 1.9, 0, 255).astype(np.uint8)
    return img


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    plan = (["ok"] * 22 + ["scratch"] * 12 + ["dent"] * 10 + ["stain"] * 10 + ["missing_screw"] * 10)
    subtle_idx = {22, 23, 24, 25, 46, 47, 48}  # 4 subtle scratches, 3 subtle stains
    issues = ["blur", "blur", "blur", "dark", "dark", "dark", "overexposed", "overexposed", "overexposed", "blur"]
    issue_for = dict(zip(rng.choice(len(plan), len(issues), replace=False).tolist(), issues))
    order = rng.permutation(len(plan))  # shuffle so the file name says nothing

    labels = []
    for file_idx, i in enumerate(order):
        defect, subtle = plan[i], int(i) in subtle_idx
        img, bbox = make_plate(defect, subtle)
        issue = issue_for.get(int(i))
        name = f"plate_{file_idx:03d}.jpg"
        cv2.imwrite(str(OUT / name), degrade(img, issue), [cv2.IMWRITE_JPEG_QUALITY, 92])
        labels.append({"file": name, "defect": defect, "subtle": subtle, "bbox": bbox,
                       "imaging_issue": issue})

    ref, _ = make_plate("ok")
    cv2.imwrite(str(OUT / "reference_good.jpg"), ref, [cv2.IMWRITE_JPEG_QUALITY, 92])

    (OUT / "labels.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    counts = {}
    for lab in labels:
        counts[lab["defect"]] = counts.get(lab["defect"], 0) + 1
    print(f"wrote {len(labels)} plates + reference to {OUT}: {counts}, "
          f"{sum(1 for x in labels if x['imaging_issue'])} with imaging issues")


if __name__ == "__main__":
    main()
