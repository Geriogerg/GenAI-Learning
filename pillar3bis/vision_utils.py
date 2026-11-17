"""
Shared config and the vision-language model (VLM) helper for pillar 3bis.

Reuses the agent machinery from pillar 3 (Agent, tool, chat), so pillar3 has to sit next to
this folder: d:\\pythoncode\\pillar3 and d:\\pythoncode\\pillar3bis.
"""
import base64
import json
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent
PLATES_DIR = ROOT / "data" / "plates"
REFERENCE = PLATES_DIR / "reference_good.jpg"
LABELS_FILE = PLATES_DIR / "labels.json"
OUTPUT_DIR = ROOT / "outputs"

sys.path.insert(0, str(ROOT.parent / "pillar3"))
from agent_utils import MODEL, TOKENS, Agent, chat, tool  # noqa: E402,F401  (pillar 3 code)

# Groq's vision model at the time of writing (preview tier). Max 3 images per request.
# Check console.groq.com/docs/vision if it gets renamed, and override in .env.
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")
MAX_IMAGES_PER_CALL = 3

DEFECT_CLASSES = ["ok", "scratch", "dent", "stain", "missing_screw"]


def load_labels():
    return {x["file"]: x for x in json.loads(LABELS_FILE.read_text(encoding="utf-8"))}


def image_to_data_url(image, max_side=1024, quality=90):
    """Path or numpy BGR image -> base64 JPEG data URL, the format the API expects for local images.
    Resized first: big images cost a lot of tokens and rarely help."""
    img = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
    if img is None:
        raise FileNotFoundError(f"cannot read image {image}")
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def vlm(images, prompt, system=None, json_mode=False, max_tokens=1500, temperature=0.1):
    """
    Ask the vision model about 1-3 images. `images` = list of paths or numpy arrays.
    An image is just another part of the user message:
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}
    Returns text, or a dict if json_mode.
    """
    images = images if isinstance(images, (list, tuple)) else [images]
    if len(images) > MAX_IMAGES_PER_CALL:
        raise ValueError(f"{VISION_MODEL} accepts at most {MAX_IMAGES_PER_CALL} images per request")

    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": image_to_data_url(im)}} for im in images]
    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": content}]

    msg = chat(messages, model=VISION_MODEL, json_mode=json_mode, max_tokens=max_tokens, temperature=temperature)
    text = (msg.content or "").strip()
    if not json_mode:
        return text
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"error": "invalid JSON from the model", "raw": text[:300]}
