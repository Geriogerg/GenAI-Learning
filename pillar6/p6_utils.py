"""
Shared helpers for pillar 6: paths, the pillar 3bis plates, Groq calls (text + vision), JSON parsing.
pillar3bis must sit next to this folder (its plate generator and labels are reused).
"""
import base64
import importlib.util
import json
import os
import re
import time
from pathlib import Path

import cv2
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs"
PILLAR3BIS = ROOT.parent / "pillar3bis"
PLATES_DIR = PILLAR3BIS / "data" / "plates"
INVOICES_DIR = DATA_DIR / "invoices"

TEXT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
VISION_MODEL = os.getenv("GROQ_VISION_MODEL", "qwen/qwen3.8-27b")


# ---------------------------------------------------------------------------
# plates (from pillar 3bis)
# ---------------------------------------------------------------------------

def load_plate_labels():
    path = PLATES_DIR / "labels.json"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python {PILLAR3BIS / 'data' / 'make_plates.py'}")
    return {x["file"]: x for x in json.loads(path.read_text(encoding="utf-8"))}


def plate_generator():
    """Import pillar3bis/data/make_plates.py as a module (to generate extra plates)."""
    path = PILLAR3BIS / "data" / "make_plates.py"
    spec = importlib.util.spec_from_file_location("make_plates", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def boxes_overlap(a, b):
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


# ---------------------------------------------------------------------------
# Groq
# ---------------------------------------------------------------------------

def image_to_data_url(image, max_side=1600, quality=90):
    img = cv2.imread(str(image)) if isinstance(image, (str, Path)) else image
    h, w = img.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def chat(messages, model=None, json_mode=False, max_tokens=2000, temperature=0.0):
    """Returns (text, seconds, usage_dict)."""
    from groq import Groq, RateLimitError

    model = model or TEXT_MODEL
    kw = dict(model=model, messages=messages, max_tokens=max_tokens, temperature=temperature)
    if json_mode:
        kw["response_format"] = {"type": "json_object"}
    if "gpt-oss" in model:
        kw["reasoning_effort"] = "low"
    if "qwen" in model:
        kw["reasoning_format"] = "parsed"
    client = Groq()
    for attempt in range(4):
        try:
            start = time.perf_counter()
            resp = client.chat.completions.create(**kw)
            usage = {"prompt": resp.usage.prompt_tokens, "completion": resp.usage.completion_tokens} if resp.usage else {}
            return resp.choices[0].message.content or "", time.perf_counter() - start, usage
        except RateLimitError:
            if attempt == 3:
                raise
            time.sleep(15 * (attempt + 1))


def vision_chat(images, prompt, json_mode=True, max_tokens=2000):
    content = [{"type": "text", "text": prompt}]
    content += [{"type": "image_url", "image_url": {"url": image_to_data_url(im)}} for im in images]
    return chat([{"role": "user", "content": content}], model=VISION_MODEL, json_mode=json_mode, max_tokens=max_tokens)


def parse_json(text):
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
