"""
Step 1 - vision-language models (VLMs): talking to a model about an image.

A VLM is an LLM with an image encoder in front (usually a ViT, often CLIP-like). The image is
cut into patches, turned into "visual tokens", and the language model reads them like text.
You already know both halves from your CV background and pillars 1-3.

From the API side, nothing new: the image is one more part of the user message (base64 here,
see vision_utils.vlm). Everything from pillar 1 applies: system prompts, JSON output...

This script: describe a photo, ask precise questions, get a structured inspection report,
and compare an image against a golden sample. 5 API calls.
"""
import json

from ultralytics.utils import ASSETS  # the 2 sample photos shipped with ultralytics

from vision_utils import PLATES_DIR, REFERENCE, VISION_MODEL, load_labels, vlm

bus = ASSETS / "bus.jpg"
labels = load_labels()
scratch = next(f for f, x in labels.items() if x["defect"] == "scratch" and not x["subtle"] and not x["imaging_issue"])
subtle = next(f for f, x in labels.items() if x["defect"] == "scratch" and x["subtle"] and not x["imaging_issue"])

print(f"Vision model: {VISION_MODEL}\n")

# --- 1. open description -----------------------------------------------------
print("=" * 80 + "\n1. Describe the image\n")
print(vlm(bus, "Describe this image in 3 sentences."))

# --- 2. precise questions ----------------------------------------------------
# VLMs are good at "what" and "roughly where", weaker at counting, exact positions and
# small details. Worth knowing before you build on them.
print("\n" + "=" * 80 + "\n2. Precise questions\n")
print(vlm(bus, "How many people are fully visible? What is written on the front of the bus? "
               "Answer as a short list."))
# Compare with YOLO in 03 (it finds the people with exact boxes, but can't read text).

# --- 3. structured inspection report (the pillar 1 mini project, with a real image this time)
print("\n" + "=" * 80 + f"\n3. Structured report on {scratch} (ground truth: scratch)\n")
INSPECTION_PROMPT = """You are a quality inspector on a production line. The image shows a brushed metal
plate that should have 4 screws and a clean, uniform surface.
Return a JSON object:
{
  "defect_found": boolean,
  "defect_type": "none" | "scratch" | "dent" | "stain" | "missing_screw" | "other",
  "location": "short description of where, e.g. 'center-left'",
  "severity": "none" | "low" | "medium" | "high",
  "image_quality_ok": boolean,
  "confidence": float between 0 and 1,
  "explanation": "one sentence"
}"""
report = vlm(PLATES_DIR / scratch, INSPECTION_PROMPT, json_mode=True)
print(json.dumps(report, indent=2))

# --- 4. a subtle defect ------------------------------------------------------
print("\n" + "=" * 80 + f"\n4. Same prompt on {subtle} (ground truth: SUBTLE scratch, ~30 px long)\n")
print(json.dumps(vlm(PLATES_DIR / subtle, INSPECTION_PROMPT, json_mode=True), indent=2))

# --- 5. with a golden sample -------------------------------------------------
# Several images in one request: here a known-good plate + the plate to inspect.
# A standard trick in inspection, the reference tells the model what "normal" looks like.
print("\n" + "=" * 80 + f"\n5. Subtle scratch again, with a reference image\n")
prompt = ("The FIRST image is a known-good reference plate. The SECOND image is the plate to inspect, "
          "taken by the same camera. Compare them carefully.\n\n" + INSPECTION_PROMPT)
print(json.dumps(vlm([REFERENCE, PLATES_DIR / subtle], prompt, json_mode=True), indent=2))

# What to notice:
#  - 1-3: impressive for zero training. This is why VLMs are everywhere in industry right now:
#    describing, captioning, extracting info from photos/documents, triaging images.
#  - 4: subtle defects are often missed, or the confidence is high while the answer is wrong.
#    VLM confidence numbers are generated text, NOT calibrated probabilities.
#  - 5: the reference image often helps, sometimes not. Step 2 measures it on 64 images
#    instead of guessing from one example.
