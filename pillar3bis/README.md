# Pillar 3bis - Agents for computer vision

Pillar 3 with eyes: vision-language models (VLMs), classic CV models as agent tools, image
search with CLIP, and an inspection pipeline where code, a VLM and an agent each handle the
part they're good at. Still $0: Groq free tier + local models (all CPU-friendly).

## Setup

```
cd d:\pythoncode\pillar3bis
conda activate first-env
pip install -r requirements.txt
python data/make_plates.py
```

- **pillar3 must be next to this folder**: the scripts reuse its `Agent` and `chat()`.
- Vision model: `qwen/qwen3.8-27b` (Groq, preview tier, max 3 images per request). If Groq
  renames it, set `GROQ_VISION_MODEL=...` in `.env` (check console.groq.com/docs/vision).
- First runs download YOLO11n (5 MB) and CLIP ViT-B/32 (~600 MB, steps 3 and 5 only).
- No GPU needed. If you have one, PyTorch uses it automatically for YOLO and CLIP.

## The dataset

`data/make_plates.py` draws 64 photos of brushed-metal plates (4 screws each) with
scratches, dents, stains and missing screws, some of them subtle, and some photos blurred,
too dark or overexposed. Plus a golden sample (`reference_good.jpg`) and the exact ground
truth (`labels.json`), so every method can be scored on the same images.

It's synthetic on purpose: known ground truth and no download or license to deal with.
The caveat is that it's easier than reality (perfect alignment, constant lighting). Swap in
real images later, for example MVTec AD, which is the standard benchmark for this.

## Steps

| # | Script | What you learn | API calls |
|---|---|---|---|
| 1 | `01_vlm_basics.py` | Images in the API, describing, questions, JSON reports, golden sample | 5 |
| 2 | `02_vlm_inspection_eval.py` | The VLM as a classifier, scored like a CV model: accuracy, confusion, escapes | ~48 |
| 3 | `03_classic_cv_tools.py` | Quality check, reference-based anomaly detection, YOLO, CLIP, and their limits | 0 |
| 4 | `04_vision_agent.py` | An LLM agent orchestrating CV specialists + the VLM on zoomed crops | ~20 |
| 5 | `05_image_search.py` | CLIP embeddings in Chroma: text-to-image and image-to-image search | 0 |
| 6 | `06_project_inspection_agent.py` | Mini project: code + VLM + agent pipeline, markdown report, scored vs ground truth | ~30-40 |
| 7 | `07_mcp_vision_server.py` | Your vision tools as an MCP server (VS Code, Claude, the pillar 3 client) | - |

Shared code: `vision_utils.py` (config, `vlm()` helper), `cv_tools.py` (the CV functions),
`vision_tools.py` (the same functions wrapped as agent tools).

## The main idea of this pillar

| | Good at | Bad at |
|---|---|---|
| VLM | naming things, open-ended questions, no training needed | small details, exact positions, counting, calibrated confidence |
| Classic CV / trained models | precise, fast, cheap, measurable | only knows what it was built or trained for |
| LLM agent | deciding what to do next, combining results, explaining | slow and expensive when looping over many items |

Good systems combine them: code for the routine, specialists for measurements, the VLM to
name what the specialists found, and the agent only for the cases nobody could decide.

## Going further (good bridges to pillar 4)

- Fine-tune YOLO on a defect dataset so it detects defect classes directly.
- Learned anomaly detection on real images: the `anomalib` library (PatchCore, EfficientAD).
- Open-vocabulary detection: Grounding DINO, YOLO-World, OWLv2 ("detect: scratch").
- Run a small VLM locally (SmolVLM, Moondream, Qwen-VL small) with `transformers`, which needs a GPU to be comfortable.
- Fine-tune a small VLM with LoRA on your inspection data: that's exactly pillar 4.

## Common issues

- `ModuleNotFoundError: agent_utils`: the pillar3 folder isn't next to pillar3bis.
- 400 error mentioning the model: the vision model was renamed or retired, set `GROQ_VISION_MODEL`.
- 400 "too many images": the model takes at most 3 images per request.
- Rate limits in steps 2 and 6: they retry automatically. Use `--n 10` (step 2) or `--limit 20` (step 6).
