# Pillar 6 - Vision foundation models and document AI

Two families of models that changed applied computer vision since 2023, used on your own data:

- **Vision foundation models**: detect things by name (YOLOE), segment anything from a click or
  a box (SAM 2), and detect anomalies with pretrained features learned from good parts only
  (PatchCore). All on the pillar 3bis plates, all scored, all on CPU.
- **Document AI**: parse PDFs and scans (Docling), then extract invoice fields with a text LLM or
  a vision-language model, validate them, and route the doubtful ones to a human.

## Setup

```
cd d:\pythoncode\pillar6
conda activate first-env
pip install -r requirements.txt
python 04_make_invoices.py
```

- **pillar3bis** must sit next to this folder (its plates, generator and `cv_tools` are reused).
  If `pillar3bis/data/plates` is missing, run `python ..\pillar3bis\data\make_plates.py`.
- YOLOE text prompts need Ultralytics' CLIP package (in requirements.txt, installed from GitHub,
  which needs `git` on your PATH). Ultralytics also tries to install it automatically on first use.
- First runs download models: YOLOE-26s (~30 MB) + its text encoder (~240 MB), SAM 2 tiny
  (78 MB), MobileSAM (40 MB), WideResNet50 ImageNet weights (~130 MB), Docling's layout, table
  and OCR models (a few hundred MB). Everything is cached afterwards.
- No GPU needed. Steps 6 and 7 call Groq (text and vision models).

## Steps

| # | Script | What you learn | API |
|---|---|---|---|
| 1 | `01_open_vocab_detection.py` | YOLOE: detection from text prompts and from one example box, scored on the plates | - |
| 2 | `02_segment_anything.py` | SAM 2 / MobileSAM: masks from a click or a box, defect area in mm² | - |
| 3 | `03_patchcore_anomaly.py` | PatchCore from scratch: trained on good plates only, robust to misalignment | - |
| 4 | `04_make_invoices.py` | Synthetic invoices (3 layouts, PDFs + scans) with exact ground truth | - |
| 5 | `05_parse_documents.py` | pypdf vs Docling (layout, tables, OCR) on a PDF and a scan | - |
| 6 | `06_extract_fields.py` | 3 extraction strategies compared: accuracy, cost, what validation catches | Groq |
| 7 | `07_project_invoice_pipeline.py` | Mini project: routing, extraction, validation, retry, review queue, report | Groq |

`doc_extract.py` holds the shared extraction code (schema, readers, prompts, validation, scoring).

## Results I got while building it (CPU)

Real runs from my test environment, to compare with yours:

- **YOLOE, zero-shot, 54 clean plates**: with defect names as prompts, 14/35 defects located but
  false alarms on 13/19 good plates (conf >= 0.1). With "visual words" ("thin bright line",
  "brown blotch"...), 22/35 located, but alarms on every good plate. One example box as a visual
  prompt found the stain on all 4 other stain plates.
- **SAM 2 tiny**: clean masks for every defect type in ~4 s per image. A scratch mask is ~10x
  smaller than its bounding box, so box-based measurement overestimates elongated defects.
- **PatchCore**: I couldn't download the ImageNet weights in my environment, so I only ran the
  `--random-weights` ablation (AUROC ~0.75). The golden-sample method of 3bis went from 35/35
  found with 0 false alarms on aligned plates to false alarms on 19/19 good plates once the
  plates were shifted by a few pixels. Your pretrained run is the interesting one.

## SAM 3 (optional, Colab)

SAM 3 (Meta, Nov 2025) adds **text prompts**: "segment every scratch" with no detector in front.
It's 3.4 GB and gated: request access on the `facebook/sam3` Hugging Face page, download `sam3.pt`,
and run it on a Colab GPU:

```python
!pip install -U ultralytics
from ultralytics.models.sam import SAM3SemanticPredictor
predictor = SAM3SemanticPredictor(overrides={"model": "sam3.pt", "conf": 0.25, "task": "segment",
                                             "mode": "predict", "save": True})
predictor.set_image("plate_010.jpg")
results = predictor(text=["stain", "scratch", "missing screw"])
```

Compare its masks with step 2's pipeline (golden-sample boxes + SAM 2) on the same plates.

## Where this goes next

- **Labeling acceleration**: YOLOE pre-annotations + SAM masks + a human correcting them in a
  labeling tool (CVAT, Label Studio), then train a regular YOLO on the result.
- **anomalib** for production anomaly detection (PatchCore, EfficientAD...), validated on MVTec AD.
- **DINOv2/v3 features**: a strong general-purpose backbone (retrieval, few-shot classification,
  anomaly detection). The weights are on Hugging Face (DINOv3 needs a license acceptance).
- **Document AI at scale**: Docling's VLM mode (granite-docling), page-level routing, comparing
  against purchase orders, and fine-tuning a small VLM on your own documents (pillar 4 notebook 08).
