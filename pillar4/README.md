# Pillar 4 - Fine-tuning and open models

Run open models on your own machine, understand quantization and LoRA from the inside,
fine-tune a small model on free Colab, bring it back to your laptop, and compare it with
the big cloud model. Still $0: Ollama locally, Colab/Kaggle free GPUs for training.

## Setup

```
cd d:\pythoncode\pillar4
conda activate first-env
pip install -r requirements.txt
```

- **Ollama** (steps 1, 5, 7): install from https://ollama.com, then `ollama pull qwen3.5:0.8b` (~1 GB).
- **GPU**: none needed on your laptop. Training runs in the notebooks on a free Colab T4
  (or Kaggle, 30 h/week of free GPU). If you have an NVIDIA GPU with 8+ GB, the notebooks
  also run locally with Unsloth.
- `.env` with `GROQ_API_KEY` in `d:\pythoncode` is picked up automatically (steps 1, 4, 5).

## The model

**Qwen3.5-0.8B** (Alibaba, March 2026, Apache-2.0): a 0.8B model that also accepts
images, so the same model is fine-tuned on text (notebook 06) and on the plates (notebook 08).
It runs on a CPU through Ollama, and Unsloth officially supports fine-tuning it on free Colab.
Unsloth discourages 4-bit QLoRA for Qwen3.5, so the notebooks use 16-bit LoRA.
To try something bigger, `Qwen/Qwen3.5-2B` still fits a T4.

## Steps

| # | File | What you learn | Where |
|---|---|---|---|
| 1 | `01_local_models_ollama.py` | Local open model vs cloud model, constrained JSON decoding | laptop |
| 2 | `02_quantization.py` | Memory math, int8/int4 by hand, why block-wise scales and outliers matter | laptop |
| 3 | `03_lora_from_scratch.py` | LoRA in 40 lines of PyTorch: params, forgetting, adapters on/off, merging | laptop |
| 4 | `04_make_dataset.py` | Building an SFT dataset (chat format, splits, optional LLM paraphrasing) | laptop |
| 5 | `05_baselines.py` | Always measure prompting first: big vs small model, basic vs expert prompt | laptop |
| 6 | `notebooks/06_finetune_llm.ipynb` | LoRA fine-tuning with Unsloth + TRL, before/after eval, GGUF export | Colab T4 |
| 7 | `07_serve_and_compare.py` | Serve your fine-tune in Ollama, compare everything in one table | laptop |
| 8 | `notebooks/08_finetune_vlm_plates.ipynb` | Fine-tune the VLM on the pillar 3bis plates, compare with the big VLM | Colab T4 |

`notes_task.py` defines the text task: messy French/English operator notes -> strict JSON,
with "house rules" for the `action` field that are never written in the basic prompt. The data
is already generated in `data/` (rerun step 4 to change it).

Recommended order: 1 -> 2 -> 3 (concepts, all local), then 4 -> 5 -> 6 -> 7 (the full
fine-tuning loop), then 8 if the vision side interests you.

## Using the notebooks on Colab

1. Go to colab.research.google.com > File > Upload notebook > pick the `.ipynb`
2. Runtime > Change runtime type > **T4 GPU**
3. Upload the files listed at the top of the notebook (folder icon on the left)
4. Runtime > Run all

Free Colab sessions disconnect after a while when idle and GPU access isn't guaranteed at
busy times. If that happens, Kaggle Notebooks work the same way (enable GPU T4 x2 in the settings).

**If a cell fails:** Unsloth, TRL and transformers move fast. Unsloth's docs link official
Qwen3.5 notebooks (text and vision) that they keep working. Compare the failing cell with
theirs, the logic of these notebooks stays the same.

## Prompt, RAG or fine-tune? (the decision you'll be asked about)

| Situation | Usually the answer |
|---|---|
| The model lacks **knowledge** that changes (docs, prices, tickets) | RAG (pillar 2) |
| The model can do it but the **output/behavior** isn't right | Better prompt first, few-shot examples |
| Prompting works but is too **expensive / slow / long** at your volume | Fine-tune a small model (distillation) |
| Strict **format or conventions** learned from many examples | Fine-tune |
| Data must **stay on-premise**, no API allowed | Open model, possibly fine-tuned |
| Not enough labeled examples (< a few hundred) | Prompting, or generate data with a big model first |

Fine-tuning teaches **behavior** (format, style, conventions, a narrow skill), not reliable
**facts**. Put facts in RAG.

## Going further

- **Distillation:** label real (unlabeled) notes with the big model + rules prompt, then
  fine-tune the small one on those labels. That's how most production fine-tunes are made.
- **Preference tuning (DPO/GRPO):** train on "better vs worse answer" pairs or a reward function
  instead of exact targets. TRL and Unsloth support it, and it's the next step after SFT.
- **Serving at scale:** vLLM or SGLang with LoRA adapters swapped per request.
- **Bigger models:** QLoRA on a 7-9B model (with an architecture where 4-bit works well) on a T4.
