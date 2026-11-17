# From classic ML to GenAI: a hands-on learning path

A self-built curriculum for a data scientist with a computer vision background, moving to the
current GenAI stack: LLM APIs, RAG, agents, vision-language models, fine-tuning, production
and security, and vision foundation models. Each pillar is a folder of numbered scripts
(one concept per script) ending with a mini project. Everything runs on a laptop CPU plus
free tiers (Groq API, Google Colab), for a total cost of $0.

The path ends with a portfolio project, **[Inspection Copilot](inspection-copilot/)**, which
combines most of the pillars into one application.

## The path at a glance

| Pillar | Topic | Mini project | Runs on |
|---|---|---|---|
| [1](pillar1/) | LLM APIs: calls, prompting, JSON outputs, function calling | ML model monitor (CLI assistant with tools) | laptop + Groq |
| [2](pillar2/) | RAG: embeddings, vector DB, chunking, hybrid search, reranking, evaluation | Research paper assistant with citations | laptop + Groq |
| [3](pillar3/) | Agents: ReAct loop, tool design, planning, memory, LangGraph, MCP, agent evals | Data analysis agent (writes and runs pandas code) | laptop + Groq |
| [3bis](pillar3bis/) | Agents for computer vision: VLMs, CV models as tools, CLIP search | Inspection pipeline (code + VLM + agent) | laptop + Groq |
| [4](pillar4/) | Fine-tuning and open models: Ollama, quantization, LoRA, SFT | Fine-tuned 0.8B model vs big cloud model | laptop + Colab T4 |
| [5](pillar5/) | Production and security: API, caching, routing, observability, prompt injection | The pillar 2 assistant as a hardened, tested, Dockerized service | laptop (+ Docker) |
| [6](pillar6/) | Vision foundation models and document AI: YOLOE, SAM 2, PatchCore, Docling | Invoice processing pipeline with validation and review queue | laptop + Groq |
| [Copilot](inspection-copilot/) | Portfolio project | Visual inspection copilot on MVTec AD | laptop (+ Colab later) |

## Setup (once)

```
d:\pythoncode\
  .env                 GROQ_API_KEY=...   (read automatically by every pillar)
  pillar1\ ... pillar6\
  inspection-copilot\
```

```
conda activate first-env                 # Python 3.11
cd d:\pythoncode\pillarN
pip install -r requirements.txt          # each pillar has its own
```

- **API key**: free at console.groq.com. The free tier has rate limits; the scripts retry
  automatically and most of them have a flag to limit the number of calls (`--n`, `--limit`).
- **Models**: the default text model is `openai/gpt-oss-120b`, the vision model is
  `qwen/qwen3.8-27b`. Groq renames and retires models regularly: override with `GROQ_MODEL` /
  `GROQ_VISION_MODEL` in `.env` instead of editing code.
- **Folders must sit side by side**: some pillars reuse earlier ones (see the dependency map below).

## Pillar by pillar

### Pillar 1 — LLMs and APIs

Calling a model from Python and getting usable output.
`01_first_call` (anatomy of a request: tokens, finish reason, reasoning) · `02_prompting` (vague vs
precise system prompt, chain-of-thought, few-shot) · `03_structured_outputs` (JSON mode for
extraction) · `04_function_calling` (the tool loop, the model chaining calls) ·
`05_project_model_monitor` (CLI assistant with memory and 6 tools).

**Key idea**: the system prompt and the output format are the cheapest quality levers;
function calling is the seed of every agent.

### Pillar 2 — RAG

Giving a model your own documents.
Embeddings and similarity → ChromaDB → first RAG → loading PDFs → comparing 4 chunking
strategies with a hit-rate metric → incremental ingestion with metadata → chat with citations,
memory and query rewriting → BM25 + hybrid search (RRF) + cross-encoder reranking → evaluation
(hit@k, MRR, LLM-as-judge) → paper assistant with structured summaries.
Data: 4 fake internal docs of a CV team (model registry, runbook, drift policy, postmortem) plus
your own arXiv PDFs.

**Key idea**: most bad RAG answers are retrieval failures. Measure retrieval first.

### Pillar 3 — Agents and tools

From one tool call to multi-step work.
ReAct loop by hand → tool design (errors the model can recover from) → agentic RAG (multi-hop
questions) → ReAct vs plan-and-execute → short and long-term memory → the same agent in LangGraph
with human approval (`interrupt`) → tools packaged as an **MCP** server, used from any client →
agent evaluation (answers AND trajectories) → data analysis agent on a CSV with planted anomalies.

**Key idea**: use the simplest design that works. Many "agents" should be workflows with one or
two LLM decisions; reliability comes from tool design, step limits and evals.

### Pillar 3bis — Agents for computer vision

Synthetic metal plates (scratches, dents, stains, missing screws, bad photos) with exact ground
truth. VLM basics → the VLM scored as a classifier → classic CV tools (quality check, golden-sample
difference, YOLO, CLIP) → an agent orchestrating CV specialists + the VLM on zoomed crops → CLIP
image search → inspection pipeline (code first, VLM on crops, agent only for doubtful cases) →
vision tools as an MCP server.

**Key idea**: VLMs know *what*, classic CV knows *where* and *how much*. Combine them, and keep
the agent for the cases nobody else could decide.

### Pillar 4 — Fine-tuning and open models

Ollama and local models, constrained JSON decoding → quantization by hand (why block-wise scales
matter) → LoRA from scratch in PyTorch (params, forgetting, merging) → building an SFT dataset →
**always measure prompting first** (baselines) → notebook: LoRA fine-tune of Qwen3.5-0.8B with
Unsloth on free Colab, GGUF export → serve the fine-tune in Ollama and compare everything →
notebook: fine-tune the same model as a VLM on the plates.
Task: messy French/English operator notes → strict JSON with "house rules" no general model knows.

**Key idea**: fine-tuning teaches behavior and conventions, not facts (those go in RAG). It wins
on cost, latency, privacy and volume once prompting has been measured and found lacking.

### Pillar 5 — Production and security

The pillar 2 assistant as a real service (FastAPI): API keys, rate limiting, streaming, exact and
semantic cache, model router, timeouts/retries/fallback, cost per request, structured logs and
metrics, input/context/output guardrails, offline mode, 18 tests, Docker, CI.
Lessons: client calls → load testing (p50/p95/p99) → cost and routing (and the semantic cache trap)
→ **prompt injection** against an agent with an email tool, 3 defense levels → red-teaming the
guardrails → observability from the logs.

**Key idea**: structural limits (what tools can do) protect; prompt hardening only lowers odds.
Test the service the way it will be used: the load test found a real deadlock.

### Pillar 6 — Vision foundation models and document AI

Open-vocabulary detection with YOLOE (text and visual prompts) → SAM 2 masks and defect area in
mm² → PatchCore from scratch (trained on good parts only, robust to misalignment) → synthetic
invoices (PDFs + scans, exact ground truth) → pypdf vs Docling → 3 extraction strategies (text
LLM, Docling + LLM, VLM on the image) scored with arithmetic validation → invoice pipeline with
routing, retry and a human review queue.

**Key idea**: foundation models remove the "no labels yet" barrier (detect by name, segment by
click, learn from good parts only). In document AI, validation rules matter as much as the model:
the metric to minimize is "accepted but wrong".

## Dependency map

```
pillar1      standalone
pillar2      standalone
pillar3      step 3 uses pillar2's retriever (run pillar2/06_ingest.py first)
pillar3bis   reuses pillar3's Agent and chat()
pillar4      notebook 08 uses pillar3bis/data/make_plates.py
pillar5      standalone (copies pillar2's docs)
pillar6      reuses pillar3bis plates, generator and cv_tools
inspection-copilot   standalone repo
```

## Shared conventions

- **Numbered scripts**, one concept each, with comments explaining what to look for in the output.
- **Synthetic data with exact ground truth** (team docs, plates, operator notes, invoices) so every
  method is scored, not eyeballed.
- **Offline / fake modes** for tests and load tests without spending API quota
  (`FAKE_LLM=1` in pillar 5, `COPILOT_FAKE_VLM=1` in the copilot).
- **Outputs** go to each pillar's `outputs/` folder (reports, charts, eval results).

## Testing status

Everything was written and checked in an environment without GPU and without access to Hugging
Face or the Groq API. What that means for each pillar:

- **Ran for real there**: the classic CV parts, YOLO/YOLOE, SAM 2, chunking/BM25/hybrid search
  logic, quantization and LoRA demos, the FastAPI services (pillar 5 and the copilot) including
  load tests, all test suites, and every script compiled under Python 3.11.
- **Tested with fake backends**: everything that calls Groq, Ollama or Docling (the logic, loops,
  scoring and reports were exercised with scripted fake models).
- **Not run**: the Colab notebooks' training cells (no GPU), real embedding/reranker/CLIP/ImageNet
  weights, Docling's models, Docker builds. First runs on your machine may need small fixes;
  the READMEs point to the official references for the fastest-moving libraries.

## Portfolio project: Inspection Copilot

[`inspection-copilot/`](inspection-copilot/) — visual inspection for industrial parts on MVTec AD:
quality gate, PatchCore anomaly detection, defect regions and masks, VLM defect naming,
inspection history, REST API + web UI, benchmark evaluation, tests, CI, Docker.

- [x] Phase 1 — MVP (pillars 3bis, 5, 6)
- [ ] Phase 2 — assistant over the inspection history (SQL tool) and quality procedures (RAG),
      with guardrails (pillars 2, 3, 5)
- [ ] Phase 3 — fine-tuned small VLM for defect naming, public demo on Hugging Face Spaces
      (pillar 4)

Next candidate: an ML monitoring agent (MLflow registry, drift on a real YOLO, incident reports
over MCP), building on pillars 1, 3 and 5.

## What this covers

| Skill | Where |
|---|---|
| Prompt engineering, structured outputs, function calling | 1 |
| Embeddings, vector databases, chunking, hybrid search, reranking | 2 |
| RAG evaluation (retrieval metrics, LLM-as-judge) | 2, 5 |
| Agent design, planning, memory, LangGraph, MCP | 3, 3bis |
| Vision-language models, CLIP, multimodal retrieval | 3bis, 6 |
| Local inference, quantization, LoRA/SFT, distillation | 4 |
| APIs, caching, routing, cost, observability, load testing, Docker, CI | 5, copilot |
| Prompt injection, guardrails, red-teaming, least privilege | 5 |
| Open-vocabulary detection, SAM, anomaly detection, document AI | 6, copilot |
