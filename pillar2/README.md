# Pillar 2 - RAG (Retrieval Augmented Generation)

Learning RAG step by step, from "what is an embedding" to a paper assistant with hybrid
search, reranking, citations and an evaluation script. Everything runs locally except the
LLM calls (Groq free tier). Total cost: $0.

## Setup

```
cd d:\pythoncode\pillar2
conda activate first-env
pip install -r requirements.txt
```

The `.env` with `GROQ_API_KEY` in `d:\pythoncode` is picked up automatically.

**Model note:** Groq shut down `llama-3.3-70b-versatile` in August 2026. The scripts default
to `openai/gpt-oss-120b`, its recommended replacement. To use another model, add a line to
`.env`, e.g. `GROQ_MODEL=qwen/qwen3.6-27b`.

First run downloads two small local models (~90MB each, cached afterwards):
`all-MiniLM-L6-v2` (embeddings) and `ms-marco-MiniLM-L-6-v2` (reranker, from step 8).

Put a few PDFs in `data/papers/` before step 4 (suggestions in `data/papers/README.txt`).

## Layout

```
pillar2/
  rag_utils.py        shared config + helpers (LLM, loading, chunking, vector store)
  retrieval.py        vector / keyword (BM25) / hybrid / rerank retriever
  01..09_*.py         one concept per script, run them in order
  10_project_*.py     the mini project
  data/
    docs/             4 fake internal docs of our CV team (registry, runbook, drift policy, postmortem)
    papers/           your PDFs
    eval_questions.json   12 questions with expected answers, used in steps 5 and 9
  chroma_db/          created automatically, the vector store on disk
```

## Steps

| # | Script | What you learn |
|---|---|---|
| 1 | `01_embeddings.py` | Text -> vector, cosine similarity, limits (negation, language) |
| 2 | `02_chroma_basics.py` | Vector DB: store, query, upsert, metadata filters |
| 3 | `03_rag_first_version.py` | Retrieval + generation, saying "I don't know" |
| 4 | `04_loading_documents.py` | What PDF extraction really gives you, cleaning |
| 5 | `05_chunking_strategies.py` | 4 chunking strategies compared with a hit-rate metric |
| 6 | `06_ingest.py` | Indexing pipeline with metadata and incremental updates |
| 7 | `07_chat_with_docs.py` | Citations, conversation memory, query rewriting |
| 8 | `08_better_retrieval.py` | BM25, hybrid search (RRF), cross-encoder reranking |
| 9 | `09_evaluation.py` | hit@k, MRR, LLM-as-judge (correct / faithful) |
| 10 | `10_project_paper_assistant.py` | Everything together + structured summaries |

Steps 1-6 and 8 make no LLM calls. 9 only calls the LLM with `--judge`.

## What about LangChain / LlamaIndex?

Not used on purpose. They wrap exactly these steps (loaders, splitters, vector stores,
retrievers, chains) and you'll meet them in pillar 3. Having written each piece yourself,
you'll know what they do under the hood and how to debug them when they misbehave,
which is most of the job.

## Common issues

- `RateLimitError`: free tier limit, `ask_llm` retries automatically. If it keeps failing,
  wait a minute or run 09 without `--judge`.
- Empty answers or `model_decommissioned` error: set `GROQ_MODEL` in `.env` to a model listed
  on console.groq.com/docs/models.
- Weird results after changing chunk size: run `python 06_ingest.py --reset`.
