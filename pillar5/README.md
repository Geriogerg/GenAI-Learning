# Pillar 5 - Production and security

The pillar 2 RAG assistant turned into a service another team could actually use: an HTTP API
with authentication, rate limiting, streaming, caching, model routing, cost tracking,
structured logs, guardrails, tests, CI and a Docker image. Plus the security lessons that
matter most for LLM apps: prompt injection and how to contain it.

## Setup

```
cd d:\pythoncode\pillar5
conda activate first-env
pip install -r requirements-dev.txt
```

`requirements.txt` = what the service needs (goes in the Docker image).
`requirements-dev.txt` = that + what the lessons and tests need.

## Run the service

```
uvicorn app.main:app --port 8000 --reload
```

Then open http://localhost:8000/docs: an interactive API page FastAPI generates from the code.
The first start downloads a small ONNX embedding model (~80 MB).

**Offline mode** (no API calls, for load tests and development), in Windows cmd:
```
set FAKE_LLM=1
set RATE_LIMIT_PER_MINUTE=100000
uvicorn app.main:app --port 8000
```

## Layout

```
app/
  main.py         FastAPI app: /ask, /ask/stream, /health, /metrics, auth, rate limit
  llm.py          Groq calls: timeout, retries, fallback, cost, streaming, router, fake mode
  retrieval.py    index built at startup (Chroma + ONNX embeddings, no PyTorch)
  cache.py        exact cache + semantic cache
  guardrails.py   input / context / output checks
  telemetry.py    JSON log line per request + live metrics
  config.py       every setting from environment variables
tests/            pytest, offline (fake LLM + fake embeddings)
data/docs/        the team docs from pillar 2
data/untrusted/   a vendor FAQ with a hidden prompt injection (lessons 4-5)
Dockerfile, docker-compose.yml, .github/workflows/ci.yml
```

## Lessons

| # | Script | What you learn | Needs |
|---|---|---|---|
| 1 | `01_call_the_api.py` | The API from a client: auth, cache hits, guardrails, errors, streaming, metrics | server |
| 2 | `02_load_test.py` | Throughput, p50/p95/p99, thread pool limits, rate limiting, workers | server (offline mode) |
| 3 | `03_cost_and_routing.py` | Small vs big model vs router, semantic cache threshold, monthly cost | server (part A) |
| 4 | `04_prompt_injection.py` | Indirect injection against an agent with an email tool, 3 defense levels | Groq |
| 5 | `05_red_team_eval.py` | Attack catch rate vs false positives of the guardrails | - (or `--live`) |
| 6 | `06_observability.py` | Latency per stage, cost per model, flags, slowest requests, one request by id | logs |

Then: `pytest -q` (18 tests, ~5 s, no network) and the Docker section below.

## Docker

Install Docker Desktop (Windows), then:
```
docker compose up --build
```
Same API on http://localhost:8000, with your `GROQ_API_KEY` read from `d:\pythoncode\.env`.
The image runs as a non-root user, has a healthcheck, and downloads the embedding model at
build time so the first request isn't slow.

To deploy it for real, any platform that runs a container works (a VM with Docker, Azure
Container Apps, Google Cloud Run, AWS App Runner, Hugging Face Spaces with the Docker SDK...).
Put the API key in the platform's secret manager, never in the image.

## CI

`.github/workflows/ci.yml` runs the tests and builds the image on every push, once this folder
is pushed as a GitHub repository. Tests use the fake LLM, so no secret is needed in CI.

## What I'd still add before real users

- **Redis** for the cache and the rate limiter (they're in memory, so per process right now)
- a real **vector DB** server and a separate ingestion job instead of indexing at startup
- **SSO / OAuth** instead of static API keys if humans use it directly
- an ML **prompt-injection classifier** in addition to the regexes (lesson 5 shows why)
- **evals in CI**: the pillar 2 eval questions run against every change of prompt or model
- **alerts** on p95 latency, error rate, daily cost and blocked-request spikes
- a **feedback button** (thumbs up/down with the request_id) that feeds the eval set

## A real bug, kept in on purpose

The first version deadlocked under load: when a client disconnected in the middle of a stream,
Python's garbage collector closed the abandoned generator while the same thread held the metrics
lock, and its cleanup tried to take the lock again. Every later request got stuck behind it. Found
with the load test, diagnosed with `py-spy dump`, fixed with a re-entrant lock and explicit
disconnect handling (`app/telemetry.py`, `app/main.py`), and covered by a regression test.
This is the kind of bug that only shows up when you test the service like it will be used.
