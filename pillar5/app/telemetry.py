"""
Observability: one structured JSON log line per request + live aggregates for /metrics.

Every line has a request_id, timings per stage, model, tokens, cost, cache hit and guardrail
flags. With that you can answer the questions that matter in production: why was this
request slow? what does the service cost per day? how often are guardrails triggered?
Tools like Langfuse, LangSmith or OpenTelemetry + Grafana do the same with a UI.
"""
import json
import threading
import time
from contextlib import contextmanager

from app.config import settings


class Timer:
    def __init__(self):
        self.stages = {}

    @contextmanager
    def stage(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages[name] = round((time.perf_counter() - start) * 1000, 1)  # ms


class Metrics:
    def __init__(self):
        # RLock, not Lock: when a client disconnects in the middle of a stream, the abandoned
        # generator is closed by the garbage collector at an arbitrary moment, possibly while this
        # same thread already holds the lock. Its cleanup logs the request -> with a plain Lock the
        # thread waits for itself forever and every later request piles up behind it.
        # (Found during the load test of lesson 2, after interrupting a streaming request.)
        self.lock = threading.RLock()
        self.latencies, self.count, self.errors, self.blocked, self.cache_hits = [], 0, 0, 0, 0
        self.cost_usd, self.tokens = 0.0, 0
        self.by_model = {}

    def record(self, rec):
        with self.lock:
            self.count += 1
            self.latencies = (self.latencies + [rec["latency_ms"]])[-1000:]
            self.errors += rec["status"] >= 500
            self.blocked += rec.get("blocked", False)
            self.cache_hits += bool(rec.get("cache"))
            self.cost_usd += rec.get("cost_usd", 0.0)
            self.tokens += rec.get("prompt_tokens", 0) + rec.get("completion_tokens", 0)
            if rec.get("model"):
                self.by_model[rec["model"]] = self.by_model.get(rec["model"], 0) + 1

    def snapshot(self):
        with self.lock:
            lat = sorted(self.latencies)
            pct = lambda p: lat[min(len(lat) - 1, int(p * len(lat)))] if lat else None
            return {"requests": self.count, "errors": self.errors, "blocked": self.blocked,
                    "cache_hit_rate": round(self.cache_hits / self.count, 3) if self.count else 0,
                    "latency_ms_p50": pct(0.5), "latency_ms_p95": pct(0.95),
                    "total_cost_usd": round(self.cost_usd, 6), "total_tokens": self.tokens,
                    "requests_by_model": dict(self.by_model)}


metrics = Metrics()
_file_lock = threading.RLock()


def log_request(rec):
    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **rec}
    metrics.record(rec)
    settings.log_file.parent.mkdir(parents=True, exist_ok=True)
    with _file_lock, settings.log_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
