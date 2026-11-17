"""
Lesson 2 - load testing: what happens with many users at once?

Sends N requests with C running concurrently and reports throughput, latency percentiles
(p50 = typical user, p95/p99 = the unlucky ones) and status codes.

Run it against the OFFLINE mode first (no quota spent, the fake LLM takes ~0.3 s per call):
    set FAKE_LLM=1                                   (Windows cmd)
    set RATE_LIMIT_PER_MINUTE=100000
    uvicorn app.main:app --port 8000 --workers 1
    python 02_load_test.py --n 200 --concurrency 20

Then try: --concurrency 1 vs 50, --workers 4, RATE_LIMIT_PER_MINUTE=30 (see the 429s),
and --same-question (everything served from cache).
Against the real LLM, keep --n small: the Groq free tier limits will be the bottleneck.
"""
import argparse
import asyncio
import random
import time
from collections import Counter

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://localhost:8000")
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--concurrency", type=int, default=10)
parser.add_argument("--same-question", action="store_true")
args = parser.parse_args()

QUESTIONS = [
    "How long does the A/B test last?", "Who owns efficientnet?", "What PSI value means drift?",
    "How do I roll back a model?", "Can I deploy on Friday afternoon?", "What caused the YOLOv8 incident?",
    "Where are drift alerts posted?", "What is the latency budget?", "Why is ViT Large not deployed?",
]


async def worker(client, queue, results):
    while True:
        try:
            i = queue.get_nowait()
        except asyncio.QueueEmpty:
            return
        # --same-question: everything should come from the cache. Otherwise the cache is turned off:
        # making each question unique isn't enough, "X (#12)" and "X (#187)" are near-identical for
        # the SEMANTIC cache (first version of this script did that, and measured the cache)
        q = QUESTIONS[0] if args.same_question else random.choice(QUESTIONS)
        start = time.perf_counter()
        try:
            r = await client.post("/ask", json={"question": q, "use_cache": args.same_question})
            results.append((r.status_code, time.perf_counter() - start))
        except httpx.HTTPError as e:
            results.append((type(e).__name__, time.perf_counter() - start))


async def main():
    queue = asyncio.Queue()
    for i in range(args.n):
        queue.put_nowait(i)
    results = []
    async with httpx.AsyncClient(base_url=args.url, headers={"X-API-Key": "dev-key-123"}, timeout=120) as client:
        health = (await client.get("/health")).json()
        print(f"server: {health}")
        start = time.perf_counter()
        await asyncio.gather(*(worker(client, queue, results) for _ in range(args.concurrency)))
        total = time.perf_counter() - start

    ok = sorted(t for s, t in results if s == 200)
    pct = lambda p: ok[min(len(ok) - 1, int(p * len(ok)))] * 1000 if ok else float("nan")
    print(f"\n{args.n} requests, concurrency {args.concurrency}, {total:.1f}s -> {args.n / total:.1f} req/s")
    print(f"status codes: {dict(Counter(s for s, _ in results))}")
    print(f"latency of successful requests: p50 {pct(0.5):.0f} ms | p95 {pct(0.95):.0f} ms | p99 {pct(0.99):.0f} ms")


asyncio.run(main())

# What you'll typically see with FAKE_LLM (0.3 s per call):
#  - concurrency 1: ~3 req/s, p50 ~ 300 ms
#  - concurrency 20, 1 worker: throughput goes up (FastAPI runs sync endpoints in a thread pool,
#    40 threads by default), latency stays OK... until the pool is full and requests queue up:
#    p95 climbs while p50 looks fine. That's why you watch p95, not the average.
#  - --same-question: the cache answers in a few ms, throughput jumps 10x
#  - --workers 4: more processes = more parallelism. But each has its OWN cache and rate limiter
#    (in memory), so cache hit rate drops and the rate limit is 4x looser. -> Redis in production.
#  - with the real LLM, the API provider's rate limit is the real ceiling: ask yourself how many
#    users the free tier would support, and what the paid tier would cost (lesson 3).
