"""
Lesson 3 - cost: model routing and caching, measured.

Part A (live, calls the API): the 12 eval questions from pillar 2, with model=small, big and auto.
    -> correctness, cost per question, latency. Does the router save money without losing answers?
Part B (offline, no LLM): semantic cache threshold. Pairs of questions that SHOULD share an answer
    (paraphrases) and pairs that should NOT (same wording, different entity). Where's the threshold?
Part C: back-of-the-envelope monthly cost.

Usage:
    uvicorn app.main:app --port 8000      (real LLM, in another terminal)
    python 03_cost_and_routing.py
    python 03_cost_and_routing.py --offline   (only parts B and C)
"""
import argparse
import json
import time

import numpy as np

from app.config import ROOT, settings
from app.llm import cost, route
from app.retrieval import embedding_function

parser = argparse.ArgumentParser()
parser.add_argument("--offline", action="store_true")
args = parser.parse_args()

questions = json.loads((ROOT / "data" / "eval_questions.json").read_text(encoding="utf-8"))

# ---------------------------------------------------------------------------
# A. routing
# ---------------------------------------------------------------------------
print("=" * 90 + "\nA. Router decisions (heuristic in app/llm.py)\n")
for q in questions[:6]:
    print(f"  {route(q['question']).split('/')[-1]:<14} {q['question']}")

if not args.offline:
    import httpx

    client = httpx.Client(base_url="http://localhost:8000", headers={"X-API-Key": "dev-key-123"}, timeout=120)
    print(f"\n{'mode':<8}{'correct':>9}{'avg cost $':>13}{'avg latency':>13}  models used")
    for mode in ["small", "big", "auto"]:
        correct, costs, lats, models = 0, [], [], {}
        for q in questions:
            body = client.post("/ask", json={"question": q["question"], "model": mode, "use_cache": False}).json()
            if "answer" not in body:
                print(f"   error: {body}")
                continue
            correct += all(s.lower() in body["answer"].lower() for s in q["must_contain"])
            costs.append(body["cost_usd"])
            lats.append(body["latency_ms"])
            models[body["model"]] = models.get(body["model"], 0) + 1
            time.sleep(0.5)  # gentle with the free tier
        n = len(costs) or 1
        print(f"{mode:<8}{correct:>5}/{len(questions):<3}{sum(costs) / n:>13.6f}{sum(lats) / n:>11.0f}ms  {models}")
    # "correct" = the expected key fact appears in the answer: crude but cheap (see pillar 2 step 9)

# ---------------------------------------------------------------------------
# B. semantic cache threshold
# ---------------------------------------------------------------------------
print("\n" + "=" * 90 + "\nB. Semantic cache: when is 'similar' similar enough?\n")
SAME = [  # a cached answer would be correct
    ("How long does the A/B test last?", "What's the duration of the A/B test?"),
    ("Who owns efficientnet?", "Who is the owner of EfficientNet?"),
    ("How do I roll back a model?", "What's the command to roll back a model?"),
    ("Where are drift alerts posted?", "In which Slack channel do drift alerts go?"),
]
DIFFERENT = [  # a cached answer would be WRONG
    ("Who owns efficientnet?", "Who owns resnet50?"),
    ("What PSI value means drift?", "What PSI value means warning?"),
    ("Can I deploy on Friday afternoon?", "Can I deploy on Monday afternoon?"),
    ("How long does the A/B test last?", "How long does the canary last?"),
]
embed = embedding_function()


def sim(a, b):
    va, vb = (np.asarray(v, dtype=np.float32) for v in embed([a, b]))
    return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb)))


same_s = [sim(a, b) for a, b in SAME]
diff_s = [sim(a, b) for a, b in DIFFERENT]
for (a, b), s in zip(SAME, same_s):
    print(f"  same      {s:.3f}  {a!r} ~ {b!r}")
for (a, b), s in zip(DIFFERENT, diff_s):
    print(f"  DIFFERENT {s:.3f}  {a!r} ~ {b!r}")
print(f"\n{'threshold':>10}{'useful hits':>13}{'wrong answers':>15}")
for t in [0.80, 0.85, 0.90, 0.95, 0.98]:
    print(f"{t:>10}{sum(s >= t for s in same_s):>9}/{len(SAME)}{sum(s >= t for s in diff_s):>11}/{len(DIFFERENT)}")
print(f"(current SEMANTIC_CACHE_THRESHOLD = {settings.semantic_cache_threshold})")
# Usually there's no threshold that catches all paraphrases without serving some wrong answers:
# "who owns resnet50" and "who owns efficientnet" can be closer than two true paraphrases.
# Safer variants: exact cache only, semantic cache only for FAQ-like questions, or check that
# the retrieved documents are the same before reusing an answer.

# ---------------------------------------------------------------------------
# C. monthly cost
# ---------------------------------------------------------------------------
print("\n" + "=" * 90 + "\nC. Monthly cost estimate\n")
users, q_per_day, days = 200, 10, 22
prompt_tokens, completion_tokens = 1300, 150  # typical for this service: system + 4 chunks + question
monthly = users * q_per_day * days
for name, model in [("all big", settings.model_big), ("all small", settings.model_small)]:
    print(f"  {name:<10} {monthly:,} questions/month -> ${monthly * cost(model, prompt_tokens, completion_tokens):,.2f}")
mix = monthly * (0.7 * cost(settings.model_small, prompt_tokens, completion_tokens)
                 + 0.3 * cost(settings.model_big, prompt_tokens, completion_tokens))
print(f"  router 70/30 + 25% cache hits -> ${mix * 0.75:,.2f}")
print("""
  At these prices the LLM bill is small; the expensive parts of an LLM product are usually
  people, evaluation and incidents. Cost matters at scale, with big models, or with agents
  (10-50 calls per task). Always compute it before choosing: it's 3 lines of arithmetic.""")
