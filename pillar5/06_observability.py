"""
Lesson 6 - observability: what is the service actually doing?

Reads logs/requests.jsonl (one line per request, written by app/telemetry.py) and answers
the questions you'd get in a weekly review or an incident:
  - how many requests, errors, blocked requests?
  - where does the time go (retrieval vs LLM vs guardrails)?
  - what does it cost, per model, per day?
  - which guardrails fire, and on what?
  - what were the slowest requests?
Saves a latency chart to logs/latency.png.

Run lessons 1-3 first so there's something in the logs.
Usage: python 06_observability.py [--request-id abc123]
"""
import argparse
import json
from collections import Counter, defaultdict

from app.config import settings

parser = argparse.ArgumentParser()
parser.add_argument("--request-id", help="show the full log line of one request")
args = parser.parse_args()

if not settings.log_file.exists():
    raise SystemExit(f"No log file yet ({settings.log_file}). Call the API first (lesson 1).")
logs = [json.loads(line) for line in settings.log_file.read_text(encoding="utf-8").splitlines() if line.strip()]

if args.request_id:  # "a user says request 3f2a... gave a weird answer": find it in one second
    for rec in logs:
        if rec["request_id"].startswith(args.request_id):
            print(json.dumps(rec, indent=2, ensure_ascii=False))
    raise SystemExit


def pct(values, p):
    v = sorted(values)
    return v[min(len(v) - 1, int(p * len(v)))] if v else 0


print(f"{len(logs)} requests from {logs[0]['ts']} to {logs[-1]['ts']}\n")

status = Counter(r["status"] for r in logs)
print("status codes:", dict(status))
print(f"blocked by guardrails: {sum(r.get('blocked', False) for r in logs)}")
print(f"cache hits: {Counter(r.get('cache') for r in logs if r.get('cache'))}  "
      f"({sum(bool(r.get('cache')) for r in logs) / len(logs):.0%} of requests)")

ok = [r for r in logs if r["status"] == 200 and not r.get("cache")]
print(f"\nLatency of answered, non-cached requests ({len(ok)}):")
print(f"  total     p50 {pct([r['latency_ms'] for r in ok], .5):>7.0f} ms   p95 {pct([r['latency_ms'] for r in ok], .95):>7.0f} ms")
stages = defaultdict(list)
for r in ok:
    for name, ms in r.get("stages", {}).items():
        stages[name].append(ms)
for name, values in stages.items():
    print(f"  {name:<9} p50 {pct(values, .5):>7.1f} ms   p95 {pct(values, .95):>7.1f} ms")
ttft = [r["time_to_first_token_ms"] for r in ok if r.get("time_to_first_token_ms")]
if ttft:
    print(f"  streaming time-to-first-token p50 {pct(ttft, .5):.0f} ms")

print("\nCost and tokens by model:")
by_model = defaultdict(lambda: [0, 0, 0.0])
for r in logs:
    if r.get("model") and not r.get("cache"):
        m = by_model[r["model"]]
        m[0] += 1
        m[1] += r.get("prompt_tokens", 0) + r.get("completion_tokens", 0)
        m[2] += r.get("cost_usd", 0.0)
for model, (n, tokens, usd) in by_model.items():
    print(f"  {model:<24} {n:>5} calls {tokens:>9,} tokens  ${usd:.5f}  (${usd / n * 1000:.3f} per 1000 requests)")

flags = Counter(f.split(":")[0] for r in logs for f in r.get("flags", []))
print(f"\nGuardrail flags: {dict(flags) or 'none'}")
for r in logs:
    if r.get("blocked"):
        print(f"  blocked {r['request_id']}: {r.get('question', '')[:70]!r} {r['flags']}")

print("\nSlowest requests:")
for r in sorted(ok, key=lambda r: -r["latency_ms"])[:3]:
    print(f"  {r['request_id']} {r['latency_ms']:>7.0f} ms {r.get('model')}: {r.get('question', '')[:60]!r} stages={r.get('stages')}")

errors = [r for r in logs if r["status"] >= 500]
for r in errors[:3]:
    print(f"\nerror {r['request_id']}: {r.get('error')}")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot([r["latency_ms"] for r in ok], ".", alpha=0.6)
    ax.set_xlabel("request (in time order)")
    ax.set_ylabel("latency (ms)")
    ax.set_title("Latency per request (non-cached, status 200)")
    out = settings.log_file.parent / "latency.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\nChart saved to {out}")
except ImportError:
    pass

# In production you'd ship these lines to a log platform (Grafana/Loki, Datadog, ELK...) or an LLM
# observability tool (Langfuse is open source and self-hostable) and set ALERTS, e.g.:
#   p95 latency > 10 s for 5 min, error rate > 2%, daily cost > budget, blocked requests spike.
