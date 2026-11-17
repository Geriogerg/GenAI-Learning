"""
Step 5 - baselines: how far do we get WITHOUT fine-tuning?

Always measure this first. Fine-tuning costs time, data and maintenance; if a better prompt
solves the problem, you're done. We evaluate on the same test notes:

    big model (Groq gpt-oss-120b)    basic prompt  |  prompt with the house rules
    small local model (qwen3.5:0.8b) basic prompt  |  prompt with the house rules
                                     (local runs use JSON-schema constrained decoding)

Notebook 06 then fine-tunes the small model with the BASIC prompt, and 07 compares everything.

Usage:
    python 05_baselines.py            # 60 test notes per config (Groq free tier friendly)
    python 05_baselines.py --n 200    # full test set
    python 05_baselines.py --skip-groq / --skip-local
"""
import argparse
import json
import time

from llm_clients import DATA_DIR, GROQ_MODEL, LOCAL_MODEL, OUTPUT_DIR, check_ollama, groq_chat, ollama_chat
from notes_task import FIELDS, JSON_SCHEMA, RULES_PROMPT, SYSTEM_PROMPT, parse_json, score, summarize

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=60)
parser.add_argument("--skip-groq", action="store_true")
parser.add_argument("--skip-local", action="store_true")
args = parser.parse_args()

test = [json.loads(line) for line in (DATA_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines()][:args.n]


def evaluate(name, call, system_prompt):
    scores, latencies, errors = [], [], []
    for ex in test:
        note = ex["messages"][1]["content"]
        text, secs = call([{"role": "system", "content": system_prompt}, {"role": "user", "content": note}])
        s = score(parse_json(text), ex["label"])
        scores.append(s)
        latencies.append(secs)
        if not s["all"] and len(errors) < 3:
            errors.append((note, text.strip()[:200], ex["label"]))
    summary = summarize(scores)
    summary["avg_latency_s"] = round(sum(latencies) / len(latencies), 2)
    print(f"\n{name}: all fields right {summary['all']:.0%} | action {summary['action']:.0%} | "
          f"valid JSON {summary['valid_json']:.0%} | {summary['avg_latency_s']}s/note")
    for note, out, label in errors:
        print(f"   note:  {note}\n   model: {out}\n   truth: {json.dumps(label, ensure_ascii=False)}\n")
    return summary


results = {}
if not args.skip_groq:
    def big(m):
        out = groq_chat(m, json_mode=True)
        time.sleep(0.3)
        return out
    results[f"{GROQ_MODEL} basic"] = evaluate(f"{GROQ_MODEL}, basic prompt", big, SYSTEM_PROMPT)
    results[f"{GROQ_MODEL} +rules"] = evaluate(f"{GROQ_MODEL}, prompt with rules", big, RULES_PROMPT)

if not args.skip_local:
    check_ollama()
    small = lambda m: ollama_chat(m, schema=JSON_SCHEMA)
    results[f"{LOCAL_MODEL} basic"] = evaluate(f"{LOCAL_MODEL}, basic prompt", small, SYSTEM_PROMPT)
    results[f"{LOCAL_MODEL} +rules"] = evaluate(f"{LOCAL_MODEL}, prompt with rules", small, RULES_PROMPT)

print("\n" + "=" * 100)
print(f"{'config':<34}{'all':>6}" + "".join(f"{f[:8]:>10}" for f in FIELDS) + f"{'s/note':>8}")
for name, s in results.items():
    print(f"{name:<34}{s['all']:>6.0%}" + "".join(f"{s[f]:>10.0%}" for f in FIELDS) + f"{s['avg_latency_s']:>8}")

OUTPUT_DIR.mkdir(exist_ok=True)
out = OUTPUT_DIR / "baselines.json"
previous = json.loads(out.read_text(encoding="utf-8")) if out.exists() else {}
out.write_text(json.dumps({**previous, **results}, indent=1), encoding="utf-8")
print(f"\nSaved to {out} (07 reads it)")

# What to look for:
#  - with the basic prompt, even the big model gets "action" wrong a lot: it can't guess
#    the house rules. Look at its errors, they're reasonable but not OUR convention.
#  - with the rules written in the prompt, the big model should jump up. That's the lesson:
#    try prompting first. If the big model + rules is good enough and affordable, stop here.
#  - the small model struggles even with the rules (following 6 ordered rules is hard at 0.8B).
#  - question for notebook 06: can fine-tuning the SMALL model with the BASIC prompt beat all
#    of this, at a fraction of the cost and latency?
