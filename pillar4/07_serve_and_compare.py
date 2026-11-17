"""
Step 7 - serve your fine-tuned model locally and compare it with everything else.

After notebook 06, unzip notes_finetune.zip into pillar4/outputs/, which gives:
    outputs/gguf_model/*.gguf          the fine-tuned model, merged and quantized (Q4_K_M)
    outputs/lora_adapter/              the LoRA adapter alone
    outputs/finetune_results.json      before/after scores measured in Colab

This script:
  1. registers the GGUF in Ollama as a new local model ("notes-ft")
  2. runs it on the test notes on YOUR machine (CPU is fine), with the basic prompt
  3. prints one table: big cloud model vs small model + prompt vs small fine-tuned model

Usage:
    python 07_serve_and_compare.py
    python 07_serve_and_compare.py --n 200 --gguf path/to/model.gguf
"""
import argparse
import json
import subprocess
from pathlib import Path

from llm_clients import DATA_DIR, LOCAL_MODEL, OUTPUT_DIR, check_ollama, ollama_chat
from notes_task import FIELDS, JSON_SCHEMA, SYSTEM_PROMPT, parse_json, score, summarize

FT_NAME = "notes-ft"

parser = argparse.ArgumentParser()
parser.add_argument("--gguf", default="")
parser.add_argument("--n", type=int, default=100)
parser.add_argument("--recreate", action="store_true", help="re-register the model in Ollama")
args = parser.parse_args()

check_ollama()  # also needs the base model pulled: its chat template is reused below

# --- 1. find the GGUF and register it in Ollama ---------------------------------
gguf = Path(args.gguf) if args.gguf else next(iter(sorted((OUTPUT_DIR / "gguf_model").rglob("*.gguf"))), None)
if not gguf or not gguf.exists():
    raise SystemExit("No .gguf found. Run notebook 06, unzip notes_finetune.zip into outputs/, or pass --gguf.")
print(f"GGUF: {gguf} ({gguf.stat().st_size / 1e9:.2f} GB)")

existing = subprocess.run(["ollama", "list"], capture_output=True, text=True).stdout
if FT_NAME not in existing or args.recreate:
    # A Modelfile tells Ollama how to wrap the raw weights: the chat template (must be the one
    # used in training!), stop tokens, default parameters. We borrow the base model's template.
    template = subprocess.run(["ollama", "show", LOCAL_MODEL, "--template"], capture_output=True,
                              text=True, encoding="utf-8").stdout
    modelfile = OUTPUT_DIR / "Modelfile"
    modelfile.write_text(
        f'FROM "{gguf.resolve()}"\n'
        f'TEMPLATE """{template}"""\n'
        'PARAMETER stop "<|im_end|>"\n'
        "PARAMETER temperature 0\n",
        encoding="utf-8",
    )
    print(f"Creating Ollama model '{FT_NAME}' from {modelfile} ...")
    subprocess.run(["ollama", "create", FT_NAME, "-f", str(modelfile)], check=True)

# --- 2. evaluate on this machine ------------------------------------------------
test = [json.loads(line) for line in (DATA_DIR / "test.jsonl").read_text(encoding="utf-8").splitlines()][:args.n]


def run(model, schema):
    scores, secs = [], []
    for ex in test:
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": ex["messages"][1]["content"]}]
        text, t = ollama_chat(msgs, model=model, schema=schema)
        scores.append(score(parse_json(text), ex["label"]))
        secs.append(t)
    s = summarize(scores)
    s["avg_latency_s"] = round(sum(secs) / len(secs), 2)
    return s


results = {}
print(f"\nEvaluating {FT_NAME} on {len(test)} test notes (local) ...")
results["notes-ft local, free JSON"] = run(FT_NAME, None)
results["notes-ft local, JSON schema"] = run(FT_NAME, JSON_SCHEMA)

# --- 3. one table with everything ------------------------------------------------
for f in ["baselines.json", "finetune_results.json"]:
    path = OUTPUT_DIR / f
    if path.exists():
        results = {**json.loads(path.read_text(encoding="utf-8")), **results}
    else:
        print(f"(no {f} yet, run {'05_baselines.py' if f.startswith('base') else 'notebook 06'} to include it)")

print("\n" + "=" * 110)
print(f"{'config':<44}{'all':>6}{'json':>6}" + "".join(f"{f[:7]:>9}" for f in FIELDS) + f"{'s/note':>8}")
for name, s in results.items():
    lat = s.get("avg_latency_s", s.get("sec_per_note", "-"))
    print(f"{name:<44}{s['all']:>6.0%}{s['valid_json']:>6.0%}" + "".join(f"{s[f]:>9.0%}" for f in FIELDS) + f"{lat:>8}")

(OUTPUT_DIR / "comparison.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

print("""
Reading the table:
  - the fine-tuned 0.8B vs the big model with the basic prompt: the small model learned the
    house rules from examples, the big one can't guess them.
  - vs the big model WITH the rules in the prompt: closer. If that's good enough, prompting wins
    (no training, no model to maintain). Fine-tuning wins on cost, latency, privacy and volume.
  - Colab (16-bit) vs local GGUF (4-bit): if the local score is clearly lower, quantization hurt.
    Try a q8_0 export in notebook 06 and compare.
  - local latency is your CPU: compare it with the API latency, and remember there's no rate limit
    and no per-token cost locally.""")
