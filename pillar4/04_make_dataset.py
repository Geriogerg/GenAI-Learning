"""
Step 4 - building the fine-tuning dataset.

The data is the real work in fine-tuning. Here we generate operator notes with their exact
JSON records (see notes_task.py), split them, and save them in the chat format that
SFT (supervised fine-tuning) expects: one conversation per line.

    data/train.jsonl  1500 examples   -> the model learns from these
    data/val.jsonl     100 examples   -> watched during training (overfitting?)
    data/test.jsonl    200 examples   -> touched only for the final comparison

Optional --paraphrase N: ask the big Groq model to rewrite N training notes in a different
style. That's "LLM-generated synthetic data" (a teacher model producing data for a student),
widely used in practice. Labels stay the ground truth, only the wording changes.

Usage:
    python 04_make_dataset.py
    python 04_make_dataset.py --paraphrase 100
"""
import argparse
import json
import random
from collections import Counter

from llm_clients import DATA_DIR, groq_chat
from notes_task import make_examples, to_chat

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=int, default=1500)
parser.add_argument("--paraphrase", type=int, default=0)
args = parser.parse_args()

# different seeds per split: no identical notes leaking from train to test
train = make_examples(args.train, seed=1)
val = make_examples(100, seed=2)
test = make_examples(200, seed=3)

train_notes = {ex["note"] for ex in train}
leaks = sum(ex["note"] in train_notes for ex in test)
print(f"train {len(train)} / val {len(val)} / test {len(test)}  (test notes also in train: {leaks})")

if args.paraphrase:
    PROMPT = ("Rewrite this production-line operator note as a different operator would write it: change the "
              "wording, order and abbreviations, keep the same language mix. Keep EVERY fact (line, station, shift, "
              "defect, severity, quantity, action if present). Don't add facts. Return only the new note.")
    rng = random.Random(0)
    for ex in rng.sample(train, args.paraphrase):
        new, _ = groq_chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": ex["note"]}],
                           temperature=0.9, max_tokens=600)
        print(f"  {ex['note']}\n  -> {new.strip()}")
        ex["note"] = new.strip()
    # A teacher can also break things (drop the quantity, change the severity...). In a real
    # project you'd check the paraphrases, e.g. by having the model re-extract the record and
    # keeping only notes where it matches the label.

DATA_DIR.mkdir(exist_ok=True)
for name, split in [("train", train), ("val", val), ("test", test)]:
    with (DATA_DIR / f"{name}.jsonl").open("w", encoding="utf-8") as f:
        for ex in split:
            f.write(json.dumps(to_chat(ex), ensure_ascii=False) + "\n")

# a quick look at the label distribution: an unbalanced dataset trains an unbalanced model
for field in ["defect_code", "severity", "action"]:
    counts = Counter(ex["label"][field] for ex in train)
    print(f"{field:<12} " + "  ".join(f"{k}:{v}" for k, v in counts.most_common()))

print(f"\nOne training example:\n{json.dumps(to_chat(train[0]), indent=2, ensure_ascii=False)}")
print(f"\nSaved to {DATA_DIR}. Upload train/val/test.jsonl + notes_task.py to Colab for notebook 06.")
