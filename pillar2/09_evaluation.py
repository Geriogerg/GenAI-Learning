"""
Step 9 - evaluating a RAG system.

"It looks good on the 3 questions I tried" is how most RAG demos are validated, and why
many of them fall apart in production. You already do this for CV models (test set, metrics),
RAG needs the same discipline. Two parts:

1. Retrieval metrics (free, no LLM call):
     hit@k  - is a chunk containing the answer in the top k?
     MRR    - mean of 1/rank of the first good chunk (1.0 = always first)
   Ground truth comes from data/eval_questions.json: the expected source file + strings
   the right chunk must contain.

2. Answer quality, with an LLM as judge (--judge, uses the API, ~2 calls per question):
     correct  - does the answer agree with the reference answer?
     faithful - is everything in the answer supported by the retrieved context?
                (a correct answer that isn't in the context = the model used its own
                knowledge, that's a hallucination risk even when it happens to be right)

Usage:
    python 09_evaluation.py              # retrieval metrics only
    python 09_evaluation.py --judge      # + answer quality (hybrid+rerank retrieval)
Run 06_ingest.py first.
"""
import argparse
import json
import time

from rag_utils import EVAL_FILE, answer_with_context, ask_llm
from retrieval import METHODS, Retriever

parser = argparse.ArgumentParser()
parser.add_argument("--judge", action="store_true", help="also grade generated answers with the LLM")
parser.add_argument("--k", type=int, default=5)
args = parser.parse_args()

questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))
retriever = Retriever()


def first_relevant_rank(hits, item):
    for rank, h in enumerate(hits, start=1):
        if h["metadata"]["source"] == item["source"] and all(
            s.lower() in h["text"].lower() for s in item["must_contain"]
        ):
            return rank
    return None


# ---------------------------------------------------------------------------
# 1. retrieval
# ---------------------------------------------------------------------------

print(f"Retrieval on {len(questions)} questions (k={args.k})\n")
print(f"{'method':<16}{'hit@1':>8}{'hit@3':>8}{'hit@' + str(args.k):>8}{'MRR':>8}")

misses = {}
for method in METHODS:
    ranks = []
    for item in questions:
        hits = retriever.search(item["question"], k=args.k, method=method)
        ranks.append(first_relevant_rank(hits, item))

    n = len(questions)
    hit = lambda k: sum(1 for r in ranks if r is not None and r <= k) / n
    mrr = sum(1 / r for r in ranks if r is not None) / n
    print(f"{method:<16}{hit(1):>8.0%}{hit(3):>8.0%}{hit(args.k):>8.0%}{mrr:>8.2f}")
    misses[method] = [q["question"] for q, r in zip(questions, ranks) if r is None]

print("\nQuestions where the answer wasn't retrieved at all:")
for method, qs in misses.items():
    if qs:
        print(f"  {method}:")
        for q in qs:
            print(f"    - {q}")

# ---------------------------------------------------------------------------
# 2. generation, LLM as judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """You grade the answers of a retrieval-augmented QA system.
You get a question, a reference answer, the context passages the system retrieved, and the system's answer.

Return a JSON object:
{
  "correct": true or false,   // does the system answer agree with the reference answer on the key facts?
  "faithful": true or false,  // is every claim in the system answer supported by the context passages?
  "comment": "one short sentence explaining your grades"
}
An answer saying the information was not found is "faithful" but not "correct"."""

if args.judge:
    print("\n" + "=" * 70)
    print("Answer quality (retrieval: hybrid+rerank, judge: same LLM)\n")
    results = []
    for i, item in enumerate(questions, start=1):
        hits = retriever.search(item["question"], k=args.k, method="hybrid+rerank")
        answer = answer_with_context(item["question"], hits)
        context = "\n\n".join(h["text"] for h in hits)

        raw = ask_llm(
            [
                {"role": "system", "content": JUDGE_PROMPT},
                {"role": "user", "content": (
                    f"Question: {item['question']}\n\nReference answer: {item['reference_answer']}\n\n"
                    f"Context passages:\n{context}\n\nSystem answer: {answer}"
                )},
            ],
            temperature=0,
            json_mode=True,
        )
        try:
            grade = json.loads(raw)
        except json.JSONDecodeError:
            grade = {"correct": False, "faithful": False, "comment": f"judge returned invalid JSON: {raw[:80]}"}

        results.append(grade)
        flag = ("OK " if grade.get("correct") else "BAD") + (" " if grade.get("faithful") else " (unfaithful)")
        print(f"[{i:>2}] {flag} {item['question']}")
        print(f"      answer: {' '.join(answer.split())[:150]}")
        print(f"      judge:  {grade.get('comment', '')}\n")
        time.sleep(1)  # be gentle with the free tier

    n = len(results)
    print(f"Correct:  {sum(bool(r.get('correct')) for r in results)}/{n}")
    print(f"Faithful: {sum(bool(r.get('faithful')) for r in results)}/{n}")

# About LLM-as-judge: it's the standard approach now (frameworks like RAGAS or DeepEval
# do the same thing with more metrics), but keep in mind:
#  - the judge is an LLM, it makes mistakes too -> read a sample of its grades yourself
#  - a model grading itself is biased; using a different model as judge is better
#  - 12 questions is a toy set. A real eval set is 50-200 questions written with the people
#    who will use the system, and it grows each time someone reports a bad answer.
