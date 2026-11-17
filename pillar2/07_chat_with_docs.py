"""
Step 7 - chat with your documents.

Compared to 03, three things change:
  1. Citations: passages are numbered, the model cites them [1][2], and we print the file
     and page behind each citation. In a company, nobody trusts an answer without a source.
  2. Conversation memory: we keep the previous turns, so you can ask follow-ups.
  3. Query rewriting: "and what about its latency?" is a terrible search query on its own.
     Before searching, the LLM rewrites it as a standalone question using the history.

Run 06_ingest.py first.
Commands: /context (show retrieved passages on/off), /reset, quit
"""
import re

from rag_utils import ask_llm, get_collection, vector_search

TOP_K = 5

SYSTEM_PROMPT = """You answer questions about an ML team's internal documentation and the research papers it indexed.

Rules:
- Use ONLY the numbered context passages. No outside knowledge.
- Cite the passages you use with their number in brackets, e.g. [2] or [1][3].
- If the context doesn't contain the answer, say "I couldn't find this in the documents." and stop.
- Be concise: a few sentences or a short list."""

REWRITE_PROMPT = """You rewrite follow-up questions into standalone search queries.
Use the conversation to resolve references like "it", "that model", "the second one".
Return only the rewritten query, nothing else."""


def rewrite_query(question, history):
    if not history:
        return question  # first question, nothing to resolve
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
    query = ask_llm(
        [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"Conversation:\n{convo}\n\nFollow-up question: {question}"},
        ],
        max_tokens=300,
        temperature=0,
    )
    return query.strip().strip('"') or question


def format_context(hits):
    blocks = []
    for i, h in enumerate(hits, start=1):
        m = h["metadata"]
        blocks.append(f"[{i}] ({m['source']}, p.{m['page']})\n{h['text']}")
    return "\n\n".join(blocks)


def answer(question, hits, history):
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    # only questions and answers go in the history, not the retrieved context
    # (otherwise every turn re-sends all the old passages and we burn tokens)
    messages += history[-6:]
    messages.append({"role": "user", "content": f"Context:\n{format_context(hits)}\n\nQuestion: {question}"})
    return ask_llm(messages)


def print_sources(text, hits):
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text) if 0 < int(n) <= len(hits)})
    if not cited:
        return
    print("\nSources:")
    for n in cited:
        m = hits[n - 1]["metadata"]
        print(f"  [{n}] {m['source']}, page {m['page']}")


def main():
    collection = get_collection()
    if collection.count() == 0:
        raise SystemExit("Collection is empty, run 06_ingest.py first")

    history = []
    show_context = False
    print(f"Chatting with {collection.count()} chunks. Commands: /context, /reset, quit\n")

    while True:
        question = input("You: ").strip()
        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            break
        if question == "/reset":
            history.clear()
            print("(history cleared)\n")
            continue
        if question == "/context":
            show_context = not show_context
            print(f"(show context: {show_context})\n")
            continue

        query = rewrite_query(question, history)
        if query != question:
            print(f"  (searching for: {query})")

        hits = vector_search(collection, query, k=TOP_K)
        if show_context:
            for i, h in enumerate(hits, start=1):
                print(f"  [{i}] {h['distance']:.3f} {h['metadata']['source']} p.{h['metadata']['page']}: "
                      f"{h['text'][:120]!r}")

        reply = answer(question, hits, history)
        print(f"\nAssistant: {reply}")
        print_sources(reply, hits)
        print("-" * 70)

        history += [{"role": "user", "content": question}, {"role": "assistant", "content": reply}]


if __name__ == "__main__":
    main()

# Conversation to try:
#   What happened to YOLOv8 in March?
#   how much did its mAP drop?            <- follow-up, look at the rewritten query
#   and what was done while waiting for the retraining?
#   who has to approve the new version before rollout?
#   what's the capital of France?         <- should refuse, it's not in the docs
