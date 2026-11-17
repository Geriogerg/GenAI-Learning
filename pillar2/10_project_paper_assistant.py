"""
Pillar 2 mini project - research paper assistant.

Chat with the PDFs in data/papers/ (and the team docs in data/docs/).
Puts everything from pillar 2 together: loading, chunking, incremental ingestion,
hybrid retrieval + reranking, cited answers, follow-up questions. The /summary command
reuses the structured JSON output from pillar 1.

Commands:
    /papers            list the indexed documents
    /focus <name>      only search inside one document (partial name is fine), "/focus off" to stop
    /summary <name>    structured summary of a paper
    /reindex           re-run the ingestion (after adding PDFs)
    /reset             forget the conversation
    quit
"""
import json
from collections import defaultdict

from rag_utils import (DATA_DIR, answer_with_context, ask_llm, chunk_documents, cited_sources,
                       get_collection, load_documents, rewrite_query, sync_collection)
from retrieval import Retriever

TOP_K = 5

SUMMARY_PROMPT = """You summarize research papers for a data scientist, using only the excerpts provided
(usually the first pages and the conclusion).
Return a JSON object with exactly these fields:
{
  "title": "string",
  "problem": "what problem the paper addresses, 1-2 sentences",
  "method": "the core idea of the approach, 2-3 sentences",
  "key_results": ["list of the main quantitative or qualitative results"],
  "limitations": ["limitations mentioned or obvious from the excerpts"],
  "worth_reading_if": "one sentence: who should read it and why"
}
If something isn't in the excerpts, write "not found in excerpts"."""


def ingest(collection):
    print("Indexing data/ ...")
    chunks = chunk_documents(load_documents(DATA_DIR))
    stats = sync_collection(collection, chunks)
    print(f"  added {stats['added']}, deleted {stats['deleted']}, unchanged {stats['unchanged']}")


def list_sources(collection):
    pages = defaultdict(set)
    chunks = defaultdict(int)
    for m in collection.get(include=["metadatas"])["metadatas"]:
        pages[m["source"]].add(m["page"])
        chunks[m["source"]] += 1
    return {s: (len(pages[s]), chunks[s]) for s in sorted(pages)}


def resolve_source(name, sources):
    """'clip' -> 'clip.pdf'. Returns None and prints the options if it's ambiguous."""
    name = name.lower().strip()
    matches = [s for s in sources if name in s.lower()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        print(f"  no document matching {name!r}, try /papers")
    else:
        print(f"  {name!r} is ambiguous: {', '.join(matches)}")
    return None


def summarize(collection, source):
    data = collection.get(where={"source": source}, include=["documents", "metadatas"])
    chunks = sorted(zip(data["metadatas"], data["documents"]), key=lambda x: (x[0]["page"], x[0]["chunk"]))
    last_page = max(m["page"] for m, _ in chunks)

    # Abstract + intro + conclusion carry most of the signal, and sending the whole paper
    # would blow the free-tier token limit. A proper version would do map-reduce: summarize
    # each section, then summarize the summaries (good exercise).
    picked = [text for m, text in chunks if m["page"] <= 2 or m["page"] >= last_page - 1]
    excerpts = "\n\n".join(picked)[:12000]

    raw = ask_llm(
        [
            {"role": "system", "content": SUMMARY_PROMPT},
            {"role": "user", "content": f"Excerpts from {source}:\n\n{excerpts}"},
        ],
        max_tokens=2000,
        json_mode=True,
    )
    return json.loads(raw)


def print_summary(s):
    print(f"\n  {s.get('title', '?')}\n")
    print(f"  Problem: {s.get('problem')}")
    print(f"  Method:  {s.get('method')}")
    print("  Key results:")
    for r in s.get("key_results", []):
        print(f"    - {r}")
    print("  Limitations:")
    for r in s.get("limitations", []):
        print(f"    - {r}")
    print(f"  Worth reading if: {s.get('worth_reading_if')}")


def main():
    collection = get_collection()
    if collection.count() == 0:
        ingest(collection)

    print("Loading retriever ...")
    retriever = Retriever(collection)
    sources = list_sources(collection)

    history = []
    focus = None
    print(f"\nPaper assistant ready, {len(sources)} documents indexed. Type /papers to see them.\n")

    while True:
        prompt = f"You [{focus}]: " if focus else "You: "
        user_input = input(prompt).strip()
        if not user_input:
            continue
        cmd, _, arg = user_input.partition(" ")

        if cmd.lower() in ("quit", "exit", "q"):
            break

        elif cmd == "/papers":
            for s, (n_pages, n_chunks) in sources.items():
                print(f"  {s:<45} {n_pages:>3} pages  {n_chunks:>4} chunks")

        elif cmd == "/focus":
            if arg.strip().lower() in ("", "off"):
                focus = None
                print("  searching in all documents")
            else:
                focus = resolve_source(arg, sources) or focus
            history.clear()  # a conversation about another doc would confuse the rewriting

        elif cmd == "/summary":
            source = resolve_source(arg, sources) if arg else focus
            if source:
                print(f"  summarizing {source} ...")
                try:
                    print_summary(summarize(collection, source))
                except json.JSONDecodeError:
                    print("  the model didn't return valid JSON, try again")

        elif cmd == "/reindex":
            ingest(collection)
            retriever = Retriever(collection)  # BM25 index has to be rebuilt
            sources = list_sources(collection)

        elif cmd == "/reset":
            history.clear()
            print("  (history cleared)")

        elif cmd.startswith("/"):
            print("  unknown command, see the docstring at the top of the file")

        else:
            query = rewrite_query(user_input, history)
            if query != user_input:
                print(f"  (searching for: {query})")
            hits = retriever.search(query, k=TOP_K, method="hybrid+rerank", source=focus)
            reply = answer_with_context(user_input, hits, history)

            print(f"\nAssistant: {reply}")
            sources_used = cited_sources(reply, hits)
            if sources_used:
                print("\nSources:")
                for n, m in sources_used:
                    print(f"  [{n}] {m['source']}, page {m['page']}")

            history += [{"role": "user", "content": user_input}, {"role": "assistant", "content": reply}]

        print("-" * 70)


if __name__ == "__main__":
    main()

# Ideas to extend it (pick one, it's the best way to make it stick):
#  - /compare <a> <b>: retrieve from both papers on the same aspect and ask for a comparison table
#  - map-reduce summary over the whole paper instead of first + last pages
#  - a Streamlit UI (pip install streamlit), ~40 lines on top of these functions
#  - answer in French while the papers are in English (system prompt + multilingual embeddings)
#  - log every question/answer/sources to a jsonl file, it becomes your eval set later
