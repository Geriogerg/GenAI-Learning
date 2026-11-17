"""
Step 8 - better retrieval: keyword search, hybrid search, reranking.

When a RAG system gives a bad answer, 8 times out of 10 the right chunk simply wasn't
retrieved. So most of the effort in real projects goes into retrieval, not into the prompt.

This script runs the same queries through the 4 methods implemented in retrieval.py and
prints the top 3 of each, so you can see where each one wins or fails.
Run 06_ingest.py first.
"""
from rag_utils import PAPERS_DIR
from retrieval import METHODS, Retriever

QUERIES = [
    # exact command name: embeddings tend to fumble this, BM25 nails it
    "mlctl rollback",
    # paraphrase with no shared keyword ("camera supplier" in the doc): embeddings win
    "why did the defect detector get worse after the hardware change?",
    # mix of both
    "PSI threshold for drift detected",
    # needs the right chunk among several close ones: that's where the reranker helps
    "who has to approve before a model goes to 100% of traffic?",
]

print("Loading retriever (BM25 index + cross-encoder) ...")
retriever = Retriever()


def preview(text, n=90):
    text = " ".join(text.split())
    return text[:n] + ("..." if len(text) > n else "")


for query in QUERIES:
    print("\n" + "=" * 90)
    print(f"QUERY: {query}")
    for method in METHODS:
        print(f"\n  {method}")
        for h in retriever.search(query, k=3, method=method):
            m = h["metadata"]
            print(f"    {m['source'][:32]:<32} p.{m['page']:<3} {preview(h['text'])}")

# Metadata filtering works with every method: search inside one document only.
pdfs = sorted(PAPERS_DIR.glob("*.pdf"))
if pdfs:
    source = pdfs[0].name
    print("\n" + "=" * 90)
    print(f"Filtered search inside {source}:")
    for h in retriever.search("main contribution of the paper", k=3, source=source):
        print(f"    p.{h['metadata']['page']:<3} {preview(h['text'])}")

# Takeaways:
#  - vector alone misses exact identifiers, keyword alone misses paraphrases
#  - hybrid (RRF) is almost never worse than the best of the two, cheap to add, do it by default
#  - the reranker fixes the ORDER of the candidates. It costs ~50-200ms per query on CPU
#    for 20 candidates. Worth it when the top-1 has to be right (answer quality depends on it).
#  - step 9 measures all this instead of eyeballing it.
