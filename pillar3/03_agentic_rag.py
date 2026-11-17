"""
Step 3 - agentic RAG: the retriever from pillar 2 becomes a tool.

Classic RAG (pillar 2) = always exactly one search with the user's question, then answer.
Agentic RAG = the agent decides IF it needs to search, WHAT to search for, and can search
several times, refining the query based on what it found.

That matters for multi-hop questions, where the answer needs facts from different documents
and the second search depends on the result of the first one.

Needs pillar 2 next to this folder (d:\\pythoncode\\pillar2) with 06_ingest.py already run.
"""
import sys

from agent_utils import PILLAR2_DIR, Agent, tool

sys.path.insert(0, str(PILLAR2_DIR))
from rag_utils import answer_with_context  # noqa: E402  (pillar 2 code)
from retrieval import Retriever            # noqa: E402

print("Loading the pillar 2 retriever ...")
retriever = Retriever()
SOURCES = sorted({c["metadata"]["source"] for c in retriever.chunks.values()})


@tool
def list_documents():
    """List the documents available in the knowledge base (team docs and research papers)."""
    return SOURCES


@tool
def search_docs(query: str, document: str = "", top_k: int = 4):
    """Search the knowledge base and return the most relevant passages with their source and page.
    Write focused queries about one fact at a time. Optionally restrict to one document
    (exact file name from list_documents)."""
    if document and document not in SOURCES:
        return f"Error: unknown document '{document}'. Use one of {SOURCES} or leave it empty."
    hits = retriever.search(query, k=min(top_k, 8), method="hybrid+rerank", source=document or None)
    return [{"source": h["metadata"]["source"], "page": h["metadata"]["page"], "text": h["text"]} for h in hits]


SYSTEM_PROMPT = """You answer questions about a computer vision team's documentation and research papers.

How to work:
- Break the question into the facts you need, and search for each one separately.
- If a search doesn't return what you need, rephrase it or search in a specific document.
- Only answer from what you found. Cite every fact as (file, p.N).
- If you can't find something after a few searches, say what's missing."""

agent = Agent([list_documents, search_docs], SYSTEM_PROMPT, max_steps=8)

QUESTIONS = [
    # needs the postmortem (dates) + simple reasoning
    "How many days passed between the camera change on line 3 and the drift alert?",
    # needs the drift policy (threshold) + the postmortem (number of images)
    "Was the number of new images collected after the March incident enough to trigger retraining "
    "automatically?",
    # needs the registry (latency, owner) + the runbook (steps, approvers)
    "If ViT Large were optimized down to 45 ms, which steps would it still need to pass before full "
    "rollout, and who would have to approve it?",
]

for q in QUESTIONS:
    print("\n" + "=" * 80)
    print(f"Q: {q}")

    # classic RAG for comparison: one search with the raw question
    hits = retriever.search(q, k=5)
    print("\n--- classic RAG (1 search) ---")
    print(answer_with_context(q, hits))

    print("\n--- agentic RAG ---")
    result = agent.run(q)
    searches = [t["args"] for t in result.trace if t["tool"] == "search_docs"]
    print(f"\n{result.answer}")
    print(f"\n({len(searches)} searches, {result.steps} steps, {result.tokens} tokens)")

# The trade-off you should see:
#  - agentic RAG answers the multi-hop questions better, the classic version often only
#    finds half the facts because one query can't match two documents well
#  - but it costs 3-10x the tokens and latency. For simple lookups ("who owns X") classic
#    RAG is the right call. A common production pattern: a cheap router decides which one to use.
