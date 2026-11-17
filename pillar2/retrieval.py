"""
Retriever with 4 modes, used by 08, 09 and the mini project:

    vector         -> embeddings + cosine similarity (what we did so far)
    keyword        -> BM25, the classic search-engine scoring on exact words
    hybrid         -> both, merged with Reciprocal Rank Fusion
    hybrid+rerank  -> hybrid gets ~20 candidates, a cross-encoder re-scores them

Why bother: embeddings are good at meaning ("the cameras changed" ~ "new camera supplier")
but bad at exact tokens (model names, error codes, CLI commands like `mlctl rollback`).
BM25 is the opposite. Combining them is the standard trick in production RAG.
"""
import re

from rank_bm25 import BM25Okapi

from rag_utils import RERANK_MODEL, get_collection, vector_search


def tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


class Retriever:
    def __init__(self, collection=None, use_reranker=True):
        self.collection = collection or get_collection()

        # BM25 needs all the chunks in memory. Fine for a few thousand chunks,
        # for bigger corpora you'd use Elasticsearch/OpenSearch or a DB with full-text search.
        data = self.collection.get(include=["documents", "metadatas"])
        if not data["ids"]:
            raise RuntimeError("The collection is empty, run 06_ingest.py first")

        self.ids = data["ids"]
        self.chunks = {
            cid: {"id": cid, "text": doc, "metadata": meta}
            for cid, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        }
        self.bm25 = BM25Okapi([tokenize(self.chunks[cid]["text"]) for cid in self.ids])

        self.reranker = None
        if use_reranker:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL)  # ~90MB, downloaded once

    # -- individual methods -------------------------------------------------

    def vector(self, query, k=5, source=None):
        where = {"source": source} if source else None
        return vector_search(self.collection, query, k, where)

    def keyword(self, query, k=5, source=None):
        scores = self.bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        hits = []
        for i in order:
            if scores[i] <= 0:
                break  # no query word in this chunk, and everything after is worse
            chunk = self.chunks[self.ids[i]]
            if source and chunk["metadata"]["source"] != source:
                continue
            hits.append({**chunk, "score": float(scores[i])})
            if len(hits) == k:
                break
        return hits

    def hybrid(self, query, k=5, source=None, candidates=20, rrf_k=60):
        """
        Reciprocal Rank Fusion: each list votes 1 / (rrf_k + rank) for its documents.
        Only ranks are used, so we don't have to make cosine distances and BM25 scores
        comparable (they aren't). rrf_k=60 is the value from the original paper.
        """
        fused = {}
        for results in (self.vector(query, candidates, source), self.keyword(query, candidates, source)):
            for rank, hit in enumerate(results, start=1):
                fused[hit["id"]] = fused.get(hit["id"], 0.0) + 1.0 / (rrf_k + rank)
        best = sorted(fused, key=fused.get, reverse=True)[:k]
        return [{**self.chunks[cid], "score": fused[cid]} for cid in best]

    def rerank(self, query, hits, k=5):
        """
        A cross-encoder reads (query, chunk) together and outputs a relevance score.
        Much more accurate than comparing two independent embeddings, but too slow to run
        on the whole corpus, hence: cheap retrieval first, rerank only the top candidates.
        """
        if not hits:
            return []
        scores = self.reranker.predict([(query, h["text"]) for h in hits])
        ranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)[:k]
        return [{**h, "score": float(s)} for h, s in ranked]

    # -- single entry point -------------------------------------------------

    def search(self, query, k=5, method="hybrid+rerank", source=None):
        if method == "vector":
            return self.vector(query, k, source)
        if method == "keyword":
            return self.keyword(query, k, source)
        if method == "hybrid":
            return self.hybrid(query, k, source)
        if method == "hybrid+rerank":
            if self.reranker is None:
                raise ValueError("Retriever was created with use_reranker=False")
            return self.rerank(query, self.hybrid(query, k=20, source=source), k)
        raise ValueError(f"unknown method {method!r}")


METHODS = ["vector", "keyword", "hybrid", "hybrid+rerank"]
