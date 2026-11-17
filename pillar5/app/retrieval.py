"""
Retrieval for the service. Same idea as pillar 2, packaged for production:
  - the index is built once at startup from DOCS_DIR (in a bigger system: a separate
    ingestion job writing to a vector DB server, and the API only reads)
  - Chroma's default embedding function runs an ONNX MiniLM model: no PyTorch, so the Docker
    image stays small (~600 MB instead of several GB)
"""
import hashlib
import re

import numpy as np

from app.config import settings


class HashEmbedding:
    """Deterministic fake embeddings for tests/CI (FAKE_EMBED=1): bag of hashed words."""

    def __init__(self, dim=256):
        self.dim = dim

    def __call__(self, input):
        out = []
        for text in input:
            v = np.zeros(self.dim, dtype=np.float32)
            for w in re.findall(r"[a-z0-9]+", text.lower()):
                v[int(hashlib.md5(w.encode()).hexdigest(), 16) % self.dim] += 1
            out.append(v / (np.linalg.norm(v) + 1e-9))
        return out

    @staticmethod
    def name():
        return "hash-embedding"

    def is_legacy(self):
        return False

    def embed_query(self, input):
        return self(input)


def embedding_function():
    if settings.fake_embed:
        return HashEmbedding()
    from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
    return DefaultEmbeddingFunction()


def chunk_markdown(text, max_chars=900):
    """Split on markdown sections, then paragraphs if a section is too long."""
    sections = re.split(r"\n(?=#{1,3} )", text)
    chunks = []
    for sec in sections:
        if len(sec) <= max_chars:
            chunks.append(sec.strip())
            continue
        current = ""
        for para in sec.split("\n\n"):
            if current and len(current) + len(para) > max_chars:
                chunks.append(current.strip())
                current = ""
            current += para + "\n\n"
        if current.strip():
            chunks.append(current.strip())
    return [c for c in chunks if len(c) > 40]


class Retriever:
    def __init__(self, docs_dir=None):
        import chromadb

        self.embed = embedding_function()
        self.collection = chromadb.EphemeralClient().get_or_create_collection(
            "docs", embedding_function=self.embed, metadata={"hnsw:space": "cosine"})
        self.add_folder(docs_dir or settings.docs_dir)

    def add_folder(self, folder):
        ids, texts, metas = [], [], []
        for path in sorted(folder.glob("*.md")):
            for i, chunk in enumerate(chunk_markdown(path.read_text(encoding="utf-8"))):
                ids.append(f"{path.name}#{i}")
                texts.append(chunk)
                metas.append({"source": path.name})
        if ids:
            self.collection.upsert(ids=ids, documents=texts, metadatas=metas)
        return len(ids)

    def search(self, query, k=4):
        k = min(k, self.collection.count())
        if not k:
            return []
        res = self.collection.query(query_texts=[query], n_results=k)
        return [{"text": d, "source": m["source"], "distance": round(dist, 3)}
                for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0])]
