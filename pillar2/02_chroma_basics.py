"""
Step 2 - storing embeddings in a vector database (ChromaDB).

Chroma does three things for us: computes the embeddings (with the embedding function we
give it), stores them on disk, and finds the nearest neighbours of a query quickly.
It also stores metadata next to each vector, which lets us filter (see the end).
"""
import chromadb
from chromadb.utils import embedding_functions

from rag_utils import CHROMA_DIR, EMBEDDING_MODEL

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

collection = client.get_or_create_collection(
    name="ml_knowledge_base",
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"},  # distance = 1 - cosine similarity, 0 means identical
)

facts = [
    ("YOLOv8 is currently experiencing data drift in production since March 2024.", "monitoring"),
    ("The ResNet50 model achieves 92.3% accuracy on ImageNet with 45ms latency.", "models"),
    ("EfficientNet was retrained on Feb 10 2024 due to performance degradation.", "models"),
    ("ViT Large has the highest accuracy at 95.1% but is not deployed in production due to high latency.", "models"),
    ("Data drift detection runs weekly on all production models using statistical tests.", "monitoring"),
    ("The team decided to deprioritize ViT Large deployment until latency improves below 50ms.", "decisions"),
    ("Model retraining requires at least 10000 new labeled samples to be triggered automatically.", "monitoring"),
    ("The CI/CD pipeline for model deployment includes automated A/B testing for 48 hours.", "deployment"),
]

# upsert instead of add: running the script twice doesn't create duplicates
# (in the last version, every run added the 8 docs again with the same ids -> warnings)
collection.upsert(
    ids=[f"fact_{i}" for i in range(len(facts))],
    documents=[text for text, _ in facts],
    metadatas=[{"topic": topic} for _, topic in facts],
)
print(f"Collection has {collection.count()} documents\n")


def show(results):
    for doc, dist, meta in zip(results["documents"][0], results["distances"][0], results["metadatas"][0]):
        print(f"  [{dist:.3f}] ({meta['topic']}) {doc}")
    print()


for query in ["Which model has drift issues?", "What is needed to trigger retraining?", "Tell me about ViT Large latency"]:
    print(f"Query: {query!r}")
    show(collection.query(query_texts=[query], n_results=3))

# Metadata filtering: only search inside one topic.
# Very useful in practice (per customer, per product line, per document type, per date...).
query = "what happens with latency?"
print(f"Query: {query!r} (no filter)")
show(collection.query(query_texts=[query], n_results=2))
print(f"Query: {query!r} (where topic == 'deployment')")
show(collection.query(query_texts=[query], n_results=2, where={"topic": "deployment"}))
