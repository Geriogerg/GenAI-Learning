"""
Step 1 - what is an embedding?

An embedding model turns a piece of text into a fixed-size vector. Texts that mean similar
things end up close to each other, even when they share no words. Everything else in
pillar 2 is built on top of this.
"""
from sentence_transformers import SentenceTransformer

from rag_utils import EMBEDDING_MODEL

model = SentenceTransformer(EMBEDDING_MODEL)  # ~90MB, downloaded once then cached

sentences = [
    "The YOLOv8 model has high latency in production",
    "Performance degradation detected on object detection model",
    "We need to retrain the detection pipeline",
    "The cat is sleeping on the sofa",
    "Data drift was detected on the CV model",
]

# normalize_embeddings=True -> every vector has norm 1, so cosine similarity is just a dot product
embeddings = model.encode(sentences, normalize_embeddings=True)
print(f"{len(sentences)} sentences -> matrix of shape {embeddings.shape}\n")

query = "object detection model is slow"
q = model.encode(query, normalize_embeddings=True)
scores = embeddings @ q

print(f"Query: {query!r}\n")
for score, sentence in sorted(zip(scores, sentences), reverse=True):
    print(f"  {score:.3f}  {sentence}")

# Full matrix, useful to calibrate what "close" means for this model.
# With MiniLM, > 0.5 is usually clearly related and < 0.2 is usually unrelated.
print("\nPairwise similarity:")
sim = embeddings @ embeddings.T
print("       " + "  ".join(f"  s{j}" for j in range(len(sentences))))
for i, row in enumerate(sim):
    print(f"  s{i}  " + "  ".join(f"{v:5.2f}" for v in row))

# The two limits I'd test next, because they matter for RAG later:
negation = model.encode(["the model is slow", "the model is not slow"], normalize_embeddings=True)
print(f"\n'slow' vs 'not slow': {negation[0] @ negation[1]:.3f}  <- embeddings barely see negation")

fr = model.encode(["le modele de detection est lent", query], normalize_embeddings=True)
print(f"french vs english query: {fr[0] @ fr[1]:.3f}  <- this model is english-only")
# for french docs, try "paraphrase-multilingual-MiniLM-L12-v2" (change EMBEDDING_MODEL in rag_utils)
