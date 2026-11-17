"""
Step 3 - first complete RAG: retrieval + generation.

Retrieve the most relevant facts from the collection built in 02, paste them into the
prompt, and ask the LLM to answer using only that context.
Run 02_chroma_basics.py first.
"""
import chromadb
from chromadb.utils import embedding_functions

from rag_utils import CHROMA_DIR, EMBEDDING_MODEL, ask_llm

client = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = client.get_or_create_collection(
    name="ml_knowledge_base",
    embedding_function=embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL),
    metadata={"hnsw:space": "cosine"},
)
if collection.count() == 0:
    raise SystemExit("Collection is empty, run 02_chroma_basics.py first")

SYSTEM_PROMPT = """You are an ML operations assistant.
Answer the user's question using ONLY the context below.
If the context doesn't contain the answer, say so clearly instead of guessing.
Be concise and mention which fact you used."""


def retrieve(query, k=3):
    res = collection.query(query_texts=[query], n_results=k)
    return res["documents"][0]


def rag_answer(query):
    facts = retrieve(query)
    context = "\n".join(f"- {f}" for f in facts)
    answer = ask_llm([
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
    ])
    return answer, context


questions = [
    "Which models currently have drift problems?",
    "Why isn't ViT Large in production?",
    "How many samples do I need to trigger retraining?",
    "What is the weather today?",  # not in the context -> should say it doesn't know
]

for q in questions:
    answer, context = rag_answer(q)
    print(f"Q: {q}")
    print(f"Context used:\n{context}")
    print(f"\nA: {answer}")
    print("=" * 70 + "\n")

# Limits of this version (fixed in the next steps):
#  - documents are hardcoded one-liners, real docs are long PDFs -> 04 and 05
#  - we always retrieve 3 facts even when only 1 is relevant
#  - no way to know where an answer comes from in a real document -> citations in 07
