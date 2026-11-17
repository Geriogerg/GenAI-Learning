"""
Step 5 - chunking strategies.

Embedding a whole 15-page paper into one vector doesn't work: the vector becomes an average
of everything and matches nothing well. So documents are cut into chunks, and each chunk
gets its own embedding.

How you cut matters more than people expect. This script compares 4 strategies on the team
docs and measures how often the chunk containing the answer comes back in the top results,
using the questions in data/eval_questions.json.
"""
import json
import re
import statistics

import chromadb
from chromadb.utils import embedding_functions

from rag_utils import DOCS_DIR, EMBEDDING_MODEL, EVAL_FILE, load_documents, recursive_chunk

CHUNK_SIZE = 400  # characters, kept small on purpose so the differences are visible on short docs
OVERLAP = 80


def fixed_size(text):
    # the naive way: cut every N characters, wherever that falls (mid-word, mid-sentence...)
    return [text[i:i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]


def fixed_size_overlap(text):
    # same, but consecutive chunks share OVERLAP characters
    step = CHUNK_SIZE - OVERLAP
    return [text[i:i + CHUNK_SIZE] for i in range(0, max(len(text) - OVERLAP, 1), step)]


def by_sentences(text):
    # pack whole sentences until we reach ~CHUNK_SIZE
    sentences = re.split(r"(?<=[.!?])\s+|\n{2,}", text)
    chunks, current = [], ""
    for s in sentences:
        if current and len(current) + len(s) > CHUNK_SIZE:
            chunks.append(current.strip())
            current = ""
        current += s + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def recursive(text):
    # paragraphs > lines > sentences > words, see rag_utils.recursive_chunk
    return recursive_chunk(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP)


STRATEGIES = {
    "fixed": fixed_size,
    "fixed+overlap": fixed_size_overlap,
    "sentences": by_sentences,
    "recursive": recursive,
}


def is_relevant(text, source, item):
    return source == item["source"] and all(s.lower() in text.lower() for s in item["must_contain"])


docs = load_documents(DOCS_DIR)
questions = json.loads(EVAL_FILE.read_text(encoding="utf-8"))

client = chromadb.EphemeralClient()  # in memory, thrown away at the end
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)

print(f"{len(docs)} docs, {len(questions)} questions, chunk_size={CHUNK_SIZE}, overlap={OVERLAP}\n")
print(f"{'strategy':<15}{'chunks':>8}{'avg len':>9}{'min':>6}{'max':>6}{'hit@1':>8}{'hit@3':>8}")

examples = {}
for name, chunker in STRATEGIES.items():
    ids, texts, metas = [], [], []
    for doc in docs:
        for i, chunk in enumerate(chunker(doc["text"])):
            ids.append(f"{doc['source']}-{i}")
            texts.append(chunk)
            metas.append({"source": doc["source"]})

    collection = client.create_collection(name=name.replace("+", "_"), embedding_function=embedding_fn,
                                          metadata={"hnsw:space": "cosine"})
    collection.add(ids=ids, documents=texts, metadatas=metas)

    hit1 = hit3 = 0
    for item in questions:
        res = collection.query(query_texts=[item["question"]], n_results=3)
        found = [is_relevant(t, m["source"], item) for t, m in zip(res["documents"][0], res["metadatas"][0])]
        hit1 += found[0]
        hit3 += any(found)

    lengths = [len(t) for t in texts]
    print(f"{name:<15}{len(texts):>8}{statistics.mean(lengths):>9.0f}{min(lengths):>6}{max(lengths):>6}"
          f"{hit1 / len(questions):>8.0%}{hit3 / len(questions):>8.0%}")
    examples[name] = texts[1]

print("\nWhat the 2nd chunk of the first doc looks like with each strategy:")
for name, text in examples.items():
    print(f"\n--- {name} ---\n{text}")

# Notes from playing with this:
#  - "fixed" cuts words and numbers in half ("10,0" | "00 samples"), which breaks both
#    retrieval and the answer. Overlap mostly hides the problem.
#  - sentences/recursive keep ideas whole. recursive also respects markdown sections and
#    tables lines, that's why it's the default in most frameworks.
#  - try CHUNK_SIZE = 200 and 1500: too small -> chunks lose context ("it", "this model"...),
#    too big -> one chunk mixes several topics and the embedding gets blurry.
#  - with 12 questions the numbers are noisy, it's a sanity check, not a benchmark (see step 9).
