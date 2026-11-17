"""
Step 6 - ingestion pipeline: load -> chunk -> embed -> store, with metadata.

This is the "indexing phase" of RAG as a proper script you can re-run:
  - every chunk keeps its source file and page number (needed for citations)
  - chunk ids are a hash of (file, page, position, content), so re-running only embeds
    what's new and removes what disappeared (incremental, like a real pipeline)

Usage:
    python 06_ingest.py                    # incremental sync of data/
    python 06_ingest.py --reset            # drop everything and rebuild
    python 06_ingest.py --chunk-size 500   # try other settings (changes ids -> full re-embed)
"""
import argparse
import time
from collections import Counter

from rag_utils import (COLLECTION_NAME, DATA_DIR, chunk_documents, get_collection, load_documents,
                       reset_collection, sync_collection)

parser = argparse.ArgumentParser()
parser.add_argument("--reset", action="store_true", help="delete the collection first")
parser.add_argument("--chunk-size", type=int, default=800)
parser.add_argument("--overlap", type=int, default=150)
args = parser.parse_args()

start = time.time()

collection = reset_collection() if args.reset else get_collection()

print(f"Loading documents from {DATA_DIR} ...")
docs = load_documents(DATA_DIR)
print(f"  {len(docs)} pages/files from {len({d['source'] for d in docs})} documents")

chunks = chunk_documents(docs, chunk_size=args.chunk_size, overlap=args.overlap)
print(f"  {len(chunks)} chunks (chunk_size={args.chunk_size}, overlap={args.overlap})")

print("Syncing with the vector store ...")
stats = sync_collection(collection, chunks)

print(f"\nDone in {time.time() - start:.1f}s - added {stats['added']}, deleted {stats['deleted']}, "
      f"unchanged {stats['unchanged']}")
print(f"Collection '{COLLECTION_NAME}' now holds {collection.count()} chunks:")

per_source = Counter(m["source"] for m in collection.get(include=["metadatas"])["metadatas"])
for source, n in sorted(per_source.items()):
    print(f"  {n:>4}  {source}")

# Run it twice: the second run should say "added 0". Then drop a new PDF in data/papers
# and run it again, only that file gets embedded.
