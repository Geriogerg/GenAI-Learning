"""
Shared bits for the pillar 2 scripts: config, LLM client, document loading, chunking,
and the vector store helpers.

I started copy-pasting the same 30 lines at the top of every script, so they live here now.
Scripts 01-05 mostly re-implement things on purpose (to see how they work), the later
ones just import from this file.
"""
import hashlib
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # walks up the folders, so the .env in d:\pythoncode is found too

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = DATA_DIR / "docs"
PAPERS_DIR = DATA_DIR / "papers"
EVAL_FILE = DATA_DIR / "eval_questions.json"
CHROMA_DIR = ROOT / "chroma_db"

COLLECTION_NAME = "pillar2_docs"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Groq shut down llama-3.3-70b-versatile in August 2026, gpt-oss-120b is their recommended
# replacement. Override with GROQ_MODEL=... in .env if you want to try another one
# (qwen/qwen3.6-27b is the other suggested option).
LLM_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

_client = None


def get_llm():
    global _client
    if _client is None:
        from groq import Groq

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not found, check your .env file")
        _client = Groq(api_key=api_key)
    return _client


def ask_llm(messages, max_tokens=1500, temperature=0.2, json_mode=False, retries=3):
    """
    Small wrapper around chat.completions with a retry on rate limits.
    The free tier returns 429s fairly quickly when you loop over questions (see 09_evaluation.py).
    """
    from groq import RateLimitError

    kwargs = dict(model=LLM_MODEL, messages=messages, max_tokens=max_tokens, temperature=temperature)
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}
    if "gpt-oss" in LLM_MODEL:
        # reasoning model: the thinking tokens count in max_tokens, low effort is plenty for RAG
        kwargs["reasoning_effort"] = "low"

    for attempt in range(retries):
        try:
            resp = get_llm().chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except RateLimitError:
            if attempt == retries - 1:
                raise
            wait = 10 * (attempt + 1)
            print(f"  (rate limited, waiting {wait}s)")
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def clean_text(text, from_pdf=False):
    text = text.replace("\x00", "").replace("\r\n", "\n")
    if from_pdf:
        # "detec-\ntion" -> "detection" (hyphenation at line breaks)
        text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
        # pdf text has a newline at the end of every visual line, which means nothing.
        # Keep blank lines (paragraph breaks), turn the rest into spaces.
        text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(folder=DATA_DIR):
    """
    Read every .pdf / .md / .txt under `folder`.
    Returns one dict per PDF page (so we can cite pages later) or per text file:
        {"text": ..., "source": "file.pdf", "page": 3}
    """
    from pypdf import PdfReader

    docs = []
    for path in sorted(Path(folder).rglob("*")):
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            try:
                reader = PdfReader(path)
            except Exception as e:  # broken pdfs happen, don't kill the whole ingestion
                print(f"  skipping {path.name}: {e}")
                continue
            for page_num, page in enumerate(reader.pages, start=1):
                text = clean_text(page.extract_text() or "", from_pdf=True)
                if len(text) < 50:  # empty or image-only page
                    continue
                docs.append({"text": text, "source": path.name, "page": page_num})
        elif suffix in (".md", ".txt") and path.name.lower() != "readme.txt":
            text = clean_text(path.read_text(encoding="utf-8"))
            if text:
                docs.append({"text": text, "source": path.name, "page": 1})
    return docs


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

SEPARATORS = ["\n\n", "\n", ". ", " "]


def _split(text, chunk_size, separators):
    """Cut text into pieces <= chunk_size, using the coarsest separator that works."""
    if len(text) <= chunk_size:
        return [text]
    if not separators:  # no separator left, hard cut
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    sep, finer = separators[0], separators[1:]
    if sep not in text:
        return _split(text, chunk_size, finer)

    pieces = []
    parts = text.split(sep)
    for i, part in enumerate(parts):
        if i < len(parts) - 1:
            part += sep  # keep the separator so sentences keep their ". "
        if part:
            pieces.extend(_split(part, chunk_size, finer))
    return pieces


def recursive_chunk(text, chunk_size=800, overlap=150):
    """
    Split on paragraphs first, then lines, then sentences, then words, and only go finer
    when a piece is still too big. Then glue small pieces back together up to chunk_size.
    Same idea as LangChain's RecursiveCharacterTextSplitter, without the dependency.

    Sizes are in characters (roughly 4 chars per token in English).
    """
    pieces = _split(text, chunk_size, SEPARATORS)
    chunks, current = [], ""
    for piece in pieces:
        if len(current) + len(piece) <= chunk_size:
            current += piece
            continue
        if current.strip():
            chunks.append(current.strip())
        # start the next chunk with the end of the previous one, so an idea that sits on
        # the boundary appears whole in at least one chunk
        tail = ""
        if overlap and len(piece) + overlap <= chunk_size:
            tail = current[-overlap:]
            if " " in tail:
                tail = tail[tail.index(" ") + 1:]  # don't start in the middle of a word
        current = tail + piece
    if current.strip():
        chunks.append(current.strip())
    return chunks


def make_id(*parts):
    return hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()


def chunk_documents(docs, chunk_size=800, overlap=150):
    chunks = []
    for doc in docs:
        for i, text in enumerate(recursive_chunk(doc["text"], chunk_size, overlap)):
            chunks.append({
                # id depends on the content: if a file changes, its chunks get new ids
                "id": make_id(doc["source"], doc["page"], i, text),
                "text": text,
                "metadata": {"source": doc["source"], "page": doc["page"], "chunk": i},
            })
    return chunks


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def _chroma_client():
    import chromadb
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _embedding_fn():
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBEDDING_MODEL)


def get_collection(name=COLLECTION_NAME):
    return _chroma_client().get_or_create_collection(
        name=name,
        embedding_function=_embedding_fn(),
        metadata={"hnsw:space": "cosine"},  # default is L2, cosine distances are easier to read
    )


def reset_collection(name=COLLECTION_NAME):
    client = _chroma_client()
    try:
        client.delete_collection(name)
    except Exception:
        pass  # didn't exist yet
    return get_collection(name)


def sync_collection(collection, chunks, batch_size=256):
    """
    Incremental ingestion: add chunks we don't have yet, delete the ones that disappeared,
    leave the rest alone. Re-running after dropping one new PDF only embeds that PDF.
    """
    existing = set(collection.get(include=[])["ids"])
    wanted = {c["id"]: c for c in chunks}

    to_add = [c for cid, c in wanted.items() if cid not in existing]
    to_delete = list(existing - wanted.keys())

    if to_delete:
        collection.delete(ids=to_delete)

    for start in range(0, len(to_add), batch_size):
        batch = to_add[start:start + batch_size]
        collection.add(
            ids=[c["id"] for c in batch],
            documents=[c["text"] for c in batch],
            metadatas=[c["metadata"] for c in batch],
        )
        print(f"  embedded {min(start + batch_size, len(to_add))}/{len(to_add)} chunks")

    return {"added": len(to_add), "deleted": len(to_delete), "unchanged": len(wanted) - len(to_add)}


def vector_search(collection, query, k=5, where=None):
    k = min(k, collection.count())
    if k == 0:
        return []
    res = collection.query(query_texts=[query], n_results=k, where=where)
    return [
        {"id": i, "text": d, "metadata": m, "distance": dist}
        for i, d, m, dist in zip(res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]


# ---------------------------------------------------------------------------
# Answering (same logic as 07_chat_with_docs.py, moved here so 09 and 10 can reuse it)
# ---------------------------------------------------------------------------

ANSWER_SYSTEM_PROMPT = """You answer questions about an ML team's internal documentation and the research papers it indexed.

Rules:
- Use ONLY the numbered context passages. No outside knowledge.
- Cite the passages you use with their number in brackets, e.g. [2] or [1][3].
- If the context doesn't contain the answer, say "I couldn't find this in the documents." and stop.
- Be concise: a few sentences or a short list."""

REWRITE_PROMPT = """You rewrite follow-up questions into standalone search queries.
Use the conversation to resolve references like "it", "that model", "the second one".
Return only the rewritten query, nothing else."""


def format_context(hits):
    blocks = []
    for i, h in enumerate(hits, start=1):
        m = h["metadata"]
        blocks.append(f"[{i}] ({m['source']}, p.{m['page']})\n{h['text']}")
    return "\n\n".join(blocks)


def rewrite_query(question, history):
    if not history:
        return question
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history[-4:])
    query = ask_llm(
        [
            {"role": "system", "content": REWRITE_PROMPT},
            {"role": "user", "content": f"Conversation:\n{convo}\n\nFollow-up question: {question}"},
        ],
        max_tokens=300,
        temperature=0,
    )
    return query.strip().strip('"') or question


def answer_with_context(question, hits, history=None):
    messages = [{"role": "system", "content": ANSWER_SYSTEM_PROMPT}]
    messages += (history or [])[-6:]
    messages.append({"role": "user", "content": f"Context:\n{format_context(hits)}\n\nQuestion: {question}"})
    return ask_llm(messages)


def cited_sources(answer, hits):
    """Which of the retrieved passages did the model actually cite?"""
    used = sorted({int(n) for n in re.findall(r"\[(\d+)\]", answer) if 0 < int(n) <= len(hits)})
    return [(n, hits[n - 1]["metadata"]) for n in used]
