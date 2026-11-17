"""
Step 4 - loading real documents.

Hardcoded one-liners were fine to get the idea, but a real knowledge base is PDFs, wiki
pages, tickets... This script looks at what you actually get when you extract text from a
PDF, because a lot of RAG quality is lost here, before the LLM is even involved.

Put one or two PDFs in data/papers/ before running (see data/papers/README.txt).
"""
from collections import Counter

from pypdf import PdfReader

from rag_utils import DATA_DIR, PAPERS_DIR, clean_text, load_documents

pdfs = sorted(PAPERS_DIR.glob("*.pdf"))

if not pdfs:
    print(f"No PDF in {PAPERS_DIR}, only the markdown docs will be loaded.")
    print("Download a paper or two from arxiv (see data/papers/README.txt) to see the PDF part.\n")
else:
    pdf = pdfs[0]
    reader = PdfReader(pdf)
    raw = reader.pages[0].extract_text() or ""

    print(f"{pdf.name}: {len(reader.pages)} pages\n")
    print("---- raw text of page 1 (first 700 chars) ----")
    print(raw[:700])
    print("\n---- after clean_text() ----")
    print(clean_text(raw, from_pdf=True)[:700])
    print()

    # Things to look for in the raw text, they're the usual PDF problems:
    #  - a line break at the end of every visual line (clean_text joins them)
    #  - words cut with a hyphen at the end of a line: "detec-\ntion" (clean_text fixes it)
    #  - two-column layouts: sometimes the columns get interleaved, check a middle page
    #  - headers / footers / page numbers repeated on every page
    #  - tables: come out as a soup of numbers, the structure is lost
    #  - equations: mostly garbage
    # For the hard cases (tables, scans) people use layout-aware parsers like
    # unstructured, docling or marker. pypdf is enough for text-heavy papers.

    lengths = [len(p.extract_text() or "") for p in reader.pages]
    empty = [i + 1 for i, n in enumerate(lengths) if n < 50]
    print(f"chars per page: min={min(lengths)} max={max(lengths)} avg={sum(lengths) // len(lengths)}")
    if empty:
        print(f"pages with (almost) no text: {empty}  <- figures only, or a scanned pdf that needs OCR")
    print()

# The loader used by the rest of pillar 2 (rag_utils.load_documents):
# one record per PDF page so we can cite page numbers, one record per markdown file.
docs = load_documents(DATA_DIR)

per_source = Counter(d["source"] for d in docs)
chars = Counter()
for d in docs:
    chars[d["source"]] += len(d["text"])

print(f"Loaded {len(docs)} records from {len(per_source)} files:")
for source, n in per_source.items():
    print(f"  {source:<40} {n:>3} page(s)  {chars[source]:>7} chars  (~{chars[source] // 4} tokens)")

total = sum(chars.values())
print(f"\nTotal: ~{total // 4} tokens. Way too much to paste into one prompt -> we need chunks (step 5).")
