"""
Step 5 - reading documents: plain text extraction vs layout-aware parsing.

In pillar 2 you used pypdf: fine for papers, but it only reads the PDF's text layer, in drawing
order. Invoices break it in two ways:
  - tables come out as a flat soup of words and numbers (which number is the unit price?)
  - scans have no text layer at all -> nothing

Docling (IBM, open source) runs a layout model (finds titles, paragraphs, tables...), a table
structure model, and OCR when there's no text, and outputs clean markdown with real tables.
Everything runs locally on CPU (a few seconds per page). The first run downloads its models.

No API calls.
"""
import time

from doc_extract import docling_markdown, load_labels, pypdf_text
from p6_utils import INVOICES_DIR

labels = load_labels()
pdf = next(x for x in labels if x["kind"] == "pdf" and x["layout"] == "fr")
scan = next(x for x in labels if x["kind"] == "scan")

for inv in [pdf, scan]:
    path = INVOICES_DIR / inv["file"]
    print("=" * 90)
    print(f"{inv['file']} ({inv['kind']}, layout {inv['layout']})")
    print("=" * 90)

    text = pypdf_text(path)
    print(f"\n--- pypdf text layer ({len(text)} chars) ---")
    print(text[:700] if text else "(nothing: no text layer, it's an image)")

    start = time.perf_counter()
    md = docling_markdown(path, cache=False)
    print(f"\n--- Docling markdown ({len(md)} chars, {time.perf_counter() - start:.1f}s) ---")
    print(md[:1500])
    print()

# The tables as structured data (useful even without any LLM)
from docling.document_converter import DocumentConverter  # noqa: E402

doc = DocumentConverter().convert(INVOICES_DIR / pdf["file"]).document
print("=" * 90)
print(f"Docling found {len(doc.tables)} table(s) in {pdf['file']}. First one as a DataFrame:\n")
if doc.tables:
    print(doc.tables[0].export_to_dataframe(doc=doc))

# What to compare:
#  - pypdf on the digital PDF: all the text is there, but the table structure is gone. An LLM can
#    often recover it, a regex can't.
#  - pypdf on the scan: empty. This is why "PDF parsing" projects fail on real archives, where a
#    large share of documents are scans or photos.
#  - Docling on the scan: OCR errors are possible (look at the numbers: 8 vs 6, comma vs dot).
#    Every downstream number depends on them -> step 6 measures it, step 7 validates it.
#
# Alternatives worth knowing: Docling's VLM mode (granite-docling, one small model doing everything),
# Marker, MinerU, and cloud services (Azure Document Intelligence, AWS Textract, Google Document AI).
