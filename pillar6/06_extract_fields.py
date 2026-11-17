"""
Step 6 - extracting invoice fields: three strategies, scored against the ground truth.

    A  pypdf text      -> LLM (gpt-oss-120b)   cheap, but blind on scans
    B  Docling markdown -> LLM (gpt-oss-120b)   layout + tables + OCR, then a text model
    C  page image      -> VLM (qwen3.8-27b)     one step, the model reads the image itself

For each: per-field accuracy, share of invoices fully correct, cost/latency, and what the
arithmetic validation catches (doc_extract.validate). Results saved to outputs/extraction_results.json.

Usage:
    python 06_extract_fields.py                   # all 24 invoices, ~72 API calls
    python 06_extract_fields.py --n 9             # quick run
    python 06_extract_fields.py --strategies B C
"""
import argparse
import json
import time

from doc_extract import (FIELDS, docling_markdown, extract_from_image, extract_from_text, load_labels, page_image,
                         pypdf_text, score, validate)
from p6_utils import INVOICES_DIR, OUTPUT_DIR, TEXT_MODEL, VISION_MODEL

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=0)
parser.add_argument("--strategies", nargs="+", default=["A", "B", "C"])
args = parser.parse_args()

labels = load_labels()[: args.n or None]
PRICES = {TEXT_MODEL: (0.15, 0.60), VISION_MODEL: (0.80, 4.00)}  # $/1M tokens (Groq, Oct 2026)

STRATEGIES = {
    "A": ("pypdf + LLM", TEXT_MODEL, lambda p: extract_from_text(pypdf_text(p))),
    "B": ("Docling + LLM", TEXT_MODEL, lambda p: extract_from_text(docling_markdown(p))),
    "C": ("image + VLM", VISION_MODEL, lambda p: extract_from_image(page_image(p))),
}

results = {}
for key in args.strategies:
    name, model, run = STRATEGIES[key]
    print(f"\n=== {key}. {name} ===")
    rows = []
    for lab in labels:
        start = time.perf_counter()
        inv, errors, meta = run(INVOICES_DIR / lab["file"])
        s = score(inv, lab)
        issues = validate(inv) if inv else errors
        cost = (meta.get("prompt", 0) * PRICES[model][0] + meta.get("completion", 0) * PRICES[model][1]) / 1e6
        rows.append({"file": lab["file"], "kind": lab["kind"], "doc_error": lab["doc_error"], "score": s,
                     "issues": issues, "cost": cost, "seconds": time.perf_counter() - start})
        wrong = [f for f in FIELDS if not s[f]]
        print(f"  {lab['file']} ({lab['kind']:<4}) {'OK ' if s['all'] else 'ERR'} "
              f"{'wrong: ' + ','.join(wrong) if wrong else '':<40} {'FLAGGED: ' + '; '.join(issues) if issues else ''}")
        time.sleep(0.5)
    results[key] = rows

# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print(f"{'strategy':<16}{'all ok':>8}{'pdf':>7}{'scan':>7}" + "".join(f"{f[:8]:>9}" for f in FIELDS[1:]) + f"{'$/doc':>9}{'s/doc':>7}")
for key, rows in results.items():
    n = len(rows)
    acc = lambda subset: sum(r["score"]["all"] for r in subset) / max(1, len(subset))
    line = f"{STRATEGIES[key][0]:<16}{acc(rows):>8.0%}{acc([r for r in rows if r['kind'] == 'pdf']):>7.0%}" \
           f"{acc([r for r in rows if r['kind'] == 'scan']):>7.0%}"
    line += "".join(f"{sum(r['score'][f] for r in rows) / n:>9.0%}" for f in FIELDS[1:])
    line += f"{sum(r['cost'] for r in rows) / n:>9.5f}{sum(r['seconds'] for r in rows) / n:>7.1f}"
    print(line)

print("\nWhat validation catches (flagged = arithmetic doesn't add up, or extraction failed):")
for key, rows in results.items():
    flagged = [r for r in rows if r["issues"]]
    passed = [r for r in rows if not r["issues"]]
    wrong_passed = [r["file"] for r in passed if not r["score"]["all"]]
    doc_errors = [r for r in rows if r["doc_error"]]
    print(f"  {STRATEGIES[key][0]:<16} flagged {len(flagged)}/{len(rows)} | invoices with a wrong printed total caught "
          f"{sum(bool(r['issues']) for r in doc_errors)}/{len(doc_errors)} | auto-accepted and correct "
          f"{len(passed) - len(wrong_passed)}/{len(passed)}" + (f" (wrong but accepted: {wrong_passed})" if wrong_passed else ""))

OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "extraction_results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

# How to read it:
#  - A fails on every scan (no text) and hesitates on tables. On digital PDFs a strong LLM often
#    reconstructs the table anyway: try it before adding heavy tooling.
#  - B vs C is the real debate in document AI today: a parsing pipeline + text LLM (cheap, local
#    parsing, auditable markdown) vs one VLM call (simpler, often better on messy scans, but more
#    expensive per page and the image goes to the API).
#  - the invoices with a wrong printed total are "correct" extractions that MUST still be flagged:
#    the extraction is right, the document is wrong. Only a business rule catches that.
#  - "wrong but accepted" is the number that matters for automation: errors that slip through
#    validation. Typically wrong text fields (supplier name, invoice number) that no arithmetic
#    can check -> cross-check with your supplier database / purchase orders (step 7 idea).
