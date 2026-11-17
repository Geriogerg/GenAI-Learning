"""
Pillar 6 mini project - an invoice processing pipeline.

What an accounts-payable automation looks like, with everything from the pillars:

  1. route        digital PDF (text layer) -> Docling + text LLM (cheap)
                  scan / photo             -> VLM
  2. extract      JSON, validated against a schema (pydantic)
  3. check        arithmetic + business rules (known supplier, duplicate invoice number)
  4. retry        if the checks fail, try the OTHER strategy once (different errors)
  5. decide       auto-accept  |  human review queue (with the reasons)
  6. output       accepted.csv, review_queue.csv, extracted.jsonl, report.md scored vs ground truth

Usage:
    python 07_project_invoice_pipeline.py
    python 07_project_invoice_pipeline.py --folder path/to/your/invoices   (no scoring without labels)
"""
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

from doc_extract import (docling_markdown, extract_from_image, extract_from_text, load_labels, page_image, pypdf_text,
                         score, validate)
from p6_utils import INVOICES_DIR, OUTPUT_DIR

parser = argparse.ArgumentParser()
parser.add_argument("--folder", default=str(INVOICES_DIR))
args = parser.parse_args()
folder = Path(args.folder)

# business data the pipeline checks against (in real life: the ERP)
KNOWN_SUPPLIERS = {"LabelCo SAS", "Nordic Optics GmbH", "Atlas Fixtures Ltd", "Visionics SARL", "CamParts Inc."}
seen_numbers = set()

run_dir = OUTPUT_DIR / f"invoices_{datetime.now():%Y%m%d_%H%M%S}"
run_dir.mkdir(parents=True)


def via_text(path):
    return extract_from_text(docling_markdown(path))


def via_image(path):
    return extract_from_image(page_image(path))


def business_checks(inv):
    issues = []
    if not any(s.lower() == inv.supplier.strip().lower() for s in KNOWN_SUPPLIERS):
        issues.append(f"unknown supplier '{inv.supplier}'")
    if inv.invoice_number in seen_numbers:
        issues.append("duplicate invoice number (already processed)")
    return issues


def process(path):
    has_text = len(pypdf_text(path)) > 50
    attempts = [("docling+llm", via_text), ("vlm", via_image)] if has_text else [("vlm", via_image), ("docling+llm", via_text)]
    log = []
    for name, fn in attempts:
        inv, errors, meta = fn(path)
        issues = errors if inv is None else validate(inv) + business_checks(inv)
        log.append({"strategy": name, "issues": issues, "seconds": round(meta.get("seconds", 0), 1)})
        if inv is not None and not issues:
            return inv, "accepted", log
        # Arithmetic says the DOCUMENT itself is wrong if both strategies read the same numbers:
        # no point retrying forever, a human has to call the supplier.
    return inv, "review", log


rows, start = [], time.time()
for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in (".pdf", ".png", ".jpg", ".jpeg")):
    inv, decision, log = process(path)
    if inv is not None and decision == "accepted":
        seen_numbers.add(inv.invoice_number)
    reasons = log[-1]["issues"] if decision == "review" else []
    rows.append({"file": path.name, "decision": decision, "invoice": inv.model_dump() if inv else None,
                 "reasons": reasons, "attempts": log})
    print(f"  {path.name:<12} {decision:<9} via {' -> '.join(a['strategy'] for a in log):<20} {'; '.join(reasons)}")

# ---------------------------------------------------------------------------
# outputs
# ---------------------------------------------------------------------------
with (run_dir / "extracted.jsonl").open("w", encoding="utf-8") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with (run_dir / "accepted.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["file", "supplier", "invoice_number", "date", "currency", "subtotal", "vat_amount", "total"])
    for r in rows:
        if r["decision"] == "accepted":
            i = r["invoice"]
            w.writerow([r["file"], i["supplier"], i["invoice_number"], i["date"], i["currency"], i["subtotal"],
                        i["vat_amount"], i["total"]])

with (run_dir / "review_queue.csv").open("w", newline="", encoding="utf-8") as f:
    w = csv.writer(f, delimiter=";")
    w.writerow(["file", "reasons", "extracted_total"])
    for r in rows:
        if r["decision"] == "review":
            w.writerow([r["file"], " | ".join(r["reasons"]), (r["invoice"] or {}).get("total")])

# ---------------------------------------------------------------------------
# scoring (only possible on the synthetic set, where we know the truth)
# ---------------------------------------------------------------------------
lines = [f"# Invoice run {datetime.now():%Y-%m-%d %H:%M}", "",
         f"{len(rows)} documents in {time.time() - start:.0f}s: "
         f"{sum(r['decision'] == 'accepted' for r in rows)} accepted, {sum(r['decision'] == 'review' for r in rows)} to review, "
         f"{sum(len(r['attempts']) > 1 for r in rows)} needed the fallback strategy.", ""]

labels = {x["file"]: x for x in load_labels()} if (folder / "labels.json").exists() else {}
if labels:
    from doc_extract import Invoice
    accepted_ok = accepted_wrong = review_needed = review_unneeded = 0
    for r in rows:
        lab = labels.get(r["file"])
        if not lab:
            continue
        correct = r["invoice"] is not None and score(Invoice.model_validate(r["invoice"]), lab)["all"] and not lab["doc_error"]
        if r["decision"] == "accepted":
            accepted_ok += correct
            accepted_wrong += not correct
        else:
            review_needed += not correct
            review_unneeded += correct
    lines += ["## Score vs ground truth", "",
              "| | count |", "|---|---|",
              f"| accepted and correct | {accepted_ok} |",
              f"| **accepted but wrong** (the costly case) | {accepted_wrong} |",
              f"| sent to review, rightly | {review_needed} |",
              f"| sent to review, was actually fine | {review_unneeded} |", ""]
    print(f"\naccepted & correct {accepted_ok} | ACCEPTED BUT WRONG {accepted_wrong} | "
          f"review needed {review_needed} | review unneeded {review_unneeded}")

lines += ["## Review queue", ""] + [f"- **{r['file']}**: {'; '.join(r['reasons'])}" for r in rows if r["decision"] == "review"]
(run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
print(f"Outputs in {run_dir}: accepted.csv, review_queue.csv, extracted.jsonl, report.md")

# The metric a finance team cares about is "accepted but wrong": a wrong amount paid automatically.
# Everything in this design pushes it to zero, at the cost of some unneeded reviews. Tune it with
# your users: how many reviews per day can they absorb vs how much risk is acceptable?
# Ideas: confidence from agreement (run both strategies, accept only if they agree), match against
# purchase orders, expose run as an API (pillar 5) or as an agent tool (pillar 3).
