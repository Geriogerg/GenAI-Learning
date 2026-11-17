"""
Invoice extraction building blocks, shared by 06 (comparison) and 07 (pipeline).

    read:     pypdf text layer | Docling markdown (layout + tables + OCR) | page image
    extract:  text -> LLM -> JSON   |   image -> VLM -> JSON
    validate: schema (pydantic) + arithmetic checks
    score:    against data/invoices/labels.json
"""
import json
import re
from pathlib import Path

import cv2
import numpy as np
from pydantic import BaseModel, ValidationError, field_validator

from p6_utils import INVOICES_DIR, OUTPUT_DIR, chat, parse_json, vision_chat

# ---------------------------------------------------------------------------
# schema
# ---------------------------------------------------------------------------


class Line(BaseModel):
    description: str
    quantity: float
    unit_price: float
    amount: float


class Invoice(BaseModel):
    supplier: str
    invoice_number: str
    date: str                 # ISO YYYY-MM-DD
    currency: str             # EUR, USD...
    lines: list[Line]
    subtotal: float
    vat_rate: float           # percent, e.g. 20
    vat_amount: float
    total: float

    @field_validator("date")
    @classmethod
    def iso_date(cls, v):
        if not re.fullmatch(r"\d{4}-\d{2}-\d{2}", v):
            raise ValueError("date must be YYYY-MM-DD")
        return v


PROMPT = """Extract the invoice data as a JSON object with exactly these fields:
{"supplier": "company that issued the invoice", "invoice_number": str, "date": "YYYY-MM-DD",
 "currency": "EUR" | "USD" | ..., "lines": [{"description": str, "quantity": number, "unit_price": number, "amount": number}],
 "subtotal": number, "vat_rate": number (percent, e.g. 20), "vat_amount": number, "total": number}
Rules: numbers as plain JSON numbers (1234.5, not "1 234,50 €"). Copy values exactly as printed on the
document, even if they look wrong: don't correct or recompute anything. Return only the JSON."""

# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------


def pypdf_text(path):
    """The text layer of a digital PDF. Scans have none."""
    path = Path(path)
    if path.suffix.lower() != ".pdf":
        return ""
    from pypdf import PdfReader
    return "\n".join(p.extract_text() or "" for p in PdfReader(path).pages).strip()


_converter = None


def docling_markdown(path, cache=True):
    """Docling: layout analysis + table structure + OCR when there's no text layer -> markdown.
    Cached in outputs/docling_md/ because it takes seconds per page on CPU."""
    global _converter
    path = Path(path)
    cached = OUTPUT_DIR / "docling_md" / (path.stem + ".md")
    if cache and cached.exists():
        return cached.read_text(encoding="utf-8")
    if _converter is None:
        from docling.document_converter import DocumentConverter
        _converter = DocumentConverter()  # first call downloads the layout/table/OCR models
    md = _converter.convert(path).document.export_to_markdown()
    cached.parent.mkdir(parents=True, exist_ok=True)
    cached.write_text(md, encoding="utf-8")
    return md


def page_image(path, dpi=150):
    """First page as a BGR image (PDFs are rendered, images are loaded)."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        import pypdfium2 as pdfium
        pil = pdfium.PdfDocument(str(path))[0].render(scale=dpi / 72).to_pil()
        return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    return cv2.imread(str(path))


# ---------------------------------------------------------------------------
# extraction
# ---------------------------------------------------------------------------


def _to_invoice(text):
    data = parse_json(text)
    if data is None:
        return None, ["invalid JSON"]
    try:
        return Invoice.model_validate(data), []
    except ValidationError as e:
        return None, [f"schema: {err['loc']} {err['msg']}" for err in e.errors()[:3]]


def extract_from_text(text):
    if len(text.strip()) < 30:
        return None, ["no text to extract from"], {}
    out, secs, usage = chat([{"role": "system", "content": PROMPT}, {"role": "user", "content": text}], json_mode=True)
    inv, errors = _to_invoice(out)
    return inv, errors, {"seconds": secs, **usage}


def extract_from_image(image):
    out, secs, usage = vision_chat([image], PROMPT)
    inv, errors = _to_invoice(out)
    return inv, errors, {"seconds": secs, **usage}


# ---------------------------------------------------------------------------
# validation: the arithmetic must hold. Catches extraction errors AND wrong invoices.
# ---------------------------------------------------------------------------


def validate(inv: Invoice, tol=0.02):
    issues = []
    for i, l in enumerate(inv.lines):
        if abs(l.quantity * l.unit_price - l.amount) > tol + 0.005 * l.amount:
            issues.append(f"line {i + 1}: qty x price != amount")
    if abs(sum(l.amount for l in inv.lines) - inv.subtotal) > tol:
        issues.append("sum of lines != subtotal")
    if abs(inv.subtotal * inv.vat_rate / 100 - inv.vat_amount) > tol:
        issues.append("subtotal x rate != VAT")
    if abs(inv.subtotal + inv.vat_amount - inv.total) > tol:
        issues.append("subtotal + VAT != total")
    return issues


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------

FIELDS = ["supplier", "invoice_number", "date", "currency", "lines", "subtotal", "vat_rate", "vat_amount", "total"]


def load_labels():
    return json.loads((INVOICES_DIR / "labels.json").read_text(encoding="utf-8"))


def score(inv: Invoice | None, label):
    if inv is None:
        return {f: False for f in FIELDS} | {"all": False}
    close = lambda a, b: abs(a - b) <= 0.01
    s = {
        "supplier": label["supplier"].lower() in inv.supplier.lower() or inv.supplier.lower() in label["supplier"].lower(),
        "invoice_number": inv.invoice_number.strip() == label["invoice_number"],
        "date": inv.date == label["date"],
        "currency": inv.currency.upper() == label["currency"],
        "lines": len(inv.lines) == len(label["lines"]) and all(
            close(a.amount, b["amount"]) and close(a.quantity, b["quantity"]) for a, b in zip(inv.lines, label["lines"])),
        "subtotal": close(inv.subtotal, label["subtotal"]),
        "vat_rate": close(inv.vat_rate, label["vat_rate"]),
        "vat_amount": close(inv.vat_amount, label["vat_amount"]),
        "total": close(inv.total, label["total"]),
    }
    s["all"] = all(s.values())
    return s
