"""
Step 4 - a synthetic invoice dataset with exact ground truth.

Document AI is easy to demo and hard to evaluate: you need documents AND the right answer for
each field. So, as with the plates, we generate them:
  - 3 layouts (English, French, US style) with different labels, date and number formats
  - digital PDFs (text layer present) and "scans" (rendered, rotated, noisy, blurred images)
  - 2 invoices whose own arithmetic is wrong (it happens with real suppliers): the pipeline
    in step 7 should catch them

    python 04_make_invoices.py
    -> data/invoices/inv_XX.pdf / inv_XX.png + data/invoices/labels.json
"""
import json
import random
from datetime import date, timedelta

import cv2
import numpy as np
import pypdfium2 as pdfium
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

from p6_utils import INVOICES_DIR

rng = random.Random(42)
styles = getSampleStyleSheet()

SUPPLIERS = [
    ("LabelCo SAS", "12 rue des Lilas, 69003 Lyon, France"),
    ("Nordic Optics GmbH", "Hafenstrasse 8, 20457 Hamburg, Germany"),
    ("Atlas Fixtures Ltd", "4 Mill Lane, Leeds LS1 4AB, United Kingdom"),
    ("Visionics SARL", "3 avenue Foch, 75116 Paris, France"),
    ("CamParts Inc.", "1200 Market St, San Jose, CA 95113, USA"),
]
ITEMS = [("Industrial camera 5MP", 480, 1900), ("C-mount lens 16mm", 90, 420), ("LED ring light", 60, 310),
         ("Image labeling, 1000 boxes", 40, 40), ("Calibration target", 35, 180), ("Mounting bracket", 12, 45),
         ("GigE cable 10m", 18, 60), ("Polarizing filter", 25, 140), ("Annual software license", 900, 2400)]

LAYOUTS = {
    "en": {"title": "INVOICE", "no": "Invoice number", "date": "Date", "to": "Bill to", "cols": ["Description", "Qty", "Unit price", "Amount"],
           "sub": "Subtotal", "vat": "VAT {r}%", "total": "Total", "currency": "EUR"},
    "fr": {"title": "FACTURE", "no": "N° de facture", "date": "Date", "to": "Facturé à", "cols": ["Désignation", "Qté", "Prix unitaire", "Montant"],
           "sub": "Total HT", "vat": "TVA {r} %", "total": "Total TTC", "currency": "EUR"},
    "us": {"title": "Invoice", "no": "Invoice #", "date": "Issued", "to": "Customer", "cols": ["Item", "Units", "Price", "Line total"],
           "sub": "Subtotal", "vat": "Sales tax {r}%", "total": "Amount due", "currency": "USD"},
}


def money(x, layout):
    if layout == "fr":
        return f"{x:,.2f}".replace(",", " ").replace(".", ",") + " €"
    return ("$" if layout == "us" else "€ ") + f"{x:,.2f}"


def fmt_date(d, layout):
    return {"fr": d.strftime("%d/%m/%Y"), "us": d.strftime("%B %d, %Y").replace(" 0", " "), "en": d.strftime("%d %b %Y")}[layout]


def make_invoice(i, layout, wrong_total=False):
    supplier, address = rng.choice(SUPPLIERS)
    d = date(2026, 1, 5) + timedelta(days=rng.randrange(300))
    lines = []
    for name, lo, hi in rng.sample(ITEMS, rng.randint(2, 5)):
        qty = rng.choice([1, 1, 2, 3, 4, 5, 10])
        unit = round(rng.uniform(lo, hi), 2)
        lines.append({"description": name, "quantity": qty, "unit_price": unit, "amount": round(qty * unit, 2)})
    subtotal = round(sum(l["amount"] for l in lines), 2)
    rate = {"fr": 20.0, "en": rng.choice([19.0, 20.0]), "us": 8.5}[layout]
    vat = round(subtotal * rate / 100, 2)
    total = round(subtotal + vat, 2)
    printed_total = round(total + rng.choice([-100, 100, 9.0]), 2) if wrong_total else total
    number = f"{supplier[:3].upper()}-{d.year}-{rng.randint(1000, 9999)}"
    return {"supplier": supplier, "address": address, "invoice_number": number, "date": d.isoformat(),
            "currency": LAYOUTS[layout]["currency"], "lines": lines, "subtotal": subtotal, "vat_rate": rate,
            "vat_amount": vat, "total": printed_total, "layout": layout, "doc_error": wrong_total}


def render_pdf(inv, path):
    L = LAYOUTS[inv["layout"]]
    lay = inv["layout"]
    doc = SimpleDocTemplate(str(path), pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm)
    head = [[Paragraph(f"<b>{inv['supplier']}</b><br/>{inv['address']}", styles["Normal"]),
             Paragraph(f"<font size=18><b>{L['title']}</b></font>", styles["Normal"])]]
    meta = [[f"{L['no']}:", inv["invoice_number"]], [f"{L['date']}:", fmt_date(date.fromisoformat(inv["date"]), lay)],
            [f"{L['to']}:", "CV Team, Acme Manufacturing"]]
    rows = [L["cols"]] + [[l["description"], str(l["quantity"]), money(l["unit_price"], lay), money(l["amount"], lay)]
                          for l in inv["lines"]]
    totals = [[L["sub"], money(inv["subtotal"], lay)],
              [L["vat"].format(r=f"{inv['vat_rate']:g}".replace(".", "," if lay == "fr" else ".")), money(inv["vat_amount"], lay)],
              [L["total"], money(inv["total"], lay)]]

    t = Table(rows, colWidths=[80 * mm, 18 * mm, 32 * mm, 34 * mm])
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dfe6ee")), ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                           ("ALIGN", (1, 1), (-1, -1), "RIGHT"), ("LINEBELOW", (0, 0), (-1, -1), 0.3, colors.grey)]))
    tt = Table(totals, colWidths=[40 * mm, 34 * mm], hAlign="RIGHT")
    tt.setStyle(TableStyle([("ALIGN", (1, 0), (1, -1), "RIGHT"), ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                            ("LINEABOVE", (0, -1), (-1, -1), 0.8, colors.black)]))
    doc.build([Table(head, colWidths=[110 * mm, 60 * mm]), Spacer(1, 10 * mm), Table(meta, hAlign="LEFT"),
               Spacer(1, 8 * mm), t, Spacer(1, 6 * mm), tt, Spacer(1, 14 * mm),
               Paragraph("Payment within 30 days. Thank you for your business.", styles["Italic"])])


def scan(pdf_path, png_path, seed):
    """Print it, scan it: render at 150 dpi, small rotation, noise, blur, JPEG artifacts."""
    r = np.random.default_rng(seed)
    page = pdfium.PdfDocument(str(pdf_path))[0]
    img = cv2.cvtColor(np.array(page.render(scale=150 / 72).to_pil()), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), r.uniform(-1.5, 1.5), 1.0)
    img = cv2.warpAffine(img, m, (w, h), borderValue=(245, 245, 240))
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    img = np.clip(img.astype(np.int16) + r.normal(0, 8, img.shape) - 10, 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 60])
    cv2.imwrite(str(png_path), cv2.imdecode(buf, cv2.IMREAD_COLOR))


def main():
    INVOICES_DIR.mkdir(parents=True, exist_ok=True)
    labels = []
    layouts = ["en", "fr", "us"] * 8
    for i, lay in enumerate(layouts):
        inv = make_invoice(i, lay, wrong_total=i in (5, 17))
        pdf = INVOICES_DIR / f"inv_{i:02d}.pdf"
        render_pdf(inv, pdf)
        if i % 3 == 2 or i in (4, 10, 16):  # 1/3 of the invoices arrive as scans
            scan(pdf, INVOICES_DIR / f"inv_{i:02d}.png", seed=i)
            pdf.unlink()
            inv["file"], inv["kind"] = f"inv_{i:02d}.png", "scan"
        else:
            inv["file"], inv["kind"] = f"inv_{i:02d}.pdf", "pdf"
        labels.append(inv)
    (INVOICES_DIR / "labels.json").write_text(json.dumps(labels, indent=1, ensure_ascii=False), encoding="utf-8")
    kinds = {k: sum(x["kind"] == k for x in labels) for k in ("pdf", "scan")}
    print(f"{len(labels)} invoices in {INVOICES_DIR}: {kinds}, layouts en/fr/us, "
          f"{sum(x['doc_error'] for x in labels)} with a wrong total printed on them")


if __name__ == "__main__":
    main()
