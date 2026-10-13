"""Build the three saved documents the interface develops against: doubtful words, an invoice with a mismatch, and a
document with unreadable pages. Each is a PDF and the `.extract.json` saved from reading it with the real models.

    python3 scripts/make_fixtures.py
"""

from __future__ import annotations

import sys
import tempfile
from decimal import Decimal
from pathlib import Path

import pypdfium2 as pdfium

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import extract  # noqa: E402
from bench.corpus import invoices  # noqa: E402
from extract.engine.render import render_page  # noqa: E402
from tests.fake_guard import DoubtingGuard  # noqa: E402
from tests.support import SAMPLE, SANS, SPANISH_LINES, SetLexicon, scanned_pdf, text_page_image  # noqa: E402

OUT = ROOT / "tests" / "fixtures"
SUGGESTIONS = SAMPLE["suggestions"]


def read(path: Path, guard) -> extract.Document:
    options = extract.Options(work_dir=Path(tempfile.mkdtemp()), backend="onnxruntime", processes=1, threads=2)
    with extract.Reader(options, guard=guard, lexicon=SetLexicon(())) as reader:
        return reader.read(path)


def save(document: extract.Document) -> None:
    document.save(OUT / f"{Path(document.name).stem}.extract.json")
    print(document.name, document.status, len(document.pages), "pages,", len(document.doubtful()), "to check")


def doubtful() -> None:
    page = text_page_image(SPANISH_LINES, SANS, 24, 1654, 2339, margin=140)
    path = scanned_pdf(OUT / "doubtful.pdf", [page])
    doubts = {**SUGGESTIONS, "Factura": None, "2026-0042": None, "IVA": None, "1.210,00": None, "Gracias": None}
    save(read(path, DoubtingGuard(doubts)))


def mismatch() -> None:
    unit = invoices.plan(3)[0]
    invoice = invoices.make_invoice(unit["id"], unit["layout"], unit["font"], 3)
    invoice["total"] = invoice["total"] + Decimal("0.30")
    digital = Path(tempfile.mkdtemp()) / "invoice.pdf"
    invoices.render(invoice, unit["layout"], unit["font"], digital)
    document = pdfium.PdfDocument(str(digital))
    image = render_page(document[0], 200)
    document.close()
    path = scanned_pdf(OUT / "mismatch.pdf", [image])
    save(read(path, DoubtingGuard({})))


def unreadable() -> None:
    full = text_page_image(SPANISH_LINES, SANS, 24, 1654, 2339, margin=140)
    sparse = text_page_image(SAMPLE["sparse_page"], SANS, 30, 1654, 2339, margin=140)
    path = scanned_pdf(OUT / "unreadable.pdf", [full, sparse, full])
    save(read(path, DoubtingGuard({})))


if __name__ == "__main__":
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):  # `--help` must show this text, not run the script
        print(__doc__)
        sys.exit(0)
    OUT.mkdir(parents=True, exist_ok=True)
    doubtful()
    mismatch()
    unreadable()
