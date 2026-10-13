"""Fixtures built by the tests: digital, scanned-like, mixed, broken-layer and rotated PDFs, and page images."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pikepdf
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

FONT_DIR = Path("/usr/share/fonts/truetype")
SANS = FONT_DIR / "dejavu" / "DejaVuSans.ttf"
SERIF = FONT_DIR / "dejavu" / "DejaVuSerif.ttf"

SAMPLE = json.loads((Path(__file__).parent / "data" / "spanish.json").read_text(encoding="utf-8"))
SPANISH_LINES = SAMPLE["lines"]


class SetLexicon:
    """A lexicon over a plain set, with the interface of the guard's lexicon."""

    def __init__(self, words) -> None:
        self.words = set(words)

    def has(self, word, languages=("es", "en")) -> bool:
        return word.lower() in self.words

    def share(self, words, languages=("es", "en")) -> float:
        return sum(w.lower() in self.words for w in words) / len(words) if words else 0.0


def text_page_image(lines, font_path=SANS, size=26, width=1654, height=2339, margin=110, spacing=1.6) -> np.ndarray:
    """A grey page with the lines of text drawn on it, 200 dpi on A4 by default."""
    image = Image.new("L", (width, height), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(str(font_path), size)
    y = margin
    for line in lines:
        draw.text((margin, y), line, font=font, fill=0)
        y += int(size * spacing)
    return np.array(image)


def digital_pdf(path: Path, pages: list[list[str]], top: int = 800) -> Path:
    pdf = canvas.Canvas(str(path), pagesize=A4)
    for lines in pages:
        pdf.setFont("Helvetica", 11)
        y = top
        for line in lines:
            pdf.drawString(50, y, line)
            y -= 18
        pdf.showPage()
    pdf.save()
    return path


def scanned_pdf(path: Path, images: list[np.ndarray], page_size=A4) -> Path:
    """Each page is one image covering it, as a scanner makes it; no text layer."""
    pdf = canvas.Canvas(str(path), pagesize=page_size)
    for array in images:
        pdf.drawImage(ImageReader(Image.fromarray(array)), 0, 0, width=page_size[0], height=page_size[1])
        pdf.showPage()
    pdf.save()
    return path


def mixed_pdf(path: Path, digital_lines: list[str], scan: np.ndarray) -> Path:
    """Page 1 digital, page 2 a scan."""
    pdf = canvas.Canvas(str(path), pagesize=A4)
    pdf.setFont("Helvetica", 11)
    y = 800
    for line in digital_lines:
        pdf.drawString(50, y, line)
        y -= 18
    pdf.showPage()
    pdf.drawImage(ImageReader(Image.fromarray(scan)), 0, 0, width=A4[0], height=A4[1])
    pdf.showPage()
    pdf.save()
    return path


def broken_layer_pdf(path: Path) -> Path:
    """A page whose text layer is garbage: a Type 3 font with no Unicode map, so pdfium reports each character
    code as a control character."""
    pdf = pikepdf.Pdf.new()
    glyphs = {f"/g{i}": pdf.make_stream(b"500 0 d0") for i in range(128, 160)}
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type3,
            FontBBox=pikepdf.Array([0, 0, 1000, 1000]),
            FontMatrix=pikepdf.Array([0.001, 0, 0, 0.001, 0, 0]),
            CharProcs=pdf.make_indirect(pikepdf.Dictionary(glyphs)),
            Encoding=pikepdf.Dictionary(
                Type=pikepdf.Name.Encoding,
                Differences=pikepdf.Array([128] + [pikepdf.Name(f"/g{i}") for i in range(128, 160)]),
            ),
            FirstChar=128,
            LastChar=159,
            Widths=pikepdf.Array([500] * 32),
        )
    )
    page = pdf.add_blank_page(page_size=(595, 842))
    page.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F1=font))
    body = "".join("\\%o" % (128 + (i * 7) % 32) if i % 6 else " " for i in range(1, 200))
    page.Contents = pdf.make_stream(f"BT /F1 12 Tf 50 700 Td ({body}) Tj ET".encode("latin-1"))
    pdf.save(path)
    return path


def rotated_cropped_pdf(path: Path, rotation: int = 90, crop=(40, 60, 500, 700)) -> Path:
    """A digital page with /Rotate and a CropBox that does not start at the origin."""
    source = path.with_suffix(".plain.pdf")
    digital_pdf(source, [[f"Line {n} of the page with some words to place" for n in range(1, 21)]], top=680)
    with pikepdf.open(source) as pdf:
        page = pdf.pages[0]
        page.Rotate = rotation
        page.CropBox = pikepdf.Array(crop)
        pdf.save(path)
    source.unlink()
    return path


def broken_page_pdf(path: Path, pages: int = 3, broken: int = 1) -> Path:
    """A PDF whose page `broken` (0-based) cannot be loaded by pdfium."""
    source = path.with_suffix(".whole.pdf")
    digital_pdf(source, [SPANISH_LINES] * pages)
    with pikepdf.open(source) as pdf:
        pdf.Root.Pages.Kids[broken] = pikepdf.Name.Foo
        pdf.save(path)
    source.unlink()
    return path


def grid_reading(number: int, texts: list[str], dpi: int = 150, size_pt=(595.28, 841.89), per_row: int = 8):
    """A page reading with the words laid out in rows, as a scan of a text page would give them."""
    from extract.types import PageReading, Word, word_id

    scale = dpi / 72
    width_px, height_px = round(size_pt[0] * scale), round(size_pt[1] * scale)
    column_px = (width_px - 160) / per_row
    words = []
    for index, text in enumerate(texts):
        row, column = divmod(index, per_row)
        left = 80 + column * column_px
        top = 100 + row * 40
        words.append(
            Word(
                word_id(number, index),
                number,
                text,
                (left, top, left + column_px * 0.8, top + 28),
                0.9,
                None,
                row,
                column,
            )
        )
    return PageReading(number, width_px, height_px, size_pt[0], size_pt[1], dpi, tuple(words), "ocr", None, None, 0.9)


def sample_document(source_pdf: Path):
    """A two-page document with doubtful, accepted and corrected words, numbers in every status and an invoice."""
    from decimal import Decimal

    from extract.engine.cache import file_sha256
    from extract.types import (
        Cell,
        DigitSuggestion,
        DocumentData,
        InvoiceFields,
        InvoiceLine,
        NumberResult,
        Reason,
        RuleResult,
        VatPair,
    )

    texts = SAMPLE["invoice_words"]
    first = grid_reading(1, texts)
    words = list(first.words)
    words[1] = replace_word(
        words[1],
        status="doubtful",
        p_wrong=0.62,
        reasons=(Reason("guard.reason.low_char_conf", {"char": "0", "conf": 0.41}),),
        suggestion="2026-0043",
        suggestion_conf=0.7,
    )
    words[5] = replace_word(words[5], status="ok", p_wrong=0.02)
    words[8] = replace_word(
        words[8],
        text=SAMPLE["word"],
        original=SAMPLE["word_plain"],
        review="corrected",
        corrected_by="user",
        status="ok",
    )
    words[9] = replace_word(words[9], status="unscored")
    first = type(first)(**{**first.__dict__, "words": tuple(words)})
    second = grid_reading(2, SAMPLE["thanks"])
    second = type(second)(**{**second.__dict__, "status": "unreadable", "reason": "page.reason.few_words"})
    cell = lambda text, ids, page=1: Cell(text, tuple(ids), page)  # noqa: E731
    numbers = (
        NumberResult(
            ("p1w3",),
            "1.000,00",
            Decimal("1000.00"),
            "amount",
            "checked",
            (RuleResult("checks.rule.total", {"expected": "1210.00", "found": "1210.00"}, "pass", True),),
        ),
        NumberResult(
            ("p1w5",),
            "210,00",
            Decimal("210.00"),
            "amount",
            "mismatch",
            (
                RuleResult(
                    "checks.rule.vat_amount", {"rate": "21", "expected": "210.00", "found": "210.00"}, "fail", True
                ),
            ),
            DigitSuggestion(2, "1", "0"),
        ),
        NumberResult(("p1w7",), "1.210,00", Decimal("1210.00"), "amount", "unverified"),
        NumberResult(("p1w11",), "14/03/2026", None, "date", "unverified"),
    )
    invoice = InvoiceFields(
        number=cell("2026-0042", ["p1w1"]),
        date=cell("14/03/2026", ["p1w11"]),
        subtotal=cell("1.000,00", ["p1w3"]),
        vat=(VatPair(cell("21", ["p1w4"]), cell("210,00", ["p1w5"]), cell("1.000,00", ["p1w3"])),),
        total=cell("1.210,00", ["p1w7"]),
        currency="EUR",
        lines=(InvoiceLine(cell("Base", ["p1w2"]), None, None, cell("1.000,00", ["p1w3"])),),
    )
    return DocumentData(
        source=source_pdf.name,
        sha256=file_sha256(source_pdf),
        pages=(first, second),
        engine="rapidocr-test|v1",
        guard="guard-test",
        options={"languages": ["es", "en"], "review": "balanced"},
        doc_type="invoice",
        locale="es",
        numbers=numbers,
        invoice=invoice,
    )


def replace_word(word, **changes):
    from dataclasses import replace

    return replace(word, **changes)
