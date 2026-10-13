"""The searchable PDF: the original pages untouched, plus an invisible text layer per page."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pikepdf

from ..fixes import tokens
from ..types import PageReading
from .helvetica import WIDTHS

FONT_NAME = "/ExtractOcr"
MIN_SCALE, MAX_SCALE = 1.0, 2000.0
FONT_HEIGHT_SHARE = 0.85
DESCENT_SHARE = 0.2


def encode_winansi(text: str) -> tuple[bytes, int]:
    """The word as WinAnsi bytes, with each character outside that encoding written as '?'; also how many were."""
    out, replaced = bytearray(), 0
    for char in text:
        try:
            out += char.encode("cp1252")
        except UnicodeEncodeError:
            out += b"?"
            replaced += 1
    return bytes(out), replaced


def pdf_string(data: bytes) -> bytes:
    escaped = bytearray(b"(")
    for byte in data:
        if byte in b"()\\":
            escaped += b"\\" + bytes([byte])
        elif 32 <= byte < 127:
            escaped.append(byte)
        else:
            escaped += b"\\%03o" % byte
    return bytes(escaped) + b")"


def text_width(data: bytes, size: float) -> float:
    return sum(WIDTHS[b] for b in data) * size / 1000


def _inherited(page: pikepdf.Object, key: str):
    node = page
    while node is not None:
        if key in node:
            return node[key]
        node = node.get("/Parent")
    return None


def _visible_box(page: pikepdf.Object) -> tuple[float, float, float, float]:
    def numbers(box):
        x0, y0, x1, y1 = (float(v) for v in box)
        return min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)

    media = numbers(_inherited(page, "/MediaBox"))
    crop = _inherited(page, "/CropBox")
    if crop is None:
        return media
    crop = numbers(crop)
    return max(media[0], crop[0]), max(media[1], crop[1]), min(media[2], crop[2]), min(media[3], crop[3])


def display_to_user(page: pikepdf.Object) -> tuple[float, float, float, float, float, float]:
    """The matrix (a b c d e f) that takes the displayed page, origin at its lower left, to the page's user space,
    whatever its /Rotate and where its CropBox starts."""
    left, bottom, right, top = _visible_box(page)
    width, height = right - left, top - bottom
    rotation = int(_inherited(page, "/Rotate") or 0) % 360
    if rotation == 90:
        return 0, 1, -1, 0, left + width, bottom
    if rotation == 180:
        return -1, 0, 0, -1, left + width, bottom + height
    if rotation == 270:
        return 0, -1, 1, 0, left, bottom + height
    return 1, 0, 0, 1, left, bottom


def text_layer(reading: PageReading) -> tuple[bytes, int]:
    """The content of the invisible text of one page and the number of characters written as '?'."""
    lines, replaced_total = [], 0
    for word in tokens(reading):
        data, replaced = encode_winansi(word.text)
        replaced_total += replaced
        left, top, right, bottom = reading.box_to_points(word.box)
        height = max(bottom - top, 1.0)
        size = height * FONT_HEIGHT_SHARE
        natural = text_width(data + b" ", size)
        scale = 100.0 if natural <= 0 else min(max(100.0 * (right - left) / natural, MIN_SCALE), MAX_SCALE)
        baseline = reading.height_pt - (bottom - DESCENT_SHARE * height)
        # A space ends every word: a viewer that guesses word breaks from the gaps (Chrome and Edge, through pdfium) joined
        # about 4 % of the neighbouring words of a page, so a search for two neighbouring words found nothing. The space
        # is inside the box of its word, scaled with it: one that stuck out over the next word was dropped by poppler
        # (Evince, Okular, pdftotext), which then joined 5 to 8 % of the neighbours instead.
        lines.append(
            b"%s %.2f Tf %.1f Tz 1 0 0 1 %.2f %.2f Tm %s Tj"
            % (FONT_NAME.encode(), size, scale, left, baseline, pdf_string(data + b" "))
        )
    return b"\n".join(lines), replaced_total


def unreadable_with_text(pages: Sequence[PageReading]) -> list[int]:
    """The pages the reader gave up on that still get an invisible text layer from the words it did find."""
    return [p.number for p in pages if p.status == "unreadable" and p.words]


def add_text_layer(source: Path, pages: Sequence[PageReading], destination: Path) -> int:
    """Write `destination`: `source` with an invisible text layer on every page that was read by OCR. Native pages
    are left as they are. Returns how many characters could not be written in WinAnsi and became '?'."""
    replaced_total = 0
    by_number = {reading.number: reading for reading in pages}
    with pikepdf.open(source) as pdf:
        font = pdf.make_indirect(
            pikepdf.Dictionary(
                Type=pikepdf.Name.Font,
                Subtype=pikepdf.Name.Type1,
                BaseFont=pikepdf.Name.Helvetica,
                Encoding=pikepdf.Name.WinAnsiEncoding,
            )
        )
        for number, page in enumerate(pdf.pages, 1):
            reading = by_number.get(number)
            if reading is None or reading.status == "native" or not reading.words:
                continue
            body, replaced = text_layer(reading)
            replaced_total += replaced
            matrix = " ".join(f"{v:g}" for v in display_to_user(page.obj))
            layer = b"Q\nq %s cm\nBT\n3 Tr\n%s\nET\nQ\n" % (matrix.encode(), body)
            _set_font(page.obj, font)
            before = pdf.make_stream(b"q\n")
            after = pdf.make_stream(layer)
            existing = page.obj.get("/Contents")
            if existing is None:
                parts = []
            elif isinstance(existing, pikepdf.Array):
                parts = list(existing)
            else:
                parts = [existing]
            page.obj["/Contents"] = pikepdf.Array([before, *parts, after])
        pdf.save(destination)
    return replaced_total


def _set_font(page: pikepdf.Object, font: pikepdf.Object) -> None:
    """Add the font to the page's own resources, copying what the page inherited so nothing of it is lost."""
    resources = page.get("/Resources")
    if resources is None:
        inherited = _inherited(page, "/Resources")
        resources = pikepdf.Dictionary(inherited) if inherited is not None else pikepdf.Dictionary()
        page["/Resources"] = resources
    fonts = resources.get("/Font")
    fonts = pikepdf.Dictionary(fonts) if fonts is not None else pikepdf.Dictionary()
    fonts[FONT_NAME] = font
    resources["/Font"] = fonts
