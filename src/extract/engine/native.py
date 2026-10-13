from __future__ import annotations

import re
from typing import Protocol, Sequence

import pypdfium2 as pdfium

from ..types import Box, PageReading, Word, word_id

NATIVE_MIN_CHARS = 80
LEGIBLE_MIN_SHARE = 0.35
NORMAL_CHARS_MIN = 0.8
LETTERS = "A-Za-z\u00c1\u00c9\u00cd\u00d3\u00da\u00dc\u00d1\u00e1\u00e9\u00ed\u00f3\u00fa\u00fc\u00f1"
PUNCTUATION = "\u2013\u2014\u00ab\u00bb\u00bf\u00a1\u00ba\u00aa\u00b0\u00a7"
_NORMAL = re.compile(rf"[{LETTERS}0-9.,;:()\[\]\-/%\"'!?=+*&{PUNCTUATION}]")
_WORD = re.compile(rf"[{LETTERS}]{{3,}}")
_LINE_BREAKS = "\r\n"


class Lexicon(Protocol):
    def has(self, word: str, languages: Sequence[str]) -> bool: ...

    def share(self, words: Sequence[str], languages: Sequence[str]) -> float: ...


def legible(text: str, lexicon: Lexicon | None, languages: Sequence[str] = ("es", "en")) -> bool:
    """Whether a text layer reads as text: mostly letters, digits and punctuation and, with at least 12 words of
    three letters or more, at least 35 % of them in the lexicon. A layer with no Unicode map comes out as garbage
    that pdftotext extracts all the same."""
    compact = re.sub(r"\s", "", text)
    if not compact:
        return False
    if len(_NORMAL.findall(compact)) / len(compact) < NORMAL_CHARS_MIN:
        return False
    words = [w.lower() for w in _WORD.findall(text)]
    if lexicon is None or len(words) < 12:
        return True
    return lexicon.share(words, languages) >= LEGIBLE_MIN_SHARE


def is_native(text: str, lexicon: Lexicon | None, languages: Sequence[str] = ("es", "en")) -> bool:
    return len(re.sub(r"\s", "", text)) >= NATIVE_MIN_CHARS and legible(text, lexicon, languages)


def page_text(page: pdfium.PdfPage) -> str:
    textpage = page.get_textpage()
    try:
        return textpage.get_text_range()
    finally:
        textpage.close()


def _to_display(page: pdfium.PdfPage):
    """A function from pdfium's page space (origin at the lower left of the page box, unrotated) to the displayed
    page's top-left space, in points."""
    left, bottom, right, top = page.get_bbox()
    width, height = right - left, top - bottom
    rotation = page.get_rotation()

    def convert(x: float, y: float) -> tuple[float, float]:
        x, y = x - left, y - bottom
        if rotation == 90:
            return y, x
        if rotation == 180:
            return width - x, y
        if rotation == 270:
            return height - y, width - x
        return x, height - y

    return convert


def read_native(
    page: pdfium.PdfPage,
    number: int,
    lexicon: Lexicon | None,
    languages: Sequence[str],
    dpi: int,
) -> PageReading | None:
    """The page's own text with boxes when its layer is legible, else None (the page goes to OCR)."""
    textpage = page.get_textpage()
    try:
        text = textpage.get_text_range()
        if not is_native(text, lexicon, languages):
            return None
        convert = _to_display(page)
        scale = dpi / 72
        runs: list[tuple[str, Box, bool]] = []
        chars: list[str] = []
        box: Box | None = None
        broke = False
        for i in range(textpage.count_chars()):
            ch = textpage.get_text_range(i, 1)
            if ch and not ch.isspace() and ch != "\x00":
                (ax, ay), (bx, by) = convert(*textpage.get_charbox(i)[:2]), convert(*textpage.get_charbox(i)[2:])
                char_box = (min(ax, bx), min(ay, by), max(ax, bx), max(ay, by))
                box = char_box if box is None else _union(box, char_box)
                chars.append(ch)
                continue
            if chars:
                runs.append(("".join(chars), box, broke))
                chars, box, broke = [], None, False
            broke = broke or (bool(ch) and ch in _LINE_BREAKS)
        if chars:
            runs.append(("".join(chars), box, broke))
        words: list[Word] = []
        line_id, idx, previous = 0, 0, None
        for text_, run_box, forced in runs:
            if previous is not None and (forced or not _same_line(previous, run_box)):
                line_id, idx = line_id + 1, 0
            if not _on_page(run_box, page.get_size()):
                continue
            px = tuple(v * scale for v in run_box)
            words.append(Word(word_id(number, len(words)), number, text_, px, None, None, line_id, idx, "native"))
            idx, previous = idx + 1, run_box
        width_pt, height_pt = page.get_size()
        return PageReading(
            number, round(width_pt * scale), round(height_pt * scale), width_pt, height_pt, dpi, tuple(words), "native"
        )
    finally:
        textpage.close()


def _on_page(box: Box, size: tuple[float, float]) -> bool:
    """Text outside the page box (under the crop) is never shown, so it is not a word of the page."""
    return 0 <= (box[0] + box[2]) / 2 <= size[0] and 0 <= (box[1] + box[3]) / 2 <= size[1]


def _union(a: Box, b: Box) -> Box:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


def _same_line(word_box: Box, char_box: Box) -> bool:
    overlap = min(word_box[3], char_box[3]) - max(word_box[1], char_box[1])
    return overlap > 0.5 * min(word_box[3] - word_box[1], char_box[3] - char_box[1])
