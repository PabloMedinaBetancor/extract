"""Words and boxes from the text layer of a born-digital PDF page, in the shape the engine gives for a native page."""

from __future__ import annotations

from pathlib import Path

import pypdfium2 as pdfium

from extract.types import PageReading, Word, word_id


def read_native_page(pdf_path: Path, number: int = 1) -> PageReading:
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        page = document[number - 1]
        width, height = page.get_size()
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        chars = []
        for i, char in enumerate(text):
            left, bottom, right, top = textpage.get_charbox(i)
            chars.append((char, left, height - top, right, height - bottom))
        words = _group(chars, number)
        textpage.close()
        page.close()
    finally:
        document.close()
    line_of = _line_ids(words)
    return PageReading(
        number=number,
        width_px=round(width),
        height_px=round(height),
        width_pt=float(width),
        height_pt=float(height),
        dpi=72,
        words=tuple(
            Word(w.id, number, w.text, w.box, line_id=line_of[w.id], idx_in_line=i)
            for w, i in _with_index(words, line_of)
        ),
        status="native",
    )


def _group(chars: list, page: int) -> list[Word]:
    words: list[Word] = []
    current: list = []

    def flush() -> None:
        if not current:
            return
        text = "".join(c[0] for c in current)
        box = (
            min(c[1] for c in current),
            min(c[2] for c in current),
            max(c[3] for c in current),
            max(c[4] for c in current),
        )
        words.append(Word(word_id(page, len(words)), page, text, box))
        current.clear()

    for char in chars:
        if char[0].isspace():
            flush()
            continue
        if current:
            previous = current[-1]
            height = max(previous[4] - previous[2], char[4] - char[2], 1.0)
            if char[1] - previous[3] > height:
                flush()
        current.append(char)
    flush()
    return words


def _line_ids(words: list[Word]) -> dict[str, int]:
    ordered = sorted(words, key=lambda w: (w.box[1] + w.box[3]) / 2)
    ids: dict[str, int] = {}
    line = -1
    last_centre = None
    for word in ordered:
        centre = (word.box[1] + word.box[3]) / 2
        if last_centre is None or centre - last_centre > 0.6 * (word.box[3] - word.box[1]):
            line += 1
        ids[word.id] = line
        last_centre = centre
    return ids


def _with_index(words: list[Word], line_of: dict[str, int]) -> list[tuple[Word, int]]:
    counters: dict[int, int] = {}
    result = []
    for word in sorted(words, key=lambda w: (line_of[w.id], w.box[0])):
        index = counters.get(line_of[word.id], 0)
        counters[line_of[word.id]] = index + 1
        result.append((word, index))
    return result
