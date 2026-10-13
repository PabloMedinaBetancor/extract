"""Document type and invoice fields from the words of a page.

A field that is not found is None, never guessed. Every value keeps the ids of the words that carry it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from extract.fields.layout import (
    ANCHORS,
    RATE,
    Row,
    cell_of,
    find_anchor,
    height,
    is_number,
    norm,
    rows_of,
    run_from,
    same_token,
    starts_with,
)
from extract.types import Cell, InvoiceFields, InvoiceLine, PageReading, VatPair, Word

INVOICE = "invoice"
GENERIC = "generic"

# What a label may be followed by. The run of words after "Factura numero" or "Fecha" goes on to the end of the row, so
# the value is cut at the first word that is not part of one (QA of 04/10/2026: a letter gave the whole row, eight
# words, as the invoice number).
MAX_NUMBER_WORDS = 4
MAX_NUMBER_CHARS = 30
MAX_DATE_WORDS = 7
DATE_CONNECTORS = {"de", "del", "of", "the", "st", "nd", "rd", "th"}
MONTHS = {
    "enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "setiembre", "octubre",
    "noviembre", "diciembre", "ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "sept", "set", "oct",
    "nov", "dic", "january", "february", "march", "april", "june", "july", "august", "september", "october",
    "november", "december", "jan", "apr", "aug", "dec",
}  # fmt: skip


def document_type(pages: Sequence[PageReading]) -> str:
    """`invoice` when the top third of page 1 says invoice and the page has at least two totals anchors."""
    if not pages or not pages[0].words:
        return GENERIC
    page = pages[0]
    top = [w for w in page.words if (w.box[1] + w.box[3]) / 2 <= page.height_px / 3]
    titled = any(same_token(norm(w.text), title) for w in top for title in ANCHORS["invoice_title"])
    totals = {anchor for anchor in ANCHORS["totals"] if _has_anchor(page.words, anchor.split())}
    return INVOICE if titled and len(totals) >= 2 else GENERIC


def _has_anchor(words: Sequence[Word], anchor: list[str]) -> bool:
    norms = [norm(w.text) for w in words]
    norms = [n for n in norms if n]
    return any(
        all(same_token(n, a) for n, a in zip(norms[i : i + len(anchor)], anchor, strict=True))
        for i in range(len(norms) - len(anchor) + 1)
    )


def extract_invoice(page: PageReading) -> InvoiceFields:
    rows = rows_of(page.words)
    header = _header_index(rows)
    totals_start = _totals_start(rows, header)
    body = rows[totals_start:]
    vat = _vat_pairs(body)
    withholding, withholding_rate = _withholding(body)
    return InvoiceFields(
        number=_labelled(rows, ANCHORS["number"], _number_words),
        date=_labelled(rows, ANCHORS["date"], _date_words),
        subtotal=_totals_value(body, ANCHORS["subtotal"]),
        vat=vat,
        withholding=withholding,
        withholding_rate=withholding_rate,
        total=_total(body),
        currency=_currency(page.words),
        lines=_lines(rows, header, totals_start),
    )


def missing_fields(fields: InvoiceFields) -> list[str]:
    names = ("number", "date", "subtotal", "total", "currency")
    found = [name for name in names if getattr(fields, name) is None]
    return found + (["vat"] if not fields.vat else []) + (["lines"] if not fields.lines else [])


def _has_digit(word: Word) -> bool:
    return any(c.isdigit() for c in word.text)


def _is_code_word(word: Word) -> bool:
    """A piece of an invoice number. It holds a digit ("F-2026-0498", "2026/0042"), or is only signs ("/", "-"), or is a
    short capital prefix ("F", "INV"). A word of letters in lower case or of any length is a word of a sentence. A
    misread character (".09/27", ",a-2025-7230") does not matter: the digits make it a piece."""
    letters = [c for c in word.text if c.isalpha()]
    if _has_digit(word) or not letters:
        return True
    return len(word.text) <= 5 and all(c.isupper() for c in letters)


def _number_words(words: list[Word]) -> list[Word]:
    """The leading words that can be an invoice number, none when they are not one: a number has digits and is short, a
    sentence is not one. The text of the OCR is not touched, only how much of the row is taken."""
    taken: list[Word] = []
    for word in words[:MAX_NUMBER_WORDS]:
        if not _is_code_word(word):
            break
        taken.append(word)
    while taken and not _has_digit(taken[-1]):
        taken.pop()  # signs or a capital word after the last digits belong to what follows
    if sum(len(w.text) for w in taken) + len(taken) - 1 > MAX_NUMBER_CHARS:
        return []
    return taken


def _is_date_word(word: Word) -> bool:
    text = word.text.casefold().strip(",.;")
    return _has_digit(word) or not any(c.isalpha() for c in word.text) or text in MONTHS or text in DATE_CONNECTORS


def _date_words(words: list[Word]) -> list[Word]:
    """The leading words that can be a date ("14/03/2026", "14 de marzo de 2026", "March 14, 2026"), none when they
    are not one: the words of a sentence after the date are left out. A misread character does not matter."""
    taken: list[Word] = []
    for word in words[:MAX_DATE_WORDS]:
        if not _is_date_word(word):
            break
        taken.append(word)
    while taken and not _has_digit(taken[-1]):
        taken.pop()  # a month or a connector with no number after it is not part of the date
    return taken


def _labelled(rows: list[Row], anchors: list[list[str]], valid: Callable[[list[Word]], list[Word]]) -> Cell | None:
    """The value written after a label on its row, or under the label when the row ends there; `valid` keeps the words
    of the run that can be that field and returns none of them when it is not one."""
    for anchor in anchors:
        for r, row in enumerate(rows):
            found = find_anchor(row, [anchor])
            if found is None:
                continue
            tail = [w for w in row.words[found[1] :] if norm(w.text)]
            value = valid(run_from(tuple(tail), 0))
            if value:
                return cell_of(value)
            below = valid(_value_below(rows, r, row.words[found[0]].box[0]))
            if below:
                return cell_of(below)
    return None


def _value_below(rows: list[Row], r: int, left: float) -> list[Word]:
    if r + 1 >= len(rows):
        return []
    row, nxt = rows[r], rows[r + 1]
    reach = max(height(w) for w in row.words)
    if nxt.top - row.bottom > 2.5 * reach:
        return []
    candidates = [w for w in nxt.words if w.box[0] >= left - 0.5 * reach]
    return run_from(tuple(candidates), 0)


def _header_index(rows: list[Row]) -> int | None:
    for i, row in enumerate(rows):
        has_description = find_anchor(row, [[a] for a in ANCHORS["header"]["description"]])
        has_amount = find_anchor(row, [[a] for a in ANCHORS["header"]["amount"]])
        if has_description and has_amount:
            return i
    return None


def _totals_start(rows: list[Row], header: int | None) -> int:
    """The first row after the lines table that opens with a totals label."""
    labels = [[a] for a in ("subtotal", "iva", "vat", "total", "retencion", "withholding", "irpf")]
    labels += [["base", "imponible"]]
    for i in range((header + 1) if header is not None else 0, len(rows)):
        if starts_with(rows[i], labels) is not None:
            return i
    return len(rows)


def _numbers_after(row: Row, start: int) -> list[Word]:
    return [w for w in row.words[start:] if is_number(w)]


def _totals_value(body: list[Row], anchors: list[list[str]]) -> Cell | None:
    for row in body:
        end = starts_with(row, anchors)
        if end is None:
            continue
        numbers = _numbers_after(row, end)
        if numbers:
            return cell_of([numbers[-1]])
    return None


def _total(body: list[Row]) -> Cell | None:
    found = None
    for row in body:
        end = starts_with(row, ANCHORS["total"])
        if end is None:
            continue
        numbers = _numbers_after(row, end)
        if numbers:
            found = cell_of([numbers[-1]])
    return found


def _rate(row: Row, start: int) -> tuple[list[Word], int]:
    """The rate words after `start` ("21 %" or "21%") and the index after them."""
    words = row.words
    for i in range(start, len(words)):
        if RATE.match(words[i].text):
            return [words[i]], i + 1
        if is_number(words[i]) and i + 1 < len(words) and words[i + 1].text == "%":
            return [words[i], words[i + 1]], i + 2
    return [], start


def _vat_pairs(body: list[Row]) -> tuple[VatPair, ...]:
    pairs = []
    for row in body:
        end = starts_with(row, ANCHORS["vat"])
        if end is None:
            continue
        rate, after = _rate(row, end)
        numbers = _numbers_after(row, after)
        if not numbers:
            continue
        base = cell_of([numbers[-2]]) if len(numbers) >= 2 else None
        pairs.append(VatPair(cell_of(rate), cell_of([numbers[-1]]), base))
    return tuple(pairs)


def _withholding(body: list[Row]) -> tuple[Cell | None, Cell | None]:
    for row in body:
        end = starts_with(row, ANCHORS["withholding"])
        if end is None:
            continue
        rate, after = _rate(row, end)
        numbers = _numbers_after(row, after)
        if numbers:
            return cell_of([numbers[-1]]), cell_of(rate)
    return None, None


def _currency(words: Sequence[Word]) -> str | None:
    codes = ANCHORS["currency"]
    for word in reversed(words):
        code = codes.get(word.text.casefold())
        if code:
            return code
    return None


def _lines(rows: list[Row], header: int | None, totals_start: int) -> tuple[InvoiceLine, ...]:
    if header is None:
        return ()
    has_quantity = find_anchor(rows[header], [[a] for a in ANCHORS["header"]["quantity"]]) is not None
    columns = 3 if has_quantity else 2
    lines = []
    for row in rows[header + 1 : totals_start]:
        numbers = [w for w in row.words if is_number(w)]
        if len(numbers) < columns:
            continue
        values = numbers[-columns:]
        description = [w for w in row.words if w.box[2] <= values[0].box[0] and w not in numbers]
        quantity = cell_of([values[0]]) if has_quantity else None
        lines.append(
            InvoiceLine(
                description=cell_of(description),
                quantity=quantity,
                unit_price=cell_of([values[-2]]),
                amount=cell_of([values[-1]]),
            )
        )
    return tuple(lines)
