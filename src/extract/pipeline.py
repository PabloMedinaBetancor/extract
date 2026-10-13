"""The stages after the engine: the guard on each page, the document type and invoice fields, and the number rules."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from decimal import Decimal
from typing import Any, Protocol

from . import checks, fields, fixes
from .events import Warning as JobWarning
from .types import (
    Cell,
    InvoiceFields,
    InvoiceLine,
    NumberResult,
    PageReading,
    Reason,
    Scored,
    VatPair,
    Word,
)

REVIEW_BUDGETS = {"fewer": 0.03, "balanced": 0.05, "more": 0.08}


class Guard(Protocol):
    """What the pipeline needs from the word checker (`extract.guard.Guard`)."""

    info: Any

    def score_page(
        self, page: PageReading, *, languages: Sequence[str], locale: str, budget: float
    ) -> Sequence[Scored]: ...


def guard_id(guard: Guard | None) -> str | None:
    if guard is None:
        return None
    info = guard.info
    if isinstance(info, dict):
        return str(info.get("model_id", info))
    return str(getattr(info, "model_id", info))


SIDEWAYS_SHARE = 0.4
SIDEWAYS_MIN_WORDS = 8


def looks_sideways(reading: PageReading) -> bool:
    """True when the lines of the page run up or down: of its words of four letters or more, 40 % or more are at least
    1.5 times taller than wide (on a page that is the right way up it is none or almost none). The engine reads a page turned a quarter
    one way with its lines as columns, the order of the lines wrong, and the other way mostly as noise; this does not
    change a word, it only lets the page say so."""
    words = [w for w in reading.words if len(w.text) >= 4]
    if reading.status not in ("ocr", "unreadable") or len(words) < SIDEWAYS_MIN_WORDS:
        return False
    tall = sum(1 for w in words if (w.box[3] - w.box[1]) >= 1.5 * (w.box[2] - w.box[0]))
    return tall / len(words) >= SIDEWAYS_SHARE


def _unscored(reading: PageReading, reason: Reason | None = None) -> PageReading:
    words = tuple(
        replace(w, status="unscored", p_wrong=None, reasons=(reason,) if reason else (), suggestion=None, fix=None)
        for w in reading.words
    )
    return replace(reading, words=words)


def score_page(
    reading: PageReading,
    guard: Guard | None,
    languages: Sequence[str],
    budget: float,
    warn: Callable[[JobWarning], None],
    locale: str = "auto",
) -> PageReading:
    """The page with the guard's verdict on each of its words. A native page keeps its words as they are; an
    unreadable one is not scored; with no guard, or when the guard fails, the words are `unscored` and the user is
    told, never `ok`."""
    if reading.status == "native":
        return reading
    if reading.status == "unreadable":
        return _unscored(reading, Reason(reading.reason) if reading.reason else None)
    if guard is None:
        warn(
            JobWarning(
                "guard.warning.unscored", {"page": reading.number, "words": len(reading.words), "cause": "no_guard"}
            )
        )
        return _unscored(reading)
    try:
        scored = {s.id: s for s in guard.score_page(reading, languages=tuple(languages), locale=locale, budget=budget)}
        reported = [
            JobWarning(w["key"], {k: v for k, v in w.items() if k != "key"})
            for w in getattr(guard, "last_warnings", ())
        ]
    except Exception as error:  # the guard's own failure must not lose the page: its words are shown as unchecked
        warn(
            JobWarning(
                "guard.warning.unscored",
                {"page": reading.number, "words": len(reading.words), "cause": "error", "detail": str(error)},
            )
        )
        return _unscored(reading)
    for warning in reported:
        warn(warning)
    words, missing = [], 0
    for word in reading.words:
        verdict = scored.get(word.id)
        if verdict is None or verdict.status not in ("ok", "doubtful"):
            missing += 1
            words.append(replace(word, status="unscored", p_wrong=None, reasons=(), suggestion=None, fix=None))
        else:
            words.append(
                replace(
                    word,
                    status=verdict.status,
                    p_wrong=verdict.p_wrong,
                    reasons=tuple(verdict.reasons),
                    suggestion=verdict.suggestion,
                    suggestion_conf=verdict.suggestion_conf,
                    fix=verdict.fix,
                    guard_status=None,
                    orig_box=None,
                )
            )
    if missing and not reported:
        warn(JobWarning("guard.warning.unscored", {"page": reading.number, "words": missing, "cause": "error"}))
    return replace(reading, words=tuple(words))


def refresh_cells(invoice: InvoiceFields, words: dict[str, Word]) -> InvoiceFields:
    """The invoice with every cell's text rebuilt from the current text of its words (after a correction)."""

    def fix(cell: Cell | None) -> Cell | None:
        if cell is None:
            return None
        text = " ".join(words[i].text for i in cell.word_ids if i in words)
        return replace(cell, text=text or cell.text)

    return replace(
        invoice,
        number=fix(invoice.number),
        date=fix(invoice.date),
        subtotal=fix(invoice.subtotal),
        withholding=fix(invoice.withholding),
        withholding_rate=fix(invoice.withholding_rate),
        total=fix(invoice.total),
        vat=tuple(VatPair(fix(v.rate), fix(v.amount), fix(v.base)) for v in invoice.vat),
        lines=tuple(
            InvoiceLine(fix(ln.description), fix(ln.quantity), fix(ln.unit_price), fix(ln.amount))
            for ln in invoice.lines
        ),
    )


def _plain(value: Any) -> Any:
    """A rule parameter as JSON can hold it: exact decimals become strings, so a saved document reads back equal."""
    return str(value) if isinstance(value, Decimal) else value


def _plain_numbers(numbers: Sequence[NumberResult]) -> tuple[NumberResult, ...]:
    return tuple(
        replace(
            n,
            rules=tuple(replace(r, params={k: _plain(v) for k, v in r.params.items()}) for r in n.rules),
        )
        for n in numbers
    )


def run_numbers(
    pages: Sequence[PageReading], invoice: InvoiceFields | None, languages: Sequence[str], locale: str | None = None
) -> tuple[str, tuple[NumberResult, ...]]:
    """The locale of the document and the status of every number in it, by the rules and nothing else."""
    pages = fixes.token_pages(pages)
    words = [w for page in pages for w in page.words]
    unreadable = [p.number for p in pages if p.status == "unreadable"]
    first = languages[0]
    locale = locale or checks.detect_locale(words, first)
    results = checks.run(
        words, invoice or InvoiceFields(), locale, unreadable_pages=unreadable, first_page_language=first
    )
    return locale, _plain_numbers(results)


def analyse(
    pages: Sequence[PageReading], languages: Sequence[str]
) -> tuple[str, InvoiceFields | None, str, tuple[NumberResult, ...]]:
    """Document type, invoice fields (page 1 of an invoice), locale and numbers."""
    pages = fixes.token_pages(pages)
    doc_type = fields.document_type(pages)
    invoice = fields.extract_invoice(pages[0]) if doc_type == fields.INVOICE and pages else None
    locale, numbers = run_numbers(pages, invoice, languages)
    return doc_type, invoice, locale, numbers


def analyse_fixed(
    pages: Sequence[PageReading], languages: Sequence[str], mode: str
) -> tuple[tuple[PageReading, ...], str, InvoiceFields | None, str, tuple[NumberResult, ...]]:
    """Steps 6, 7 and 7b: the extraction and the number rules on the token view, then the digit fixes the verifying
    rules allow in `mode`, and a second pass of the rules over the changed numbers (A2.6)."""
    doc_type, invoice, locale, numbers = analyse(pages, languages)
    pages, changed = fixes.apply_digit_fixes(pages, numbers, mode)
    if changed:
        words = {w.id: w for p in fixes.token_pages(pages) for w in p.words}
        if invoice is not None:
            invoice = refresh_cells(invoice, words)
        locale, numbers = run_numbers(pages, invoice, languages, locale)
    return tuple(pages), doc_type, invoice, locale, numbers
