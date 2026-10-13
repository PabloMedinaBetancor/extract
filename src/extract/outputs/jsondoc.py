"""The JSON form of a document: every word with its box, confidence and status, the numbers and the invoice."""

from __future__ import annotations

import dataclasses
import json
from decimal import Decimal
from pathlib import Path
from typing import Any

from .. import fixes
from ..errors import UnsupportedFile
from ..types import (
    Cell,
    DigitSuggestion,
    DocumentData,
    Fix,
    InvoiceFields,
    InvoiceLine,
    NumberResult,
    PageReading,
    Reason,
    RuleResult,
    VatPair,
    Word,
    page_of,
)
from .fields import cell_status, missing_fields

SCHEMA_VERSION = 2
READABLE_VERSIONS = (1, 2)
NAME_SUFFIX = ".extract.json"


def exported(data: DocumentData) -> DocumentData:
    """The document as the JSON holds it: the engine's own detail (character alternatives, masked reading, second
    readings and the page report) stays in the page cache."""
    pages = tuple(
        dataclasses.replace(
            page,
            words=tuple(dataclasses.replace(w, char_alts=None, masked=None, readings=None) for w in page.words),
            rereads={},
            reread_report={},
        )
        for page in data.pages
    )
    return dataclasses.replace(data, pages=pages)


def _cell(cell: Cell | None, data: DocumentData) -> dict | None:
    if cell is None:
        return None
    return {"text": cell.text, "word_ids": list(cell.word_ids), "page": cell.page, "status": cell_status(cell, data)}


def _cell_from(raw: dict | None) -> Cell | None:
    return None if raw is None else Cell(raw["text"], tuple(raw["word_ids"]), raw["page"])


def _fix(fix: Fix | None, applied: bool) -> dict | None:
    if fix is None:
        return None
    return {
        "op": fix.op,
        "kind": fix.kind,
        "text": fix.text,
        "ids": list(fix.ids),
        "source": fix.source,
        "sources": list(fix.sources),
        "p_fix": fix.p_fix,
        "p_keep": fix.p_keep,
        "rule": fix.rule,
        "modes": list(fix.modes),
        "reasons": [{"key": r.key, "params": dict(r.params)} for r in fix.reasons],
        "applied": applied,
    }


def fix_json(word: Word) -> dict | None:
    """The word's proposal as the JSON and the interface carry it, with whether it is applied."""
    return _fix(word.fix, fixes.is_applied(word))


def _fix_from(raw: dict | None) -> Fix | None:
    if raw is None:
        return None
    return Fix(
        op=raw["op"],
        kind=raw["kind"],
        text=raw["text"],
        ids=tuple(raw["ids"]),
        source=raw["source"],
        sources=tuple(raw["sources"]),
        p_fix=raw["p_fix"],
        p_keep=raw["p_keep"],
        rule=raw["rule"],
        modes=tuple(raw["modes"]),
        reasons=tuple(Reason(r["key"], r["params"]) for r in raw["reasons"]),
    )


def _word(word: Word, page: PageReading) -> dict:
    return {
        "id": word.id,
        "text": word.text,
        "box_px": list(word.box),
        "box_pt": list(page.box_to_points(word.box)) if page.width_px else list(word.box),
        "conf": word.conf,
        "char_conf": None if word.char_conf is None else list(word.char_conf),
        "line_id": word.line_id,
        "idx_in_line": word.idx_in_line,
        "status": word.status,
        "p_wrong": word.p_wrong,
        "reasons": [{"key": r.key, "params": dict(r.params)} for r in word.reasons],
        "suggestion": word.suggestion,
        "suggestion_conf": word.suggestion_conf,
        "review": word.review,
        "original": word.original,
        "corrected_by": word.corrected_by,
        "absorbed_by": word.absorbed_by,
        "guard_status": word.guard_status,
        "orig_box_px": None if word.orig_box is None else list(word.orig_box),
        "fix": _fix(word.fix, fixes.is_applied(word)),
    }


def _word_from(raw: dict, page: int) -> Word:
    return Word(
        id=raw["id"],
        page=page,
        text=raw["text"],
        box=tuple(raw["box_px"]),
        conf=raw["conf"],
        char_conf=None if raw["char_conf"] is None else tuple(raw["char_conf"]),
        line_id=raw["line_id"],
        idx_in_line=raw["idx_in_line"],
        status=raw["status"],
        p_wrong=raw["p_wrong"],
        reasons=tuple(Reason(r["key"], r["params"]) for r in raw["reasons"]),
        suggestion=raw["suggestion"],
        suggestion_conf=raw["suggestion_conf"],
        review=raw["review"],
        original=raw["original"],
        corrected_by=raw["corrected_by"],
        absorbed_by=raw.get("absorbed_by"),
        guard_status=raw.get("guard_status"),
        orig_box=None if raw.get("orig_box_px") is None else tuple(raw["orig_box_px"]),
        fix=_fix_from(raw.get("fix")),
    )


def _page(page: PageReading) -> dict:
    return {
        "number": page.number,
        "status": page.status,
        "reason": page.reason,
        "error": page.error,
        "dpi": page.dpi,
        "width_px": page.width_px,
        "height_px": page.height_px,
        "width_pt": page.width_pt,
        "height_pt": page.height_pt,
        "mean_conf": page.mean_conf,
        "unaligned_words": page.unaligned_words,
        "words": [_word(w, page) for w in page.words],
    }


def _page_from(raw: dict) -> PageReading:
    return PageReading(
        raw["number"],
        raw["width_px"],
        raw["height_px"],
        raw["width_pt"],
        raw["height_pt"],
        raw["dpi"],
        tuple(_word_from(w, raw["number"]) for w in raw["words"]),
        raw["status"],
        raw["reason"],
        raw["error"],
        raw["mean_conf"],
        raw["unaligned_words"],
    )


def _number(number: NumberResult) -> dict:
    suggestion = number.suggestion
    return {
        "word_ids": list(number.word_ids),
        "page": page_of(number.word_ids[0]) if number.word_ids else None,
        "text": number.text,
        "value": None if number.value is None else str(number.value),
        "kind": number.kind,
        "status": number.status,
        "rules": [
            {"key": r.key, "params": dict(r.params), "result": r.result, "verifying": r.verifying} for r in number.rules
        ],
        "suggestion": None
        if suggestion is None
        else {
            "position": suggestion.position,
            "from": suggestion.old,
            "to": suggestion.new,
            "verified": suggestion.verified,
            "rule": suggestion.rule,
        },
    }


def _number_from(raw: dict) -> NumberResult:
    suggestion = raw["suggestion"]
    return NumberResult(
        tuple(raw["word_ids"]),
        raw["text"],
        None if raw["value"] is None else Decimal(raw["value"]),
        raw["kind"],
        raw["status"],
        tuple(RuleResult(r["key"], r["params"], r["result"], r["verifying"]) for r in raw["rules"]),
        None
        if suggestion is None
        else DigitSuggestion(
            suggestion["position"],
            suggestion["from"],
            suggestion["to"],
            suggestion.get("verified", False),
            suggestion.get("rule", ""),
        ),
    )


def _invoice(invoice: InvoiceFields, data: DocumentData) -> dict:
    def cell(value):
        return _cell(value, data)

    return {
        "fields": {
            "number": cell(invoice.number),
            "date": cell(invoice.date),
            "subtotal": cell(invoice.subtotal),
            "vat": [{"rate": cell(v.rate), "amount": cell(v.amount), "base": cell(v.base)} for v in invoice.vat],
            "withholding": cell(invoice.withholding),
            "withholding_rate": cell(invoice.withholding_rate),
            "total": cell(invoice.total),
            "currency": invoice.currency,
        },
        "lines": [
            {
                "description": cell(line.description),
                "quantity": cell(line.quantity),
                "unit_price": cell(line.unit_price),
                "amount": cell(line.amount),
            }
            for line in invoice.lines
        ],
        "missing": missing_fields(invoice),
    }


def _invoice_from(raw: dict) -> InvoiceFields:
    fields = raw["fields"]
    return InvoiceFields(
        number=_cell_from(fields["number"]),
        date=_cell_from(fields["date"]),
        subtotal=_cell_from(fields["subtotal"]),
        vat=tuple(
            VatPair(_cell_from(v["rate"]), _cell_from(v["amount"]), _cell_from(v["base"])) for v in fields["vat"]
        ),
        withholding=_cell_from(fields["withholding"]),
        total=_cell_from(fields["total"]),
        currency=fields["currency"],
        lines=tuple(
            InvoiceLine(
                _cell_from(line["description"]),
                _cell_from(line["quantity"]),
                _cell_from(line["unit_price"]),
                _cell_from(line["amount"]),
            )
            for line in raw["lines"]
        ),
        withholding_rate=_cell_from(fields["withholding_rate"]),
    )


def to_dict(data: DocumentData) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "source": {"name": data.source, "sha256": data.sha256},
        "page_count": len(data.pages),
        "engine": data.engine,
        "guard": data.guard,
        "options": dict(data.options),
        "fixes": fixes.summary(data),
        "status": data.status,
        "unreadable_pages": [
            {"page": p.number, "reason": p.reason, "error": p.error} for p in data.pages if p.status == "unreadable"
        ],
        "doc_type": data.doc_type,
        "locale": data.locale,
        "pages": [_page(p) for p in data.pages],
        "numbers": [_number(n) for n in data.numbers],
        "invoice": None if data.invoice is None else _invoice(data.invoice, data),
    }


def from_dict(raw: dict[str, Any]) -> DocumentData:
    if not isinstance(raw, dict) or raw.get("schema_version") not in READABLE_VERSIONS:
        raise UnsupportedFile("not an Extract document of schema version 1 or 2")
    return DocumentData(
        source=raw["source"]["name"],
        sha256=raw["source"]["sha256"],
        pages=tuple(_page_from(p) for p in raw["pages"]),
        engine=raw["engine"],
        guard=raw["guard"],
        options=raw["options"],
        doc_type=raw["doc_type"],
        locale=raw["locale"],
        numbers=tuple(_number_from(n) for n in raw["numbers"]),
        invoice=None if raw["invoice"] is None else _invoice_from(raw["invoice"]),
    )


def dumps(data: DocumentData) -> str:
    return json.dumps(to_dict(data), ensure_ascii=False, indent=1) + "\n"


def save(data: DocumentData, path: str | Path) -> None:
    Path(path).write_text(dumps(data), encoding="utf-8")


def load(path: str | Path) -> DocumentData:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (ValueError, UnicodeDecodeError) as error:
        raise UnsupportedFile(f"{Path(path).name} is not a JSON document: {error}") from error
    return from_dict(raw)
