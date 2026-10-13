"""Invoice detection and extraction (F1-F3): invoice detection, field accuracy and line accuracy against the generator's sidecar.

`evaluate(items)` is a pure function of pages and truths; the corrector's test run calls it inside its single read of
the sealed test set, so it never opens a file or looks at which split it was given.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from extract.fields import INVOICE, document_type, extract_invoice
from extract.types import Cell, PageReading


@dataclass(frozen=True)
class Item:
    """A page and the sidecar of the invoice it shows (None for a page that is not an invoice)."""

    page: PageReading
    truth: dict | None


def squash(text: str | None) -> str | None:
    return None if text is None else "".join(text.split())


def cell_text(cell: Cell | None) -> str | None:
    return None if cell is None else squash(cell.text)


def field_pairs(truth: dict, page: PageReading) -> dict[str, tuple[object, object]]:
    """Per field, (expected, found) for one invoice, both as whitespace-free strings or lists of strings."""
    fields = extract_invoice(page)
    several = len(truth["vat"]) > 1
    withholding = truth["withholding"]
    pairs = {
        "number": (squash(truth["number"]), cell_text(fields.number)),
        "date": (squash(truth["date"]), cell_text(fields.date)),
        "subtotal": (squash(truth["subtotal"]), cell_text(fields.subtotal)),
        "total": (squash(truth["total"]), cell_text(fields.total)),
        "currency": (truth["currency"], fields.currency),
        "vat_rate": ([squash(v["rate"]) for v in truth["vat"]], [cell_text(v.rate) for v in fields.vat]),
        "vat_amount": ([squash(v["amount"]) for v in truth["vat"]], [cell_text(v.amount) for v in fields.vat]),
        "withholding": (squash(withholding["amount"]) if withholding else None, cell_text(fields.withholding)),
        "withholding_rate": (squash(withholding["rate"]) if withholding else None, cell_text(fields.withholding_rate)),
    }
    if several:
        pairs["vat_base"] = ([squash(v["base"]) for v in truth["vat"]], [cell_text(v.base) for v in fields.vat])
    return pairs


def line_pairs(truth: dict, page: PageReading) -> tuple[int, int, list[tuple[bool, bool]]]:
    """Expected and found row counts, and for each expected row whether its amount and its quantity and unit price
    (together) are exactly right."""
    found = extract_invoice(page).lines
    rows = []
    for i, line in enumerate(truth["lines"]):
        got = found[i] if i < len(found) else None
        amount_ok = got is not None and cell_text(got.amount) == squash(line["amount"])
        parts_ok = (
            got is not None
            and cell_text(got.unit_price) == squash(line["unit_price"])
            and cell_text(got.quantity) == squash(line["quantity"])
        )
        rows.append((amount_ok, parts_ok))
    return len(truth["lines"]), len(found), rows


def evaluate(items: Iterable[Item]) -> dict:
    tp = fp = fn = tn = 0
    field_hits: dict[str, list[bool]] = {}
    count_hits: list[bool] = []
    amount_hits: list[bool] = []
    parts_hits: list[bool] = []
    for item in items:
        detected = document_type([item.page]) == INVOICE
        if item.truth is None:
            fp += detected
            tn += not detected
            continue
        tp += detected
        fn += not detected
        for name, (expected, found) in field_pairs(item.truth, item.page).items():
            field_hits.setdefault(name, []).append(expected == found)
        expected_rows, found_rows, rows = line_pairs(item.truth, item.page)
        count_hits.append(expected_rows == found_rows)
        amount_hits.extend(ok for ok, _ in rows)
        parts_hits.extend(ok for _, ok in rows)
    return {
        "f1": {
            "invoices": tp + fn,
            "others": fp + tn,
            "recall": _ratio(tp, tp + fn),
            "precision": _ratio(tp, tp + fp),
            "false_positives": fp,
        },
        "f2": {name: _share(hits) for name, hits in field_hits.items()},
        "f3": {
            "row_count": _share(count_hits),
            "row_amount": _share(amount_hits),
            "row_quantity_and_price": _share(parts_hits),
            "rows": len(amount_hits),
        },
    }


def _ratio(hits: int, total: int) -> float | None:
    return hits / total if total else None


def _share(hits: list[bool]) -> float | None:
    return _ratio(sum(hits), len(hits))


def items_from_records(records: Iterable[tuple[Path, dict]], invoices: Path) -> list[Item]:
    """Pages read by the engine, from (json path, record) pairs: an invoice page gets its sidecar from
    `<invoices>/<set>/<doc_id>.json`, any other page is a page that is not an invoice. A page with no reading is left
    out. The sealed test split comes through here from the corrector's only reader."""
    from extract.engine.cache import reading_from_dict

    items = []
    for _, record in records:
        ocr = record.get("ocr")
        if not ocr or "reading" not in ocr:
            continue
        truth = None
        if record.get("source") == "invoice":
            truth = json.loads((invoices / record["set"] / f"{record['doc_id']}.json").read_text(encoding="utf-8"))
        items.append(Item(reading_from_dict(ocr["reading"]), truth))
    return items


def load_ocr_items(data: Path, split: str, invoices: Path) -> list[Item]:
    """The engine's readings of one split that is not sealed: every page JSON under `<data>/pages/<split>/`."""
    paths = sorted((data / "pages" / split).glob("*/*.json"))
    return items_from_records(((p, json.loads(p.read_text(encoding="utf-8"))) for p in paths), invoices)


def load_sidecar_items(folder: Path, split: str) -> list[Item]:
    """The generator's invoices of one split, read from their PDF text layer (the truth words, no OCR)."""
    from bench.corpus.pdf_words import read_native_page

    items = []
    for sidecar in sorted((folder / split).glob("*.json")):
        truth = json.loads(sidecar.read_text(encoding="utf-8"))
        items.append(Item(read_native_page(sidecar.with_suffix(".pdf")), truth))
    return items
