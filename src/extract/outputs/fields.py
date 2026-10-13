from __future__ import annotations

from ..types import Cell, DocumentData, InvoiceFields

FIELD_ORDER = ("number", "date", "subtotal", "vat", "withholding", "withholding_rate", "total", "currency")


def cell_status(cell: Cell | None, data: DocumentData) -> str:
    """The status of a field: that of the checked number its words make up, `unverified` for anything else (a field the
    invoice does not have, `None`, included)."""
    if cell is None:
        return "unverified"
    ids = set(cell.word_ids)
    best = "unverified"
    for number in data.numbers:
        if ids & set(number.word_ids):
            if number.status == "mismatch":
                return "mismatch"
            if number.status == "checked":
                best = "checked"
    return best


def named_cells(invoice: InvoiceFields) -> list[tuple[str, Cell]]:
    """The cells of an invoice with the name each goes by in the outputs, in a fixed order."""
    cells: list[tuple[str, Cell]] = []
    for name in ("number", "date", "subtotal"):
        if getattr(invoice, name) is not None:
            cells.append((name, getattr(invoice, name)))
    for index, pair in enumerate(invoice.vat, 1):
        for part in ("base", "rate", "amount"):
            if getattr(pair, part) is not None:
                cells.append((f"vat.{index}.{part}", getattr(pair, part)))
    for name in ("withholding_rate", "withholding", "total"):
        if getattr(invoice, name) is not None:
            cells.append((name, getattr(invoice, name)))
    for index, line in enumerate(invoice.lines, 1):
        for part in ("description", "quantity", "unit_price", "amount"):
            if getattr(line, part) is not None:
                cells.append((f"line.{index}.{part}", getattr(line, part)))
    return cells


def missing_fields(invoice: InvoiceFields) -> list[str]:
    missing = [n for n in ("number", "date", "subtotal", "total") if getattr(invoice, n) is None]
    if invoice.currency is None:
        missing.append("currency")
    if not invoice.vat:
        missing.append("vat")
    if not invoice.lines:
        missing.append("lines")
    return missing
