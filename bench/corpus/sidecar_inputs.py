"""Words and `InvoiceFields` built straight from a generator sidecar, so the number rules are tested with no OCR and
no extraction in between. `checks.run` reads every value from the words; the cells only say which words are which."""

from __future__ import annotations

import dataclasses
import functools
import json
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from extract.types import Cell, InvoiceFields, InvoiceLine, VatPair, Word

# For each digit, the digits an engine plausibly reads in its place (1/7, 3/8, 5/6, 0/8, 2/7, 4/9, both ways).
CONFUSABLE = {"1": "7", "7": "12", "3": "8", "8": "30", "5": "6", "6": "5", "0": "8", "2": "7", "4": "9", "9": "4"}


@dataclass
class Inputs:
    words: list[Word]
    fields: InvoiceFields
    ids: dict = field(default_factory=dict)

    def text_of(self, word_id: str) -> str:
        return next(w.text for w in self.words if w.id == word_id)

    def with_text(self, word_id: str, text: str) -> Inputs:
        words = [dataclasses.replace(w, text=text) if w.id == word_id else w for w in self.words]
        return Inputs(words, self.fields, self.ids)

    def with_texts(self, changes: dict[str, str]) -> Inputs:
        words = [dataclasses.replace(w, text=changes[w.id]) if w.id in changes else w for w in self.words]
        return Inputs(words, self.fields, self.ids)

    def with_conf(self, word_id: str, conf: float, char_conf: tuple[float, ...] | None) -> Inputs:
        words = [
            dataclasses.replace(w, conf=conf, char_conf=char_conf, status="ok", p_wrong=0.0001)
            if w.id == word_id
            else w
            for w in self.words
        ]
        return Inputs(words, self.fields, self.ids)


class Builder:
    def __init__(self, page: int = 1) -> None:
        self.page = page
        self.words: list[Word] = []

    def add(self, *texts: str) -> tuple[str, ...]:
        ids = []
        for text in texts:
            index = len(self.words)
            word_id = f"p{self.page}w{index}"
            self.words.append(Word(word_id, self.page, text, (10.0 * index, 0.0, 10.0 * index + 8.0, 10.0)))
            ids.append(word_id)
        return tuple(ids)

    def cell(self, *texts: str) -> tuple[Cell, tuple[str, ...]]:
        ids = self.add(*texts)
        return Cell(" ".join(texts), ids, self.page), ids


DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"
CIF_CONTROL_LETTERS = "JABCDEFGHI"


def checkable_tax_id(text: str, kind: str) -> str:
    """The invented tax number of a sidecar (one character short, so that it cannot exist) completed so that it passes
    its check digits, for the number rules to be tried on it. Built in memory for one test and never written down."""
    if kind == "dni":
        number = int(f"{1 + sum(map(int, text[:7])) % 9}{text[:7]}")
        return f"{number}{DNI_LETTERS[number % 23]}"
    digits = [int(c) for c in text[1:8]]
    odd = sum(sum(divmod(2 * d, 10)) for d in digits[0::2])
    control = (10 - (odd + sum(digits[1::2])) % 10) % 10
    letter = text[0] in "PQRSNW" or (text[0] not in "ABEH" and sum(digits) % 2 == 1)
    return f"{text}{CIF_CONTROL_LETTERS[control] if letter else control}"


def checkable_iban(text: str) -> str:
    """The invented IBAN of a sidecar ("ESXX" and eighteen digits) completed to a Spanish IBAN whose check digits hold."""
    bban = text.replace(" ", "")[4:] + "00"
    raw = f"ES{98 - int(bban + '142800') % 97:02d}{bban}"
    return " ".join(raw[i : i + 4] for i in range(0, len(raw), 4))


def build(truth: dict, page: int = 1) -> Inputs:
    """The sidecar of one invoice as words and fields, in reading order. The identifiers of a sidecar are invented
    and cannot exist, so the words carry them completed to pass their check digits (see `checkable_tax_id`)."""
    b = Builder(page)
    ids: dict = {}
    for key in ("supplier", "customer"):
        party = truth[key]
        ids[f"{key}_tax_id"] = b.add(checkable_tax_id(party["tax_id"], party["tax_id_kind"]))[0]
    number, number_ids = b.cell(truth["number"])
    date, date_ids = b.cell(truth["date"])
    ids["number"], ids["date"] = number_ids[0], date_ids[0]
    lines = []
    for i, line in enumerate(truth["lines"]):
        quantity = price = amount = description = None
        if line["quantity"] is not None:
            quantity, q_ids = b.cell(line["quantity"])
            ids[("line_quantity", i)] = q_ids[0]
        price, p_ids = b.cell(line["unit_price"])
        amount, a_ids = b.cell(line["amount"])
        ids[("line_unit_price", i)], ids[("line_amount", i)] = p_ids[0], a_ids[0]
        lines.append(InvoiceLine(description, quantity, price, amount))
    subtotal, s_ids = b.cell(truth["subtotal"])
    ids["subtotal"] = s_ids[0]
    pairs = []
    several = len(truth["vat"]) > 1
    for j, vat in enumerate(truth["vat"]):
        rate_text, _, percent = vat["rate"].partition(" ")
        rate, r_ids = b.cell(rate_text, percent)
        base = None
        if several:
            base, b_ids = b.cell(vat["base"])
            ids[("vat_base", j)] = b_ids[0]
        amount, v_ids = b.cell(vat["amount"])
        ids[("vat_rate", j)], ids[("vat_amount", j)] = r_ids[0], v_ids[0]
        pairs.append(VatPair(rate, amount, base))
    withholding = withholding_rate = None
    if truth["withholding"]:
        rate_text, _, percent = truth["withholding"]["rate"].partition(" ")
        withholding_rate, wr_ids = b.cell(rate_text, percent)
        withholding, w_ids = b.cell(truth["withholding"]["amount"])
        ids["withholding_rate"], ids["withholding"] = wr_ids[0], w_ids[0]
    total, t_ids = b.cell(truth["total"])
    ids["total"] = t_ids[0]
    iban_groups = checkable_iban(truth["iban"]).split()
    ids["iban"] = b.add(*iban_groups)
    fields = InvoiceFields(
        number=number,
        date=date,
        subtotal=subtotal,
        vat=tuple(pairs),
        withholding=withholding,
        withholding_rate=withholding_rate,
        total=total,
        currency=truth["currency"],
        lines=tuple(lines),
    )
    return Inputs(b.words, fields, ids)


def load_truths(folder: Path) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(folder.glob("*/*.json"))]


@functools.cache
def generated_truths(seed: int = 5) -> tuple[dict, ...]:
    """The synthetic invoices of one seed (210 of them), generated once per process."""
    from bench.corpus import invoices

    folder = Path(tempfile.mkdtemp(prefix="checks-invoices-"))
    invoices.generate(folder, seed)
    return tuple(load_truths(folder))


def digit_positions(text: str) -> list[int]:
    return [i for i, c in enumerate(text) if c.isdigit()]


def reprice(truth: dict) -> None:
    """Make subtotal, VAT, withholding and total of a one-rate sidecar consistent with its lines again, in place."""
    from decimal import ROUND_HALF_UP, Decimal

    from bench.corpus.invoices import format_amount

    lang = truth["language"]

    def read(text: str) -> Decimal:
        plain = text.replace(".", "").replace(",", ".") if lang == "es" else text.replace(",", "")
        return Decimal(plain)

    cent = Decimal("0.01")
    if len(truth["vat"]) != 1:
        raise ValueError("reprice handles one VAT rate")
    subtotal = sum((read(line["amount"]) for line in truth["lines"]), Decimal(0))
    vat = truth["vat"][0]
    vat_amount = (subtotal * read(vat["rate"].replace(" %", "")) / 100).quantize(cent, rounding=ROUND_HALF_UP)
    withheld = Decimal(0)
    if truth["withholding"]:
        rate = read(truth["withholding"]["rate"].replace(" %", ""))
        withheld = (subtotal * rate / 100).quantize(cent, rounding=ROUND_HALF_UP)
        truth["withholding"]["amount"] = "-" + format_amount(withheld, lang)
    truth["subtotal"] = vat["base"] = format_amount(subtotal, lang)
    vat["amount"] = format_amount(vat_amount, lang)
    truth["total"] = format_amount(subtotal + vat_amount - withheld, lang)
