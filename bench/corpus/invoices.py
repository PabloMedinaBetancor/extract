"""Synthetic invoices: a PDF with a faithful text layer and a sidecar JSON with the exact truth.

Six layouts and eight fonts. Layouts 1-4 and fonts 1-6 are for training and validation; the test split holds only
layouts 5-6, with any font, and fonts 7-8 appear nowhere else. Every amount is a ``Decimal`` and is written to the
sidecar as the exact string printed on the page, so the sidecar is the truth for field extraction, for the number rules
and for the injection harness.

    python -m bench.corpus.invoices --out bench/data/invoices --seed 1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

CENT = Decimal("0.01")
FONT_DIR = Path("/usr/share/fonts/truetype")


@dataclass(frozen=True)
class FontSpec:
    key: str
    family: str
    regular: str
    bold: str
    licence: str
    held_out: bool


FONTS = (
    FontSpec(
        "dejavu-sans",
        "DejaVu Sans",
        "dejavu/DejaVuSans.ttf",
        "dejavu/DejaVuSans-Bold.ttf",
        "Bitstream Vera",
        False,
    ),
    FontSpec(
        "dejavu-serif",
        "DejaVu Serif",
        "dejavu/DejaVuSerif.ttf",
        "dejavu/DejaVuSerif-Bold.ttf",
        "Bitstream Vera",
        False,
    ),
    FontSpec(
        "liberation-sans",
        "Liberation Sans",
        "liberation2/LiberationSans-Regular.ttf",
        "liberation2/LiberationSans-Bold.ttf",
        "SIL OFL 1.1",
        False,
    ),
    FontSpec(
        "liberation-serif",
        "Liberation Serif",
        "liberation2/LiberationSerif-Regular.ttf",
        "liberation2/LiberationSerif-Bold.ttf",
        "SIL OFL 1.1",
        False,
    ),
    FontSpec(
        "liberation-mono",
        "Liberation Mono",
        "liberation2/LiberationMono-Regular.ttf",
        "liberation2/LiberationMono-Bold.ttf",
        "SIL OFL 1.1",
        False,
    ),
    FontSpec(
        "ubuntu",
        "Ubuntu",
        "ubuntu/Ubuntu-R.ttf",
        "ubuntu/Ubuntu-B.ttf",
        "Ubuntu Font Licence 1.0",
        False,
    ),
    FontSpec(
        "free-sans",
        "FreeSans",
        "freefont/FreeSans.ttf",
        "freefont/FreeSansBold.ttf",
        "GPL-3 with font exception",
        True,
    ),
    FontSpec(
        "free-serif",
        "FreeSerif",
        "freefont/FreeSerif.ttf",
        "freefont/FreeSerifBold.ttf",
        "GPL-3 with font exception",
        True,
    ),
)


@dataclass(frozen=True)
class Layout:
    key: str
    label_above: bool
    qty_column: bool
    totals_left: bool
    header_right: bool
    grid: bool
    held_out: bool


LAYOUTS = (
    Layout(
        "L1",
        label_above=False,
        qty_column=True,
        totals_left=False,
        header_right=False,
        grid=True,
        held_out=False,
    ),
    Layout(
        "L2",
        label_above=True,
        qty_column=True,
        totals_left=False,
        header_right=True,
        grid=False,
        held_out=False,
    ),
    Layout(
        "L3",
        label_above=False,
        qty_column=False,
        totals_left=True,
        header_right=True,
        grid=True,
        held_out=False,
    ),
    Layout(
        "L4",
        label_above=True,
        qty_column=False,
        totals_left=True,
        header_right=False,
        grid=False,
        held_out=False,
    ),
    Layout(
        "L5",
        label_above=False,
        qty_column=True,
        totals_left=True,
        header_right=False,
        grid=False,
        held_out=True,
    ),
    Layout(
        "L6",
        label_above=True,
        qty_column=False,
        totals_left=False,
        header_right=True,
        grid=True,
        held_out=True,
    ),
)

TEXT = json.loads((Path(__file__).with_name("invoice_text.json")).read_text(encoding="utf-8"))
WORDS = TEXT["words"]
ITEMS = TEXT["items"]
COMPANIES = tuple(TEXT["companies"])
SUFFIXES = tuple(TEXT["suffixes"])
STREETS = tuple(TEXT["streets"])
CITIES = tuple(TEXT["cities"])
MONTHS = {lang: tuple(names) for lang, names in TEXT["months"].items()}


def money(value: Decimal) -> Decimal:
    return value.quantize(CENT, rounding=ROUND_HALF_UP)


def format_amount(value: Decimal, lang: str) -> str:
    text = f"{value:,.2f}"
    if lang == "es":
        return text.replace(",", "_").replace(".", ",").replace("_", ".")
    return text


def format_rate(rate: Decimal, lang: str) -> str:
    text = f"{rate.normalize():f}"
    return (text.replace(".", ",") if lang == "es" else text) + " %"


def format_quantity(qty: Decimal, lang: str) -> str:
    text = f"{qty:f}"
    return text.replace(".", ",") if lang == "es" else text


# Every name and identifier a generated invoice carries is INVENTED and cannot exist (nothing in the
# public data may belong to anyone). The identifiers are written in forms no registry issues: a Spanish tax number has
# nine characters and an IBAN of Spain twenty-four, so the numbers below are one character short. Each function still
# makes the same random draws as the version that produced valid check digits, so the amounts, dates and layouts of the
# corpus do not move; only these strings and the invented names in invoice_text.json differ.
INVENTED_NOTE = "Invented sample: every name and identifier is made up and cannot exist."


def make_dni(rng: random.Random) -> str:
    """An invented personal tax number: seven digits and a letter (a real one has eight digits and a letter)."""
    number = rng.randrange(10_000_000, 99_999_999)
    return f"{number % 10_000_000:07d}{'TRWAGMYFPDXBNJZSQVHLCKE'[number % 23]}"


def make_cif(rng: random.Random) -> str:
    """An invented company tax number: a letter and seven digits, without the control character a real one ends in."""
    org = rng.choice("ABCDEFGHJNPQRSUVW")
    digits = [rng.randrange(10) for _ in range(7)]
    return f"{org}{''.join(map(str, digits))}"


def make_iban(rng: random.Random) -> str:
    """An invented account: "ESXX" and eighteen digits, in groups of four. A real IBAN has two check digits where the
    X are and a Spanish one twenty digits after them, and Extract does not take this for an IBAN at all."""
    digits = "".join(str(rng.randrange(10)) for _ in range(20))
    raw = f"ESXX{digits[:18]}"
    return " ".join(raw[i : i + 4] for i in range(0, len(raw), 4))


def make_date(rng: random.Random, lang: str, style: int) -> tuple[str, str]:
    year, month, day = (
        rng.choice((2025, 2026)),
        rng.randrange(1, 13),
        rng.randrange(1, 29),
    )
    iso = f"{year}-{month:02d}-{day:02d}"
    if style == 0:
        numeric = f"{day:02d}/{month:02d}/{year}" if lang == "es" else f"{month:02d}/{day:02d}/{year}"
        return numeric, iso
    if lang == "es":
        return f"{day} de {MONTHS['es'][month - 1]} de {year}", iso
    return f"{MONTHS['en'][month - 1]} {day}, {year}", iso


def make_party(rng: random.Random, lang: str, company: bool) -> dict:
    name = (
        f"{rng.choice(COMPANIES)} {rng.choice(SUFFIXES)}"
        if company
        else f"{rng.choice(COMPANIES)} {rng.choice(COMPANIES)}"
    )
    return {
        "name": name,
        "address": f"{rng.choice(STREETS)} {rng.randrange(1, 120)}, {rng.choice(CITIES)}",
        "tax_id": make_cif(rng) if company else make_dni(rng),
        "tax_id_kind": "cif" if company else "dni",
    }


def make_invoice(unit_id: str, layout: Layout, font: FontSpec, seed: int) -> dict:
    digest = hashlib.sha256(f"{seed}:{unit_id}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    lang = rng.choice(("es", "en"))
    rates = [Decimal(21)] if rng.random() < 0.75 else [Decimal(21), Decimal(10)]
    lines = []
    for _ in range(rng.randrange(2, 7)):
        qty = Decimal(rng.randrange(1, 40)) if rng.random() < 0.7 else Decimal(rng.randrange(5, 120)) / 10
        price = Decimal(rng.randrange(500, 250_000)) / 100
        lines.append(
            {
                "description": rng.choice(ITEMS[lang]),
                "quantity": qty,
                "unit_price": price,
                "amount": money(qty * price),
                "vat_rate": rng.choice(rates),
            }
        )
    subtotal = sum((line["amount"] for line in lines), Decimal(0))
    vat = []
    for rate in rates:
        base = sum((line["amount"] for line in lines if line["vat_rate"] == rate), Decimal(0))
        if base:
            vat.append({"rate": rate, "base": base, "amount": money(base * rate / 100)})
    withholding = None
    if rng.random() < 0.5:
        w_rate = rng.choice((Decimal(15), Decimal(7)))
        withholding = {"rate": w_rate, "amount": money(subtotal * w_rate / 100)}
    total = subtotal + sum((v["amount"] for v in vat), Decimal(0)) - (withholding["amount"] if withholding else 0)
    date_text, date_iso = make_date(rng, lang, rng.randrange(2))
    supplier = make_party(rng, lang, True)
    customer = make_party(rng, lang, rng.random() < 0.5)
    return {
        "id": unit_id,
        "layout": layout.key,
        "font": font.key,
        "language": lang,
        "seed": seed,
        "currency": "EUR",
        "number": f"{rng.choice(('F', 'INV', 'A'))}-{rng.randrange(2025, 2027)}-{rng.randrange(1, 9999):04d}",
        "date": date_text,
        "date_iso": date_iso,
        "supplier": supplier,
        "customer": customer,
        "iban": make_iban(rng),
        "lines": lines,
        "subtotal": subtotal,
        "vat": vat,
        "withholding": withholding,
        "total": total,
    }


class Page:
    """A canvas that records every string it draws, in the order drawn."""

    def __init__(self, path: Path, font: FontSpec) -> None:
        self.width, self.height = A4
        self.canvas = canvas.Canvas(str(path), pagesize=A4, pageCompression=0)
        self.canvas.setSubject(INVENTED_NOTE)
        self.regular, self.bold = _register(font)
        self.drawn: list[str] = []

    def text(
        self,
        x: float,
        y: float,
        value: str,
        size: float = 9.5,
        bold: bool = False,
        right: bool = False,
    ) -> None:
        self.canvas.setFont(self.bold if bold else self.regular, size)
        (self.canvas.drawRightString if right else self.canvas.drawString)(x, y, value)
        self.drawn.append(value)

    def rule(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.canvas.setLineWidth(0.5)
        self.canvas.line(x1, y1, x2, y2)

    def box(self, x: float, y: float, w: float, h: float) -> None:
        self.canvas.setLineWidth(0.5)
        self.canvas.rect(x, y, w, h)


_REGISTERED: set[str] = set()


def _register(font: FontSpec) -> tuple[str, str]:
    regular, bold = f"{font.key}-r", f"{font.key}-b"
    if regular not in _REGISTERED:
        pdfmetrics.registerFont(TTFont(regular, str(FONT_DIR / font.regular)))
        pdfmetrics.registerFont(TTFont(bold, str(FONT_DIR / font.bold)))
        _REGISTERED.add(regular)
    return regular, bold


def render(invoice: dict, layout: Layout, font: FontSpec, pdf_path: Path) -> list[dict]:
    """Draw the invoice and return the printed numbers with their role."""
    lang = invoice["language"]
    words = WORDS[lang]
    page = Page(pdf_path, font)
    left, right = 50.0, page.width - 50.0
    numbers: list[dict] = []

    def note(role: str, text: str, **extra: object) -> None:
        numbers.append({"role": role, "text": text, **extra})

    y = page.height - 60
    page.text(
        right if layout.header_right else left,
        y,
        words["title"],
        20,
        bold=True,
        right=layout.header_right,
    )

    def field(x: float, y_pos: float, label: str, value: str, width: float = 200) -> float:
        if layout.label_above:
            page.text(x, y_pos, label, 7.5)
            page.text(x, y_pos - 12, value, 10.5, bold=True)
            return y_pos - 32
        page.text(x, y_pos, f"{label}:", 9.5)
        page.text(x + width * 0.55, y_pos, value, 9.5, bold=True)
        return y_pos - 16

    fy = page.height - 100
    fy = field(left, fy, words["number"], invoice["number"])
    note("number", invoice["number"])
    fy = field(left, fy, words["date"], invoice["date"])
    note("date", invoice["date"])

    party_y = fy - 6
    for column, key in ((left, "supplier"), (left + 260, "customer")):
        party = invoice[key]
        page.text(column, party_y, words[key], 8, bold=True)
        page.text(column, party_y - 13, party["name"], 10.5, bold=True)
        page.text(column, party_y - 26, party["address"])
        page.text(column, party_y - 39, f"{words['tax_id']}: {party['tax_id']}")
        note(f"{key}_tax_id", party["tax_id"], kind=party["tax_id_kind"])

    y = party_y - 78
    cols = _columns(layout, left, right)
    headers = [
        ("description", False),
        ("quantity", True),
        ("unit_price", True),
        ("amount", True),
    ]
    if layout.grid:
        page.box(left - 4, y - 5, right - left + 8, 17)
    for name, is_num in headers:
        if name == "quantity" and not layout.qty_column:
            continue
        page.text(cols[name], y, words[name], 9, bold=True, right=is_num)
    y -= 22
    for i, line in enumerate(invoice["lines"]):
        page.text(cols["description"], y, line["description"])
        if layout.qty_column:
            q = format_quantity(line["quantity"], lang)
            page.text(cols["quantity"], y, q, right=True)
            note("line_quantity", q, line=i)
        p = format_amount(line["unit_price"], lang)
        a = format_amount(line["amount"], lang)
        page.text(cols["unit_price"], y, p, right=True)
        page.text(cols["amount"], y, a, right=True)
        note("line_unit_price", p, line=i)
        note("line_amount", a, line=i)
        if layout.grid:
            page.rule(left - 4, y - 5, right + 4, y - 5)
        y -= 18
    page.rule(left, y + 6, right, y + 6)

    ty = y - 14
    label_x = left if layout.totals_left else right - 230
    value_x = left + 230 if layout.totals_left else right
    rows = [("subtotal", words["subtotal"], invoice["subtotal"])]
    for pair in invoice["vat"]:
        label = f"{words['vat']} {format_rate(pair['rate'], lang)}"
        if len(invoice["vat"]) > 1:
            label += f" s/ {format_amount(pair['base'], lang)}"
        rows.append(("vat", label, pair))
    if invoice["withholding"]:
        rows.append(
            (
                "withholding",
                f"{words['withholding']} {format_rate(invoice['withholding']['rate'], lang)}",
                invoice["withholding"],
            )
        )
    rows.append(("total", words["total"], invoice["total"]))
    for kind, label, payload in rows:
        value = payload["amount"] if isinstance(payload, dict) else payload
        shown = ("-" if kind == "withholding" else "") + format_amount(value, lang)
        strong = kind == "total"
        page.text(label_x, ty, label, 10 if strong else 9.5, bold=strong)
        page.text(value_x - 12, ty, shown, 10 if strong else 9.5, bold=strong, right=True)
        page.text(value_x - 8, ty, "€", 10 if strong else 9.5, bold=strong)
        note(
            kind,
            shown,
            rate=format_rate(payload["rate"], lang) if isinstance(payload, dict) else None,
        )
        ty -= 17
    page.text(left, 90, f"{words['iban']}: {invoice['iban']}")
    note("iban", invoice["iban"])
    page.text(left, 76, words["pay"])
    page.canvas.showPage()
    page.canvas.save()
    return numbers


def _columns(layout: Layout, left: float, right: float) -> dict[str, float]:
    if layout.qty_column:
        return {
            "description": left,
            "quantity": right - 230,
            "unit_price": right - 115,
            "amount": right,
        }
    return {"description": left, "unit_price": right - 115, "amount": right}


def sidecar(invoice: dict, layout: Layout, font: FontSpec, numbers: list[dict], split: str) -> dict:
    lang = invoice["language"]
    return {
        "id": invoice["id"],
        "split": split,
        "layout": layout.key,
        "font": font.key,
        "language": lang,
        "seed": invoice["seed"],
        "currency": "EUR",
        "number": invoice["number"],
        "date": invoice["date"],
        "date_iso": invoice["date_iso"],
        "subtotal": format_amount(invoice["subtotal"], lang),
        "vat": [
            {
                "rate": format_rate(v["rate"], lang),
                "base": format_amount(v["base"], lang),
                "amount": format_amount(v["amount"], lang),
            }
            for v in invoice["vat"]
        ],
        "withholding": None
        if invoice["withholding"] is None
        else {
            "rate": format_rate(invoice["withholding"]["rate"], lang),
            "amount": "-" + format_amount(invoice["withholding"]["amount"], lang),
        },
        "total": format_amount(invoice["total"], lang),
        "supplier": invoice["supplier"],
        "customer": invoice["customer"],
        "iban": invoice["iban"],
        "lines": [
            {
                "description": line["description"],
                "quantity": format_quantity(line["quantity"], lang) if layout.qty_column else None,
                "unit_price": format_amount(line["unit_price"], lang),
                "amount": format_amount(line["amount"], lang),
            }
            for line in invoice["lines"]
        ],
        "numbers": numbers,
    }


def plan(seed: int, n_train_val: int = 150, n_test: int = 60) -> list[dict]:
    """The units to generate: the split decides train or validation; layouts 5-6 are all test."""
    from bench.corpus.split import assign_split

    train_layouts = [lay for lay in LAYOUTS if not lay.held_out]
    test_layouts = [lay for lay in LAYOUTS if lay.held_out]
    train_fonts = [f for f in FONTS if not f.held_out]
    units = []
    for i in range(n_train_val):
        split = assign_split(f"inv-tv-{i:04d}", seed, {"train": 0.8, "val": 0.2})
        units.append(
            {
                "id": f"inv-tv-{i:04d}",
                "split": split,
                "layout": train_layouts[i % len(train_layouts)],
                "font": train_fonts[(i * 5 + i // 4) % len(train_fonts)],
            }
        )
    for i in range(n_test):
        units.append(
            {
                "id": f"inv-te-{i:04d}",
                "split": "test",
                "layout": test_layouts[i % len(test_layouts)],
                "font": FONTS[(i * 3 + i // 2) % len(FONTS)],
            }
        )
    return units


V5_SEED = 20261004
V5_HELD_LAYOUTS = ("L3", "L4")
V5_HELD_FONTS = ("liberation-mono", "ubuntu")


def held_out_in_v5(layout: str, font: str) -> bool:
    """True for an invoice of a layout or a font that v5 keeps away from training (the sealed test T5 carries them)."""
    return layout in V5_HELD_LAYOUTS or font in V5_HELD_FONTS


def plan_v5(seed: int = V5_SEED, n: int = 60) -> list[dict]:
    """The invoices of the v5 sealed test: only the two held-out layouts, and a held-out font for every other invoice.

    The v4 test invoices (layouts 5-6, any font) are development data now, so v5 holds out two layouts and two fonts
    that the v4 training documents had; `held_out_in_v5` names the development invoices that set aside.
    """
    layouts = [lay for lay in LAYOUTS if lay.key in V5_HELD_LAYOUTS]
    held_fonts = [f for f in FONTS if f.key in V5_HELD_FONTS]
    other_fonts = [f for f in FONTS if f.key not in V5_HELD_FONTS]
    units = []
    for i in range(n):
        font = held_fonts[(i // 4) % len(held_fonts)] if i % 2 == 0 else other_fonts[(i // 2) % len(other_fonts)]
        units.append(
            {"id": f"inv-t5-{i:04d}", "split": "test5", "layout": layouts[(i // 2) % len(layouts)], "font": font}
        )
    return units


def generate(out: Path, seed: int, units: list[dict] | None = None) -> list[dict]:
    manifest = []
    for unit in units if units is not None else plan(seed):
        layout, font, split = unit["layout"], unit["font"], unit["split"]
        folder = out / split
        folder.mkdir(parents=True, exist_ok=True)
        invoice = make_invoice(unit["id"], layout, font, seed)
        numbers = render(invoice, layout, font, folder / f"{unit['id']}.pdf")
        data = sidecar(invoice, layout, font, numbers, split)
        (folder / f"{unit['id']}.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        manifest.append(
            {
                "id": unit["id"],
                "split": split,
                "layout": layout.key,
                "font": font.key,
                "language": data["language"],
            }
        )
    if units is None:
        (out / "invoices.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("bench/data/invoices"))
    parser.add_argument("--seed", type=int, help="default 1, or the v5 seed with --v5-test")
    parser.add_argument("--v5-test", action="store_true", help="generate the 60 invoices of the v5 sealed test")
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else (V5_SEED if args.v5_test else 1)
    manifest = generate(args.out, seed, plan_v5(seed) if args.v5_test else None)
    print(f"{len(manifest)} invoices in {args.out}")


if __name__ == "__main__":
    main()
