"""Number rules: readers, check digits, invoice arithmetic and the one-digit suggestion.

Pure functions over words and `InvoiceFields`: no OCR, no I/O, no floats. Every value is read from the words; the
cells of `InvoiceFields` only say which words play which part. A number is `checked` only when a verifying rule that
is sensitive to it passes and no rule that involves it fails; confidence and the guard's scores play no part.

The amount of a withholding is never `checked`: a document may print the deduction with or without a minus, so no
equation can cover that sign. Its digits are still caught, because the total and the withholding rule fail.
"""

from __future__ import annotations

import dataclasses
import datetime
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

from extract.types import (
    Cell,
    DigitSuggestion,
    InvoiceFields,
    NumberResult,
    RuleResult,
    Word,
)

LOCALES = ("es", "en")
CENT = Decimal("0.01")
DIGITS = frozenset("0123456789")
CURRENCY_SYMBOLS = "€$£"
CURRENCY_CODES = ("EUR", "USD", "GBP")

DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"
CIF_ORGANISATIONS = frozenset("ABCDEFGHJNPQRSUVW")
CIF_LETTER_ONLY = frozenset("PQRSNW")
CIF_DIGIT_ONLY = frozenset("ABEH")
CIF_CONTROL_LETTERS = "JABCDEFGHI"

# BBAN layout after the country code and check digits: n = digits, a = capital letters, c = letters or digits.
IBAN_LAYOUTS = {
    "AD": "n8 c12",
    "AT": "n16",
    "BE": "n12",
    "BG": "a4 n6 c8",
    "CH": "n5 c12",
    "CY": "n8 c16",
    "CZ": "n20",
    "DE": "n18",
    "DK": "n14",
    "EE": "n16",
    "ES": "n20",
    "FI": "n14",
    "FR": "n10 c11 n2",
    "GB": "a4 n14",
    "GR": "n7 c16",
    "HR": "n17",
    "HU": "n24",
    "IE": "a4 n14",
    "IS": "n22",
    "IT": "a1 n10 c12",
    "LI": "n5 c12",
    "LT": "n16",
    "LU": "n3 c13",
    "LV": "a4 c13",
    "MC": "n10 c11 n2",
    "MT": "a4 n5 c18",
    "NL": "a4 n10",
    "NO": "n11",
    "PL": "n24",
    "PT": "n21",
    "RO": "a4 c16",
    "SE": "n20",
    "SI": "n15",
    "SK": "n20",
    "SM": "a1 n10 c12",
    "VA": "n18",
}
IBAN_MIN_LENGTH = 15
IBAN_MAX_LENGTH = 34

# Rates that exist in Spain (VAT) and in the Canary Islands (IGIC). A rate outside the list can only fail.
PLAUSIBLE_RATES = frozenset(Decimal(r) for r in ("0", "2", "3", "4", "5", "7", "9.5", "10", "15", "20", "21"))

MONTHS = {
    "enero": 1, "febrero": 2, "marzo": 3, "abril": 4, "mayo": 5, "junio": 6, "julio": 7, "agosto": 8,
    "septiembre": 9, "setiembre": 9, "octubre": 10, "noviembre": 11, "diciembre": 12,
    "january": 1, "february": 2, "march": 3, "april": 4, "june": 6, "july": 7, "august": 8,
    "september": 9, "october": 10, "november": 11, "december": 12,
    "ene": 1, "feb": 2, "mar": 3, "abr": 4, "may": 5, "jun": 6, "jul": 7, "ago": 8, "sep": 9, "sept": 9,
    "oct": 10, "nov": 11, "dic": 12, "jan": 1, "apr": 4, "aug": 8, "dec": 12,
}  # fmt: skip

AMOUNT_ES = re.compile(r"^(?:[0-9]{1,3}(?:\.[0-9]{3})+|[0-9]+)(?:,[0-9]+)?$")
AMOUNT_EN = re.compile(r"^(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)(?:\.[0-9]+)?$")
PERCENT_ES = re.compile(r"^[0-9]+(?:,[0-9]+)?$")
PERCENT_EN = re.compile(r"^[0-9]+(?:\.[0-9]+)?$")
INTEGER_ES = re.compile(r"^(?:[0-9]{1,3}(?:\.[0-9]{3})+|[0-9]+)$")
INTEGER_EN = re.compile(r"^(?:[0-9]{1,3}(?:,[0-9]{3})+|[0-9]+)$")
NUMERIC_DATE = re.compile(r"^([0-9]{1,4})([/.-])([0-9]{1,2})\2([0-9]{1,4})$")
AMOUNT_LIKE = re.compile(r"^[-+(]?[0-9][0-9.,]*[)-]?$")
DNI_SHAPE = re.compile(r"^[0-9]{8}[A-Z]$")
NIE_SHAPE = re.compile(r"^[XYZ][0-9]{7}[A-Z]$")
CIF_SHAPE = re.compile(r"^[A-Z][0-9]{7}[0-9A-Z]$")
IBAN_SHAPE = re.compile(r"^[A-Z]{2}[0-9]{2}[0-9A-Z]+$")
IBAN_START = re.compile(r"^[A-Z]{2}[0-9]{2}")
IBAN_GROUP = re.compile(r"^[0-9A-Z]{1,4}$")


def _upper(text: str) -> str:
    """Uppercase the ASCII letters only: other scripts must never turn into something valid."""
    return "".join(c.upper() if "a" <= c <= "z" else c for c in text)


def _has_digit(text: str) -> bool:
    return any(c in DIGITS for c in text)


def _ascii_digits(text: str) -> bool:
    return bool(text) and all(c in DIGITS for c in text)


def _is_locale(locale: str) -> str:
    if locale not in LOCALES:
        raise ValueError(f"locale must be one of {LOCALES}, not {locale!r}")
    return locale


# Readers -----------------------------------------------------------------------------------------------------------


def _strip_currency(text: str) -> str | None:
    """The text without one currency symbol or code at either end, or None when there are two or a stray one."""
    body = text
    stripped = False
    for code in CURRENCY_CODES:
        if body.startswith(code):
            body, stripped = body[len(code) :], True
            break
        if body.endswith(code):
            body, stripped = body[: -len(code)], True
            break
    if not stripped:
        if body[:1] in tuple(CURRENCY_SYMBOLS):
            body, stripped = body[1:], True
        elif body[-1:] in tuple(CURRENCY_SYMBOLS):
            body, stripped = body[:-1], True
    if not body or any(c in CURRENCY_SYMBOLS for c in body):
        return None
    return body


def _split_sign(body: str) -> tuple[bool, str] | None:
    if body.startswith("-") and not body.endswith("-"):
        return True, body[1:]
    if body.endswith("-") and not body.startswith("-"):
        return True, body[:-1]
    if body.startswith("(") and body.endswith(")"):
        return True, body[1:-1]
    return False, body


def _plain(core: str, locale: str) -> str:
    return core.replace(".", "").replace(",", ".") if locale == "es" else core.replace(",", "")


def parse_amount(text: str, locale: str) -> Decimal | None:
    """An amount in the locale's notation, or None: a format that is not the locale's is refused, never guessed."""
    _is_locale(locale)
    body = _strip_currency(text)
    if body is None:
        return None
    signed = _split_sign(body)
    if signed is None:
        return None
    negative, core = signed
    pattern = AMOUNT_ES if locale == "es" else AMOUNT_EN
    if not pattern.match(core):
        return None
    value = Decimal(_plain(core, locale))
    return -value if negative else value


def parse_percent(text: str, locale: str) -> Decimal | None:
    _is_locale(locale)
    if not text.endswith("%"):
        return None
    core = text[:-1]
    if core.endswith(" "):
        core = core[:-1]
    pattern = PERCENT_ES if locale == "es" else PERCENT_EN
    if not pattern.match(core):
        return None
    return Decimal(_plain(core, locale))


def parse_integer(text: str, locale: str) -> int | None:
    _is_locale(locale)
    negative = text.startswith("-")
    core = text[1:] if negative else text
    pattern = INTEGER_ES if locale == "es" else INTEGER_EN
    if not pattern.match(core):
        return None
    value = int(core.replace(".", "").replace(",", ""))
    return -value if negative else value


def parse_date(text: str, locale: str) -> datetime.date | None:
    _is_locale(locale)
    numeric = NUMERIC_DATE.match(text)
    if numeric:
        first, _, middle, last = numeric.groups()
        if len(first) == 4 and len(last) <= 2:
            year, month, day = first, middle, last
        elif len(last) == 4 and len(first) <= 2:
            day, month = (first, middle) if locale == "es" else (middle, first)
            year = last
        else:
            return None
        return _calendar(year, month, day)
    tokens = [t for t in text.casefold().replace(",", " ").replace("-", " ").split() if t != "de"]
    if len(tokens) != 3:
        return None
    if tokens[1] in MONTHS:
        day, month, year = tokens[0], MONTHS[tokens[1]], tokens[2]
    elif tokens[0] in MONTHS:
        day, month, year = tokens[1], MONTHS[tokens[0]], tokens[2]
    else:
        return None
    return _calendar(year, str(month), day)


def _calendar(year: str, month: str, day: str) -> datetime.date | None:
    if len(year) != 4 or not (_ascii_digits(year) and _ascii_digits(month) and _ascii_digits(day)):
        return None
    try:
        return datetime.date(int(year), int(month), int(day))
    except ValueError:
        return None


def _separator_locale(token: str) -> str | None:
    """The locale a bare numeric token can only belong to, or None when it could be either or is not a number."""
    if not token or not (token[0] in DIGITS and token[-1] in DIGITS):
        return None
    if any(c not in DIGITS and c not in ".," for c in token):
        return None
    dots, commas = token.count("."), token.count(",")
    if not dots and not commas:
        return None
    if dots and commas:
        last_dot, last_comma = token.rfind("."), token.rfind(",")
        return "es" if last_comma > last_dot and AMOUNT_ES.match(token) else "en" if AMOUNT_EN.match(token) else None
    separator, count = ("." if dots else ",", dots or commas)
    lead, _, tail = token.partition(separator)
    if count >= 2:
        valid = (INTEGER_ES if separator == "." else INTEGER_EN).match(token)
        return ("es" if separator == "." else "en") if valid else None
    if len(tail) == 3 and 1 <= len(lead) <= 3:
        return None
    return "es" if separator == "," else "en"


def detect_locale(words: Sequence[Word], first_page_language: str = "es") -> str:
    """`es` or `en` by majority of the unambiguous numeric tokens; a tie goes to the first page's language."""
    votes = {"es": 0, "en": 0}
    for word in words:
        locale = _separator_locale(word.text.strip("-()"))
        if locale:
            votes[locale] += 1
    if votes["es"] == votes["en"]:
        return _is_locale(first_page_language)
    return "es" if votes["es"] > votes["en"] else "en"


# Check digits ------------------------------------------------------------------------------------------------------


def check_dni(text: str) -> bool:
    text = _upper(text)
    return bool(DNI_SHAPE.match(text)) and DNI_LETTERS[int(text[:8]) % 23] == text[8]


def check_nie(text: str) -> bool:
    text = _upper(text)
    if not NIE_SHAPE.match(text):
        return False
    number = int(str("XYZ".index(text[0])) + text[1:8])
    return DNI_LETTERS[number % 23] == text[8]


def _cif_control(digits: str) -> tuple[str, str]:
    values = [int(c) for c in digits]
    odd = sum(sum(divmod(2 * v, 10)) for v in values[0::2])
    control = (10 - (odd + sum(values[1::2])) % 10) % 10
    return str(control), CIF_CONTROL_LETTERS[control]


def check_cif(text: str) -> bool:
    text = _upper(text)
    if not CIF_SHAPE.match(text) or text[0] not in CIF_ORGANISATIONS:
        return False
    digit, letter = _cif_control(text[1:8])
    if text[0] in CIF_LETTER_ONLY:
        return text[8] == letter
    if text[0] in CIF_DIGIT_ONLY:
        return text[8] == digit
    return text[8] in (digit, letter)


def _iban_classes(country: str) -> str:
    return "".join(segment[0] * int(segment[1:]) for segment in IBAN_LAYOUTS[country].split())


def _iban_compact(text: str) -> str:
    return _upper(text.replace(" ", ""))


def iban_expected_length(country: str) -> int | None:
    return 4 + len(_iban_classes(country)) if country in IBAN_LAYOUTS else None


def _iban_layout_ok(compact: str) -> bool:
    if not IBAN_SHAPE.match(compact) or compact[:2] not in IBAN_LAYOUTS:
        return False
    classes = _iban_classes(compact[:2])
    if len(compact) != 4 + len(classes):
        return False
    for char, cls in zip(compact[4:], classes, strict=True):
        if cls == "n" and char not in DIGITS or cls == "a" and not ("A" <= char <= "Z"):
            return False
    return True


def _iban_mod97(compact: str) -> bool:
    rearranged = compact[4:] + compact[:4]
    number = int("".join(c if c in DIGITS else str(ord(c) - 55) for c in rearranged))
    return number % 97 == 1


def check_iban(text: str) -> bool:
    compact = _iban_compact(text)
    return _iban_layout_ok(compact) and _iban_mod97(compact)


def iban_covers_every_character(text: str) -> bool:
    """False when the country's BBAN has letter-or-digit positions: a digit read as a letter there changes the length
    of the number mod 97 reads, and one swap in 97 goes unseen, so a pass does not verify every character."""
    country = _iban_compact(text)[:2]
    return country in IBAN_LAYOUTS and "c" not in _iban_classes(country)


def find_ibans(words: Sequence[Word]) -> list[tuple[tuple[str, ...], str]]:
    """IBANs written in one word or in groups of four over consecutive words, each with the ids of all its words."""
    found = []
    i = 0
    while i < len(words):
        start = words[i].text
        if not IBAN_START.match(start) or not _has_digit(start[:4]):
            i += 1
            continue
        taken = [words[i]]
        length = len(start)
        expected = iban_expected_length(start[:2])
        limit = expected or IBAN_MAX_LENGTH
        while length < limit and len(start) <= 4 and i + len(taken) < len(words):
            nxt = words[i + len(taken)]
            if not IBAN_GROUP.match(nxt.text) or length + len(nxt.text) > limit:
                break
            taken.append(nxt)
            length += len(nxt.text)
            if len(nxt.text) < 4:
                break
        if length >= IBAN_MIN_LENGTH:
            found.append((tuple(w.id for w in taken), " ".join(w.text for w in taken)))
            i += len(taken)
        else:
            i += 1
    return found


# Rules and results -------------------------------------------------------------------------------------------------

PASS, FAIL, NOT_APPLICABLE = "pass", "fail", "not_applicable"


@dataclass(frozen=True)
class Operand:
    word_id: str
    value: Decimal
    step: Decimal


@dataclass
class Equation:
    """One arithmetic rule over some operands. `verified` lists the operands this rule is sensitive to."""

    key: str
    ident: tuple
    operands: tuple[str, ...]
    passes: bool
    params: dict
    verified: frozenset[str] = frozenset()
    applicable: bool = True


def _quantize(value: Decimal, rounding: str) -> Decimal:
    return value.quantize(CENT, rounding=rounding)


def _cents(value: Decimal) -> frozenset[Decimal]:
    return frozenset({_quantize(value, ROUND_HALF_UP), _quantize(value, ROUND_HALF_EVEN)})


def _equal(left: Iterable[Decimal], right: Iterable[Decimal]) -> tuple[bool, bool]:
    """Whether some accepted rounding of each side agree, and whether every accepted rounding gives the same cents."""
    left_set = frozenset().union(*(_cents(v) for v in left))
    right_set = frozenset().union(*(_cents(v) for v in right))
    return bool(left_set & right_set), len(left_set) == 1 and len(right_set) == 1


def _step_of(text: str, locale: str) -> Decimal:
    separator = "," if locale == "es" else "."
    decimals = len(text.rsplit(separator, 1)[1]) if separator in text else 0
    return Decimal(1).scaleb(-decimals)


def _sensitive(coefficient: Decimal, operand: Operand) -> bool:
    """A change of one in the operand's last printed decimal moves the result by at least a cent."""
    return abs(coefficient) * operand.step >= CENT


@dataclass
class _Doc:
    words: dict[str, Word]
    fields: InvoiceFields
    locale: str
    values: dict[str, Operand | None] = field(default_factory=dict)

    def operand(self, cell: Cell | None, kind: str) -> Operand | None:
        """The one number a cell stands for, or None if it is missing, split over words or not a number."""
        word_id = single_number_word(cell, self.words)
        return None if word_id is None else self.read(word_id, kind)

    def read(self, word_id: str, kind: str) -> Operand | None:
        key = (word_id, kind)
        if key in self.values:
            return self.values[key]
        text = self.words[word_id].text
        if kind == "percent":
            value = parse_percent(text if text.endswith("%") else text + "%", self.locale)
            step = _step_of(text.rstrip("%"), self.locale)
        else:
            value = parse_amount(text, self.locale)
            step = _step_of(text, self.locale)
        operand = None if value is None else Operand(word_id, value, step)
        self.values[key] = operand
        return operand


def single_number_word(cell: Cell | None, words: Mapping[str, Word]) -> str | None:
    if cell is None:
        return None
    ids = [i for i in cell.word_ids if i in words and _has_digit(words[i].text)]
    return ids[0] if len(ids) == 1 else None


def _equations(doc: _Doc, only: Collection[tuple] | None = None) -> list[Equation]:
    """Every arithmetic rule the fields allow; with `only`, just the rules with those identities."""

    def wanted(ident: tuple) -> bool:
        return only is None or ident in only

    fields = doc.fields
    found: list[Equation] = []
    subtotal = doc.operand(fields.subtotal, "amount")
    lines = [
        (
            index,
            doc.operand(line.quantity, "amount"),
            doc.operand(line.unit_price, "amount"),
            doc.operand(line.amount, "amount"),
        )
        for index, line in enumerate(fields.lines)
    ]
    amounts = [amount for _, _, _, amount in lines]
    all_amounts = bool(lines) and all(a is not None for a in amounts)

    for index, quantity, price, amount in lines:
        ident = ("line_amount", index)
        if price is None or amount is None or not wanted(ident):
            continue
        if quantity is None:
            found.append(
                Equation("checks.rule.line_amount", ident, (price.word_id, amount.word_id), True, {}, applicable=False)
            )
            continue
        product = quantity.value * price.value
        ok, exact = _equal([product], [amount.value])
        exact = exact and quantity.value != 0
        covered = ((quantity, price.value), (price, quantity.value), (amount, Decimal(1)))
        params = {
            "quantity": quantity.value,
            "unit_price": price.value,
            "expected": _quantize(product, ROUND_HALF_UP),
            "found": amount.value,
        }
        found.append(
            Equation(
                "checks.rule.line_amount",
                ident,
                (quantity.word_id, price.word_id, amount.word_id),
                ok,
                params,
                frozenset(o.word_id for o, c in covered if exact and _sensitive(c, o)),
            )
        )

    if subtotal is not None and all_amounts and wanted(("lines_sum",)):
        expected = sum((a.value for a in amounts), Decimal(0))
        ok, exact = _equal([expected], [subtotal.value])
        operands = [*amounts, subtotal]
        found.append(
            Equation(
                "checks.rule.lines_sum",
                ("lines_sum",),
                tuple(o.word_id for o in operands),
                ok,
                {"expected": _quantize(expected, ROUND_HALF_UP), "found": subtotal.value},
                frozenset(o.word_id for o in operands if exact and _sensitive(Decimal(1), o)),
            )
        )

    several = len(fields.vat) > 1
    for j, pair in enumerate(fields.vat):
        if not wanted(("vat_amount", j)):
            continue
        rate = doc.operand(pair.rate, "percent")
        amount = doc.operand(pair.amount, "amount")
        base = doc.operand(pair.base, "amount") if pair.base is not None else (None if several else subtotal)
        if rate is None or amount is None or base is None:
            continue
        factor = rate.value.scaleb(-2)
        candidates = [base.value * factor]
        if pair.base is None and not several and all_amounts:
            for rounding in (ROUND_HALF_UP, ROUND_HALF_EVEN):
                candidates.append(sum((_quantize(a.value * factor, rounding) for a in amounts), Decimal(0)))
        ok, exact = _equal(candidates, [amount.value])
        exact = exact and amount.step == CENT
        covered = ((amount, Decimal(1)), (rate, base.value.scaleb(-2)), (base, factor))
        params = {"rate": rate.value, "expected": _quantize(candidates[0], ROUND_HALF_UP), "found": amount.value}
        found.append(
            Equation(
                "checks.rule.vat_amount",
                ("vat_amount", j),
                (rate.word_id, base.word_id, amount.word_id),
                ok,
                params,
                frozenset(o.word_id for o, c in covered if exact and _sensitive(c, o)),
            )
        )

    withholding = doc.operand(fields.withholding, "amount")
    withholding_rate = doc.operand(fields.withholding_rate, "percent")
    if withholding is not None and withholding_rate is not None and subtotal is not None and wanted(("withholding",)):
        factor = withholding_rate.value.scaleb(-2)
        expected = subtotal.value * factor
        ok, exact = _equal([expected], [abs(withholding.value)])
        covered = ((withholding, Decimal(1)), (withholding_rate, subtotal.value.scaleb(-2)), (subtotal, factor))
        params = {
            "rate": withholding_rate.value,
            "expected": _quantize(expected, ROUND_HALF_UP),
            "found": abs(withholding.value),
        }
        found.append(
            Equation(
                "checks.rule.withholding_amount",
                ("withholding",),
                (withholding_rate.word_id, subtotal.word_id, withholding.word_id),
                ok,
                params,
                frozenset(o.word_id for o, c in covered if exact and _sensitive(c, o)) - {withholding.word_id},
            )
        )

    total = doc.operand(fields.total, "amount")
    vat_amounts = [doc.operand(pair.amount, "amount") for pair in fields.vat]
    complete = vat_amounts and all(v is not None for v in vat_amounts)
    if subtotal is not None and total is not None and complete and wanted(("total",)):
        if fields.withholding is None or withholding is not None:
            expected = subtotal.value + sum((v.value for v in vat_amounts), Decimal(0))
            if withholding is not None:
                expected -= abs(withholding.value)
            ok, exact = _equal([expected], [total.value])
            operands = [subtotal, *vat_amounts, *([withholding] if withholding is not None else []), total]
            found.append(
                Equation(
                    "checks.rule.total",
                    ("total",),
                    tuple(o.word_id for o in operands),
                    ok,
                    {"expected": _quantize(expected, ROUND_HALF_UP), "found": total.value},
                    frozenset(o.word_id for o in operands if exact and _sensitive(Decimal(1), o))
                    - {w.word_id for w in (withholding,) if w is not None},
                )
            )
    return found


# Words -> numbers --------------------------------------------------------------------------------------------------


@dataclass
class _Entry:
    word_ids: tuple[str, ...]
    text: str
    kind: str
    value: Decimal | None
    rules: list[RuleResult] = field(default_factory=list)
    suggestion: DigitSuggestion | None = None


def _rule(key: str, params: Mapping, passes: bool, verifying: bool) -> RuleResult:
    return RuleResult(key, dict(params), PASS if passes else FAIL, verifying)


def _identifier_rules(kind: str, text: str) -> list[RuleResult]:
    if kind == "dni":
        return [_rule("checks.rule.dni_letter", {}, check_dni(text), True)]
    if kind == "nie":
        return [_rule("checks.rule.nie_letter", {}, check_nie(text), True)]
    if kind == "cif":
        return [_rule("checks.rule.cif_control", {}, check_cif(text), False)]
    compact = _iban_compact(text)
    country = compact[:2]
    expected = iban_expected_length(country)
    rules = []
    if expected is not None and len(compact) != expected:
        rules.append(_rule("checks.rule.iban_length", {"country": country, "expected": expected}, False, False))
    elif expected is None:
        rules.append(_rule("checks.rule.iban_length", {"country": country, "expected": 0}, False, False))
    else:
        verifying = iban_covers_every_character(compact)
        rules.append(_rule("checks.rule.iban_mod97", {"country": country}, check_iban(compact), verifying))
    return rules


IDENTIFIER_CHECKS = {"dni": check_dni, "nie": check_nie, "cif": check_cif, "iban": check_iban}


def _classify(word_text: str, locale: str) -> tuple[str, Decimal | None, list[RuleResult]]:
    text = word_text.rstrip(".,;:")
    upper = _upper(text)
    for kind, shape in (("dni", DNI_SHAPE), ("nie", NIE_SHAPE)):
        if shape.match(upper):
            return kind, None, _identifier_rules(kind, upper)
    if CIF_SHAPE.match(upper) and upper[0] in CIF_ORGANISATIONS:
        return "cif", None, _identifier_rules("cif", upper)
    if text.endswith("%") and len(text) > 1:
        percent = parse_percent(text, locale)
        return "percent", percent, [_rule("checks.rule.number_format", {"locale": locale}, percent is not None, False)]
    if NUMERIC_DATE.match(text):
        return "date", None, [_rule("checks.rule.date_valid", {}, parse_date(text, locale) is not None, False)]
    if AMOUNT_LIKE.match(text):
        if _ascii_digits(text.lstrip("-")):
            integer = parse_integer(text, locale)
            return "integer", None if integer is None else Decimal(integer), []
        value = parse_amount(text, locale)
        return "amount", value, [_rule("checks.rule.number_format", {"locale": locale}, value is not None, False)]
    return "other", None, []


def _roles(fields: InvoiceFields, words: Mapping[str, Word]) -> dict[str, str]:
    """The kind of number the fields give to each single-word cell: amount, quantity or percent."""
    roles: dict[str, str] = {}

    def put(cell: Cell | None, kind: str) -> None:
        word_id = single_number_word(cell, words)
        if word_id is not None:
            roles[word_id] = kind

    put(fields.subtotal, "amount")
    put(fields.total, "amount")
    put(fields.withholding, "amount")
    put(fields.withholding_rate, "percent")
    for pair in fields.vat:
        put(pair.rate, "percent")
        put(pair.amount, "amount")
        put(pair.base, "amount")
    for line in fields.lines:
        put(line.quantity, "quantity")
        put(line.unit_price, "amount")
        put(line.amount, "amount")
    return roles


def _entry_for_role(word: Word, kind: str, locale: str) -> _Entry:
    if kind == "percent":
        text = word.text if word.text.endswith("%") else word.text + "%"
        value = parse_percent(text, locale)
    else:
        value = parse_amount(word.text, locale)
    rule = _rule("checks.rule.number_format", {"locale": locale}, value is not None, False)
    return _Entry((word.id,), word.text, kind, value, [rule])


def _build_entries(
    words: Sequence[Word], fields: InvoiceFields, locale: str, by_id: Mapping[str, Word]
) -> list[_Entry]:
    roles = _roles(fields, by_id)
    ibans = {ids[0]: (ids, text) for ids, text in find_ibans(words)}
    claimed = {i for ids, _ in ibans.values() for i in ids}
    entries = []
    for word in words:
        if word.id in ibans:
            ids, text = ibans[word.id]
            entries.append(_Entry(ids, text, "iban", None, _identifier_rules("iban", text)))
        elif word.id in claimed or not _has_digit(word.text):
            continue
        elif word.id in roles:
            entries.append(_entry_for_role(word, roles[word.id], locale))
        else:
            kind, value, rules = _classify(word.text, locale)
            entries.append(_Entry((word.id,), word.text, kind, value, rules))
    return entries


def _status(rules: Sequence[RuleResult]) -> str:
    if any(r.result == FAIL for r in rules):
        return "mismatch"
    if any(r.verifying and r.result == PASS for r in rules):
        return "checked"
    return "unverified"


# One-digit suggestion ----------------------------------------------------------------------------------------------


def _digit_positions(text: str) -> list[int]:
    return [i for i, c in enumerate(text) if c in DIGITS]


def _suggest_identifier(entry: _Entry) -> DigitSuggestion | None:
    """The one digit change that makes the identifier pass its failing rule, if there is exactly one.

    A tie gives no suggestion at all. The suggestion is `verified` when that rule is a verifying one: a DNI or NIE
    letter, an IBAN whose mod-97 covers every character. A CIF never verifies and an IBAN with letters in its account
    part does not either, so their suggestions are offered with `verified` false."""
    check = IDENTIFIER_CHECKS[entry.kind]
    failing = next(r for r in entry.rules if r.result == FAIL)
    fixes = []
    for i in _digit_positions(entry.text):
        for new in "0123456789":
            if new != entry.text[i] and check(entry.text[:i] + new + entry.text[i + 1 :]):
                fixes.append(DigitSuggestion(i, entry.text[i], new, failing.verifying, failing.key))
                if len(fixes) > 1:
                    return None
    return fixes[0] if fixes else None


def _suggest_arithmetic(
    failing: Sequence[Equation],
    equations: Sequence[Equation],
    doc: _Doc,
    entries: Mapping[str, _Entry],
    roles: Mapping[str, str],
    rated: Collection[str],
) -> dict[str, DigitSuggestion]:
    """Per group of failing rules that share numbers: the one digit change, if exactly one, that fixes all of them.

    The suggestion is `verified` when one of those rules is a verifying rule for the changed number (sensitive to
    every digit of it) and passes after the change; `rule` is its key, or the first fixed rule's when none is."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for eq in failing:
        for operand in eq.operands[1:]:
            parent[find(operand)] = find(eq.operands[0])
        find(eq.operands[0])
    groups: dict[str, list[str]] = {}
    for word_id in list(parent):
        groups.setdefault(find(word_id), []).append(word_id)
    touching: dict[str, set[tuple]] = {}
    for eq in equations:
        for operand in eq.operands:
            touching.setdefault(operand, set()).add(eq.ident)
    suggestions = {}
    for members in groups.values():
        idents = {eq.ident for eq in failing if eq.operands[0] in members}
        fixes: list[tuple[str, DigitSuggestion]] = []
        for word_id in members:
            text = doc.words[word_id].text
            for i in _digit_positions(text):
                for new in "0123456789":
                    if new == text[i]:
                        continue
                    after = _fixed_equations(
                        doc, word_id, text[:i] + new + text[i + 1 :], idents, touching[word_id], roles, rated
                    )
                    if after is not None:
                        fixes.append((word_id, _arithmetic_fix(i, text[i], new, word_id, idents, after)))
                        if len(fixes) > 1:
                            break
                if len(fixes) > 1:
                    break
            if len(fixes) > 1:
                break
        if len(fixes) == 1 and fixes[0][0] in entries:
            suggestions[fixes[0][0]] = fixes[0][1]
    return suggestions


def _arithmetic_fix(
    position: int, old: str, new: str, word_id: str, idents: Collection[tuple], after: Sequence[Equation]
) -> DigitSuggestion:
    fixed = sorted((eq for eq in after if eq.ident in idents), key=lambda eq: (eq.key, eq.ident))
    verifying = [eq for eq in fixed if word_id in eq.verified]
    chosen = verifying[0] if verifying else fixed[0]
    return DigitSuggestion(position, old, new, bool(verifying), chosen.key)


def _fixed_equations(
    doc: _Doc,
    word_id: str,
    text: str,
    idents: Collection[tuple],
    touching: Collection[tuple],
    roles: Mapping[str, str],
    rated: Collection[str],
) -> list[Equation] | None:
    """The rules that apply to the number, checked again with `text` in its place, when every one of them passes and
    no rule of the group went missing; None otherwise."""
    changed = dataclasses.replace(doc.words[word_id], text=text)
    trial = _Doc({**doc.words, word_id: changed}, doc.fields, doc.locale)
    kind = roles.get(word_id, "amount")
    operand = trial.read(word_id, kind)
    if operand is None:
        return None
    if word_id in rated and operand.value not in PLAUSIBLE_RATES:
        return None
    after = _equations(trial, only={*idents, *touching})
    present = {eq.ident for eq in after}
    if not set(idents) <= present:
        return None
    relevant = [eq for eq in after if eq.applicable and (eq.ident in idents or word_id in eq.operands)]
    return relevant if all(eq.passes for eq in relevant) else None


# Entry point -------------------------------------------------------------------------------------------------------


def run(
    words: Sequence[Word],
    invoice: InvoiceFields,
    locale: str = "auto",
    unreadable_pages: Collection[int] = (),
    *,
    first_page_language: str = "es",
    suggest: bool = True,
) -> list[NumberResult]:
    """Every number of the document with its status, the rules that touch it and, for a mismatch, a one-digit fix.

    `locale` is `es`, `en` or `auto` (the majority of the unambiguous numeric tokens, a tie going to
    `first_page_language`). A number on a page in `unreadable_pages` is never `checked`. The suggestion is offered,
    never applied, and does not depend on any confidence.
    """
    if locale == "auto":
        locale = detect_locale(words, first_page_language)
    _is_locale(locale)
    by_id = {w.id: w for w in words}
    doc = _Doc(by_id, invoice, locale)
    entries = _build_entries(words, invoice, locale, by_id)
    by_word = {word_id: entry for entry in entries for word_id in entry.word_ids}
    roles = _roles(invoice, by_id)

    equations = _equations(doc)
    for eq in equations:
        result = NOT_APPLICABLE if not eq.applicable else PASS if eq.passes else FAIL
        for word_id in eq.operands:
            if word_id in by_word:
                params = eq.params if eq.applicable else {}
                by_word[word_id].rules.append(RuleResult(eq.key, dict(params), result, word_id in eq.verified))
    rated = set()
    for pair in invoice.vat:
        word_id = single_number_word(pair.rate, by_id)
        if word_id is None or word_id not in by_word:
            continue
        rated.add(word_id)
        operand = doc.read(word_id, "percent")
        if operand is not None:
            plausible = operand.value in PLAUSIBLE_RATES
            by_word[word_id].rules.append(
                _rule("checks.rule.vat_rate_plausible", {"rate": operand.value}, plausible, False)
            )

    if suggest:
        failing = [eq for eq in equations if eq.applicable and not eq.passes]
        for word_id, suggestion in _suggest_arithmetic(failing, equations, doc, by_word, roles, rated).items():
            by_word[word_id].suggestion = suggestion
        for entry in entries:
            if entry.kind in IDENTIFIER_CHECKS and any(r.result == FAIL for r in entry.rules):
                entry.suggestion = _suggest_identifier(entry)

    unreadable = set(unreadable_pages)
    results = []
    for entry in entries:
        status = _status(entry.rules)
        if status == "checked" and any(by_id[i].page in unreadable for i in entry.word_ids):
            status = "unverified"
        results.append(
            NumberResult(
                entry.word_ids, entry.text, entry.value, entry.kind, status, tuple(entry.rules), entry.suggestion
            )
        )
    return results
