"""An independent oracle for the check-digit tests: generators of valid identifiers and the facts the tests rely on.

None of it is imported by `extract.checks`; the algorithms are written out again here on purpose, so a mistake in one
place does not hide in the other.

No identifier is written out in the tree. Every example below is built when this module loads, from numbers and letters
taken from the published rules (the DNI letter table, the CIF control table, the IBAN structure per country), and it
lives in memory for the length of a test: it is never saved, shown or shipped. They are invented and carry no
person, company or account. The files of the repository hold only identifiers that cannot exist
(`tests/test_public_identifiers.py` fails on any other).
"""

from __future__ import annotations

import random

from extract.types import Word

DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"
ALPHABET = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CIF_ORGANISATIONS = "ABCDEFGHJNPQRSUVW"
LETTER_ONLY = "PQRSNW"
DIGIT_ONLY = "ABEH"
CONTROL_LETTERS = "JABCDEFGHI"

# BBAN structure after the two check digits: n digits, a capital letters, c letters or digits.
STRUCTURE = {
    "ES": "n20",
    "DE": "n18",
    "FR": "n10 c11 n2",
    "GB": "a4 n14",
    "IT": "a1 n10 c12",
    "PT": "n21",
    "NL": "a4 n10",
    "BE": "n12",
    "AT": "n16",
    "IE": "a4 n14",
    "CH": "n5 c12",
    "LU": "n3 c13",
    "DK": "n14",
    "SE": "n20",
    "NO": "n11",
    "FI": "n14",
    "PL": "n24",
    "GR": "n7 c16",
    "CZ": "n20",
    "SK": "n20",
    "HU": "n24",
    "RO": "a4 c16",
    "BG": "a4 n4 n2 c8",
    "HR": "n17",
    "SI": "n15",
    "LT": "n16",
    "LV": "a4 c13",
    "EE": "n16",
    "CY": "n8 c16",
    "MT": "a4 n5 c18",
    "IS": "n22",
    "LI": "n5 c12",
    "MC": "n10 c11 n2",
    "SM": "a1 n10 c12",
    "AD": "n8 c12",
    "VA": "n18",
}


def classes_of(country: str) -> str:
    """One class letter per BBAN character, for example "nnnnccc"."""
    return "".join(seg[0] * int(seg[1:]) for seg in STRUCTURE[country].split())


def dni_letter(number: int) -> str:
    return DNI_LETTERS[number % 23]


def make_dni(rng: random.Random) -> str:
    n = rng.randrange(0, 100_000_000)
    return f"{n:08d}{dni_letter(n)}"


def make_nie(rng: random.Random) -> str:
    prefix = rng.choice("XYZ")
    n = rng.randrange(0, 10_000_000)
    return f"{prefix}{n:07d}{dni_letter(int(str('XYZ'.index(prefix)) + f'{n:07d}'))}"


def cif_control(digits: str) -> tuple[str, str]:
    """The control digit and its letter twin for the seven digits of a CIF."""
    values = [int(c) for c in digits]
    odd = sum(sum(divmod(2 * v, 10)) for v in values[0::2])
    even = sum(values[1::2])
    control = (10 - (odd + even) % 10) % 10
    return str(control), CONTROL_LETTERS[control]


def make_cif(rng: random.Random, org: str | None = None, letter_control: bool | None = None) -> str:
    org = org or rng.choice(CIF_ORGANISATIONS)
    digits = f"{rng.randrange(10_000_000):07d}"
    digit, letter = cif_control(digits)
    if org in LETTER_ONLY:
        use_letter = True
    elif org in DIGIT_ONLY:
        use_letter = False
    else:
        use_letter = rng.random() < 0.5 if letter_control is None else letter_control
    return f"{org}{digits}{letter if use_letter else digit}"


def cif_allows(org: str, control: str, digits: str) -> bool:
    digit, letter = cif_control(digits)
    if org not in CIF_ORGANISATIONS:
        return False
    if org in LETTER_ONLY:
        return control == letter
    if org in DIGIT_ONLY:
        return control == digit
    return control in (digit, letter)


def iban_mod97(compact: str) -> int:
    rearranged = compact[4:] + compact[:4]
    return int("".join(str(int(c, 36)) for c in rearranged)) % 97


def make_iban(rng: random.Random, country: str) -> str:
    bban = ""
    for cls in classes_of(country):
        pool = {"n": "0123456789", "a": "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "c": ALPHABET}[cls]
        bban += rng.choice(pool)
    check = 98 - iban_mod97(f"{country}00{bban}")
    compact = f"{country}{check:02d}{bban}"
    return " ".join(compact[i : i + 4] for i in range(0, len(compact), 4))


def words(*texts: str, page: int = 1) -> list[Word]:
    """Words laid out left to right on one row; only the text matters to the number rules."""
    return [Word(f"p{page}w{i}", page, text, (10.0 * i, 0.0, 10.0 * i + 8.0, 10.0)) for i, text in enumerate(texts)]


# Invented examples, built here so that none is written out as a string in the tree ---------------------------------


def fullwidth(text: str) -> str:
    """The same text with its ASCII digits replaced by the full-width digits (a different character each)."""
    return text.translate({ord(d): ord(d) + 0xFEE0 for d in "0123456789"})


def dni(number: int, letter: str) -> str:
    return f"{number:08d}{letter}"


def nie(prefix: str, number: int, letter: str) -> str:
    return f"{prefix}{number:07d}{letter}"


def cif(org: str, digits: str, control: str) -> str:
    return f"{org}{digits}{control}"


def iban_with_check(country: str, bban: str) -> str:
    """An IBAN of `country` with this BBAN and the check digits that make mod 97 hold, in groups of four."""
    check = 98 - iban_mod97(f"{country}00{bban}")
    compact = f"{country}{check:02d}{bban}"
    return " ".join(compact[i : i + 4] for i in range(0, len(compact), 4))


def filler(country: str) -> str:
    """A BBAN that fills the country's structure with a plain pattern (digits 1-9 in turn, the letter A for letters)."""
    out, digit = "", 0
    for cls in classes_of(country):
        if cls == "n":
            digit = digit % 9 + 1
            out += str(digit)
        else:
            out += "A"
    return out


# The letters come from the published tables, not from the generators above: 12345678 mod 23 = 2 is Z, 0 is T,
# 87654321 mod 23 = 10 is X; X/Y/Z stand for 0/1/2 in front of 1234567; the CIF controls are worked by hand
# (digits 1234567 sum to 26, control 4; seven zeros, control 0; seven nines sum to 63, control 7).
DNI = dni(12345678, "Z")
NIE = nie("X", 1234567, "L")
GOLDEN = {
    "dni": [DNI, dni(0, "T"), dni(87654321, "X")],
    "nie": [NIE, nie("Y", 1234567, "X"), nie("Z", 1234567, "R")],
    "cif": [cif("A", "1234567", "4"), cif("B", "0000000", "0"), cif("Q", "9999999", "G")],
    "iban": [iban_with_check(c, filler(c)) for c in ("ES", "DE", "GB", "FR")],
}
IBAN_ES = GOLDEN["iban"][0]
GOLDEN_INVALID = {
    "dni": dni(12345678, "A"),
    "nie": nie("X", 1234567, "A"),
    "cif": cif("A", "1234567", "5"),
    "iban": IBAN_ES[:-1] + ("1" if IBAN_ES[-1] != "1" else "2"),
}


def valid_registry() -> list[str]:
    """One valid IBAN per country of the table, built from its structure."""
    return [iban_with_check(country, filler(country)) for country in STRUCTURE]
