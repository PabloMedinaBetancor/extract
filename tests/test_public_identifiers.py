"""No identifier in the repository, or in what its generators make, may belong to anyone.

Extract goes public with synthetic invoices, saved readings and test texts. A tax number, an IBAN, a phone number, a card
number, a plate or an e-mail address that is VALID by its own check digits could be a real person's or company's, so the
tree may hold none: every identifier written in it has a form that cannot exist (a Spanish tax number has nine characters,
a Spanish IBAN twenty-four; an e-mail lives on a reserved domain). The tests that need valid ones, to try the check
digits, build them in memory (`tests/checks_oracle.py`) and never write them down.

This test reads every text file and every PDF text layer under the repository root (the same tree the export ships) and
what the invoice generator makes, and fails on the first valid identifier, naming the file and line. A new one is either
a real value that must go, or an example that must be rewritten as an invented form.

What no code can judge is the name of a real person or company; those are checked by hand when a sample is written, and
the generator draws its names from invented lists (`bench/corpus/invoice_text.json`).
"""

import re
import tempfile
import unittest
from pathlib import Path

import pypdfium2 as pdfium

from bench.corpus import invoices
from tests import checks_oracle as oracle

STRUCTURE = oracle.STRUCTURE
ROOT = Path(__file__).resolve().parents[1]
SKIP_FOLDERS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "dist", "work", "node_modules", "site"}
SKIP_PREFIXES = ("bench/data/",)
BINARY = {".png", ".jpg", ".avif", ".npy", ".gz", ".model", ".whl", ".onnx", ".zip", ".woff", ".woff2", ".ttf"}

# E-mail domains that cannot reach anyone: RFC 2606 reserved names.
SAFE_DOMAINS = {"example.com", "example.org", "example.net", "example.invalid"}

DNI_LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"
CIF_ORGANISATIONS = "ABCDEFGHJNPQRSUVW"
CIF_LETTERS = "JABCDEFGHI"
# A match must not be part of a longer word or number; a full stop or comma that ends a sentence does not count.
BEFORE = r"(?<![A-Za-z0-9])(?<!\d[.,/-])"
AFTER = r"(?![A-Za-z0-9])(?![.,/-]\d)"
TAX = re.compile(BEFORE + r"([XYZ]?\d{7,8}[A-Z]|[A-Z]\d{7}[0-9A-Z])" + AFTER)
IBAN = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?(?![A-Za-z0-9])")
EMAIL = re.compile(r"[A-Za-z0-9._%+-]+@([A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)")
PHONE = re.compile(
    BEFORE
    + r"(?:\+\d{1,3}[ .-]?)?(?:[6-9]\d{2}[ .-]?\d{2}[ .-]?\d{2}[ .-]?\d{2}|[6-9]\d{2}[ .-]\d{3}[ .-]\d{3})"
    + AFTER
)
CARD = re.compile(BEFORE + r"\d(?:[ -]?\d){12,18}" + AFTER)
IBAN_LIKE = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2}[A-Z0-9]{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?(?![A-Za-z0-9])")
PLATE = re.compile(r"(?<![A-Za-z0-9])\d{4}[ -]?[BCDFGHJKLMNPRSTVWXYZ]{3}(?![A-Za-z0-9])")


def dni_ok(text: str) -> bool:
    return len(text) == 9 and text[:8].isdigit() and DNI_LETTERS[int(text[:8]) % 23] == text[8]


def nie_ok(text: str) -> bool:
    return (
        len(text) == 9
        and text[0] in "XYZ"
        and text[1:8].isdigit()
        and (DNI_LETTERS[int(str("XYZ".index(text[0])) + text[1:8]) % 23] == text[8])
    )


def cif_ok(text: str) -> bool:
    if len(text) != 9 or text[0] not in CIF_ORGANISATIONS or not text[1:8].isdigit():
        return False
    values = [int(c) for c in text[1:8]]
    odd = sum(sum(divmod(2 * v, 10)) for v in values[0::2])
    control = (10 - (odd + sum(values[1::2])) % 10) % 10
    if text[0] in "PQRSNW":
        return text[8] == CIF_LETTERS[control]
    if text[0] in "ABEH":
        return text[8] == str(control)
    return text[8] in (str(control), CIF_LETTERS[control])


def iban_ok(text: str) -> bool:
    """mod 97 holds, the check digits are 02-98 (00, 01 and 99 are never issued) and, for a country of the table, the
    length is that country's. A country outside the table is judged by mod 97 alone, the stricter way."""
    compact = text.replace(" ", "")
    if not 15 <= len(compact) <= 34 or not compact[:2].isalpha() or not compact[2:4].isdigit():
        return False
    if not 2 <= int(compact[2:4]) <= 98:
        return False
    country = compact[:2]
    if country in STRUCTURE and len(compact) != 4 + len(oracle.classes_of(country)):
        return False
    return int("".join(str(int(c, 36)) for c in compact[4:] + compact[:4])) % 97 == 1


def luhn_ok(text: str) -> bool:
    digits = [int(c) for c in text if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(reversed(digits)))
    return total % 10 == 0


def findings(text: str) -> list[tuple[str, str]]:
    """The valid identifiers in `text`, as (kind, value)."""
    found = []
    for match in TAX.finditer(text):
        value = match.group(1)
        for kind, check in (("DNI", dni_ok), ("NIE", nie_ok), ("CIF", cif_ok)):
            if check(value):
                found.append((kind, value))
    for match in IBAN.finditer(text):
        if iban_ok(match.group(0)):
            found.append(("IBAN", match.group(0)))
    for match in EMAIL.finditer(text):
        if match.group(1).lower() not in SAFE_DOMAINS and not match.group(1).lower().endswith(".invalid"):
            found.append(("e-mail", match.group(0)))
    in_iban = [
        match.span() for match in IBAN_LIKE.finditer(text)
    ]  # digits of an IBAN-shaped text are judged as an IBAN

    def inside(match) -> bool:
        return any(start <= match.start() and match.end() <= end for start, end in in_iban)

    for match in PHONE.finditer(text):
        if not inside(match):
            found.append(("phone", match.group(0)))
    for match in CARD.finditer(text):
        if luhn_ok(match.group(0)) and not inside(match):
            found.append(("card", match.group(0)))
    for match in PLATE.finditer(text):
        found.append(("plate", match.group(0)))
    return found


def pdf_text(path: Path) -> str:
    document = pdfium.PdfDocument(str(path))
    try:
        parts = []
        for index in range(len(document)):
            page = document[index]
            parts.append(page.get_textpage().get_text_range())
            page.close()
        return "\n".join(parts)
    finally:
        document.close()


def tree_files() -> list[Path]:
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if not path.is_file() or SKIP_FOLDERS & set(relative.parts) or relative.as_posix().startswith(SKIP_PREFIXES):
            continue
        if path.suffix.lower() not in BINARY:
            files.append(path)
    return files


def scan_file(path: Path) -> list[str]:
    if path.suffix.lower() == ".pdf":
        text = pdf_text(path)
    else:
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return []
    problems = []
    for number, line in enumerate(text.splitlines(), 1):
        for kind, value in findings(line):
            problems.append(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}:{number}: {kind} {value}")
    return problems


class DetectorTest(unittest.TestCase):
    """The detectors themselves, tried on identifiers built here from the published rules (never written out)."""

    def test_valid_identifiers_are_found_and_impossible_forms_are_not(self):
        dni = f"{12345678:08d}{DNI_LETTERS[12345678 % 23]}"
        self.assertEqual(findings(f"NIF: {dni}."), [("DNI", dni)])
        self.assertEqual(findings(f"NIF: {dni[1:]}"), [])  # seven digits and a letter: no such number
        self.assertEqual(findings(f"NIF: {dni[:8]}A"), [])  # a wrong control letter
        nie = "X" + f"{1234567:07d}" + DNI_LETTERS[int("0" + f"{1234567:07d}") % 23]
        self.assertEqual(findings(nie), [("NIE", nie)])
        cif = "A1234567" + "4"
        self.assertEqual(findings(cif), [("CIF", cif)])
        self.assertEqual(findings(cif[:-1]), [])
        compact = "ES00" + "123456789" * 2 + "12"
        check = 98 - int("".join(str(int(c, 36)) for c in compact[4:] + "ES00")) % 97
        iban = f"ES{check:02d}{compact[4:]}"
        self.assertEqual([k for k, _ in findings(iban)], ["IBAN"])
        self.assertEqual(findings("ES00 1111 2222 3333 4444 55"), [])
        self.assertEqual(findings("someone@" + "gmail.com")[0][0], "e-mail")
        self.assertEqual(findings("maria@" + "example.com, a@" + "example.invalid"), [])
        self.assertEqual(findings("Call 6" + "12 34 56 78 today")[0][0], "phone")
        self.assertEqual(findings("total 1.210,00 on 14/03/2026, sha 649126673ab"), [])
        self.assertEqual(findings("plate 1234 " + "BCD")[0][0], "plate")
        card = "4" + "1" * 15  # a number that passes the Luhn sum
        self.assertEqual(findings(f"card {card}")[0][0], "card")


class PublicTreeTest(unittest.TestCase):
    def test_no_valid_identifier_is_written_in_the_repository(self):
        problems = []
        for path in tree_files():
            problems += scan_file(path)
        self.assertEqual(
            problems, [], "valid identifiers in the tree; rewrite each as an invented form that cannot exist"
        )

    def test_no_valid_identifier_is_made_by_the_invoice_generator(self):
        problems = []
        for seed in (1, 7):
            with tempfile.TemporaryDirectory() as folder:
                invoices.generate(Path(folder), seed)
                for path in sorted(Path(folder).rglob("*")):
                    if path.suffix in (".json", ".pdf"):
                        problems += scan_file(path)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main()
