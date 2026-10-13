import json
import re
import tempfile
import unittest
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pypdfium2 as pdfium

from bench.corpus import invoices
from extract import checks

SEED = 7
AMOUNT = re.compile(r"(?<![\d.,])-?\d{1,3}(?:[.,]\d{3})*[.,]\d{2}(?![\d.,])")


def parse(text, lang):
    plain = text.replace(".", "").replace(",", ".") if lang == "es" else text.replace(",", "")
    return Decimal(plain)


def page_text(pdf_path):
    document = pdfium.PdfDocument(str(pdf_path))
    page = document[0]
    text = page.get_textpage().get_text_range()
    page.close()
    document.close()
    return " ".join(text.split())


def own_cif_control(digits):
    values = [int(c) for c in digits]
    odd = sum(v * 2 // 10 + v * 2 % 10 for v in values[0::2])
    total = odd + sum(values[1::2])
    control = (10 - total % 10) % 10
    return str(control), "JABCDEFGHI"[control]


class InvoicesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.out = Path(cls.dir.name)
        invoices.generate(cls.out, SEED)
        cls.sidecars = [json.loads(p.read_text(encoding="utf-8")) for p in sorted(cls.out.glob("*/*.json"))]

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    def test_F4_six_layouts_and_eight_fonts_in_both_languages(self):
        self.assertEqual(len({s["layout"] for s in self.sidecars}), 6)
        self.assertEqual(len({s["font"] for s in self.sidecars}), 8)
        self.assertEqual({s["language"] for s in self.sidecars}, {"es", "en"})

    def test_F4_test_split_holds_only_the_two_held_out_layouts(self):
        held_out = {lay.key for lay in invoices.LAYOUTS if lay.held_out}
        self.assertEqual(len(held_out), 2)
        self.assertEqual({s["layout"] for s in self.sidecars if s["split"] == "test"}, held_out)
        self.assertFalse({s["layout"] for s in self.sidecars if s["split"] != "test"} & held_out)

    def test_F4_test_split_has_two_fonts_that_training_never_sees(self):
        seen = {s["font"] for s in self.sidecars if s["split"] != "test"}
        unseen_in_test = {s["font"] for s in self.sidecars if s["split"] == "test"} - seen
        self.assertGreaterEqual(len(unseen_in_test), 2)

    def test_F4_split_sizes_follow_the_plan(self):
        counts = Counter(s["split"] for s in self.sidecars)
        self.assertEqual(counts["test"], 60)
        self.assertTrue(100 <= counts["train"] <= 135)
        self.assertTrue(20 <= counts["val"] <= 45)

    def test_F4_every_font_file_exists_and_has_a_licence(self):
        for font in invoices.FONTS:
            self.assertTrue((invoices.FONT_DIR / font.regular).is_file())
            self.assertTrue((invoices.FONT_DIR / font.bold).is_file())
            self.assertTrue(font.licence)

    def test_F4_text_layer_contains_every_sidecar_value(self):
        for data in self.sidecars:
            text = page_text(self.out / data["split"] / f"{data['id']}.pdf")
            for entry in data["numbers"]:
                self.assertIn(entry["text"], text, (data["id"], entry))
            self.assertIn(data["number"], text)
            self.assertIn(data["date"], text)
            for line in data["lines"]:
                self.assertIn(line["description"], text, (data["id"], line))

    def test_F4_text_layer_has_no_amount_the_sidecar_does_not_know(self):
        for data in self.sidecars:
            text = page_text(self.out / data["split"] / f"{data['id']}.pdf")
            known = {e["text"] for e in data["numbers"]} | {v["base"] for v in data["vat"]}
            found = set(AMOUNT.findall(text))
            self.assertLessEqual(found, known, (data["id"], found - known))

    def test_truth_arithmetic_is_exact(self):
        for data in self.sidecars:
            lang = data["language"]
            lines = data["lines"]
            for line in lines:
                if line["quantity"] is None:
                    continue
                expected = (parse(line["quantity"], lang) * parse(line["unit_price"], lang)).quantize(
                    Decimal("0.01"), rounding="ROUND_HALF_UP"
                )
                self.assertEqual(expected, parse(line["amount"], lang))
            subtotal = sum(parse(line["amount"], lang) for line in lines)
            self.assertEqual(subtotal, parse(data["subtotal"], lang))
            vat = sum(parse(v["amount"], lang) for v in data["vat"])
            withheld = parse(data["withholding"]["amount"], lang) if data["withholding"] else Decimal(0)
            self.assertEqual(parse(data["total"], lang), subtotal + vat + withheld)

    def test_own_cif_check_agrees_with_the_golden_cases(self):
        # worked by hand: 1234567 sums to 26 (control 4, letter D); seven zeros, control 0 (J); seven nines sum to 63 (7, G)
        self.assertEqual(own_cif_control("1234567"), ("4", "D"))
        self.assertEqual(own_cif_control("0000000"), ("0", "J"))
        self.assertEqual(own_cif_control("9999999"), ("7", "G"))
        self.assertNotEqual(own_cif_control("1234567")[0], "5")

    def test_truth_identifiers_are_invented_and_cannot_exist(self):
        """No identifier in the public data may belong to anyone. A Spanish tax number has nine
        characters and a Spanish IBAN twenty-four, and the generated ones are a character short, so none can exist."""
        for data in self.sidecars:
            iban = data["iban"].replace(" ", "")
            self.assertEqual((iban[:4], len(iban)), ("ESXX", 22), data["id"])
            for party in (data["supplier"], data["customer"]):
                tax_id = party["tax_id"]
                self.assertEqual(len(tax_id), 8, tax_id)
                if party["tax_id_kind"] == "cif":
                    self.assertTrue(tax_id[0].isalpha() and tax_id[1:].isdigit(), tax_id)
                if party["tax_id_kind"] == "dni":
                    self.assertTrue(tax_id[:7].isdigit() and tax_id[7].isalpha(), tax_id)
                self.assertFalse(
                    any(check(tax_id) for check in (checks.check_dni, checks.check_nie, checks.check_cif)), tax_id
                )
            self.assertFalse(checks.check_iban(iban))

    def test_the_names_and_streets_come_from_invented_lists(self):
        for word in invoices.COMPANIES + invoices.STREETS + invoices.CITIES:
            self.assertNotIn(word, ("Madrid", "Valencia", "Sevilla", "Calle Mayor", "Camino Real", "Altamira"))
        self.assertIn("invented", invoices.INVENTED_NOTE.lower())

    def test_truth_marks_the_role_of_every_number(self):
        roles = {e["role"] for data in self.sidecars for e in data["numbers"]}
        self.assertLessEqual(
            {"number", "date", "line_amount", "subtotal", "vat", "total", "withholding", "iban"}, roles
        )

    def test_generation_is_deterministic(self):
        a = invoices.make_invoice("inv-x", invoices.LAYOUTS[0], invoices.FONTS[0], 3)
        b = invoices.make_invoice("inv-x", invoices.LAYOUTS[0], invoices.FONTS[0], 3)
        self.assertEqual(a, b)
        self.assertNotEqual(a, invoices.make_invoice("inv-x", invoices.LAYOUTS[0], invoices.FONTS[0], 4))


if __name__ == "__main__":
    unittest.main()
