import dataclasses
import json
import math
import tempfile
import unittest
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

from bench import f_eval
from bench.corpus import invoices
from bench.corpus.pdf_words import read_native_page
from extract import fields
from extract.types import PageReading

SEED = 11
FLOOR = 0.98
GENERIC_PAGES = json.loads((Path(__file__).parent / "data" / "generic_pages.json").read_text(encoding="utf-8"))


def blank_page(path, lines):
    pdf = canvas.Canvas(str(path), pagesize=A4)
    y = A4[1] - 70
    for line in lines:
        pdf.drawString(60, y, line)
        y -= 18
    pdf.showPage()
    pdf.save()
    return read_native_page(path)


def without(page: PageReading, drop) -> PageReading:
    return dataclasses.replace(page, words=tuple(w for w in page.words if not drop(w)))


def rotated(page: PageReading, degrees: float) -> PageReading:
    """The same words with their boxes turned about the page centre, as a skewed scan would give them."""
    angle = math.radians(degrees)
    cx, cy = page.width_px / 2, page.height_px / 2
    words = []
    for w in page.words:
        x, y = (w.box[0] + w.box[2]) / 2 - cx, (w.box[1] + w.box[3]) / 2 - cy
        nx = cx + x * math.cos(angle) - y * math.sin(angle)
        ny = cy + x * math.sin(angle) + y * math.cos(angle)
        half_w, half_h = (w.box[2] - w.box[0]) / 2, (w.box[3] - w.box[1]) / 2
        words.append(dataclasses.replace(w, box=(nx - half_w, ny - half_h, nx + half_w, ny + half_h)))
    return dataclasses.replace(page, words=tuple(words))


class FieldsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        cls.root = Path(cls.dir.name)
        invoices.generate(cls.root, SEED)
        cls.items = [item for split in ("train", "val", "test") for item in f_eval.load_sidecar_items(cls.root, split)]
        cls.generic = cls._generic_items()

    @classmethod
    def tearDownClass(cls):
        cls.dir.cleanup()

    @classmethod
    def _generic_items(cls):
        pages = {name: lines for name, lines in GENERIC_PAGES.items() if name != "title_only"}
        return [f_eval.Item(blank_page(cls.root / f"{name}.pdf", lines), None) for name, lines in pages.items()]

    def test_F1_invoices_and_other_pages_are_told_apart(self):
        result = f_eval.evaluate(self.items + self.generic)["f1"]
        self.assertEqual(result["invoices"], 210)
        self.assertEqual(result["others"], 4)
        self.assertEqual(result["recall"], 1.0)
        self.assertEqual(result["false_positives"], 0)

    def test_F1_an_invoice_title_alone_is_not_enough(self):
        page = blank_page(self.root / "title.pdf", GENERIC_PAGES["title_only"])
        self.assertEqual(fields.document_type([page]), fields.GENERIC)

    def test_F1_totals_without_the_title_are_not_an_invoice(self):
        page = blank_page(self.root / "totals.pdf", ["Resumen", "Subtotal 10,00", "IVA 21 % 2,10", "TOTAL 12,10"])
        self.assertEqual(fields.document_type([page]), fields.GENERIC)

    def test_F1_the_title_must_be_in_the_top_third(self):
        lines = ["Resumen"] + ["relleno"] * 30 + ["FACTURA", "Subtotal 10,00", "IVA 21 % 2,10", "TOTAL 12,10"]
        self.assertEqual(fields.document_type([blank_page(self.root / "low.pdf", lines)]), fields.GENERIC)

    def test_F2_every_field_is_exactly_right_on_each_split(self):
        for split in ("train", "val", "test"):
            result = f_eval.evaluate(f_eval.load_sidecar_items(self.root, split))["f2"]
            for name, share in result.items():
                self.assertGreaterEqual(share, FLOOR, (split, name))

    def test_F2_readings_stored_in_page_records_are_evaluated_like_the_text_layer(self):
        from extract.engine.cache import reading_to_dict

        records = []
        for item in self.items[:6] + self.generic:
            ocr = {"engine": "x", "reading": reading_to_dict(item.page)}
            if item.truth is None:
                records.append(
                    (Path("page.json"), {"set": "train", "doc_id": "generic", "source": "catalog", "ocr": ocr})
                )
            else:
                records.append(
                    (
                        Path("page.json"),
                        {"set": item.truth["split"], "doc_id": item.truth["id"], "source": "invoice", "ocr": ocr},
                    )
                )
        records.append((Path("unread.json"), {"set": "train", "doc_id": "none", "source": "catalog", "ocr": None}))
        items = f_eval.items_from_records(records, self.root)
        self.assertEqual(len(items), 6 + len(self.generic))
        result = f_eval.evaluate(items)
        self.assertEqual(result["f1"]["invoices"], 6)
        self.assertEqual(result["f1"]["recall"], 1.0)
        self.assertEqual(result["f2"]["total"], 1.0)

    def test_F2_a_skewed_page_is_read_as_well_as_a_straight_one(self):
        for degrees in (-2.0, -1.0, 1.0, 2.0):
            skewed = [f_eval.Item(rotated(item.page, degrees), item.truth) for item in self.items[::5]]
            result = f_eval.evaluate(skewed)
            for name, share in result["f2"].items():
                self.assertGreaterEqual(share, FLOOR, (degrees, name))
            self.assertGreaterEqual(result["f3"]["row_count"], FLOOR, degrees)
            self.assertGreaterEqual(result["f3"]["row_amount"], FLOOR, degrees)

    def _retyped(self, item, old, new):
        words = tuple(dataclasses.replace(w, text=new) if w.text == old else w for w in item.page.words)
        return dataclasses.replace(item.page, words=words)

    def test_F2_a_label_with_one_misread_letter_is_still_found(self):
        item = next(i for i in self.items if i.truth["language"] == "es" and i.truth["layout"] == "L3")
        page = self._retyped(item, "imponible", "imponžble")
        self.assertEqual(fields.extract_invoice(page).subtotal.text, item.truth["subtotal"])
        item = next(i for i in self.items if i.truth["language"] == "en")
        page = self._retyped(item, "Invoice", "Invoxce")
        self.assertEqual(fields.extract_invoice(page).date.text.replace(" ", ""), item.truth["date"].replace(" ", ""))

    def test_F2_short_labels_and_other_words_are_not_matched_loosely(self):
        item = next(i for i in self.items if i.truth["language"] == "es")
        page = self._retyped(item, "IVA", "IVO")
        self.assertEqual(fields.extract_invoice(page).vat, ())
        page = self._retyped(item, "Fecha", "Rocha")
        self.assertIsNone(fields.extract_invoice(page).date)

    def test_F2_a_speck_in_front_of_an_amount_does_not_hide_it(self):
        item = next(i for i in self.items if i.truth["withholding"] and i.truth["language"] == "es")
        amount = item.truth["withholding"]["amount"]
        page = self._retyped(item, amount, "." + amount[1:])
        self.assertEqual(fields.extract_invoice(page).withholding.text, "." + amount[1:])

    def test_F2_the_evaluation_counts_a_wrong_field(self):
        item = self.items[0]
        wrong = dict(item.truth, total="0,01")
        result = f_eval.evaluate([f_eval.Item(item.page, wrong)])["f2"]
        self.assertEqual(result["total"], 0.0)
        self.assertEqual(result["number"], 1.0)

    def test_F2_both_label_positions_and_both_languages_are_read(self):
        seen = {(item.truth["layout"], item.truth["language"]) for item in self.items}
        above = {lay.key for lay in invoices.LAYOUTS if lay.label_above}
        left = {lay.key for lay in invoices.LAYOUTS if not lay.label_above}
        self.assertTrue({k for k, _ in seen} & above and {k for k, _ in seen} & left)
        self.assertEqual({lang for _, lang in seen}, {"es", "en"})

    def test_F3_rows_and_their_amounts_are_exactly_right(self):
        result = f_eval.evaluate(self.items)["f3"]
        self.assertGreaterEqual(result["row_count"], FLOOR)
        self.assertGreaterEqual(result["row_amount"], FLOOR)
        self.assertGreaterEqual(result["row_quantity_and_price"], FLOOR)
        self.assertGreaterEqual(result["rows"], 800)

    def test_F3_a_page_with_no_total_gives_none_and_the_rest_is_kept(self):
        item = next(i for i in self.items if i.truth["withholding"] is None)
        total_row = {w.line_id for w in item.page.words if w.text.casefold() == "total"}
        page = without(item.page, lambda w: w.line_id in total_row)
        extracted = fields.extract_invoice(page)
        self.assertIsNone(extracted.total)
        self.assertIsNotNone(extracted.subtotal)
        self.assertIn("total", fields.missing_fields(extracted))

    def test_F3_a_field_that_is_not_there_is_none_never_guessed(self):
        item = self.items[0]
        label = {w.line_id for w in item.page.words if w.text in ("Fecha", "Invoice", "Date") and w.line_id < 6}
        words_of_date = {w.id for w in item.page.words if w.line_id in label or w.text == item.truth["date"]}
        stripped = without(item.page, lambda w: w.id in words_of_date or w.line_id - 1 in label)
        self.assertIsNone(fields.extract_invoice(stripped).date)

    def test_F3_a_row_without_amounts_is_not_a_line(self):
        item = next(i for i in self.items if i.truth["layout"] == "L1")
        row = [w for w in item.page.words if w.text == item.truth["lines"][0]["description"].split()[0]][0].line_id
        page = without(item.page, lambda w: w.line_id == row and any(c.isdigit() for c in w.text))
        self.assertEqual(len(fields.extract_invoice(page).lines), len(item.truth["lines"]) - 1)

    def test_F3_a_row_with_one_loose_number_is_not_a_line(self):
        for layout, columns in (("L1", 3), ("L3", 2)):
            item = next(i for i in self.items if i.truth["layout"] == layout)
            first = item.truth["lines"][0]
            row = [w for w in item.page.words if w.text == first["amount"]][0].line_id
            numbers = [w for w in item.page.words if w.line_id == row and any(c.isdigit() for c in w.text)]
            self.assertEqual(len(numbers), columns)
            lone = {w.id for w in numbers[1:]}
            page = without(item.page, lambda w, lone=lone: w.id in lone)
            self.assertEqual(len(fields.extract_invoice(page).lines), len(item.truth["lines"]) - 1, layout)

    def test_F3_every_extracted_value_keeps_the_ids_of_its_words(self):
        item = self.items[0]
        extracted = fields.extract_invoice(item.page)
        by_id = {w.id: w.text for w in item.page.words}
        for cell in (extracted.number, extracted.date, extracted.subtotal, extracted.total):
            self.assertEqual(" ".join(by_id[i] for i in cell.word_ids), cell.text)
            self.assertEqual(cell.page, 1)
        for line in extracted.lines:
            self.assertEqual(" ".join(by_id[i] for i in line.amount.word_ids), line.amount.text)


if __name__ == "__main__":
    unittest.main()
