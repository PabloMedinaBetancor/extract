import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import extract
from bench.corpus import invoices
from bench.corpus.pdf_words import read_native_page
from extract.api import Reader
from tests.fake_guard import FakeGuard
from tests.support import SANS, SPANISH_LINES, SetLexicon, scanned_pdf, text_page_image

FAKE = "tests.fake_engine:load_engine"
DPI = 150


def page_json(path: Path, source_pdf: Path, change=None, conf: float = 0.97) -> Path:
    """The words of a born-digital invoice, as the fake engine will give them for a scan at 150 dpi."""
    reading = read_native_page(source_pdf)
    lines: dict[int, list] = {}
    scale = DPI / 72
    for word in reading.words:
        text = change(word) if change else word.text
        lines.setdefault(word.line_id, []).append([text, conf, [v * scale for v in word.box]])
    path.write_text(json.dumps(list(lines.values())))
    return path


class InvoicePipelineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp())
        unit = invoices.plan(3)[0]
        cls.invoice = invoices.make_invoice(unit["id"], unit["layout"], unit["font"], 3)
        cls.pdf = cls.tmp / "invoice.pdf"
        invoices.render(cls.invoice, unit["layout"], unit["font"], cls.pdf)
        scan = text_page_image(SPANISH_LINES[:2], SANS, 24, 1240, 1754)
        cls.scan = scanned_pdf(cls.tmp / "scan.pdf", [scan])

    def setUp(self):
        patcher = mock.patch.object(Reader, "_engine_factory", FAKE)
        patcher.start()
        self.addCleanup(patcher.stop)

    def read(self, page_words: Path):
        options = extract.Options(work_dir=self.tmp / f"work-{page_words.stem}", processes=1, threads=1)
        with mock.patch.dict(os.environ, {"EXTRACT_FAKE_PAGE": str(page_words)}):
            reader = Reader(options, guard=FakeGuard(), lexicon=SetLexicon(()))
            self.addCleanup(reader.close)
            return reader.read(self.scan)

    def test_A2_an_invoice_page_gives_its_type_fields_and_checked_numbers(self):
        document = self.read(page_json(self.tmp / "ok.json", self.pdf))
        self.assertEqual(document.doc_type, "invoice")
        self.assertIsNotNone(document.invoice.total)
        statuses = {n.status for n in document.numbers}
        self.assertIn("checked", statuses)
        self.assertNotIn("mismatch", statuses)
        self.assertEqual(document.field_status(document.invoice.total), "checked")

    def test_A3_a_misread_digit_is_a_mismatch_with_a_one_digit_suggestion_and_fixing_it_checks_the_number(self):
        good = self.read(page_json(self.tmp / "good.json", self.pdf))
        total_text = good.invoice.total.text
        total_id = good.invoice.total.word_ids[0]

        wrong = total_text[:-1] + str((int(total_text[-1]) + 3) % 10)

        def misread(word):
            return wrong if word.id == total_id else word.text

        document = self.read(page_json(self.tmp / "bad.json", self.pdf, misread))
        number = next(n for n in document.numbers if total_id in n.word_ids)
        self.assertEqual(number.status, "mismatch")
        self.assertIsNotNone(number.suggestion)
        self.assertEqual(document.field_status(document.invoice.total), "mismatch")
        document.accept(total_id)
        self.assertEqual(next(n for n in document.numbers if total_id in n.word_ids).status, "mismatch")
        document.use_suggestion(total_id)
        fixed = next(n for n in document.numbers if total_id in n.word_ids)
        self.assertEqual((document.word(total_id).text, fixed.status), (total_text, "checked"))
        self.assertEqual(document.invoice.total.text, total_text)

    def test_A3_rule_parameters_are_plain_so_a_saved_document_reads_back_equal(self):
        document = self.read(page_json(self.tmp / "save.json", self.pdf))
        values = [v for n in document.numbers for r in n.rules for v in r.params.values()]
        self.assertTrue(values)
        self.assertTrue(all(isinstance(v, (str, int, float, bool, type(None))) for v in values))
        path = self.tmp / "saved.extract.json"
        document.save(path)
        self.assertEqual(extract.load(path, self.scan), document)

    def test_A2_a_whole_valid_invoice_on_an_unreadable_page_has_no_checked_number(self):
        document = self.read(page_json(self.tmp / "dim.json", self.pdf, conf=0.5))
        self.assertEqual(document.unreadable_pages, (1,))
        self.assertNotIn("checked", {n.status for n in document.numbers})
        self.assertEqual({w.status for w in document.words()}, {"unscored"})
        sound = self.read(page_json(self.tmp / "sound.json", self.pdf))
        self.assertIn("checked", {n.status for n in sound.numbers})

    def test_A3_accepting_a_number_never_checks_it(self):
        document = self.read(page_json(self.tmp / "ok2.json", self.pdf))
        unverified = next((n for n in document.numbers if n.status == "unverified"), None)
        if unverified is not None:
            document.accept(unverified.word_ids[0])
            self.assertEqual(
                next(n for n in document.numbers if n.word_ids == unverified.word_ids).status, "unverified"
            )


if __name__ == "__main__":
    unittest.main()
