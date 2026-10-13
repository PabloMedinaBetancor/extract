import tempfile
import unittest
from pathlib import Path

import extract

FIXTURES = Path(__file__).resolve().parent / "fixtures"


class SavedDocumentsTest(unittest.TestCase):
    """The three documents the interface develops against, before the guard has a trained model."""

    def load(self, name):
        return extract.load(FIXTURES / f"{name}.extract.json")

    def test_X1_the_document_with_doubtful_words_has_reasons_and_suggestions(self):
        document = self.load("doubtful")
        doubtful = document.doubtful()
        self.assertGreaterEqual(len(doubtful), 6)
        self.assertTrue(all(w.reasons for w in doubtful))
        self.assertGreaterEqual(sum(1 for w in doubtful if w.suggestion), 3)
        self.assertEqual({w.status for w in document.words()}, {"ok", "doubtful"})

    def test_X1_the_invoice_has_a_mismatch_with_a_one_digit_suggestion(self):
        document = self.load("mismatch")
        self.assertEqual(document.doc_type, "invoice")
        mismatches = [n for n in document.numbers if n.status == "mismatch"]
        self.assertGreaterEqual(len(mismatches), 1)
        self.assertTrue(any(n.suggestion for n in mismatches))
        self.assertEqual(document.field_status(document.invoice.total), "mismatch")

    def test_X1_the_document_with_unreadable_pages_is_partial_and_its_words_are_unscored(self):
        document = self.load("unreadable")
        self.assertEqual((document.status, document.unreadable_pages), ("partial", (2,)))
        words = document.pages[1].words
        self.assertTrue(words)
        self.assertEqual({w.status for w in words}, {"unscored"})
        self.assertEqual(words[0].reasons[0].key, "page.reason.few_words")

    def test_X1_every_fixture_comes_with_its_source_for_the_page_images(self):
        for name in ("doubtful", "mismatch", "unreadable"):
            with self.subTest(name=name):
                document = self.load(name)
                self.assertEqual(document.source, FIXTURES / f"{name}.pdf")
                self.assertTrue(document.page_image(1).startswith(b"\x89PNG"))
                word = document.pages[0].words[0]
                self.assertTrue(document.word_crop(word.id).startswith(b"\x89PNG"))

    def test_X1_a_fixture_exports_in_every_format(self):
        paths = self.load("mismatch").export(tempfile.mkdtemp())
        self.assertEqual(len(paths), 4)


if __name__ == "__main__":
    unittest.main()
