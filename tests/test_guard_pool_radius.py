import unittest

from bench.guard import pool_radius
from extract.guard import dictionary


class RadiusTest(unittest.TestCase):
    def test_G15_the_exhaustive_search_finds_a_word_three_edits_away_that_the_index_cannot(self):
        book = dictionary.load("es")
        near = {word for word, _ in pool_radius.within_radius(book, "ordxnxxza")}
        self.assertIn("ordenanza", near)
        self.assertEqual([c for c in book.candidates("ordxnxxza") if book.words[c[0]] == "ordenanza"], [])

    def test_G15_the_texts_are_written_in_the_form_of_the_token_and_capped(self):
        books = [dictionary.load("es")]
        texts = pool_radius.radius_texts(books, "(Ordxnxxza),", 5)
        self.assertLessEqual(len(texts), 5)
        self.assertTrue(all(t.startswith("(") and t.endswith("),") for t in texts))
        self.assertIn("(Ordenanza),", pool_radius.radius_texts(books, "(Ordxnxxza),", 50))
