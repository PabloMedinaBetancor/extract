"""The guard and the engine agree on what a second reading is: the readings that `second_pass` really makes (a group of
the word's own id for a word read alone) reach the candidates and the pilot's counts."""

import unittest

from bench.guard import pilot_cells
from extract.engine.second import second_pass
from extract.guard import candidates
from tests.test_engine_second import CONFIG, FixedReaders, doubtful_line, page_of

LANGUAGES = ("es", "en")


class RealSecondPassTest(unittest.TestCase):
    def pass2(self, texts, read):
        reading, crops, _ = page_of([doubtful_line(texts)] * 8)
        readers = FixedReaders(("latin",), {"latin": [read] * 8})
        return second_pass(reading, crops, readers, "line", CONFIG)

    def test_a_reader_that_reads_a_word_differently_gives_a_text_candidate_through_the_real_pass_2(self):
        page = self.pass2(["Ccntro", "de", "Salud", "publica"], "Centro de Salud publica")
        word = page.words[0]
        self.assertTrue(word.readings)
        self.assertEqual(word.readings[0].group, (word.id,))  # the aligner's convention: the word itself
        ctx = candidates.PageContext(page.words, LANGUAGES)
        self.assertEqual(candidates.by_source(ctx, 0)["reading:latin"], ["Centro"])
        self.assertIn("Centro", [c.text for c in candidates.generate(ctx, 0)])

    def test_the_pilot_counts_that_reading_as_a_reading_of_the_word(self):
        page = self.pass2(["Ccntro", "de", "Salud", "publica"], "Centro de Salud publica")
        built = self.labelled(page, ["Centro", "de", "Salud", "publica"] * 8)
        harm = pilot_cells.harm_counts([built])
        self.assertEqual((harm["fixed"], harm["harmed"]), (8, 0))  # the same line eight times on the page
        self.assertGreaterEqual(harm["right_read"], 1)

    def test_a_reader_that_cuts_a_word_or_joins_two_gives_structure_evidence_not_a_text(self):
        cut = self.pass2(["Ccntro", "de", "Salud", "publica"], "Cen tro de Salud publica")
        ctx = candidates.PageContext(cut.words, LANGUAGES)
        self.assertFalse([s for s in candidates.by_source(ctx, 0) if s.startswith("reading:")])
        harm = pilot_cells.harm_counts([self.labelled(cut, ["Centro", "de", "Salud", "publica"] * 8)])
        self.assertEqual((harm["fixed"], harm["changed"]), (0, 0))  # a cut is not a reading of the word
        joined = self.pass2(["Cen", "tro", "de", "Salud"], "Centro de Salud")
        ctx = candidates.PageContext(joined.words, LANGUAGES)
        self.assertFalse([s for s in candidates.by_source(ctx, 0) if s.startswith("reading:")])
        self.assertTrue(candidates.joins_words(joined.words[0].readings[0]))

    def labelled(self, page, truth):
        from tests.guard_pages import labelled_page

        built = labelled_page([w.text for w in page.words], [(t, [i], None) for i, t in enumerate(truth)])
        import dataclasses

        return dataclasses.replace(built, words=page.words)


if __name__ == "__main__":
    unittest.main()
