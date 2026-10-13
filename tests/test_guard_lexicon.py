import subprocess
import sys
import textwrap
import time
import tracemalloc
import unittest
from unittest import mock

from extract.guard import lexicon
from tests.words import WORDS


def fresh():
    lexicon._SETS.clear()


class LexiconTest(unittest.TestCase):
    def setUp(self):
        fresh()
        self.addCleanup(fresh)

    def test_G1_known_words_are_found_whatever_their_case(self):
        self.assertTrue(lexicon.has("casa"))
        self.assertTrue(lexicon.has("Casa"))
        self.assertTrue(lexicon.has("invoice", ("en",)))
        self.assertTrue(lexicon.has("factura", ("es",)))

    def test_G1_a_word_of_another_language_is_not_found_in_the_list_of_the_first(self):
        self.assertFalse(lexicon.has("invoice", ("es",)))

    def test_G1_accents_count(self):
        self.assertTrue(lexicon.has(WORDS["informacion_acc"], ("es",)))
        self.assertFalse(lexicon.has("informacion", ("es",)))

    def test_G1_noise_is_not_a_word(self):
        self.assertFalse(lexicon.has("xqzvbn"))
        self.assertFalse(lexicon.has("Ie11er"))

    def test_G1_share_is_the_fraction_of_known_words(self):
        self.assertEqual(lexicon.share(["la", "casa", "xqzvbn", "kwjzpq"]), 0.5)
        self.assertEqual(lexicon.share([]), 0.0)

    def test_G1_a_language_without_a_list_is_ignored_but_none_at_all_is_an_error(self):
        self.assertTrue(lexicon.has("casa", ("fr", "es")))
        self.assertEqual(lexicon.available(("fr", "es", "en")), ("es", "en"))
        with self.assertRaises(lexicon.NoLexicon):
            lexicon.has("casa", ("fr",))
        with self.assertRaises(lexicon.NoLexicon):
            lexicon.share(["casa"], ("fr",))

    def test_G10_both_lists_load_in_half_a_second(self):
        started = time.perf_counter()
        lexicon.has("casa")
        lexicon.has("invoice")
        self.assertLess(time.perf_counter() - started, 0.5)
        self.assertGreater(sum(len(s) for s in lexicon._SETS.values()), 150_000)

    def test_G10_both_lists_take_less_than_30_mb(self):
        tracemalloc.start()
        lexicon.has("casa")
        lexicon.has("invoice")
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        self.assertLess(peak / 1e6, 30)

    def test_G10_importing_the_lexicon_pulls_in_nothing_heavy(self):
        code = textwrap.dedent(
            """
            import sys
            import extract.guard.lexicon
            heavy = [m for m in ("numpy", "lightgbm", "rapidfuzz", "scipy") if m in sys.modules]
            print(heavy)
            """
        )
        run = subprocess.run([sys.executable, "-s", "-c", code], capture_output=True, text=True, check=True)
        self.assertEqual(run.stdout.strip(), "[]")

    def test_G1_a_missing_data_file_is_not_hidden(self):
        with mock.patch.object(lexicon, "DATA_DIR", lexicon.DATA_DIR / "nowhere"):
            with self.assertRaises(lexicon.NoLexicon):
                lexicon.has("casa")


if __name__ == "__main__":
    unittest.main()
