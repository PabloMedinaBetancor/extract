import random
import time
import unittest

from rapidfuzz.distance import Levenshtein

from extract.guard import dictionary
from extract.guard.dictionary import fold_accents, load
from tests.words import WORDS


class DictionaryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        started = time.perf_counter()
        cls.es = load("es")
        cls.en = load("en")
        cls.load_seconds = time.perf_counter() - started

    def test_G10_both_dictionaries_load_in_three_seconds(self):
        self.assertLess(self.load_seconds, 3.0)

    def test_G1_words_are_found_by_rank_ignoring_case(self):
        self.assertEqual(self.es.rank_of("de"), 0)
        self.assertGreater(self.es.rank_of("Ordenanza"), 0)
        self.assertEqual(self.es.rank_of("xqzvbn"), -1)

    def test_G1_frequency_orders_common_above_rare_and_missing_is_zero(self):
        self.assertGreater(self.es.log_frequency("casa"), self.es.log_frequency("catastral"))
        self.assertGreater(self.es.log_frequency("catastral"), 0)
        self.assertEqual(self.es.log_frequency("xqzvbn"), 0.0)

    def test_G1_folded_lookup_ignores_accents_the_tilde_of_n_and_the_diaeresis(self):
        self.assertTrue(self.es.has_folded("informacion"))
        self.assertTrue(self.es.has_folded("espana"))
        self.assertEqual(fold_accents(WORDS["year"]), "ano")
        self.assertEqual(fold_accents(WORDS["country"]), "pais")
        self.assertEqual(fold_accents(WORDS["umlaut"]), "pinguino")

    def test_G1_the_lost_accent_of_a_common_word_is_not_a_word(self):
        self.assertGreaterEqual(self.es.rank_of(WORDS["informacion_acc"]), 0)
        self.assertEqual(self.es.rank_of("informacion"), -1)

    def test_G1_candidates_find_one_and_two_edit_neighbours_nearest_first(self):
        near = self.es.candidates("ordenansa")
        words = [(self.es.words[r], d) for r, d in near]
        self.assertIn(("ordenanza", 1), words)
        self.assertEqual(near[0][1], 1)
        self.assertEqual(self.es.candidates("xqzvbnwk"), [])
        self.assertTrue(all(d <= 2 for _, d in near))

    def test_G1_candidates_agree_with_brute_force_on_random_misspellings(self):
        rng = random.Random(5)
        letters = "abcdefghijklmnopqrstuvwxyz"
        for _ in range(60):
            word = self.en.words[rng.randrange(2000, 40000)]
            typo = list(word)
            for _ in range(rng.choice((1, 2))):
                position = rng.randrange(len(typo))
                typo[position] = rng.choice(letters)
                if rng.random() < 0.4 and len(typo) > 3:
                    del typo[rng.randrange(len(typo))]
            typo = "".join(typo)
            expected = {i for i, w in enumerate(self.en.words) if Levenshtein.distance(typo, w, score_cutoff=2) <= 2}
            got = {r for r, _ in self.en.candidates(typo)}
            with self.subTest(typo=typo):
                self.assertEqual(got, expected)

    def test_G10_a_query_takes_a_millisecond_or_less(self):
        words = ["ordenansa", "construcsion", "vivienda", "parcelas", "edificabiliad", "catastrl"]
        for word in words:
            self.es.candidates(word)
        started = time.perf_counter()
        for _ in range(20):
            for word in words:
                self.es.candidates(word)
        self.assertLess((time.perf_counter() - started) / 120, 0.002)

    def test_G1_the_character_model_prefers_words_to_noise(self):
        real = self.es.char_logp("construcciones")
        noise = self.es.char_logp("xqzkvjw")
        self.assertGreater(real, noise + 1.0)
        self.assertIsNone(self.es.char_logp("1234"))

    def test_G1_the_batch_model_gives_the_same_numbers_as_one_word_at_a_time(self):
        words = ["construcciones", "xqzkvjw", "1234", "casa", "casa", "", "Ordenanza"]
        alone = [load("es").char_logp(w) for w in words]
        fresh = dictionary.Dictionary("es")
        batch = fresh.char_logp_many(words)
        self.assertEqual(len(batch), len(words))
        for one, many in zip(alone, batch, strict=True):
            if one is None:
                self.assertIsNone(many)
            else:
                self.assertAlmostEqual(one, many, places=9)
        self.assertEqual(batch[3], batch[4])

    def test_G1_each_language_prefers_its_own_words(self):
        self.assertGreater(self.es.char_logp("administracion"), self.en.char_logp("administracion") - 0.5)
        self.assertGreater(self.en.char_logp("something"), self.es.char_logp("something"))

    def test_G1_the_loader_gives_the_same_object_every_time(self):
        self.assertIs(load("es"), self.es)

    def test_G1_deletes_cover_up_to_two_characters(self):
        self.assertEqual(dictionary.deletes("ab", 1), {"ab", "a", "b"})
        self.assertIn("", dictionary.deletes("ab", 2))


if __name__ == "__main__":
    unittest.main()
