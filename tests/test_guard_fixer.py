import math
import unittest

import numpy as np

from extract.guard import fixer
from extract.guard.fixer import Candidate, apply_case, generate, split_word, weighted_distance
from tests.words import WORDS

LANGUAGES = ("es", "en")


class SplitTest(unittest.TestCase):
    def test_G8_plain_words_keep_their_attached_punctuation(self):
        self.assertEqual(split_word("(obra),"), ("(", "obra", "),"))
        self.assertEqual(split_word("obra"), ("", "obra", ""))

    def test_G8_digits_hyphens_apostrophes_and_symbols_are_not_plain_words(self):
        for text in ("a1b", "casa-obra", "don't", "1234", "---", "", "a b"):
            with self.subTest(text=text):
                self.assertIsNone(split_word(text))


class WeightedDistanceTest(unittest.TestCase):
    def test_G8_ocr_swaps_cost_less_than_other_edits(self):
        self.assertLess(weighted_distance("modelo", "rnodelo"), weighted_distance("modelo", "podelo"))
        self.assertEqual(weighted_distance("modelo", "rnodelo"), 0.25)
        self.assertLess(weighted_distance("casa", "easa"), 1.0)
        self.assertEqual(weighted_distance("casa", "caxa"), 1.0)

    def test_G8_case_and_accents_do_not_count(self):
        self.assertEqual(weighted_distance("Informacion", WORDS["informacion_acc"]), 0.0)

    def test_G8_insertions_and_deletions_cost_one_each(self):
        self.assertEqual(weighted_distance("casa", "casas"), 1.0)
        self.assertEqual(weighted_distance("casas", "ca"), 3.0)


class CaseTest(unittest.TestCase):
    def test_G8_a_candidate_takes_the_case_pattern_of_the_word(self):
        self.assertEqual(apply_case("obra", "Obrx"), "Obra")
        self.assertEqual(apply_case("obra", "OBRX"), "OBRA")
        self.assertEqual(apply_case("obra", "obrx"), "obra")
        self.assertEqual(apply_case("obra", "oBrx"), "obra")
        self.assertEqual(apply_case("a", "A"), "A")


class GenerateTest(unittest.TestCase):
    def test_G8_the_truth_is_among_the_candidates_of_a_one_edit_misreading(self):
        texts = [c.text for c in generate("ordenansa", LANGUAGES)]
        self.assertIn("ordenanza", texts)
        self.assertLessEqual(len(texts), fixer.MAX_CANDIDATES + 3)

    def test_G8_candidates_keep_punctuation_and_case(self):
        texts = [c.text for c in generate("(Ordenansa),", LANGUAGES)]
        self.assertIn("(Ordenanza),", texts)

    def test_G8_a_lost_accent_is_a_candidate(self):
        texts = [c.text for c in generate("informacion", LANGUAGES)]
        self.assertIn(WORDS["informacion_acc"], texts)

    def test_G8_the_other_casings_of_the_word_are_candidates(self):
        texts = [c.text for c in generate("plan", LANGUAGES)]
        self.assertIn("Plan", texts)
        self.assertIn("PLAN", texts)

    def test_G7_tokens_with_digits_get_no_candidates(self):
        self.assertEqual(generate("1.234,56", LANGUAGES), [])
        self.assertEqual(generate("A12b", LANGUAGES), [])

    def test_G8_no_candidate_repeats_and_the_word_itself_is_not_proposed(self):
        candidates = generate("casa", LANGUAGES)
        texts = [c.text for c in candidates]
        self.assertEqual(len(texts), len(set(texts)))
        self.assertNotIn("casa", texts)

    def test_G8_a_language_without_a_list_gives_no_candidates_from_it(self):
        self.assertEqual([c for c in generate("ordenansa", ("fr",)) if c.log_frequency > 0], [])


class FeatureTest(unittest.TestCase):
    def test_G8_every_candidate_gets_a_row_of_the_declared_features(self):
        candidates = generate("ordenansa", LANGUAGES)
        rows = fixer.features(
            "ordenansa", (0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.4, 0.9), 0.8, 0.0, candidates, LANGUAGES
        )
        self.assertEqual(rows.shape, (len(candidates), len(fixer.FEATURES)))
        names = {n: i for i, n in enumerate(fixer.FEATURES)}
        best = next(i for i, c in enumerate(candidates) if c.text == "ordenanza")
        self.assertEqual(rows[best, names["distance"]], 1.0)
        self.assertAlmostEqual(rows[best, names["lowest_changed_confidence"]], 0.4, places=5)
        self.assertEqual(rows[best, names["candidates"]], len(candidates))
        self.assertGreater(rows[best, names["lm_gain"]], 0.0)

    def test_G8_a_case_only_candidate_is_marked_and_costs_nothing(self):
        candidates = generate("plan", LANGUAGES)
        rows = fixer.features("plan", None, None, 1.0, candidates, LANGUAGES)
        names = {n: i for i, n in enumerate(fixer.FEATURES)}
        title = next(i for i, c in enumerate(candidates) if c.text == "Plan")
        self.assertEqual((rows[title, names["case_equal"]], rows[title, names["weighted_distance"]]), (1.0, 0.0))
        self.assertTrue(math.isnan(rows[title, names["lowest_changed_confidence"]]))

    def test_G8_the_most_frequent_candidate_has_frequency_rank_zero(self):
        candidates = generate("ordenansa", LANGUAGES)
        rows = fixer.features("ordenansa", None, None, 0.0, candidates, LANGUAGES)
        rank = rows[:, fixer.FEATURES.index("frequency_rank")]
        self.assertEqual(sorted(rank.tolist()), list(range(len(candidates))))
        top = int(np.argmin(rank))
        self.assertEqual(candidates[top].log_frequency, max(c.log_frequency for c in candidates))

    def test_G8_the_feature_fingerprint_changes_with_the_list(self):
        base = fixer.feature_fingerprint()
        self.assertRegex(base, "^[0-9a-f]{64}$")
        old = fixer.FEATURES
        try:
            fixer.FEATURES = old[:-1]
            self.assertNotEqual(fixer.feature_fingerprint(), base)
        finally:
            fixer.FEATURES = old


class CandidateTypeTest(unittest.TestCase):
    def test_G8_candidates_are_immutable_values(self):
        candidate = Candidate("obra", "obra", 0, 1.0)
        with self.assertRaises(AttributeError):
            candidate.text = "another"


if __name__ == "__main__":
    unittest.main()
