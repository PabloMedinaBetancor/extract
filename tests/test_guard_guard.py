import tempfile
import unittest
import unittest.mock
from pathlib import Path
from unittest import mock

import numpy as np

from bench.guard import fixer_train, train
from extract.errors import ModelMismatch
from extract.guard import MAX_WORDS, Guard
from extract.types import PageReading, Word
from tests.test_guard_fixer_train import simulated_rows
from tests.test_guard_train import synthetic


def page(texts, conf=0.95, number=1):
    words = tuple(
        Word(
            f"p{number}w{i}",
            number,
            t,
            (i * 60.0, 0.0, i * 60.0 + 8.0 * len(t), 12.0),
            conf,
            tuple([conf] * len(t)),
            0,
            i,
        )
        for i, t in enumerate(texts)
    )
    return PageReading(number, 800, 600, 400.0, 300.0, 144, words)


class GuardTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model, _, _ = train.fit_model(synthetic(60, 1), synthetic(30, 2), {"leaves": 7, "trees": 100})
        cls.guard = Guard(model)

    def test_G7_every_word_gets_a_verdict_and_no_suggestion_or_new_text(self):
        reading = page(["obra", "plano", "casa"])
        scored = self.guard.score_page(reading, ("es", "en"), "es", 0.05)
        self.assertEqual([s.id for s in scored], [w.id for w in reading.words])
        for verdict in scored:
            self.assertIn(verdict.status, ("ok", "doubtful"))
            self.assertTrue(0.0 <= verdict.p_wrong <= 1.0)
            self.assertIsNone(verdict.suggestion)
        self.assertEqual([w.text for w in reading.words], ["obra", "plano", "casa"])

    def test_G6_low_confidence_words_are_the_doubtful_ones(self):
        confident = self.guard.score_page(page(["obra"] * 20, conf=0.99), ("es",), "es", 0.05)
        shaky = self.guard.score_page(page(["obra"] * 20, conf=0.30), ("es",), "es", 0.05)
        self.assertLess(sum(s.status == "doubtful" for s in confident), sum(s.status == "doubtful" for s in shaky))
        self.assertGreater(min(s.p_wrong for s in shaky), max(s.p_wrong for s in confident))

    def test_G6_a_bigger_budget_never_flags_fewer_words(self):
        reading = page(["obra"] * 10 + ["x1"] * 10, conf=0.6)
        counts = [
            sum(s.status == "doubtful" for s in self.guard.score_page(reading, ("es",), "es", b))
            for b in (0.03, 0.05, 0.08)
        ]
        self.assertEqual(counts, sorted(counts))

    def test_G13_an_empty_page_gives_no_verdicts(self):
        self.assertEqual(self.guard.score_page(page([]), ("es",), "es", 0.05), [])

    def test_G13_a_page_over_5000_words_is_unscored_with_a_warning_and_never_ok(self):
        self.assertEqual(MAX_WORDS, 5000)
        scored = self.guard.score_page(page(["a"] * (MAX_WORDS + 1)), ("es",), "es", 0.05)
        self.assertEqual({s.status for s in scored}, {"unscored"})
        self.assertIsNone(scored[0].p_wrong)
        warning = self.guard.last_warnings[0]
        self.assertEqual(
            (warning["key"], warning["cause"], warning["words"]),
            ("guard.warning.unscored", "too_many_words", MAX_WORDS + 1),
        )

    def test_G13_all_nan_confidences_are_unscored_with_a_warning(self):
        words = tuple(Word(f"p1w{i}", 1, "obra", (i * 60.0, 0.0, i * 60.0 + 32, 12.0)) for i in range(5))
        reading = PageReading(1, 800, 600, 400.0, 300.0, 144, words)
        scored = self.guard.score_page(reading, ("es",), "es", 0.05)
        self.assertEqual({s.status for s in scored}, {"unscored"})
        self.assertEqual(self.guard.last_warnings[0]["cause"], "no_confidence")

    def test_G13_an_unknown_language_still_scores_and_a_good_page_clears_the_warnings(self):
        scored = self.guard.score_page(page(["obra", "casa"]), ("fr",), "es", 0.05)
        self.assertEqual({s.status for s in scored} <= {"ok", "doubtful"}, True)
        self.assertEqual(self.guard.last_warnings, [])

    def test_G13_one_word_is_scored_like_any_other_page(self):
        scored = self.guard.score_page(page(["obra"]), ("es",), "es", 0.05)
        self.assertEqual(len(scored), 1)
        self.assertIn(scored[0].status, ("ok", "doubtful"))

    def test_G13_five_thousand_words_are_scored_and_one_more_is_not(self):
        scored = self.guard.score_page(page(["obra"] * MAX_WORDS), ("es",), "es", 0.05)
        self.assertEqual({s.status for s in scored} <= {"ok", "doubtful"}, True)
        self.assertEqual(self.guard.last_warnings, [])
        scored = self.guard.score_page(page(["obra"] * (MAX_WORDS + 1)), ("es",), "es", 0.05)
        self.assertEqual({s.status for s in scored}, {"unscored"})
        self.assertEqual(self.guard.last_warnings[0]["cause"], "too_many_words")

    def test_G6_a_token_with_digits_is_doubtful_from_a_third_of_the_word_threshold_and_a_word_is_not(self):
        probability = np.array([0.2, 0.2])
        with (
            mock.patch.object(self.guard.model, "probability", return_value=probability),
            mock.patch.object(self.guard.model, "thresholds", return_value=(0.3, 0.1)),
        ):
            scored = self.guard.score_page(page(["obra", "a12"]), ("es",), "es", 0.05)
        self.assertEqual([s.status for s in scored], ["ok", "doubtful"])

    def test_G6_p_wrong_is_the_calibration_table_of_the_model_applied_to_the_raw_score(self):
        raw = np.array([-4.0, 1.5])
        table = self.guard.model.meta["calibration"]
        with mock.patch.object(self.guard.model, "raw", return_value=raw):
            scored = self.guard.score_page(page(["obra", "casa"]), ("es",), "es", 0.05)
        expected = np.interp(raw, table["scores"], table["probs"])
        self.assertEqual([s.p_wrong for s in scored], expected.tolist())

    def test_G13_an_internal_error_leaves_the_words_unscored_with_a_warning_and_never_ok(self):
        broken = unittest.mock.patch("extract.guard.signals.extract", side_effect=RuntimeError("boom"))
        with broken, self.assertLogs("extract.guard.checker", level="ERROR"):
            scored = self.guard.score_page(page(["obra", "casa"]), ("es",), "es", 0.05)
        self.assertEqual({s.status for s in scored}, {"unscored"})
        self.assertEqual(self.guard.last_warnings[0]["cause"], "error")

    def test_G13_a_budget_the_model_does_not_know_is_a_caller_error_not_an_unscored_page(self):
        with self.assertRaises(ValueError):
            self.guard.score_page(page(["obra"]), ("es",), "es", 0.5)

    def test_G6_an_automatic_locale_is_decided_from_the_numbers_of_the_page(self):
        reading = page(["1.234,56", "2.500,00", "obra"])
        auto = self.guard.score_page(reading, ("es",), "auto", 0.05)
        spanish = self.guard.score_page(reading, ("es",), "es", 0.05)
        self.assertEqual([s.p_wrong for s in auto], [s.p_wrong for s in spanish])

    def test_G6_the_same_page_scores_the_same_every_time(self):
        reading = page(["obra", "plano", "1.234,56", "xqzvbn"], conf=0.7)
        first = self.guard.score_page(reading, ("es", "en"), "es", 0.05)
        again = self.guard.score_page(reading, ("es", "en"), "es", 0.05)
        self.assertEqual(first, again)

    def test_G1_info_names_the_model_and_its_training(self):
        info = self.guard.info
        self.assertEqual(len(info["model_id"]), 12)
        self.assertEqual(info["training"]["validation_words"], 3600)
        self.assertEqual(set(info["thresholds"]), {"0.03", "0.05", "0.08"})

    def test_G1_load_reads_a_saved_model_and_refuses_a_missing_one(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "guard.model"
            self.guard.model.save(path)
            again = Guard.load(path)
            self.assertEqual(again.info["model_id"], self.guard.info["model_id"])
            with self.assertRaises(ModelMismatch):
                Guard.load(Path(folder) / "none.model")

    def test_G7_the_guard_scores_are_in_the_order_and_ids_of_the_page_even_with_repeated_text(self):
        scored = self.guard.score_page(page(["a", "a", "a"]), ("es",), "es", 0.05)
        self.assertEqual([s.id for s in scored], ["p1w0", "p1w1", "p1w2"])
        self.assertTrue(np.isfinite([s.p_wrong for s in scored]).all())


class SuggestionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model, _, _ = train.fit_model(synthetic(60, 1), synthetic(30, 2), {"leaves": 7, "trees": 100})
        rows = simulated_rows(600, 1)
        queries = fixer_train.build_queries(rows, range(len(rows.y)))
        model.ranker = fixer_train.fit_ranker(queries)
        _, scores = fixer_train.top_choices(model.ranker, queries)
        correct = np.array(queries.truth_at) == fixer_train.top_choices(model.ranker, queries)[0]
        model.meta["fixer"] = {
            "calibration": fixer_train.calibrate_top_choice(scores, correct),
            "t_suggest": 0.0,
            "feature_fingerprint": "test",
        }
        cls.guard = Guard(model)

    def shaky_page(self, texts):
        words = tuple(
            Word(f"p1w{i}", 1, t, (i * 90.0, 0.0, i * 90.0 + 8.0 * len(t), 12.0), 0.35, tuple([0.35] * len(t)), 0, i)
            for i, t in enumerate(texts)
        )
        return PageReading(1, 800, 600, 400.0, 300.0, 144, words)

    def test_G8_a_doubtful_misspelling_gets_a_suggestion_with_its_confidence(self):
        scored = self.guard.score_page(self.shaky_page(["ordenansa", "construcsion"]), ("es",), "es", 0.08)
        shown = [s for s in scored if s.suggestion]
        self.assertTrue(shown)
        for verdict in shown:
            self.assertEqual(verdict.status, "doubtful")
            self.assertTrue(0.0 <= verdict.suggestion_conf <= 1.0)
        self.assertIn("ordenanza", [s.suggestion for s in scored])

    def test_G7_a_token_with_digits_never_gets_a_suggestion_from_the_guard(self):
        scored = self.guard.score_page(
            self.shaky_page(["1.234,56", "A12b", "ordenansa", "0rdenansa"]), ("es",), "es", 0.08
        )
        self.assertIsNone(scored[0].suggestion)
        self.assertIsNone(scored[1].suggestion)
        self.assertIsNone(scored[3].suggestion)

    def test_G7_the_guard_never_changes_the_text_it_was_given(self):
        page = self.shaky_page(["ordenansa", "construcsion"])
        self.guard.score_page(page, ("es",), "es", 0.08)
        self.assertEqual([w.text for w in page.words], ["ordenansa", "construcsion"])

    def test_G8_a_suggestion_below_the_threshold_is_not_shown(self):
        self.guard.model.meta["fixer"]["t_suggest"] = 1.01
        try:
            scored = self.guard.score_page(self.shaky_page(["ordenansa"]), ("es",), "es", 0.08)
        finally:
            self.guard.model.meta["fixer"]["t_suggest"] = 0.0
        self.assertIsNone(scored[0].suggestion)

    def test_G9_doubtful_words_carry_at_most_three_reasons_and_ok_words_none(self):
        scored = self.guard.score_page(self.shaky_page(["ordenansa", "xqzvbn"]), ("es",), "es", 0.08)
        for verdict in scored:
            self.assertLessEqual(len(verdict.reasons), 3)
            if verdict.status == "ok":
                self.assertEqual(verdict.reasons, ())
        self.assertTrue(any(s.reasons for s in scored if s.status == "doubtful"))

    def test_G8_a_model_without_a_ranker_suggests_nothing(self):
        plain, _, _ = train.fit_model(synthetic(60, 1), synthetic(30, 2), {"leaves": 7, "trees": 100})
        scored = Guard(plain).score_page(self.shaky_page(["ordenansa"]), ("es",), "es", 0.08)
        self.assertIsNone(scored[0].suggestion)


if __name__ == "__main__":
    unittest.main()
