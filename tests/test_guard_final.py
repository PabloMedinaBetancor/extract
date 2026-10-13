import unittest

import numpy as np

from bench.guard import baselines, final, fixer_train, metrics, train
from extract.guard import model
from tests.test_guard_fixer_train import simulated_rows
from tests.test_guard_run_validation import rows

RESAMPLES = 60


class FinalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = rows(60, 1)
        cls.validation = rows(30, 2)
        cls.test = rows(40, 5)
        cls.test.band = np.where(np.arange(len(cls.test.y)) % 5 == 0, "heavy", cls.test.band).astype(object)
        cls.test.counts = {"unreadable_pages": 1, "missed": 7}
        cls.test.page = cls.test.group
        cls.guard, _, _ = train.fit_model(cls.train, cls.validation, {"leaves": 7, "trees": 100})
        cls.logistic = baselines.fit_logistic(cls.train.x, cls.train.y, cls.validation.x, cls.validation.y)
        cls.baselines = {
            "lowest_mean_confidence": baselines.lowest_mean_confidence(cls.test.x),
            "logistic_regression": cls.logistic.score(cls.test.x),
        }
        cls.report = final.evaluate(cls.test, cls.guard, cls.baselines, resamples=RESAMPLES)

    def test_G5_the_detector_and_every_baseline_have_recall_precision_and_ap_with_an_interval(self):
        block = self.report["at_review_5pct"]
        self.assertEqual(set(block), {"detector", "lowest_mean_confidence", "logistic_regression"})
        for figures in block.values():
            for name in ("recall", "precision", "ap"):
                self.assertLess(figures[name]["low"], figures[name]["high"])

    def test_G5_the_recall_at_five_percent_is_the_rank_matched_one_computed_by_hand(self):
        raw = self.guard.raw(self.test.x)
        expected = metrics.recall_at_share(self.test.y, raw, 0.05)
        self.assertAlmostEqual(self.report["at_review_5pct"]["detector"]["recall"]["value"], expected)

    def test_G5_the_best_baseline_is_the_one_with_the_highest_recall_and_the_difference_is_paired(self):
        block = self.report["at_review_5pct"]
        best = max(self.baselines, key=lambda n: block[n]["recall"]["value"])
        self.assertEqual(self.report["best_baseline"], best)
        difference = self.report["detector_minus_best_baseline"]
        self.assertAlmostEqual(
            difference["recall"]["value"], block["detector"]["recall"]["value"] - block[best]["recall"]["value"]
        )
        self.assertIn("excludes_zero", difference["average_precision"])

    def test_G6_the_shipped_thresholds_flag_about_their_budget_and_numbers_get_the_lower_threshold(self):
        shares = [self.report["shipped"][f"{budget:.2f}"]["flagged_share"]["value"] for budget in model.BUDGETS]
        self.assertEqual(shares, sorted(shares))
        for budget, share in zip(model.BUDGETS, shares, strict=True):
            self.assertGreater(share, budget - 0.02)
            self.assertIn("recall", self.report["numbers"][f"{budget:.2f}"])
        self.assertGreater(self.report["numbers"]["digit_tokens"], 0)

    def test_G6_calibration_is_reported_pooled_and_per_band_with_the_reliability_tables(self):
        self.assertEqual(set(self.report["calibration"]["ece"]), {"pooled", "light", "medium", "heavy"})
        self.assertEqual(len(self.report["calibration"]["reliability"]["pooled"]), 10)

    def test_G12_each_band_reports_its_words_and_the_half_width_of_the_pooled_recall_is_given(self):
        self.assertEqual(sum(b["words"] for b in self.report["by_band"].values()), len(self.test.y))
        self.assertGreater(self.report["half_width_pooled_recall"], 0)

    def test_G3_pages_that_the_guard_did_not_cover_are_reported_beside_the_recall(self):
        self.assertEqual((self.report["rows"]["unreadable_pages"], self.report["rows"]["missed"]), (1, 7))

    def test_G8_a_model_without_a_ranker_reports_that_instead_of_a_number(self):
        self.assertEqual(self.report["fixer"], {"note": "the model has no ranker"})

    def test_G5_the_misses_are_listed_by_kind_and_confidence_decile(self):
        self.assertEqual(len(self.report["misses"]["by_confidence_decile"]), 10)
        self.assertEqual(set(self.report["misses"]["by_kind"]), {"substitution", "merge"})


class FixerFinalTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard, _, _ = train.fit_model(rows(60, 1), rows(30, 2), {"leaves": 7, "trees": 100})
        train_rows = simulated_rows(600, 1)
        queries = fixer_train.build_queries(train_rows, range(len(train_rows.y)))
        cls.guard.ranker = fixer_train.fit_ranker(queries)
        best, scores = fixer_train.top_choices(cls.guard.ranker, queries)
        table = fixer_train.calibrate_top_choice(scores, np.array(queries.truth_at) == best)
        cls.guard.meta["fixer"] = {"calibration": table, "t_suggest": 0.5, "feature_fingerprint": "test"}
        cls.rows = simulated_rows(300, 7)
        n = len(cls.rows.y)
        cls.rows.group = np.array([f"d{i % 12}" for i in range(n)], dtype=object)
        cls.rows.band = np.array(["light"] * n, dtype=object)
        cls.report = final.evaluate_fixer(cls.rows, cls.guard, np.ones(n, dtype=bool), resamples=RESAMPLES)

    def test_G8_the_queries_get_candidate_recall_and_top1_accuracy_with_intervals(self):
        self.assertGreater(self.report["queries"], 100)
        for name in ("candidate_recall", "top1_all", "top1_given_candidate"):
            self.assertLessEqual(self.report[name]["low"], self.report[name]["value"])
            self.assertLessEqual(self.report[name]["value"], self.report[name]["high"])
        self.assertGreater(self.report["candidate_recall"]["value"], 0.5)

    def test_G8_the_suggestions_shown_are_the_ones_over_t_suggest_and_their_precision_is_measured(self):
        shown = self.report["shown"]
        self.assertEqual(shown["t_suggest"], 0.5)
        self.assertGreater(shown["suggested"], 0)
        self.assertGreater(shown["precision"]["value"], 0.5)
        self.assertLessEqual(shown["coverage"]["value"], 1.0)


if __name__ == "__main__":
    unittest.main()
