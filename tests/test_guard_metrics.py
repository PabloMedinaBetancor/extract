import unittest

import numpy as np

from bench.guard import metrics


class RankingTest(unittest.TestCase):
    def test_G5_average_precision_of_known_rankings(self):
        y = np.array([1, 0, 1, 0, 0])
        self.assertAlmostEqual(metrics.average_precision(y, np.array([5, 4, 3, 2, 1])), (1 + 2 / 3) / 2)
        self.assertEqual(metrics.average_precision(y, np.array([5, 1, 4, 2, 3])), 1.0)
        self.assertTrue(np.isnan(metrics.average_precision(np.zeros(4), np.arange(4.0))))

    def test_G5_average_precision_of_a_random_score_is_the_base_rate(self):
        rng = np.random.default_rng(0)
        y = (rng.random(20000) < 0.1).astype(int)
        self.assertAlmostEqual(metrics.average_precision(y, rng.random(20000)), 0.1, delta=0.01)

    def test_G4_auc_of_perfect_reversed_and_tied_scores(self):
        y = np.array([0, 0, 1, 1])
        self.assertEqual(metrics.auc(y, np.array([1.0, 2.0, 3.0, 4.0])), 1.0)
        self.assertEqual(metrics.auc(y, np.array([4.0, 3.0, 2.0, 1.0])), 0.0)
        self.assertEqual(metrics.auc(y, np.zeros(4)), 0.5)
        self.assertTrue(np.isnan(metrics.auc(np.zeros(4), np.arange(4.0))))

    def test_G4_auc_of_random_scores_is_one_half(self):
        rng = np.random.default_rng(1)
        y = (rng.random(20000) < 0.1).astype(int)
        self.assertAlmostEqual(metrics.auc(y, rng.random(20000)), 0.5, delta=0.02)

    def test_G5_flagging_the_top_share_gives_recall_and_precision(self):
        y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 1, 0])
        score = np.array([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], dtype=float)
        flagged = metrics.flag_top(score, 0.3)
        self.assertEqual(flagged.sum(), 3)
        self.assertEqual(metrics.recall_precision(y, flagged), (2 / 3, 2 / 3))
        self.assertAlmostEqual(metrics.recall_at_share(y, score, 0.3), 2 / 3)
        self.assertAlmostEqual(metrics.precision_at_share(y, score, 0.3), 2 / 3)

    def test_G5_ties_are_broken_by_row_order_so_results_do_not_change(self):
        score = np.zeros(6)
        self.assertEqual(metrics.flag_top(score, 0.5).tolist(), [True, True, True, False, False, False])

    def test_G5_nothing_to_divide_by_gives_nan_not_zero(self):
        recall, precision = metrics.recall_precision(np.zeros(5), np.zeros(5, dtype=bool))
        self.assertTrue(np.isnan(recall) and np.isnan(precision))


class CalibrationErrorTest(unittest.TestCase):
    def test_G6_a_perfectly_calibrated_score_has_a_small_error(self):
        rng = np.random.default_rng(1)
        p = rng.random(50000)
        y = (rng.random(50000) < p).astype(int)
        self.assertLess(metrics.expected_calibration_error(y, p), 0.02)

    def test_G6_an_over_confident_score_has_a_large_error(self):
        rng = np.random.default_rng(1)
        p = rng.random(50000)
        y = (rng.random(50000) < p).astype(int)
        self.assertGreater(metrics.expected_calibration_error(y, np.clip(p * 0.4 + 0.6, 0, 1)), 0.2)

    def test_G6_the_reliability_table_has_equal_mass_bins(self):
        p = np.linspace(0, 1, 1000)
        table = metrics.reliability_table((p > 0.5).astype(int), p, bins=10)
        self.assertEqual([row["words"] for row in table], [100] * 10)
        self.assertLess(table[0]["observed"], table[-1]["observed"])


class BootstrapTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.groups = np.repeat(np.arange(60), 100)
        self.y = (rng.random(6000) < 0.1).astype(int)
        self.score = self.y * 0.5 + rng.random(6000)

    def test_G5_documents_are_resampled_whole(self):
        documents = metrics.document_index(self.groups)
        self.assertEqual(len(documents), 60)
        rows = metrics.bootstrap_rows(documents, np.random.default_rng(0))
        self.assertEqual(len(rows), 6000)
        counts = np.bincount(self.groups[rows], minlength=60)
        self.assertTrue(set(counts.tolist()) <= {0, 100, 200, 300, 400, 500})

    def test_G5_the_interval_covers_the_value_and_is_reproducible(self):
        statistics = {"recall": lambda rows: metrics.recall_at_share(self.y[rows], self.score[rows], 0.1)}
        first = metrics.cluster_bootstrap(self.groups, statistics, resamples=300, seed=5)["recall"]
        again = metrics.cluster_bootstrap(self.groups, statistics, resamples=300, seed=5)["recall"]
        self.assertEqual(first, again)
        self.assertLess(first["low"], first["value"])
        self.assertGreater(first["high"], first["value"])
        self.assertGreater(first["se"], 0)

    def test_G5_a_paired_difference_excludes_zero_only_when_one_score_is_better(self):
        good = lambda rows: metrics.recall_at_share(self.y[rows], self.score[rows], 0.1)  # noqa: E731
        random_score = np.random.default_rng(9).random(6000)
        bad = lambda rows: metrics.recall_at_share(self.y[rows], random_score[rows], 0.1)  # noqa: E731
        self.assertTrue(metrics.paired_difference(self.groups, good, bad, resamples=300)["excludes_zero"])
        self.assertFalse(metrics.paired_difference(self.groups, good, good, resamples=300)["excludes_zero"])

    def test_G12_the_spread_across_documents_sizes_the_test(self):
        flagged = metrics.flag_top(self.score, 0.1)
        recalls = metrics.per_document_recall(self.y, flagged, self.groups)
        self.assertGreater(len(recalls), 40)
        self.assertEqual(metrics.documents_needed(0.15), 55)
        self.assertGreater(metrics.documents_needed(0.3), metrics.documents_needed(0.15))


if __name__ == "__main__":
    unittest.main()
