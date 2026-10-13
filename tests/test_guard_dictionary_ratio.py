import unittest
from types import SimpleNamespace

import numpy as np

from bench.guard import dictionary_ratio as ratio


class DecideTest(unittest.TestCase):
    def results(self, accent_8, ap_8, accent_20=0.30, ap_20=0.60):
        return {20: {"accent_recall": accent_20, "ap": ap_20}, 8: {"accent_recall": accent_8, "ap": ap_8}}

    def test_G12_the_strict_ratio_wins_with_two_points_of_accent_recall_and_a_small_ap_loss(self):
        self.assertEqual(ratio.decide(self.results(0.33, 0.597))["chosen"], 8)

    def test_G12_it_loses_when_the_accent_gain_is_below_two_points(self):
        self.assertEqual(ratio.decide(self.results(0.315, 0.60))["chosen"], 20)

    def test_G12_it_loses_when_average_precision_falls_by_more_than_half_a_point(self):
        self.assertEqual(ratio.decide(self.results(0.40, 0.59))["chosen"], 20)

    def test_G12_the_rule_is_quoted_in_the_decision(self):
        decision = ratio.decide(self.results(0.33, 0.60))
        self.assertIn("at least 2 points", decision["rule"])
        self.assertAlmostEqual(decision["accent_gain_points"], 3.0, places=6)


class AccentRecallTest(unittest.TestCase):
    def test_G12_only_errors_labelled_accent_count(self):
        rows = SimpleNamespace(
            y=np.array([1, 1, 1, 0]),
            detail=np.array(["accent", "accent", "letters", ""], dtype=object),
        )
        score = np.array([4.0, 1.0, 3.0, 2.0])
        self.assertEqual(ratio.accent_recall(rows, score, share=0.5), 0.5)

    def test_G12_without_accent_errors_the_recall_is_nan(self):
        rows = SimpleNamespace(y=np.array([1, 0]), detail=np.array(["letters", ""], dtype=object))
        self.assertTrue(np.isnan(ratio.accent_recall(rows, np.array([1.0, 0.0]))))


if __name__ == "__main__":
    unittest.main()
