import unittest

import numpy as np

from bench.guard import baselines, calibrate, metrics
from extract.guard.signals import COLUMNS, INDEX


def matrix(rows):
    x = np.full((len(rows), len(COLUMNS)), np.nan, dtype=np.float32)
    for i, values in enumerate(rows):
        for name, value in values.items():
            x[i, INDEX[name]] = value
    return x


class IsotonicTest(unittest.TestCase):
    def test_G6_the_table_is_non_decreasing_and_pools_violators(self):
        rng = np.random.default_rng(0)
        raw = rng.normal(size=4000)
        y = (rng.random(4000) < 1 / (1 + np.exp(-2 * raw))).astype(int)
        table = calibrate.fit_isotonic(raw, y)
        self.assertTrue(np.all(np.diff(table["probs"]) >= 0))
        self.assertTrue(np.all(np.diff(table["scores"]) > 0))
        probability = calibrate.apply_table(table, raw)
        self.assertLess(metrics.expected_calibration_error(y, probability), 0.02)

    def test_G6_pooling_gives_the_mean_of_the_pooled_labels(self):
        raw = np.arange(1200, dtype=float)
        y = np.zeros(1200, dtype=int)
        y[:600] = np.arange(600) % 2 == 0
        y[600:] = 1
        table = calibrate.fit_isotonic(raw, y)
        self.assertAlmostEqual(calibrate.apply_table(table, np.array([100.0]))[0], 0.5, places=6)
        self.assertAlmostEqual(calibrate.apply_table(table, np.array([1100.0]))[0], 1.0, places=6)

    def test_G6_scores_outside_the_table_take_its_end_values(self):
        raw = np.linspace(0, 1, 1500)
        table = calibrate.fit_isotonic(raw, (raw > 0.5).astype(int))
        outside = calibrate.apply_table(table, np.array([-5.0, 5.0]))
        self.assertEqual(outside[0], table["probs"][0])
        self.assertEqual(outside[1], table["probs"][-1])

    def test_G6_a_model_fitted_on_too_few_words_is_refused(self):
        with self.assertRaises(ValueError):
            calibrate.fit_isotonic(np.arange(999.0), np.zeros(999))

    def test_G6_equal_scores_get_equal_probabilities(self):
        raw = np.repeat(np.arange(30.0), 50)
        y = (np.arange(1500) % 3 == 0).astype(int)
        probability = calibrate.apply_table(calibrate.fit_isotonic(raw, y), raw)
        for value in range(30):
            self.assertEqual(len(set(probability[raw == value])), 1)


class ThresholdTest(unittest.TestCase):
    def test_G6_each_budget_flags_about_its_share_of_the_words(self):
        probability = np.random.default_rng(3).random(10000)
        thresholds = calibrate.word_thresholds(probability, (0.03, 0.05, 0.08))
        shares = calibrate.realised_shares(probability, thresholds)
        for budget, share in shares.items():
            self.assertAlmostEqual(share, float(budget), delta=0.001)
        self.assertGreater(thresholds["0.03"], thresholds["0.05"])
        self.assertGreater(thresholds["0.05"], thresholds["0.08"])

    def test_G6_ties_can_flag_more_and_the_realised_share_says_so(self):
        probability = np.array([0.9] * 20 + [0.1] * 80)
        thresholds = calibrate.word_thresholds(probability, (0.05,))
        self.assertEqual(calibrate.realised_shares(probability, thresholds)["0.05"], 0.2)


class BaselineTest(unittest.TestCase):
    def test_G5_the_heuristics_rank_the_way_their_names_say(self):
        x = matrix(
            [
                {"conf_mean": 0.9, "conf_min": 0.9, "in_dictionary": 1},
                {"conf_mean": 0.5, "conf_min": 0.8, "in_dictionary": 1},
                {"conf_mean": 0.8, "conf_min": 0.3, "in_dictionary": 1},
                {"conf_mean": 0.95, "conf_min": 0.95, "in_dictionary": 0},
            ]
        )
        self.assertEqual(np.argmax(baselines.lowest_mean_confidence(x)), 1)
        self.assertEqual(np.argmax(baselines.lowest_minimum_confidence(x)), 2)
        self.assertEqual(np.argmax(baselines.outside_the_dictionary_first(x)), 3)

    def test_G5_missing_confidence_or_dictionary_does_not_make_a_word_suspicious(self):
        x = matrix(
            [
                {"conf_mean": np.nan, "conf_min": np.nan, "in_dictionary": np.nan},
                {"conf_mean": 0.5, "conf_min": 0.5, "in_dictionary": 1},
            ]
        )
        for score in (
            baselines.lowest_mean_confidence,
            baselines.lowest_minimum_confidence,
            baselines.outside_the_dictionary_first,
        ):
            self.assertLess(score(x)[0], score(x)[1])

    def test_G5_the_logistic_baseline_learns_a_separable_signal_and_copes_with_nan(self):
        rng = np.random.default_rng(4)
        n = 4000
        conf = rng.random(n)
        y = (conf < 0.3).astype(int)
        digits = rng.random(n) < 0.2
        x = matrix(
            [
                {"conf_mean": c, "has_digit": float(d), "number_format_ok": np.nan if not d else 1.0}
                for c, d in zip(conf, digits, strict=True)
            ]
        )
        model = baselines.LogisticBaseline(1.0).fit(x[:3000], y[:3000])
        self.assertGreater(metrics.average_precision(y[3000:], model.score(x[3000:])), 0.95)

    def test_G5_stronger_regularisation_shrinks_the_weights(self):
        rng = np.random.default_rng(5)
        x = matrix([{"conf_mean": c} for c in rng.random(2000)])
        y = (rng.random(2000) < x[:, INDEX["conf_mean"]]).astype(int)
        weak = baselines.LogisticBaseline(0.01).fit(x, y)
        strong = baselines.LogisticBaseline(1e6).fit(x, y)
        self.assertLess(np.abs(strong.weights[:-1]).sum(), np.abs(weak.weights[:-1]).sum())

    def test_G5_the_regularisation_is_chosen_on_validation(self):
        rng = np.random.default_rng(6)
        conf = rng.random(3000)
        y = (conf < 0.25).astype(int)
        x = matrix([{"conf_mean": c} for c in conf])
        model = baselines.fit_logistic(x[:2000], y[:2000], x[2000:], y[2000:])
        self.assertIn(model.l2, baselines.L2_CHOICES)


if __name__ == "__main__":
    unittest.main()
