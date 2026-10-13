import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from bench.guard import calibrate, metrics, train
from extract.errors import ModelMismatch
from extract.guard import model, signals
from extract.guard.signals import INDEX


def synthetic(n_documents, seed, words=120):
    """Rows whose wrongness follows low confidence and being outside the dictionary, by document."""
    rng = np.random.default_rng(seed)
    n = n_documents * words
    x = np.full((n, len(signals.COLUMNS)), np.nan, dtype=np.float32)
    conf = rng.beta(8, 1.5, n)
    inside = (rng.random(n) < 0.9).astype(float)
    logit = -3.5 + 6.0 * (1.0 - conf) + 1.5 * (1.0 - inside)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-logit))).astype(np.uint8)
    x[:, INDEX["conf_mean"]] = conf
    x[:, INDEX["conf_min"]] = conf * rng.uniform(0.6, 1.0, n)
    x[:, INDEX["in_dictionary"]] = inside
    x[:, INDEX["length"]] = rng.integers(2, 12, n)
    group = np.repeat(np.array([f"d{i}" for i in range(n_documents)], dtype=object), words)
    band = np.where(conf > 0.9, "light", "medium").astype(object)
    return SimpleNamespace(x=x, y=y, group=group, band=band)


class GridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = synthetic(60, 1)
        cls.validation = synthetic(30, 2)
        cls.cells = train.run_grid(cls.train, cls.validation)

    def test_G5_the_grid_has_six_cells_with_an_ap_and_a_cluster_se(self):
        self.assertEqual([(c["leaves"], c["trees"]) for c in self.cells], list(train.GRID))
        self.assertEqual(len(train.GRID), 6)
        for cell in self.cells:
            self.assertTrue(0.0 < cell["ap"] <= 1.0 and cell["se"] > 0)

    def test_G5_the_chosen_cell_is_the_cheapest_within_one_se_of_the_best(self):
        chosen = train.choose_cell(self.cells)
        best = max(self.cells, key=lambda c: c["ap"])
        self.assertGreaterEqual(chosen["ap"], best["ap"] - best["se"])
        for cell in self.cells:
            if cell["ap"] >= best["ap"] - best["se"]:
                self.assertLessEqual((chosen["trees"], chosen["leaves"]), (cell["trees"], cell["leaves"]))

    def test_G5_every_cell_is_scored_on_the_validation_rows_and_never_on_the_training_rows(self):
        with mock.patch.object(train, "GRID", ((7, 30),)):
            cell = train.run_grid(self.train, self.validation)[0]
        booster = model.train_detector(self.train.x, self.train.y, 7, 30, seed=train.SEED, threads=1)
        on_validation = metrics.average_precision(self.validation.y, model.raw_scores(booster, self.validation.x, 1))
        on_training = metrics.average_precision(self.train.y, model.raw_scores(booster, self.train.x, 1))
        self.assertAlmostEqual(cell["ap"], on_validation)
        self.assertGreater(abs(on_training - on_validation), 1e-3)

    def test_G5_a_cheaper_cell_within_one_se_of_the_best_wins_over_the_best(self):
        cells = [
            {"leaves": 15, "trees": 200, "ap": 0.80, "se": 0.02},
            {"leaves": 7, "trees": 100, "ap": 0.785, "se": 0.02},
            {"leaves": 7, "trees": 200, "ap": 0.77, "se": 0.02},
        ]
        chosen = train.choose_cell(cells)
        self.assertEqual((chosen["leaves"], chosen["trees"]), (7, 100))

    def test_G5_a_clearly_worse_cell_is_never_chosen(self):
        cells = [
            {"leaves": 7, "trees": 100, "ap": 0.50, "se": 0.01},
            {"leaves": 15, "trees": 200, "ap": 0.80, "se": 0.01},
            {"leaves": 31, "trees": 200, "ap": 0.795, "se": 0.01},
        ]
        self.assertEqual(train.choose_cell(cells)["leaves"], 15)


class FitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = synthetic(60, 1)
        cls.validation = synthetic(30, 2)
        cls.guard, cls.raw, cls.probability = train.fit_model(cls.train, cls.validation, {"leaves": 7, "trees": 100})

    def test_G6_the_calibrated_probabilities_match_the_observed_rates_on_validation(self):
        self.assertLess(metrics.expected_calibration_error(self.validation.y, self.probability), 0.03)

    def test_G6_the_model_says_it_is_calibrated_and_keeps_its_summary(self):
        meta = self.guard.meta
        self.assertTrue(meta["calibrated"])
        self.assertEqual(meta["training"]["validation_words"], 3600)
        self.assertEqual(meta["schema"], signals.schema_fingerprint())
        self.assertEqual(meta["number_divisor"], 3.0)

    def test_G6_thresholds_flag_each_budget_on_validation_and_numbers_get_a_third(self):
        for budget in model.BUDGETS:
            word, number = self.guard.thresholds(budget)
            self.assertAlmostEqual(number, word / 3.0)
            self.assertAlmostEqual(float((self.probability >= word).mean()), budget, delta=0.005)
        with self.assertRaises(ValueError):
            self.guard.thresholds(0.5)

    def test_G5_the_detector_beats_the_lowest_confidence_baseline_on_the_signals_it_was_built_from(self):
        from bench.guard import baselines

        flagged_model = metrics.recall_at_share(self.validation.y, self.raw, 0.05)
        flagged_conf = metrics.recall_at_share(
            self.validation.y, baselines.lowest_mean_confidence(self.validation.x), 0.05
        )
        self.assertGreater(flagged_model, flagged_conf)

    def test_G6_the_calibration_report_has_the_pooled_error_and_one_per_band(self):
        report = train.calibration_report(self.validation, self.probability, self.raw)
        self.assertEqual(set(report["ece_in_sample"]), {"pooled", "light", "medium"})
        self.assertEqual(set(report["ece_out_of_sample"]), {"pooled", "light", "medium"})
        self.assertEqual(len(report["reliability"]["pooled"]), 10)

    def test_G6_the_error_on_the_rows_that_fitted_the_table_is_smaller_than_on_rows_it_never_saw(self):
        report = train.calibration_report(self.validation, self.probability, self.raw)
        self.assertLess(report["ece_in_sample"]["pooled"], report["ece_out_of_sample"]["pooled"])

    def test_G6_each_half_of_the_documents_is_calibrated_by_the_table_of_the_other_half(self):
        held_out = calibrate.out_of_sample_probability(self.raw, self.validation.y, self.validation.group)
        documents = np.unique(self.validation.group)
        first = np.isin(self.validation.group, documents[::2])
        table = calibrate.fit_isotonic(self.raw[~first], self.validation.y[~first])
        np.testing.assert_allclose(held_out[first], calibrate.apply_table(table, self.raw[first]))


class ModelFileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.guard, cls.raw, _ = train.fit_model(synthetic(60, 1), synthetic(30, 2), {"leaves": 7, "trees": 100})

    def test_G1_a_saved_model_loads_back_with_the_same_predictions(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "guard.model"
            self.guard.save(path)
            again = model.GuardModel.load(path)
        x = synthetic(5, 9).x
        np.testing.assert_allclose(again.raw(x), self.guard.raw(x), atol=1e-9)
        self.assertEqual(again.meta["thresholds"], self.guard.meta["thresholds"])

    def test_G13_saving_twice_gives_the_same_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            first, second = Path(folder) / "a.model", Path(folder) / "b.model"
            self.guard.save(first)
            self.guard.save(second)
            self.assertEqual(first.read_bytes(), second.read_bytes())

    def test_G1_a_model_with_another_schema_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.model"
            self.guard.meta["schema"] = "0" * 64
            try:
                self.guard.save(path)
                with self.assertRaises(ModelMismatch):
                    model.GuardModel.load(path)
            finally:
                self.guard.meta["schema"] = signals.schema_fingerprint()

    def test_G1_a_model_from_another_lightgbm_major_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "old.model"
            self.guard.meta["lightgbm_major"] = "3"
            try:
                self.guard.save(path)
                with self.assertRaises(ModelMismatch):
                    model.GuardModel.load(path)
            finally:
                self.guard.meta["lightgbm_major"] = model.lgb.__version__.split(".")[0]

    def test_G1_a_missing_or_broken_file_is_a_model_mismatch_not_a_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ModelMismatch):
                model.GuardModel.load(Path(folder) / "none.model")
            broken = Path(folder) / "broken.model"
            broken.write_bytes(b"not a zip")
            with self.assertRaises(ModelMismatch):
                model.GuardModel.load(broken)

    def test_G13_the_same_seed_gives_the_same_predictions_with_one_and_four_threads(self):
        data = synthetic(40, 3)
        one = model.train_detector(data.x, data.y, 15, 100, seed=7, threads=1)
        four = model.train_detector(data.x, data.y, 15, 100, seed=7, threads=4)
        probe = synthetic(5, 11).x
        np.testing.assert_allclose(model.raw_scores(one, probe), model.raw_scores(four, probe, threads=4), atol=1e-9)

    def test_G11_a_higher_confidence_never_makes_a_word_more_suspicious(self):
        data = synthetic(40, 4)
        booster = model.train_detector(data.x, data.y, 15, 100)
        base = synthetic(8, 12).x
        for name, sign in signals.SCHEMA:
            if sign == 0:
                continue
            column = INDEX[name]
            low, high = base.copy(), base.copy()
            filled = np.nan_to_num(base[:, column], nan=0.0)
            low[:, column] = filled
            high[:, column] = filled + 1.0
            difference = model.raw_scores(booster, high) - model.raw_scores(booster, low)
            self.assertTrue(np.all(sign * difference >= -1e-9), name)

    def test_G9_contributions_add_up_to_the_raw_score(self):
        x = synthetic(5, 13).x
        parts = model.contributions(self.guard.detector, x)
        self.assertEqual(parts.shape, (len(x), len(signals.COLUMNS) + 1))
        np.testing.assert_allclose(parts.sum(axis=1), self.guard.raw(x), atol=1e-6)


class SignalGroupTest(unittest.TestCase):
    def test_G12_the_five_groups_cover_the_21_signals_once_each(self):
        groups = train.signal_groups()
        every = sorted(i for members in groups.values() for i in members)
        self.assertEqual(every, list(range(len(signals.COLUMNS))))
        self.assertEqual(set(groups), {"engine", "dictionary", "shape", "layout", "context"})


if __name__ == "__main__":
    unittest.main()
