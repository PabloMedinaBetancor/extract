import unittest
from unittest import mock

import numpy as np

from bench.guard import analysis, train
from extract.guard.signals import INDEX
from tests.test_guard_train import synthetic

CELL = {"leaves": 7, "trees": 100}


class AnalysisTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = synthetic(60, 1)
        cls.validation = synthetic(30, 2)

    def test_G12_the_learning_curve_uses_growing_shares_of_whole_documents(self):
        curve = analysis.learning_curve(self.train, self.validation, CELL)
        self.assertEqual([c["fraction"] for c in curve], [0.25, 0.5, 1.0])
        self.assertEqual([c["documents"] for c in curve], [15, 30, 60])
        self.assertTrue(curve[0]["words"] < curve[1]["words"] < curve[2]["words"])
        self.assertGreater(curve[2]["ap"], 0.0)

    def test_G12_the_subsets_are_nested_and_follow_the_document_hash_not_the_row_order(self):
        masks = analysis.document_subsets(self.train, (0.25, 0.5, 1.0))
        self.assertTrue(np.all(masks[0.5][masks[0.25]]))
        self.assertTrue(np.all(masks[1.0]))
        shuffled = np.random.default_rng(0).permutation(len(self.train.y))
        again = analysis.document_subsets(
            type(self.train)(
                x=self.train.x[shuffled],
                y=self.train.y[shuffled],
                group=self.train.group[shuffled],
                band=self.train.band[shuffled],
            ),
            (0.25,),
        )
        self.assertEqual(set(self.train.group[masks[0.25]]), set(self.train.group[shuffled][again[0.25]]))

    def test_G12_dropping_the_group_that_holds_the_signal_hurts_and_dropping_another_does_not(self):
        groups = {"engine": [INDEX["conf_mean"], INDEX["conf_min"]], "layout": [INDEX["height_ratio"]]}
        results = analysis.ablation(self.train, self.validation, CELL, groups)
        self.assertLess(results["without_engine"]["recall_at_review"], results["none"]["recall_at_review"] - 0.05)
        self.assertAlmostEqual(results["without_layout"]["ap"], results["none"]["ap"], delta=0.02)

    def test_G4_the_permutation_canary_scores_chance_on_the_true_labels(self):
        canary = analysis.permutation_canary(self.train, self.validation, CELL, repeats=12, constrained_repeats=4)
        self.assertTrue(canary["passes"], canary)
        self.assertGreater(canary["auc_with_monotone_constraints"], canary["auc"])
        self.assertLess(abs(canary["auc_both_shuffled"] - 0.5), 0.05)

    def test_G4_scores_that_know_the_true_labels_fail_the_canary_on_the_no_leak_side(self):
        knowing = self.validation.y.astype(float) + np.random.default_rng(3).random(len(self.validation.y)) * 0.1
        with mock.patch.object(analysis, "_fit_score", return_value=knowing):
            canary = analysis.permutation_canary(self.train, self.validation, CELL, repeats=5, constrained_repeats=1)
        self.assertFalse(canary["no_leak"])
        self.assertTrue(canary["clean_null"])
        self.assertFalse(canary["passes"])

    def test_G4_a_feature_that_copies_the_label_shows_up_as_an_auc_near_one(self):
        leaky_train = self.train
        leaky_validation = type(self.validation)(
            x=self.validation.x.copy(), y=self.validation.y, group=self.validation.group, band=self.validation.band
        )
        leaky_train.x[:, INDEX["page_conf_mean"]] = leaky_train.y
        leaky_validation.x[:, INDEX["page_conf_mean"]] = leaky_validation.y
        try:
            fresh = analysis._summary(leaky_validation, analysis._fit_score(leaky_train, leaky_validation, CELL))
            self.assertGreater(fresh["auc"], 0.99)
        finally:
            self.train.x[:, INDEX["page_conf_mean"]] = np.nan

    def test_G11_the_monotone_constraints_cost_is_measured_in_points_of_recall(self):
        cost = analysis.monotone_cost(self.train, self.validation, CELL)
        self.assertIn("recall_cost_points", cost)
        self.assertLess(abs(cost["recall_cost_points"]), 5.0)

    def test_G12_the_pilot_turns_the_spread_across_documents_into_a_document_count(self):
        score = np.random.default_rng(3).random(len(self.validation.y)) + self.validation.y
        pilot = analysis.pilot_sample_size(self.validation.y, score, self.validation.group)
        self.assertGreater(pilot["documents_with_errors"], 20)
        self.assertGreater(pilot["recall_sd"], 0)
        self.assertGreater(pilot["documents_needed_per_band"], 0)
        self.assertLess(pilot["documents_needed_per_band"], pilot["documents_needed_pooled"])

    def test_G5_the_grid_helper_and_the_analysis_use_the_same_seed(self):
        self.assertEqual(train.SEED, 1)


if __name__ == "__main__":
    unittest.main()
