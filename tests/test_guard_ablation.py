import unittest
from types import SimpleNamespace

import numpy as np

from bench.guard import ablation
from extract.guard import features
from tests.test_guard_operating_ops import raw_of, table_of


class MaskingTest(unittest.TestCase):
    def test_G12_the_groups_cover_the_features_once_each(self):
        names = [n for g in ablation.GROUPS.values() for n in g]
        self.assertEqual(sorted(names), sorted(features.FEATURES))

    def test_G12_a_dropped_group_is_held_at_zero_in_training_and_in_scoring(self):
        seen = {}

        def base(x, y):
            seen["fit"] = x.copy()
            return lambda rows: (seen.__setitem__("score", rows.copy()), np.zeros(len(rows)))[1]

        columns = ablation.columns_of("readings")
        learner = ablation.without_columns(base, columns)
        x = np.ones((4, len(features.FEATURES)), dtype=np.float32)
        learner(x, np.zeros(4))(x)
        for key in ("fit", "score"):
            self.assertTrue((seen[key][:, columns] == 0).all())
            kept = [i for i in range(x.shape[1]) if i not in columns]
            self.assertTrue((seen[key][:, kept] == 1).all())
        self.assertTrue((x == 1).all())  # the caller's matrix is not touched

    def test_G12_a_ranking_learner_keeps_its_groups_through_the_mask(self):
        def ranker(x, y, groups):
            return lambda rows: np.zeros(len(rows))

        ranker.needs_groups = True
        masked = ablation.without_columns(ranker, [0])
        self.assertTrue(masked.needs_groups)
        masked(np.ones((3, len(features.FEATURES)), dtype=np.float32), np.zeros(3), np.array([3]))(
            np.ones((3, len(features.FEATURES)), dtype=np.float32)
        )

    def test_G12_there_is_a_learner_for_the_full_model_and_one_without_each_group(self):
        learners = ablation.ablation_learners(lambda x, y: lambda rows: rows[:, 0])
        self.assertEqual(set(learners), {"all", *[f"without_{g}" for g in ablation.GROUPS]})


class EvaluateTest(unittest.TestCase):
    def test_G12_the_loss_of_a_dropped_group_is_the_difference_in_correct_fixes(self):
        rng = np.random.default_rng(1)
        sites = [("text", int(rng.random() < 0.7), i // 3, (i,), f"d{i % 30}") for i in range(900)]
        table = table_of(sites)
        good, poor = raw_of(table, rng, 0.02), raw_of(table, rng, 2.0)
        run = SimpleNamespace(table=table, raw={"all": good, "without_page": poor})
        out = ablation.evaluate(run, resamples=10)
        self.assertIsNone(out["all"]["safe_fixed_lost"])
        self.assertGreater(out["without_page"]["safe_fixed_lost"], 100)
        self.assertEqual(
            out["without_page"]["safe_fixed_lost"],
            round(out["all"]["safe_fixed"] - out["without_page"]["safe_fixed"], 1),
        )


if __name__ == "__main__":
    unittest.main()
