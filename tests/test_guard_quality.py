"""What the applied edits do to the words: signed precision, the precision that never makes a word worse, the damage,
and the changes per 100 applied."""

import unittest
from types import SimpleNamespace

import numpy as np

from bench.guard import operating
from tests.test_guard_operating_ops import table_of

# (token as read, truth, candidate applied, token was right)
CASES = [
    ("casa", "casas", "casas", False),  # fixed: 1 -> 0
    ("casa", "casa", "caso", True),  # broken: 0 -> 1
    ("CARACTERISTICAS", "CARACTERÍSTICAS.", "CARACTERÍSTICAS", False),  # still wrong but closer: 2 -> 1
    ("abcd", "abce", "abcf", False),  # still wrong, as far: 1 -> 1
    ("abc", "abcd", "xyz", False),  # still wrong, farther: 1 -> 4
]


def build(cases=CASES, documents=None):
    table = table_of([("text", 1, 0, (i,), (documents or {}).get(i, f"d{i}")) for i in range(len(cases))])
    table.token = np.array([c[0] for c in cases], dtype=object)
    table.truth = np.array([c[1] for c in cases], dtype=object)
    table.token_right = np.array([c[3] for c in cases], dtype=bool)
    table.found = [[None, SimpleNamespace(text=c[2], kind="letters")] for c in cases]
    outcome = [operating.FIXED if c[1] == c[2] else operating.BROKEN if c[3] else operating.STILL_WRONG for c in cases]
    n = len(cases)
    choice = operating.Choice(
        np.ones(n, dtype=np.int64), np.linspace(1.0, 0.5, n), np.zeros(n), np.array(outcome, dtype=np.int8)
    )
    return table, choice


class QualityTest(unittest.TestCase):
    def test_the_distances_of_each_site_before_and_after_and_the_length_of_the_truth(self):
        table, choice = build()
        d = operating.edit_distances(table, choice)
        self.assertEqual(d.before.tolist(), [1, 0, 2, 1, 1])
        self.assertEqual(d.after.tolist(), [0, 1, 1, 1, 4])
        self.assertEqual(d.reference.tolist(), [5, 4, 16, 4, 4])

    def test_a_site_with_no_candidate_has_no_distance(self):
        table, choice = build()
        choice.p_fix[2] = -np.inf
        d = operating.edit_distances(table, choice)
        self.assertTrue(np.isnan(d.before[2]) and np.isnan(d.after[2]))

    def test_the_two_promises_and_the_damage_over_the_applied_sites(self):
        table, choice = build()
        out = operating.quality(table, choice, np.ones(5, dtype=bool), resamples=20)
        self.assertEqual(
            {k: out[k] for k in ("applied", "fixed", "broken", "improved", "equal", "worse")},
            {"applied": 5, "fixed": 1, "broken": 1, "improved": 1, "equal": 1, "worse": 1},
        )
        value = {k: v["value"] for k, v in out["intervals"].items()}
        self.assertAlmostEqual(value["precision_signed"], 1 / 5)
        self.assertAlmostEqual(value["precision_no_worse"], 2 / 5)  # fixed + improved
        self.assertAlmostEqual(value["damage_strict"], 2 / 5)  # broken + worse
        self.assertAlmostEqual(value["damage_with_equal"], 3 / 5)
        self.assertAlmostEqual(value["broken_share"], 1 / 5)
        self.assertAlmostEqual(value["improved_share"], 1 / 5)

    def test_the_changes_per_100_applied_and_the_character_error_rate_over_the_applied_truth(self):
        table, choice = build()
        out = operating.quality(table, choice, np.ones(5, dtype=bool), resamples=20)
        value = {k: v["value"] for k, v in out["intervals"].items()}
        # edit distance: -1 +1 -1 +0 +3 = +2 over 5 applied
        self.assertAlmostEqual(value["edit_distance_change_per_100"], 100 * 2 / 5)
        # word errors: before 4 wrong (the broken one was right), after 4 wrong (all but the fixed one): 0 change
        self.assertAlmostEqual(value["word_errors_change_per_100"], 0.0)
        self.assertAlmostEqual(value["char_error_rate_change"], 2 / (5 + 4 + 16 + 4 + 4))

    def test_only_the_masked_sites_count_and_an_empty_mask_has_no_rate(self):
        table, choice = build()
        mask = np.array([True, False, True, False, False])
        out = operating.quality(table, choice, mask, resamples=20)
        value = {k: v["value"] for k, v in out["intervals"].items()}
        self.assertEqual((out["applied"], out["fixed"], out["broken"], out["improved"]), (2, 1, 0, 1))
        self.assertAlmostEqual(value["precision_no_worse"], 1.0)
        self.assertAlmostEqual(value["damage_strict"], 0.0)
        self.assertAlmostEqual(value["edit_distance_change_per_100"], 100 * -2 / 2)
        nothing = operating.quality(table, choice, np.zeros(5, dtype=bool), resamples=20)
        self.assertTrue(np.isnan(nothing["intervals"]["precision_signed"]["value"]))

    def test_the_intervals_are_over_documents(self):
        cases = CASES * 8
        docs = {i: f"d{i % 8}" for i in range(len(cases))}
        table, choice = build(cases, docs)
        out = operating.quality(table, choice, np.ones(len(cases), dtype=bool), resamples=50)
        interval = out["intervals"]["precision_signed"]
        self.assertAlmostEqual(interval["value"], 1 / 5)
        self.assertLessEqual(interval["low"], interval["value"])
        self.assertGreaterEqual(interval["high"], interval["value"])

    def test_the_best_n_by_probability_for_the_n_the_sites_can_fill(self):
        table, choice = build()
        masks = operating.top_masks(choice, (2, 4, 10))
        self.assertEqual(sorted(masks), [2, 4])
        self.assertEqual(np.nonzero(masks[2])[0].tolist(), [0, 1])  # p_fix falls with the site number
        self.assertEqual(int(masks[4].sum()), 4)
        choice.p_fix[0] = -np.inf  # a site with no candidate is never in a top
        self.assertEqual(np.nonzero(operating.top_masks(choice, (2,))[2])[0].tolist(), [1, 2])


class PromiseTest(unittest.TestCase):
    def test_V5B37_no_worse_counts_the_edits_that_stay_wrong_but_closer_as_hits_and_exact_does_not(self):
        table, choice = build()
        exact = operating.safe_outcome(table, choice, "exact")
        self.assertEqual(exact.tolist(), choice.outcome.tolist())
        no_worse = operating.safe_outcome(table, choice, "no_worse")
        # site 2 (CARACTERISTICAS: 2 -> 1) becomes a hit; equal, farther and broken stay what they were
        self.assertEqual(
            no_worse.tolist(),
            [operating.FIXED, operating.BROKEN, operating.FIXED, operating.STILL_WRONG, operating.STILL_WRONG],
        )
        self.assertEqual(choice.outcome[2], operating.STILL_WRONG)  # the input is not changed
        with self.assertRaises(ValueError):
            operating.safe_outcome(table, choice, "whatever")

    def test_V5B37_the_promise_only_moves_the_safe_threshold_not_the_more_one(self):
        table, choice = build(CASES * 100)
        ops = operating.site_ops(table)
        minima = {"text": 10, "join": 10, "segment": 10, "drop": 10}
        # p_fix falls with the site: the improved sites are interleaved with the exact ones
        exact = operating.thresholds_by_op(choice, ops, floor=0.99, minima=minima)
        hits = operating.safe_outcome(table, choice, "no_worse")
        no_worse = operating.thresholds_by_op(choice, ops, floor=0.99, minima=minima, safe_outcome=hits)
        self.assertEqual(no_worse["text"]["m"], exact["text"]["m"])
        self.assertIsNone(exact["text"]["t_safe"])  # one in five ends exact: nowhere near 99 %
        self.assertIsNone(no_worse["text"]["t_safe"])  # nor two in five
        ideal = hits.copy()
        ideal[:] = operating.FIXED
        self.assertIsNotNone(
            operating.thresholds_by_op(choice, ops, floor=0.99, minima=minima, safe_outcome=ideal)["text"]["t_safe"]
        )

    def test_the_floor_of_the_sweep_is_99_5_percent_and_t5_asks_99(self):
        self.assertEqual((operating.SAFE_FLOOR, operating.T5_FLOOR), (0.995, 0.99))
        self.assertEqual(operating.PROMISES, ("exact", "no_worse"))


if __name__ == "__main__":
    unittest.main()
