"""The per-group door: an op whose `t_safe` misses the floor gets one scan per (op, kind) group that has a calibration
table of its own; the group's threshold overrides the op's in `safe` only, in the report and at run time."""

import unittest

import numpy as np

from bench.guard import first_result, operating
from extract.guard import fixing
from tests.test_guard_fixing import MINIMA, site
from tests.test_guard_operating_ops import table_of


def mixed_table(seed=1, n=400):
    """Text sites of two kinds: `accent` fixes are nearly always right and `letters` ones half of the time; joins are right."""
    rng = np.random.default_rng(seed)
    sites = [("text", int(rng.random() < 0.999), 0, (i,), f"d{i % 20}", "accent") for i in range(n)]
    sites += [("text", int(rng.random() < 0.5), 0, (i + 1000,), f"d{i % 20}", "letters") for i in range(n)]
    sites += [("join", 1, 0, (i + 2000, i + 2001), f"d{i % 20}", "structure") for i in range(60)]
    sites += [("text", 1, 0, (i + 3000,), f"d{i % 20}", "case") for i in range(20)]  # a group too small to have a table
    table = table_of(sites)
    # the raw score follows the label, with some noise: a calibration that is good for both kinds
    raw = np.where(table.y == 1, 0.9, 0.1) + np.random.default_rng(seed + 1).normal(0, 0.05, len(table.y))
    return table, raw


class EligibleGroupsTest(unittest.TestCase):
    def test_V5B33_a_group_needs_its_own_calibration_table(self):
        table, _ = mixed_table()
        self.assertEqual(
            operating.eligible_groups(table, min_group=300, min_positive=50), {"text|accent", "text|letters"}
        )
        self.assertEqual(
            operating.eligible_groups(table, min_group=50, min_positive=50),
            {"text|accent", "text|letters", "join|structure"},
        )
        self.assertEqual(operating.eligible_groups(table, min_group=300, min_positive=500), set())

    def test_V5B33_the_eligible_groups_are_those_the_calibration_tables_hold(self):
        table, raw = mixed_table()
        tables = operating.calibration_tables(table, raw, min_group=300, min_positive=50)
        wanted = {k for k in tables if "|" in k}
        self.assertEqual(operating.eligible_groups(table, 300, 50), wanted)


class DoorThresholdsTest(unittest.TestCase):
    def setUp(self):
        self.table, self.raw = mixed_table()
        n = self.table.sites
        # the calibrated probability does not tell the kinds apart (as if it were not good enough): the fixes of the two
        # kinds are interleaved on `p_fix`, the `accent` ones are all right and half of the `letters` ones are wrong
        p_fix = np.random.default_rng(3).uniform(0.5, 1.0, n)
        label = self.table.y[self.table.starts + 1]
        outcome = np.where(label == 1, operating.FIXED, operating.STILL_WRONG).astype(np.int8)
        self.choice = operating.Choice(np.ones(n, dtype=np.int64), p_fix, np.zeros(n), outcome)
        self.ops = operating.site_ops(self.table)
        self.kinds = operating.site_kinds(self.table, self.choice)
        self.eligible = operating.eligible_groups(self.table, 300, 50)

    def points(self, floor=0.99, minima=None):
        return operating.thresholds_by_op(
            self.choice, self.ops, floor, minima or {**MINIMA, "text": 100}, kinds=self.kinds, eligible=self.eligible
        )

    def test_V5B33_the_kinds_are_those_of_the_best_candidates(self):
        self.assertEqual(set(self.kinds[self.ops == "text"]), {"accent", "case", "letters"})
        self.assertEqual(set(self.kinds[self.ops == "join"]), {"structure"})

    def test_V5B33_an_op_that_misses_its_floor_gets_a_threshold_per_group_and_the_clean_group_reaches_it(self):
        points = self.points()
        self.assertIsNone(points["text"]["t_safe"])  # half of the letters fixes are wrong: the op as a whole misses
        self.assertIsNotNone(points["text|accent"]["t_safe"])
        self.assertIsNone(points["text|letters"]["t_safe"])  # a group that misses stays off
        self.assertIsNone(points["text|accent"]["m"])
        self.assertNotIn("join|structure", points)  # no table of its own: not eligible
        self.assertNotIn("text|case", points)
        applied = operating.applied_by_op(self.choice, self.ops, "safe", points, self.kinds)
        self.assertTrue(applied[(self.ops == "text") & (self.kinds == "accent")].any())
        self.assertFalse(applied[(self.ops == "text") & (self.kinds == "letters")].any())

    def test_V5B33_the_door_opens_only_for_an_op_that_misses(self):
        points = self.points(floor=0.3)  # the op as a whole reaches a floor this low
        self.assertIsNotNone(points["text"]["t_safe"])
        self.assertFalse([k for k in points if "|" in k])

    def test_V5B33_the_group_thresholds_use_the_minimum_count_of_their_op(self):
        points = self.points(minima={**MINIMA, "text": 10_000})
        self.assertIsNone(points["text|accent"]["t_safe"])

    def test_V5B33_without_kinds_there_is_no_door(self):
        points = operating.thresholds_by_op(self.choice, self.ops, 0.99, {**MINIMA, "text": 100})
        self.assertFalse([k for k in points if "|" in k])

    def test_V5B33_a_group_threshold_never_applies_in_more_and_the_op_entries_still_do(self):
        points = self.points()
        points["text"]["m"] = 0.0
        more = operating.applied_by_op(self.choice, self.ops, "more", points, self.kinds)
        letters = (self.ops == "text") & (self.kinds == "letters")
        self.assertTrue(more[letters].any())
        self.assertEqual(
            int(more.sum()),
            int(
                operating.applied_by_op(
                    self.choice, self.ops, "more", {"text": points["text"], "join": points["join"]}
                ).sum()
            ),
        )

    def test_V5B33_the_report_counts_each_group_and_the_threshold_of_a_group_overrides_its_op(self):
        shipped = {
            "text": {"t_safe": None, "m": 0.0},
            "join": {"t_safe": None, "m": None},
            "text|accent": {"t_safe": 0.5, "m": None},
            "text|letters": {"t_safe": None, "m": None},
        }
        report = operating.report_by_op(
            self.table, self.raw, cross_fit=False, resamples=5, thresholds=shipped, min_group=300, min_positive=50
        )
        self.assertGreater(report["safe"]["text|accent"]["applied"], 0)
        self.assertEqual(report["safe"]["text|accent"]["broken"], 0)
        self.assertEqual(report["safe"]["text|letters"]["applied"], 0)
        self.assertEqual(report["safe"]["text"]["applied"], report["safe"]["text|accent"]["applied"])  # the op is off
        self.assertNotIn("text|accent", report["more"])
        more = report["more"]["text"]
        for key in ("broken_per_100_applied", "still_wrong_share"):
            self.assertIn(key, more)
        self.assertAlmostEqual(more["broken_per_100_applied"], 100 * more["broken"] / more["applied"])
        self.assertAlmostEqual(more["still_wrong_share"], more["changed_still_wrong"] / more["applied"])


class RuntimeDoorTest(unittest.TestCase):
    BLOCK = {
        "t_safe": {"text": None, "text|accent": 0.9, "text|letters": None, "join": 0.95},
        "m": {"text": 0.3, "join": 0.3},
    }

    def modes(self, op, kind, p_fix, p_keep=0.0, block=None):
        kept = [(site(op, (1,), kind=kind), 1, p_fix, p_keep)]
        return fixing._modes(kept, block or self.BLOCK, None)[0]

    def test_V5B33_the_group_threshold_decides_in_safe_and_a_null_group_is_off(self):
        self.assertEqual(self.modes("text", "accent", 0.95), ("safe", "more"))
        self.assertEqual(self.modes("text", "accent", 0.85), ("more",))  # under its t_safe, over m
        self.assertEqual(self.modes("text", "letters", 0.99), ("more",))  # null: off in safe, margin still gives more
        self.assertEqual(self.modes("text", "case", 0.99), ("more",))  # no key for the group: the op's, which is None

    def test_V5B33_a_null_group_is_off_even_when_its_op_has_a_threshold(self):
        block = {"t_safe": {"text": 0.5, "text|letters": None}, "m": {"text": 0.3}}
        self.assertEqual(self.modes("text", "letters", 0.99, block=block), ("more",))
        self.assertEqual(self.modes("text", "case", 0.99, block=block), ("safe", "more"))

    def test_V5B33_a_group_without_a_key_uses_the_threshold_of_its_op(self):
        self.assertEqual(self.modes("join", "structure", 0.96), ("safe", "more"))
        self.assertEqual(self.modes("join", "structure", 0.90), ("more",))

    def test_V5B33_a_group_key_overrides_a_threshold_of_the_op_that_is_there(self):
        block = {**self.BLOCK, "t_safe": {"text": 0.5, "text|accent": 0.99}}
        self.assertEqual(self.modes("text", "accent", 0.8, block=block), ("more",))
        self.assertEqual(self.modes("text", "letters", 0.8, block=block), ("safe", "more"))


class ModelDoorTest(unittest.TestCase):
    def test_V5B33_the_model_file_has_the_group_thresholds_in_t_safe_and_never_in_m(self):
        from unittest import mock

        from bench.guard import final_fix, train_chooser
        from tests import guard_pages
        from tests.test_guard_fixing import CELL

        pages = guard_pages.corpus(80, errors=0.3)
        with (
            mock.patch.object(train_chooser, "DETECTOR_CELL", CELL),
            mock.patch.object(
                operating,
                "safe_outcome",
                lambda table, choice, promise, *a: np.full(len(choice.outcome), 77, dtype=np.int8),
            ),
            mock.patch.object(operating, "thresholds_by_op", wraps=operating.thresholds_by_op) as sweep,
            mock.patch.object(operating, "protect_proper", wraps=operating.protect_proper),
        ):
            model, report = final_fix.build(
                pages,
                chooser_cell=CELL,
                threads=1,
                minima={**MINIMA, "text": 10**6},  # no text fix count reaches this: the door opens
                min_group=50,
                min_positive=10,
                floor=0.9,
                sensor=0.05,
                promise="no_worse",
            )
        self.assertTrue(
            (sweep.call_args_list[0].kwargs["safe_outcome"] == 77).all()
        )  # what the promise says is what is swept
        fix = model.fix
        self.assertEqual((fix["safe_promise"], fix["t5_floor"]), ("no_worse", 0.99))
        self.assertIsNone(fix["t_safe"]["text"])
        groups = [k for k in fix["t_safe"] if "|" in k]
        self.assertTrue(groups)
        self.assertTrue(all(k.split("|")[0] in fix["m"] for k in groups))
        self.assertFalse([k for k in fix["m"] if "|" in k])
        block = fix["by_language"]["es"]
        self.assertEqual(block["t_safe"], fix["t_safe"])
        self.assertFalse([k for k in block["m"] if "|" in k])
        self.assertEqual(report["delivered"], {k: v for k, v in report["cross_fitted"]["thresholds"].items()})


class PrecisionByGroupTest(unittest.TestCase):
    def test_the_precision_of_the_k_best_of_each_group_and_the_most_it_could_apply_at_a_floor(self):
        table, raw = mixed_table()
        choice = operating.choose(table, operating.calibrate_rows_by_op(table, raw, False, 300, 50))
        out = first_result.precision_by_group(table, choice)
        accent = out["text|accent"]
        self.assertEqual(accent["sites"], 400)
        self.assertGreaterEqual(accent["precision_at"]["300"], 0.99)
        self.assertGreaterEqual(accent["largest_k_at_0.99"], 300)
        self.assertLess(out["text|letters"]["precision_at"]["300"], 0.9)
        self.assertIn("join|structure", out)
        self.assertNotIn("800", accent["precision_at"])  # no more than 400 sites


if __name__ == "__main__":
    unittest.main()
