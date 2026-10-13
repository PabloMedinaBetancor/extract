"""Calibration by (op, kind), the thresholds of each op, and the overlaps."""

import unittest

import numpy as np

from bench.guard import calibrate, operating
from bench.guard.operating import BROKEN, FIXED, Choice
from bench.guard.pairs import Table
from extract.guard.features import FEATURES, INDEX, KINDS, OPS


def table_of(sites):
    """`sites` is a list of (op, label of the candidate, page, span, doc): each site has its KEEP row and one candidate."""
    n = len(sites)
    x = np.zeros((2 * n, len(FEATURES)), dtype=np.float32)
    y = np.zeros(2 * n, dtype=np.uint8)
    for s, (op, label, _, _, _, *kind) in enumerate(sites):
        x[2 * s, INDEX["kind"]] = KINDS.index("keep")
        x[2 * s + 1, INDEX["kind"]] = KINDS.index(kind[0] if kind else "letters")
        x[2 * s : 2 * s + 2, INDEX["op"]] = OPS.index(op)
        y[2 * s + 1] = label
    blank = np.array([""] * n, dtype=object)
    return Table(
        x=x,
        y=y,
        site=np.repeat(np.arange(n), 2),
        starts=np.arange(0, 2 * n, 2),
        page=np.array([s[2] for s in sites]),
        word=np.array([s[3][0] for s in sites]),
        doc=np.array([s[4] for s in sites], dtype=object),
        token_right=np.zeros(n, bool),
        reachable=np.ones(n, bool),
        found=[[None, None] for _ in sites],
        token=blank,
        truth=blank,
        label_kind=blank,
        family=np.array([s[0] for s in sites], dtype=object),
        span=[s[3] for s in sites],
    )


def raw_of(table, rng, noise=0.05):
    """A raw score that follows the label: high for a true candidate, low for the rest."""
    return np.where(table.y == 1, 0.9, 0.1) + rng.normal(0, noise, len(table.y))


class CalibrationByOpTest(unittest.TestCase):
    def test_B4_each_op_is_calibrated_by_its_own_table_when_it_has_enough_true_candidates(self):
        rng = np.random.default_rng(1)
        # the same raw score means 90 % in the text sites and 30 % in the join sites
        sites = [("text", int(rng.random() < 0.9), 0, (i,), f"d{i % 20}") for i in range(600)]
        sites += [("join", int(rng.random() < 0.3), 0, (i,), f"d{i % 20}") for i in range(600)]
        table = table_of(sites)
        raw = np.where(np.arange(len(table.y)) % 2 == 1, 0.5, 0.1) + rng.normal(0, 0.01, len(table.y))
        probability = operating.calibrate_rows_by_op(table, raw, min_group=100, min_positive=20)
        candidate = np.arange(len(table.y)) % 2 == 1
        text = candidate & (table.family[table.site] == "text")
        join = candidate & (table.family[table.site] == "join")
        self.assertAlmostEqual(float(probability[text].mean()), 0.9, delta=0.06)
        self.assertAlmostEqual(float(probability[join].mean()), 0.3, delta=0.06)
        pooled = operating.calibrate_rows(table, raw, min_group=100, min_positive=20)
        self.assertGreater(abs(float(pooled[join].mean()) - 0.3), 0.1)

    def test_B4_an_op_with_too_few_true_candidates_falls_back_to_the_pooled_table(self):
        rng = np.random.default_rng(2)
        sites = [("text", int(rng.random() < 0.5), 0, (i,), f"d{i % 10}") for i in range(400)]
        sites += [("drop", 0, 0, (i,), f"d{i % 10}") for i in range(30)]
        table = table_of(sites)
        raw = raw_of(table, rng)
        probability = operating.calibrate_rows_by_op(table, raw, min_group=100, min_positive=20)
        drop_rows = np.nonzero(table.family[table.site] == "drop")[0]
        pooled = calibrate.apply_table(calibrate.fit_isotonic(raw, table.y, min_words=1), raw)
        self.assertTrue(np.allclose(probability[drop_rows], pooled[drop_rows]))

    def test_B4_inside_an_op_each_kind_with_enough_rows_has_its_own_table(self):
        rng = np.random.default_rng(4)
        sites = [("text", int(rng.random() < 0.3), 0, (i,), f"d{i % 20}", "letters") for i in range(500)]
        sites += [("text", int(rng.random() < 0.9), 0, (i,), f"d{i % 20}", "accent") for i in range(500)]
        table = table_of(sites)
        raw = np.where(np.arange(len(table.y)) % 2 == 1, 0.5, 0.1) + rng.normal(0, 0.01, len(table.y))
        probability = operating.calibrate_rows_by_op(table, raw, min_group=100, min_positive=20)
        candidate = np.arange(len(table.y)) % 2 == 1
        kind = table.x[:, INDEX["kind"]].astype(int)
        self.assertAlmostEqual(float(probability[candidate & (kind == KINDS.index("letters"))].mean()), 0.3, delta=0.07)
        self.assertAlmostEqual(float(probability[candidate & (kind == KINDS.index("accent"))].mean()), 0.9, delta=0.07)

    def test_B4_cross_fitting_calibrates_each_half_with_the_tables_of_the_other(self):
        rng = np.random.default_rng(6)
        sites = [("text", int(rng.random() < 0.5), 0, (i,), f"d{i % 10}") for i in range(600)]
        table = table_of(sites)
        raw = raw_of(table, rng, 0.3)
        crossed = operating.calibrate_rows_by_op(table, raw, cross_fit=True, min_group=50, min_positive=10)
        docs = table.doc[table.site]
        first = np.isin(docs, np.unique(docs)[::2])
        kind = table.x[:, INDEX["kind"]].astype(int)
        op = table.family[table.site]
        expected = operating._apply_by_op(
            raw[~first], table.y[~first], op[~first], kind[~first], raw[first], op[first], kind[first], 50, 10
        )
        self.assertTrue(np.allclose(crossed[first], expected))

    def test_B4_cross_fitting_never_calibrates_a_row_with_a_table_it_helped_fit(self):
        rng = np.random.default_rng(3)
        sites = [("text", int(rng.random() < 0.5), 0, (i,), f"d{i % 10}") for i in range(600)]
        table = table_of(sites)
        raw = raw_of(table, rng, 0.3)
        plain = operating.calibrate_rows_by_op(table, raw, min_group=50, min_positive=10)
        crossed = operating.calibrate_rows_by_op(table, raw, cross_fit=True, min_group=50, min_positive=10)
        self.assertFalse(np.allclose(plain, crossed))
        self.assertEqual(len(crossed), len(raw))


def choice_of(p_fix, outcome):
    n = len(p_fix)
    return Choice(np.ones(n, dtype=np.int64), np.array(p_fix, float), np.full(n, 0.1), np.array(outcome, dtype=np.int8))


class ThresholdsByOpTest(unittest.TestCase):
    def test_B4_each_op_has_its_own_minimum_count_and_an_op_that_cannot_meet_it_is_off(self):
        text = [(0.99 - i * 1e-4, FIXED) for i in range(400)]
        join = [(0.99 - i * 1e-4, FIXED) for i in range(60)]
        ops = np.array(["text"] * 400 + ["join"] * 60, dtype=object)
        choice = choice_of([p for p, _ in text + join], [o for _, o in text + join])
        points = operating.thresholds_by_op(choice, ops)
        self.assertIsNotNone(points["text"]["t_safe"])
        self.assertIsNone(points["join"]["t_safe"])  # 60 applied are under the 100 a structure op needs

    def test_B4_a_structure_op_with_a_hundred_applied_passes_where_text_with_a_hundred_does_not(self):
        sites = [(0.99 - i * 1e-4, FIXED) for i in range(100)]
        choice = choice_of([p for p, _ in sites], [o for _, o in sites])
        self.assertIsNotNone(
            operating.thresholds_by_op(choice, np.array(["segment"] * 100, dtype=object))["segment"]["t_safe"]
        )
        self.assertIsNone(operating.thresholds_by_op(choice, np.array(["text"] * 100, dtype=object))["text"]["t_safe"])

    def test_B4_an_op_that_breaks_the_floor_early_does_not_borrow_the_precision_of_another(self):
        good = [(0.99 - i * 1e-4, FIXED) for i in range(400)]
        bad = [(0.99 - i * 1e-4, BROKEN if i % 2 else FIXED) for i in range(150)]
        ops = np.array(["text"] * 400 + ["join"] * 150, dtype=object)
        choice = choice_of([p for p, _ in good + bad], [o for _, o in good + bad])
        points = operating.thresholds_by_op(choice, ops)
        self.assertIsNotNone(points["text"]["t_safe"])
        self.assertIsNone(points["join"]["t_safe"])

    def test_B4_each_op_applies_against_its_own_threshold_and_an_op_that_is_off_applies_nothing(self):
        choice = choice_of([0.9, 0.9, 0.9, 0.9], [FIXED] * 4)
        ops = np.array(["text", "text", "join", "drop"], dtype=object)
        mask = operating.applied_by_op(
            choice,
            ops,
            "safe",
            {
                "text": {"t_safe": 0.8, "m": None},
                "join": {"t_safe": None, "m": None},
                "drop": {"t_safe": 0.95, "m": None},
            },
        )
        self.assertEqual(mask.tolist(), [True, True, False, False])
        more = operating.applied_by_op(choice, ops, "more", {"text": {"t_safe": None, "m": 0.5}})
        self.assertEqual(more.tolist(), [True, True, False, False])


class OverlapTest(unittest.TestCase):
    def test_B4_of_two_applied_sites_that_share_a_word_the_larger_gain_stays(self):
        table = table_of(
            [
                ("text", 1, 0, (3,), "d1"),
                ("join", 1, 0, (2, 3), "d1"),
                ("text", 1, 0, (7,), "d1"),
                ("text", 1, 1, (3,), "d2"),
            ]
        )
        choice = Choice(
            np.ones(4, dtype=np.int64),
            np.array([0.9, 0.95, 0.9, 0.9]),
            np.array([0.1, 0.05, 0.1, 0.1]),
            np.full(4, FIXED, dtype=np.int8),
        )
        mask = operating.without_overlaps(table, choice, np.ones(4, dtype=bool))
        self.assertEqual(mask.tolist(), [False, True, True, True])

    def test_B4_the_gain_that_decides_is_the_margin_not_p_fix(self):
        table = table_of([("text", 1, 0, (3,), "d1"), ("join", 1, 0, (2, 3), "d1")])
        choice = Choice(
            np.ones(2, dtype=np.int64), np.array([0.95, 0.90]), np.array([0.85, 0.10]), np.full(2, FIXED, dtype=np.int8)
        )
        self.assertEqual(operating.without_overlaps(table, choice, np.ones(2, dtype=bool)).tolist(), [False, True])

    def test_B4_a_tie_in_gain_resolves_the_same_way_every_time(self):
        table = table_of([("text", 1, 0, (3,), "d1"), ("join", 1, 0, (2, 3), "d1")])
        choice = Choice(
            np.ones(2, dtype=np.int64), np.array([0.9, 0.9]), np.array([0.1, 0.1]), np.full(2, FIXED, dtype=np.int8)
        )
        runs = {tuple(operating.without_overlaps(table, choice, np.ones(2, dtype=bool)).tolist()) for _ in range(5)}
        self.assertEqual(len(runs), 1)

    def test_B4_a_site_that_is_not_applied_never_blocks_another(self):
        table = table_of([("text", 1, 0, (3,), "d1"), ("join", 1, 0, (2, 3), "d1")])
        choice = Choice(
            np.ones(2, dtype=np.int64), np.array([0.9, 0.99]), np.array([0.1, 0.0]), np.full(2, FIXED, dtype=np.int8)
        )
        mask = operating.without_overlaps(table, choice, np.array([True, False]))
        self.assertEqual(mask.tolist(), [True, False])


class ReportByOpTest(unittest.TestCase):
    def test_B4_the_report_has_every_op_the_total_and_the_overlaps_resolved(self):
        rng = np.random.default_rng(5)
        sites = [("text", 1, i // 3, (i,), f"d{i % 30}") for i in range(900)]
        sites += [("join", 1, i // 3, (i, i + 1), f"d{i % 30}") for i in range(200)]
        table = table_of(sites)
        raw = raw_of(table, rng, 0.02)
        report = operating.report_by_op(table, raw, cross_fit=False, resamples=20)
        self.assertEqual(set(report["thresholds"]), {"text", "join"})
        self.assertGreater(report["safe"]["total"]["applied"], 0)
        by_op = sum(report["safe"][op]["applied"] for op in ("text", "join"))
        self.assertEqual(by_op, report["safe"]["total"]["applied"])
        self.assertEqual(report["safe"]["total"]["broken"], 0)


class EmptyGroupTest(unittest.TestCase):
    def test_a_group_with_no_site_that_safe_can_apply_has_no_threshold_instead_of_failing(self):
        # found by g28 (a source dropped from the pool left a group with only -inf scores)
        nothing = np.full(5, -np.inf)
        outcome = np.full(5, operating.FIXED, dtype=np.int8)
        self.assertIsNone(operating.safe_threshold(nothing, outcome, 0.995, 1))
        self.assertIsNone(operating.more_margin(nothing, outcome))
        self.assertIsNone(operating.safe_threshold(np.array([]), np.array([], dtype=np.int8), 0.995, 1))


if __name__ == "__main__":
    unittest.main()
