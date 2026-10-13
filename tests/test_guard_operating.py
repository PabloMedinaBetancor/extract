import unittest

import numpy as np

from bench.guard import operating
from bench.guard.operating import BROKEN, FIXED, STILL_WRONG
from bench.guard.pairs import Table
from extract.guard.candidates import Candidate
from extract.guard.features import FEATURES, INDEX, KINDS


def table_of(x, y, site, docs, found):
    """A table of one block per site where `site` says which site each row belongs to."""
    n = len(docs)
    starts = np.array([int(np.argmax(site == s)) for s in range(len(docs))]) if len(site) else np.zeros(0, int)
    blank = np.array([""] * n, dtype=object)
    return Table(
        x,
        y,
        site,
        starts,
        np.zeros(n, int),
        np.zeros(n, int),
        docs,
        np.zeros(n, bool),
        np.ones(n, bool),
        found,
        blank,
        blank,
        blank,
    )


def outcomes(spec):
    codes = {"f": FIXED, "b": BROKEN, "w": STILL_WRONG}
    return np.array([codes[c] for c in spec], dtype=np.int8)


class SafeThresholdTest(unittest.TestCase):
    def test_G21_the_cut_is_before_the_first_failure_past_the_minimum_count(self):
        p = np.linspace(0.999, 0.5, 400)
        spec = "f" * 150 + "b" + "f" * 249
        threshold = operating.safe_threshold(p, outcomes(spec), 0.995, 100)
        self.assertAlmostEqual(threshold, p[149])
        self.assertEqual(int((p >= threshold).sum()), 150)

    def test_G21_a_lucky_dip_lower_down_is_never_picked(self):
        p = np.linspace(0.999, 0.5, 2000)
        spec = "f" * 120 + "bb" + "f" * 1878
        threshold = operating.safe_threshold(p, outcomes(spec), 0.995, 100)
        self.assertEqual(int((p >= threshold).sum()), 120)

    def test_G21_a_failure_before_the_minimum_count_leaves_the_operation_off(self):
        p = np.linspace(0.999, 0.5, 400)
        self.assertIsNone(operating.safe_threshold(p, outcomes("f" * 50 + "b" + "f" * 349), 0.995, 100))

    def test_G21_a_changed_word_that_stays_wrong_is_a_miss(self):
        p = np.linspace(0.999, 0.5, 400)
        spec = "f" * 150 + "w" + "f" * 249
        self.assertEqual(int((p >= operating.safe_threshold(p, outcomes(spec), 0.995, 100)).sum()), 150)

    def test_G21_equal_scores_enter_together(self):
        p = np.array([0.9] * 3 + [0.8] * 3)
        threshold = operating.safe_threshold(p, outcomes("fffbff"), 0.6, 1)
        self.assertEqual(threshold, 0.8)

    def test_G21_without_a_failure_every_site_is_applied(self):
        p = np.linspace(0.99, 0.6, 300)
        self.assertAlmostEqual(operating.safe_threshold(p, outcomes("f" * 300), 0.995, 100), 0.6)

    def test_G21_sites_with_no_candidate_are_ignored(self):
        p = np.array([-np.inf, 0.99, 0.98])
        self.assertEqual(operating.safe_threshold(p, outcomes("wff"), 0.995, 1), 0.98)


class MoreMarginTest(unittest.TestCase):
    def test_G22_the_margin_stops_where_broken_words_pass_five_percent_of_the_fixed_ones(self):
        spec = "f" * 100 + "b" + "f" * 10 + "b" * 10 + "f" * 200
        margin = np.linspace(0.9, 0.0, len(spec))
        m = operating.more_margin(margin, outcomes(spec))
        taken = outcomes(spec)[margin >= m]
        self.assertEqual((len(taken), int((taken == FIXED).sum()), int((taken == BROKEN).sum())), (115, 110, 5))

    def test_G22_an_operation_that_breaks_more_than_it_fixes_has_no_margin(self):
        self.assertIsNone(operating.more_margin(np.array([0.9, 0.8]), outcomes("bf")))


class ChooseTest(unittest.TestCase):
    def table(self, labels, token_right):
        n = len(labels)
        found = [[Candidate("a", "keep", "keep", ("keep",)), Candidate("b", "text", "letters", ("dictionary",))]]
        table = table_of(
            np.full((n, len(FEATURES)), np.nan, dtype=np.float32),
            np.array(labels, dtype=np.uint8),
            np.zeros(n, int),
            np.array(["d"], dtype=object),
            found,
        )
        table.token_right = np.array([token_right], dtype=bool)
        return table

    def test_G21_a_site_where_the_token_as_read_is_the_likeliest_breaks_when_its_best_candidate_is_applied(self):
        table = self.table([1, 0], True)
        choice = operating.choose(table, np.array([0.9, 0.05]))
        self.assertEqual((int(choice.index[0]), int(choice.outcome[0])), (1, BROKEN))
        self.assertAlmostEqual(float(choice.p_fix[0]), 0.05)
        self.assertAlmostEqual(float(choice.p_keep[0]), 0.9)

    def test_G21_a_wrong_token_whose_best_candidate_is_not_the_truth_is_changed_still_wrong(self):
        choice = operating.choose(self.table([0, 0], False), np.array([0.3, 0.6]))
        self.assertEqual(int(choice.outcome[0]), STILL_WRONG)


class IntervalsTest(unittest.TestCase):
    def test_G17_the_bootstrap_resamples_documents_not_rows(self):
        n = 40
        outcome = np.array([FIXED, BROKEN, FIXED, STILL_WRONG] * (n // 4), dtype=np.int8)
        choice = operating.Choice(np.ones(n, int), np.linspace(1, 0.5, n), np.zeros(n), outcome)
        mask = np.ones(n, dtype=bool)
        one_document = operating.with_intervals(choice, mask, np.array(["d1"] * n, dtype=object), resamples=200)
        for name in ("fixed", "net", "precision"):
            low_high = one_document["intervals"][name]
            self.assertEqual(low_high["low"], low_high["high"], name)
        two_documents = operating.with_intervals(
            choice, mask, np.array(["d1"] * 5 + ["d2"] * (n - 5), dtype=object), resamples=200
        )
        self.assertLess(two_documents["intervals"]["fixed"]["low"], two_documents["intervals"]["fixed"]["high"])

    def test_G21_the_constants_of_the_design_are_the_numbers_it_states(self):
        self.assertEqual(operating.SAFE_FLOOR, 0.995)
        self.assertEqual((operating.MIN_GROUP, operating.MIN_POSITIVE), (500, 100))
        self.assertEqual(operating.MORE_BROKEN_SHARE, 0.05)

    def test_G22_a_margin_that_fixes_nothing_has_no_point_even_when_it_breaks_nothing(self):
        self.assertIsNone(operating.more_margin(np.array([0.9, 0.8]), outcomes("ww")))

    def test_G20_a_group_one_row_short_of_the_minimum_does_not_get_its_own_table(self):
        n = 499
        x = np.full((n + 600, len(FEATURES)), np.nan, dtype=np.float32)
        x[:n, INDEX["kind"]] = KINDS.index("accent")
        x[n:, INDEX["kind"]] = KINDS.index("letters")
        rng = np.random.default_rng(3)
        raw = rng.normal(size=n + 600)
        y = np.where(np.arange(n + 600) < n, raw > 0, raw > 1.2).astype(np.uint8)
        table = table_of(x, y, np.arange(n + 600), np.array(["d"] * (n + 600), dtype=object), [[]] * (n + 600))
        pooled = operating.calibrate_rows(table, raw, min_group=10**9)
        small = operating.calibrate_rows(table, raw)
        np.testing.assert_allclose(pooled[:n], small[:n])


class CalibrationTest(unittest.TestCase):
    def table(self, kinds, y, docs):
        n = len(y)
        x = np.full((n, len(FEATURES)), np.nan, dtype=np.float32)
        x[:, INDEX["kind"]] = [KINDS.index(k) for k in kinds]
        return table_of(x, np.array(y, dtype=np.uint8), np.arange(n), np.array(docs, dtype=object), [[]] * n)

    def test_G20_a_kind_with_enough_rows_has_its_own_table_and_the_others_share_the_pooled_one(self):
        rng = np.random.default_rng(1)
        n = 1200
        kinds = ["accent"] * 700 + ["letters"] * 500
        raw = rng.normal(size=n)
        y = np.where(np.array(kinds) == "accent", raw > 0, raw > 1).astype(int)
        probability = operating.calibrate_rows(self.table(kinds, y, [f"d{i % 10}" for i in range(n)]), raw)
        accent, letters = np.array(kinds) == "accent", np.array(kinds) == "letters"
        self.assertGreater(probability[accent & (raw > 0.2)].mean(), probability[letters & (raw > 0.2)].mean())

    def test_G20_a_cross_fitted_probability_does_not_use_the_row_that_it_calibrates(self):
        n = 400
        docs = [f"d{i % 2}" for i in range(n)]
        raw = np.linspace(-1, 1, n)
        y = np.array([(raw[i] > 0) if docs[i] == "d0" else (raw[i] < 0) for i in range(n)], dtype=int)
        table = self.table(["accent"] * n, y, docs)
        own = operating.calibrate_rows(table, raw)
        crossed = operating.calibrate_rows(table, raw, cross_fit=True)
        self.assertFalse(np.allclose(own, crossed))
        high = (raw > 0.8) & (np.array(docs) == "d1")
        self.assertEqual(crossed[high].mean(), 1.0)
        self.assertLess(own[high].mean(), 0.9)


class ChoiceTest(unittest.TestCase):
    def test_G21_the_best_candidate_of_a_site_and_what_applying_it_does(self):
        found = [
            [
                Candidate("x", "keep", "keep", ("keep",)),
                Candidate("a", "text", "letters", ("dictionary",)),
                Candidate("b", "text", "letters", ("dictionary",)),
            ]
        ]
        table = table_of(
            np.zeros((3, len(FEATURES)), np.float32),
            np.array([0, 0, 1], dtype=np.uint8),
            np.zeros(3, int),
            np.array(["d"], dtype=object),
            found,
        )
        choice = operating.choose(table, np.array([0.2, 0.3, 0.7]))
        self.assertEqual((choice.index[0], choice.p_fix[0], choice.p_keep[0]), (2, 0.7, 0.2))
        self.assertEqual(choice.outcome[0], FIXED)
        self.assertAlmostEqual(choice.margin[0], 0.5)
