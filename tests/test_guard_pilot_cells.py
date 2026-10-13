"""The pilot's analysis : the reach of a variant, the gain with its interval, the cost files and the rule that
chooses the reader."""

import dataclasses
import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bench.guard import pilot_cells as pc
from extract.engine.cache import reading_from_dict, rereads_to_dict
from tests.guard_pages import labelled_page, record
from tests.guard_words import Reading

FAR = ("xqzw", "obra")  # a token the dictionary cannot bring back to its truth: only a reading can


def page_with(readings=None, tokens=("la", "xqzw", "casa"), truth=("la", "obra", "casa")):
    extras = {"1": {"readings": tuple(readings)}} if readings else {}
    return labelled_page(list(tokens), [(t, [i], None) for i, t in enumerate(truth)], **extras)


class VariantTest(unittest.TestCase):
    def test_a_reading_of_the_right_text_brings_a_wrong_word_into_the_pool_only_with_the_readings(self):
        p = page_with([Reading("deep", "obra", rank=0)])
        self.assertFalse(pc.reach_vector([pc.without_readings(p)]).any())
        self.assertTrue(pc.reach_vector([p]).all())

    def test_the_budget_hides_the_units_of_a_rank_at_or_over_it(self):
        p = page_with([Reading("deep", "obra", rank=3)])
        self.assertFalse(pc.reach_vector([pc.with_readings(p, 3, None)]).any())
        self.assertTrue(pc.reach_vector([pc.with_readings(p, 4, None)]).all())

    def test_a_reader_alone_keeps_only_its_own_readings(self):
        p = page_with([Reading("latin", "obrx", rank=0), Reading("deep", "obra", rank=0)])
        self.assertTrue(pc.reach_vector([pc.with_readings(p, 8, frozenset({"deep"}))]).all())
        self.assertFalse(pc.reach_vector([pc.with_readings(p, 8, frozenset({"latin"}))]).any())

    def test_the_rows_are_the_wrong_components_weighted_and_the_plain_pairs_are_marked(self):
        rows = pc.wrong_rows([page_with()])
        self.assertEqual((rows.weight, rows.family, rows.plain), ([1], ["replace"], [True]))

    def test_a_split_is_reached_by_a_join_reading_of_its_two_words(self):
        tokens = ["la", "orde", "nanza", "del"]
        truth = [("la", [0], None), ("ordenanza", [1, 2], None), ("del", [3], None)]
        p = labelled_page(tokens, truth)
        self.assertTrue(pc.reach_vector([p]).all())  # the dictionary alone joins it
        self.assertEqual(pc.wrong_rows([p]).family, ["join"])

    def test_the_harm_counts_the_right_words_a_reading_contradicts_and_the_wrong_words_it_fixes(self):
        p = labelled_page(
            ["la", "xqzw", "casa", "xqzv"],
            [("la", [0], None), ("obra", [1], None), ("casa", [2], None), ("plano", [3], None)],
            **{
                "0": {"readings": (Reading("deep", "lo", rank=0),)},
                "1": {"readings": (Reading("deep", "obra", rank=0),)},
                "2": {"readings": (Reading("deep", "casa", rank=0),)},
                "3": {"readings": (Reading("deep", "plana", rank=0),)},
            },
        )
        counts = pc.harm_counts([p])
        self.assertEqual((counts["right_read"], counts["harmed"], counts["wrong_read"], counts["fixed"]), (2, 1, 2, 1))
        self.assertEqual(counts["changed"], 3)
        self.assertEqual(counts["changed_right"], 1)

    def test_a_reading_aligned_one_to_one_counts_as_a_reading_of_the_word_not_as_a_join(self):
        p = labelled_page(
            ["la", "xqzw", "casa"],
            [("la", [0], None), ("obra", [1], None), ("casa", [2], None)],
            **{"1": {"readings": (Reading("deep", "obra", rank=0, group=("p1w1",)),)}},
        )
        self.assertTrue(pc.reach_vector([p]).all())
        self.assertEqual(pc.harm_counts([p])["fixed"], 1)

    def test_the_plain_wrong_words_out_of_the_pool_are_counted_by_cause(self):
        far = page_with()  # xqzw for obra: a dictionary word four edits away
        outside = labelled_page(["la", "zzqyw", "casa"], [("la", [0], None), ("zzqyx", [1], None), ("casa", [2], None)])
        pages = [dataclasses.replace(far, key="k1"), dataclasses.replace(outside, doc="d2", key="k2")]
        rows = pc.wrong_rows(pages)
        causes = pc.unreached_causes(pages, pc.reach_vector(pages), rows)
        self.assertEqual(causes, {"farther_than_3_edits": 1, "truth_outside_the_dictionary": 1})
        with_reading = [dataclasses.replace(page_with([Reading("deep", "obra", rank=0)]), key="k1"), pages[1]]
        self.assertEqual(
            pc.unreached_causes(with_reading, pc.reach_vector(with_reading), rows), {"truth_outside_the_dictionary": 1}
        )

    def test_H2_a_reading_of_a_join_or_a_cut_is_not_counted_as_a_reading_of_the_word(self):
        p = labelled_page(
            ["la", "xqzw", "casa"],
            [("la", [0], None), ("obra", [1], None), ("casa", [2], None)],
            **{
                "0": {"readings": (Reading("deep", "lacasa", rank=0, group=("p1w0", "p1w1")),)},
                "1": {"readings": (Reading("deep", "ob ra", rank=0, group=("p1w1",)),)},
            },
        )
        counts = pc.harm_counts([p])
        self.assertEqual((counts["right_read"], counts["harmed"], counts["changed"]), (0, 0, 0))
        self.assertEqual(counts["right_unread"] + counts["wrong_unread"], 3)

    def test_the_report_gives_the_gain_over_the_base_and_net100(self):
        p = page_with([Reading("deep", "obra", rank=0)])
        base = pc.without_readings(p)
        rows = pc.wrong_rows([base]).array()
        report = pc.variant_report(pc.reach_vector([base]), pc.reach_vector([p]), pc.harm_counts([p]), rows, 50)
        self.assertEqual((report["base"], report["with_readings"], report["gain"]), (0.0, 1.0, 1.0))
        self.assertEqual(report["plain_wrong_pairs"]["with_readings"], 1.0)
        self.assertEqual(report["harm"]["net100"], 100.0)


def rows_of(n_docs=30, per_doc=4):
    docs = np.repeat(np.arange(n_docs), per_doc)
    return np.ones(len(docs)), docs


def gain_with(share_of_rows, n, seed):
    rng = np.random.default_rng(seed)
    v = np.zeros(n)
    v[rng.permutation(n)[: int(share_of_rows * n)]] = 1.0
    return v


class ChooseReaderTest(unittest.TestCase):
    def setUp(self):
        self.weights, self.groups = rows_of()
        self.n = len(self.weights)
        self.cheap_ok = (1.0, 2.0, 1000.0)

    def option(self, cell, budget, gain, cost=None, **kw):
        mean, p95, rss = cost or self.cheap_ok
        return pc.Option(cell, budget, mean, p95, rss, gain, **kw)

    def choose(self, opts):
        return pc.choose_reader(opts, self.weights, self.groups, resamples=300)

    def test_B24_the_cell_with_the_clearly_larger_gain_at_the_cap_wins(self):
        big, small = np.ones(self.n), gain_with(0.2, self.n, 1)
        out = self.choose([self.option("L", 16, small), self.option("Z", 40, big, (1.5, 2.5, 1500.0))])
        self.assertEqual((out["decision"], out["cell"], out["budget"]), ("choice", "Z", 40))
        self.assertTrue(out["difference_first_minus_second"]["excludes_zero"])

    def test_B24_equal_gains_go_to_the_cheaper_in_seconds(self):
        gain = gain_with(0.5, self.n, 2)
        out = self.choose(
            [self.option("L", 16, gain, (1.5, 3.0, 1000.0)), self.option("Z", 40, gain, (0.8, 2.0, 1000.0))]
        )
        self.assertEqual(out["cell"], "Z")
        out = self.choose(
            [self.option("L", 16, gain, (0.8, 2.0, 1000.0)), self.option("Z", 40, gain, (1.5, 3.0, 1000.0))]
        )
        self.assertEqual(out["cell"], "L")
        self.assertIn("cheaper", out["reason"])

    def test_B24_equal_gains_and_equal_cost_go_to_z(self):
        gain = gain_with(0.5, self.n, 3)
        for order in ((("L", 16), ("Z", 40)), (("Z", 40), ("L", 16))):
            out = self.choose([self.option(cell, budget, gain, (1.0, 2.0, 1000.0)) for cell, budget in order])
            self.assertEqual(out["cell"], "Z")
            self.assertIn("Z", out["reason"])

    def test_B24_a_cost_within_the_tie_margin_is_still_a_tie(self):
        gain = gain_with(0.5, self.n, 4)
        out = self.choose(
            [self.option("L", 16, gain, (1.00, 2.0, 1000.0)), self.option("Z", 40, gain, (1.04, 2.0, 1000.0))]
        )
        self.assertEqual(out["cell"], "Z")
        out = self.choose(
            [self.option("L", 16, gain, (1.00, 2.0, 1000.0)), self.option("Z", 40, gain, (1.07, 2.0, 1000.0))]
        )
        self.assertEqual(out["cell"], "L")

    def test_B24_the_cap_of_a_cell_is_its_largest_feasible_budget(self):
        gain = gain_with(0.3, self.n, 5)
        opts = [
            self.option("L", 8, gain),
            self.option("L", 16, np.ones(self.n), (2.5, 4.0, 1000.0)),
            self.option("Z", 10, np.zeros(self.n)),
        ]
        out = self.choose(opts)
        self.assertEqual((out["cell"], out["budget"]), ("L", 8))
        self.assertEqual(out["caps"], {"L": 8, "Z": 10})

    def test_B24_with_two_feasible_budgets_the_cap_is_the_larger_one(self):
        opts = [
            self.option("L", 8, gain_with(0.2, self.n, 10), (0.5, 1.0, 900.0)),
            self.option("L", 16, np.ones(self.n), (1.2, 2.0, 1000.0)),
            self.option("Z", 10, gain_with(0.2, self.n, 11)),
        ]
        out = self.choose(opts)
        self.assertEqual((out["cell"], out["budget"], out["caps"]), ("L", 16, {"L": 16, "Z": 10}))

    def test_B24_each_limit_makes_an_option_infeasible_and_the_reason_is_named(self):
        gain = np.ones(self.n)
        cases = {
            "mean_seconds_over_R6": (1.61, 2.0, 1000.0),
            "p95_seconds_over_R6": (1.0, 3.51, 1000.0),
            "memory_over_R7": (1.0, 2.0, 1700.0),
        }
        for reason, cost in cases.items():
            opt = self.option("Z", 40, gain, cost)
            self.assertEqual(pc.infeasible_reasons(opt), [reason])
        self.assertEqual(pc.infeasible_reasons(self.option("Z", 40, gain, (1.6, 3.5, 1500.0))), [])
        self.assertEqual(pc.infeasible_reasons(self.option("Z", 40, gain, licence_ok=False)), ["licence_or_download"])
        unmeasured = pc.Option("Z", 40, None, None, None, gain)
        self.assertEqual(pc.infeasible_reasons(unmeasured), ["seconds_not_measured", "memory_not_measured"])
        no_memory = pc.Option("Z", 40, 1.0, 2.0, None, gain)
        self.assertEqual(pc.infeasible_reasons(no_memory), ["memory_not_measured"])

    def test_B24_when_nothing_fits_the_options_go_to_the_maintainer_with_their_costs(self):
        opts = [
            self.option("L", 16, np.ones(self.n), (2.0, 4.0, 1000.0)),
            self.option("Z", 40, gain_with(0.5, self.n, 6), (3.0, 5.0, 1000.0)),
        ]
        out = self.choose(opts)
        self.assertEqual(out["decision"], "options_for_maintainer")
        self.assertEqual([o["cell"] for o in out["options"]], ["L", "Z"])
        self.assertTrue(all(o["infeasible"] for o in out["options"]))

    def test_B24_a_single_feasible_cell_is_chosen_without_a_comparison(self):
        out = self.choose(
            [self.option("L", 16, np.ones(self.n), (2.0, 4.0, 1000.0)), self.option("Z", 40, np.zeros(self.n))]
        )
        self.assertEqual((out["cell"], out["reason"]), ("Z", "the only feasible cell"))

    def test_B24_the_choice_is_the_same_on_every_run(self):
        opts = [self.option("L", 16, gain_with(0.5, self.n, 7)), self.option("Z", 40, gain_with(0.5, self.n, 8))]
        self.assertEqual(self.choose(opts), self.choose(opts))

    def test_B24_a_ratio_is_not_the_rule_a_tiny_cheap_budget_does_not_beat_a_larger_gain(self):
        tiny = self.option("L", 4, gain_with(0.05, self.n, 9), (0.01, 0.02, 500.0))
        large = self.option("Z", 40, np.ones(self.n), (1.5, 3.0, 1500.0))
        out = self.choose([tiny, large])
        self.assertEqual(out["cell"], "Z")


class CostAndKneeTest(unittest.TestCase):
    def test_the_budgets_of_reader_speed_json_give_mean_and_p95_and_the_cell_memory(self):
        speed = {
            "cells": {
                "L": {
                    "peak_reader_rss_mb": 1043,
                    "budgets": {"4": {"mean": 1.12, "p95": 1.842}, "8": {"mean": 2.0, "p95": 3.4}},
                }
            }
        }
        self.assertEqual(pc.cost_of(speed, "L", 4, 16), {"mean_s": 1.12, "p95_s": 1.842, "rss_mb": 1043.0})
        self.assertEqual(pc.cost_of(speed, "L", 8, 16)["mean_s"], 2.0)
        self.assertIsNone(pc.cost_of(speed, "L", 16, 16))

    def test_the_cost_is_read_from_a_summary_a_budget_table_or_a_cell_at_budget_key(self):
        summary = {"seconds_per_page_mean": 1.2, "seconds_per_page_p95": 2.5, "peak_rss_mb": 900}
        expected = {"mean_s": 1.2, "p95_s": 2.5, "rss_mb": 900.0}
        self.assertEqual(pc.cost_of({"cells": {"L": summary}}, "L", 16, 16), expected)
        self.assertIsNone(pc.cost_of({"cells": {"L": summary}}, "L", 8, 16))
        self.assertEqual(pc.cost_of({"cells": {"L": {**summary, "budget": 8}}}, "L", 8, 16), expected)
        self.assertEqual(pc.cost_of({"cells": {"L": {"budgets": {"8": summary}}}}, "L", 8, 16), expected)
        self.assertEqual(pc.cost_of({"cells": {"L@8": summary}}, "L", 8, 16), expected)
        speed = {"pass2_seconds_mean": 3.0, "pass2_seconds_p95": 5.0, "peak_reader_rss_mb": 1000}
        self.assertEqual(pc.cost_of({"cells": {"L": speed}}, "L", 16, 16)["mean_s"], 3.0)
        self.assertIsNone(pc.cost_of({"cells": {}}, "Z", 40, 40))

    def test_the_knee_is_the_point_farthest_from_the_chord(self):
        self.assertEqual(pc.knee([(0.2, 0.30), (0.6, 0.55), (1.0, 0.60), (2.0, 0.62)]), 1)
        self.assertEqual(pc.knee([(1.0, 0.5), (2.0, 0.6)]), 1)

    def test_gate_p_passes_at_85_stops_under_80_and_says_so_between(self):
        self.assertEqual(
            [pc.gate_p(x) for x in (0.85, 0.8499, 0.80, 0.7999, None)],
            ["pass", "between_80_and_85", "between_80_and_85", "stop", "no_data"],
        )


class CostFromUnitsTest(unittest.TestCase):
    pages = [{"fixed_s": 0.3, "unit_s": [0.5, 0.4, 0.3, 0.2]}, {"fixed_s": 0.1, "unit_s": [0.6, 0.6, 0.6, 0.6]}]

    def test_the_cost_of_a_budget_is_the_fixed_part_plus_the_units_below_it(self):
        arms = {"cells": {"L": {"per_page": self.pages, "peak_rss_mb": 900}}}
        two = pc.cost_of(arms, "L", 2, 4)
        self.assertAlmostEqual(two["mean_s"], (0.3 + 0.9 + 0.1 + 1.2) / 2)
        self.assertEqual(two["rss_mb"], 900.0)
        self.assertAlmostEqual(pc.cost_of(arms, "L", 4, 4)["mean_s"], (0.3 + 1.4 + 0.1 + 2.4) / 2)
        self.assertLess(pc.cost_of(arms, "L", 1, 4)["mean_s"], two["mean_s"])

    def test_the_batched_figures_of_the_full_budget_scale_the_per_unit_times(self):
        arms = {"cells": {"L": {"per_page": self.pages, "seconds_per_page_mean": 4.0, "seconds_per_page_p95": 5.0}}}
        full = pc.cost_of(arms, "L", 4, 4)
        self.assertAlmostEqual(full["mean_s"], 4.0)
        self.assertAlmostEqual(full["p95_s"], 5.0)
        half = pc.cost_of(arms, "L", 2, 4)
        plain = pc.cost_of({"cells": {"L": {"per_page": self.pages}}}, "L", 2, 4)
        self.assertAlmostEqual(half["mean_s"] / plain["mean_s"], full["batch_scale"][0], places=2)
        self.assertGreater(full["batch_scale"][0], 1.0)

    def test_the_p95_comes_from_the_pages_of_that_budget(self):
        pages = [{"fixed_s": 0.0, "unit_s": [1.0, 1.0]} for _ in range(19)] + [{"fixed_s": 0.0, "unit_s": [1.0, 9.0]}]
        arms = {"cells": {"Z": {"per_page": pages}}}
        self.assertAlmostEqual(pc.cost_of(arms, "Z", 1, 2)["p95_s"], 1.0)
        self.assertAlmostEqual(pc.cost_of(arms, "Z", 2, 2)["p95_s"], 2.4)  # the 95th percentile, not the maximum (10)

    def test_a_cell_with_unit_times_makes_every_budget_choosable_not_only_the_largest(self):
        arms = {"cells": {"L": {"per_page": self.pages, "peak_rss_mb": 900}}}
        self.assertTrue(all(pc.cost_of(arms, "L", b, 16) for b in (1, 2, 3, 4)))

    def test_the_pilots_own_file_replaces_the_first_sample_for_the_same_cell_and_a_missing_one_is_skipped(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / "speed.json").write_text(json.dumps({"cells": {"L": {"pass2_seconds_mean": 3.0}, "Z": {"x": 1}}}))
            (folder / "arms.json").write_text(json.dumps({"cells": {"L": {"pass2_seconds_mean": 1.0}}}))
            arms = pc.load_arms(folder / "speed.json", folder / "none.json", folder / "arms.json")
        self.assertEqual(arms["L"]["pass2_seconds_mean"], 1.0)
        self.assertIn("Z", arms)

    def test_a_later_file_adds_to_a_cell_without_erasing_the_budgets_of_the_earlier_one(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            (folder / "speed.json").write_text(
                json.dumps(
                    {
                        "cells": {
                            "L": {
                                "peak_reader_rss_mb": 1043,
                                "budgets": {"4": {"mean": 1.1, "p95": 1.8}, "8": {"mean": 2.0, "p95": 3.4}},
                            }
                        }
                    }
                )
            )
            (folder / "arms.json").write_text(
                json.dumps(
                    {
                        "cells": {
                            "L": {
                                "seconds_per_page_mean": 2.7,
                                "seconds_per_page_p95": 4.7,
                                "pages": 60,
                                "budgets": {"8": {"mean": 1.9, "p95": 3.3}},
                            }
                        }
                    }
                )
            )
            arms = pc.load_arms(folder / "speed.json", folder / "arms.json")
        self.assertEqual(pc.cost_of(arms, "L", 4, 16)["mean_s"], 1.1)
        self.assertEqual(pc.cost_of(arms, "L", 8, 16)["mean_s"], 1.9)
        self.assertEqual(pc.cost_of(arms, "L", 4, 16)["rss_mb"], 1043.0)

    def test_the_reader_process_cap_is_the_one_of_the_reader_process(self):
        from extract.engine import readerproc

        self.assertEqual(pc.RSS_CAP_MB, readerproc.RSS_CAP_MB)


class LicenceTest(unittest.TestCase):
    good = {"licence": "Apache-2.0", "url": "https://x/m.onnx", "sha256": "ab" * 32}

    def test_the_licence_is_in_order_when_every_model_of_the_cell_is_permissive_with_url_and_sha256(self):
        bake = {"models": {n: self.good for n in (*pc.CHEAP_MODELS, *pc.DEEP_MODELS)}}
        self.assertTrue(pc.licence_in_order(bake, "L"))
        self.assertTrue(pc.licence_in_order(bake, "Z"))

    def test_a_missing_file_a_missing_model_a_closed_licence_or_a_missing_hash_is_not_in_order(self):
        base = {n: self.good for n in (*pc.CHEAP_MODELS, *pc.DEEP_MODELS)}
        self.assertFalse(pc.licence_in_order(None, "Z"))
        self.assertFalse(pc.licence_in_order({"models": {}}, "Z"))
        self.assertFalse(
            pc.licence_in_order({"models": {**base, "ppocrv6-medium": {**self.good, "licence": "CC-BY-NC-4.0"}}}, "Z")
        )
        self.assertFalse(pc.licence_in_order({"models": {**base, "ppocrv6-medium": {**self.good, "sha256": ""}}}, "Z"))
        self.assertFalse(
            pc.licence_in_order({"models": {**base, "latin-ppocrv5-mobile": {**self.good, "url": None}}}, "L")
        )
        self.assertTrue(
            pc.licence_in_order({"models": {**base, "latin-ppocrv5-mobile": {**self.good, "url": None}}}, "Z")
        )


class RecallWeightsAndHarmTest(unittest.TestCase):
    def test_P18_a_pair_on_a_token_that_is_not_plain_is_never_reached(self):
        p = labelled_page(
            ["la", "a1b", "casa"],
            [("la", [0], None), ("abc", [1], None), ("casa", [2], None)],
            **{"1": {"readings": (Reading("deep", "abc", rank=0),)}},
        )
        self.assertFalse(pc.reach_vector([p]).any())

    def test_P23_the_gain_weights_a_component_by_the_words_of_its_truth(self):
        tokens = ["la", "ordenanzaplan", "xqzw", "casa"]
        truth = [
            ("la", [0], None),
            ("ordenanza", [1], (0, 2)),
            ("plan", [1], (1, 2)),
            ("obra", [2], None),
            ("casa", [3], None),
        ]
        p = labelled_page(tokens, truth, **{"2": {"readings": (Reading("deep", "obra", rank=0),)}})
        base = pc.without_readings(p)
        rows = pc.wrong_rows([base])
        self.assertEqual(sorted(zip(rows.family, rows.weight, strict=True)), [("replace", 1), ("segment", 2)])
        arrays = rows.array()
        report = pc.variant_report(pc.reach_vector([base]), pc.reach_vector([p]), pc.harm_counts([p]), arrays, 20)
        # the segment is reached by the dictionary in both (weight 2 of 3), the pair only with the reading (1 of 3)
        self.assertAlmostEqual(report["base"], round(2 / 3, 4))
        self.assertEqual(report["with_readings"], 1.0)
        self.assertAlmostEqual(report["gain"], round(1 / 3, 4))

    def test_P23_the_published_gain_counts_a_gained_component_by_the_words_of_its_truth(self):
        # a merge (truth of two words) that only a reader's cut reaches, and a pair that only a reading reaches
        tokens = ["la", "zwhqvvgqhk", "xqzw", "casa"]
        truth = [
            ("la", [0], None),
            ("zwhq", [1], (0, 2)),
            ("vvgqhk", [1], (1, 2)),
            ("obra", [2], None),
            ("casa", [3], None),
        ]
        p = labelled_page(
            tokens,
            truth,
            **{
                "1": {"readings": (Reading("deep", "zwhq vvgqhk", rank=0, group=("p1w1",)),)},
                "2": {"readings": (Reading("deep", "obra", rank=0),)},
            },
        )
        base = pc.without_readings(p)
        self.assertFalse(pc.reach_vector([base]).any())
        arrays = pc.wrong_rows([base]).array()
        self.assertEqual(sorted(arrays["weight"].tolist()), [1.0, 2.0])
        report = pc.variant_report(pc.reach_vector([base]), pc.reach_vector([p]), pc.harm_counts([p]), arrays, 20)
        self.assertEqual((report["base"], report["with_readings"], report["gain"]), (0.0, 1.0, 1.0))
        unweighted = pc.variant_report(
            pc.reach_vector([base]), pc.reach_vector([p]), pc.harm_counts([p]), {**arrays, "weight": np.ones(2)}, 20
        )
        self.assertEqual(unweighted["gain"], 1.0)
        # only the merge gained: weighted 2 of 3 (the pair stays out), so a count of rows (1 of 2) would say 0.5
        only_merge = pc.reach_vector([p]) & (arrays["family"] == "segment")
        partial = pc.variant_report(pc.reach_vector([base]), only_merge, pc.harm_counts([p]), arrays, 20)
        self.assertAlmostEqual(partial["gain"], round(2 / 3, 4))

    def test_P27_net100_subtracts_the_right_words_a_reader_contradicts(self):
        p = labelled_page(
            ["la", "xqzw", "casa"],
            [("la", [0], None), ("obra", [1], None), ("casa", [2], None)],
            **{
                "0": {"readings": (Reading("deep", "lo", rank=0),)},
                "1": {"readings": (Reading("deep", "obra", rank=0),)},
            },
        )
        base = pc.without_readings(p)
        arrays = pc.wrong_rows([base]).array()
        report = pc.variant_report(pc.reach_vector([base]), pc.reach_vector([p]), pc.harm_counts([p]), arrays, 20)
        self.assertEqual(report["harm"]["right_words_contradicted"], 1)
        self.assertEqual(report["harm"]["net100"], 0.0)  # one fixed, one broken, one wrong word


class GateForTest(unittest.TestCase):
    def result(self, recall, decision="choice"):
        plain = {"n": 200, "base": 0.6, "with_readings": recall, "with_readings_ci95": [recall - 0.03, recall + 0.03]}
        choice = {"decision": decision, "cell": "Z", "budget": 16}
        if decision != "choice":
            choice = {
                "decision": decision,
                "options": [{"cell": "L", "budget": 8, "gain": 0.1}, {"cell": "Z", "budget": 16, "gain": 0.3}],
            }
        return {
            "choice": choice,
            "cells": {
                "Z": {"budgets": {"16": {"plain_wrong_pairs": plain}}},
                "L": {"budgets": {"8": {"plain_wrong_pairs": plain}}},
            },
        }

    def test_P_the_gate_is_the_one_of_the_chosen_option_with_n_interval_and_the_action_between_the_bars(self):
        for recall, verdict, action in (
            (0.9, "pass", None),
            (0.82, "between_80_and_85", pc.BETWEEN_ACTION),
            (0.7, "stop", "stop the build and tell the maintainer with the table"),
        ):
            gate = pc.gate_for(self.result(recall))
            self.assertEqual(
                (gate["option"], gate["cell"], gate["budget"], gate["decision"], gate["action"]),
                ("chosen", "Z", 16, verdict, action),
            )
            self.assertEqual(
                (gate["n_plain_wrong_words"], gate["recall_with_readings"], len(gate["recall_ci95"])), (200, recall, 2)
            )

    def test_P_when_nothing_fits_the_gate_is_read_on_the_option_with_the_larger_gain_and_says_so(self):
        gate = pc.gate_for(self.result(0.9, "options_for_maintainer"))
        self.assertEqual((gate["option"], gate["cell"], gate["budget"]), ("best_gain_not_chosen", "Z", 16))

    def test_P_the_interval_of_the_plain_recall_is_over_documents(self):
        pages = []
        for d in range(6):
            p = page_with([Reading("deep", "obra", rank=0)])
            pages.append(dataclasses.replace(p, doc=f"d{d}"))
        base = [pc.without_readings(p) for p in pages]
        arrays = pc.wrong_rows(base).array()
        report = pc.variant_report(pc.reach_vector(base), pc.reach_vector(pages), pc.harm_counts(pages), arrays, 30)
        plain = report["plain_wrong_pairs"]
        self.assertEqual((plain["n"], plain["with_readings"], plain["with_readings_ci95"]), (6, 1.0, [1.0, 1.0]))


SPEED_OF_OCR_APP = {
    "cells": {
        "L": {
            "peak_reader_rss_mb": 1043,
            "budgets": {
                "4": {"mean": 1.12, "p95": 1.842},
                "8": {"mean": 2.019, "p95": 3.465},
                "16": {"mean": 2.86, "p95": 4.778},
            },
        },
        "LD": {
            "peak_reader_rss_mb": 1034,
            "budgets": {
                "4": {"mean": 1.297, "p95": 2.004},
                "8": {"mean": 2.294, "p95": 3.797},
                "16": {"mean": 3.435, "p95": 6.001},
            },
        },
        "Z": {
            "peak_reader_rss_mb": 808,
            "budgets": {
                "5": {"mean": 0.793, "p95": 1.044},
                "10": {"mean": 1.269, "p95": 1.829},
                "20": {"mean": 1.645, "p95": 2.791},
                "40": {"mean": 2.192, "p95": 4.955},
            },
        },
        "ZC": {
            "peak_reader_rss_mb": 707,
            "budgets": {
                "5": {"mean": 0.602, "p95": 0.768},
                "10": {"mean": 0.896, "p95": 1.357},
                "20": {"mean": 1.218, "p95": 2.045},
                "40": {"mean": 1.561, "p95": 3.343},
            },
        },
    }
}


class IntegrationWithTheSpeedFileTest(unittest.TestCase):
    """The cost files as the speed measurement writes them (the quiet measures of 2026-10-02), read by `cost_of` for every cell and
    budget the pilot uses, and what the rule makes of them."""

    def check(self, arms):
        found = {}
        for cell, (unit, _) in pc.CELLS.items():
            for budget in pc.BUDGETS[unit]:
                cost = pc.cost_of(arms, cell, budget, max(pc.BUDGETS[unit]))
                self.assertIsNotNone(cost, f"{cell}@{budget}")
                self.assertTrue(all(cost[k] is not None for k in ("mean_s", "p95_s", "rss_mb")), f"{cell}@{budget}")
                found[(cell, budget)] = pc.infeasible_reasons(pc.Option(cell, budget, *cost.values(), np.zeros(1)))
        return found

    def test_every_cell_and_budget_of_the_pilot_has_a_cost_and_the_rule_sees_what_fits(self):
        found = self.check(SPEED_OF_OCR_APP)
        fits = sorted(k for k, why in found.items() if not why)
        self.assertEqual(
            fits, [("L", 4), ("LD", 4), ("Z", 5), ("Z", 10), ("ZC", 5), ("ZC", 10), ("ZC", 20), ("ZC", 40)]
        )
        self.assertEqual(found[("Z", 20)], ["mean_seconds_over_R6"])
        self.assertEqual(found[("L", 8)], ["mean_seconds_over_R6"])

    def test_the_versioned_speed_file_has_a_cost_for_every_cell_and_budget_once_ocr_app_has_merged_it(self):
        path = Path(__file__).resolve().parent.parent / "bench" / "results" / "reader-speed.json"
        arms = json.loads(path.read_text(encoding="utf-8"))
        if "budgets" not in arms["cells"]["L"]:
            self.skipTest("reader-speed.json is the first sample (no budgets yet)")
        self.check(arms)


SPEED_WITH_READERS = {
    "cells": {
        "L": {
            "readers": {
                "latin": {"mean": 1.0, "p95": 9.0},
                **{r: {"mean": 1.0, "p95": 1.0} for r in ("bin", "clahe", "wide")},
            },
            "peak_reader_rss_mb": 1000,
            "budgets": {"8": {"mean": 2.0, "p95": 3.0}, "4": {"mean": 1.0, "p95": 1.5}},
        },
        "LD": {
            "readers": {"deep": {"mean": 3.0}},
            "peak_reader_rss_mb": 900,
            "budgets": {"8": {"mean": 1.5, "p95": 2.0}},
        },
    }
}


class ReadersAblationTest(unittest.TestCase):
    def pages(self):
        cheap = page_with([Reading("latin", "obra", rank=0), Reading("bin", "obre", rank=0)])
        deep = page_with([Reading("deep", "obrx", rank=0)])
        return cheap, deep

    def test_the_cheap_and_the_deep_cell_of_a_unit_are_one_page_with_the_readings_of_both(self):
        cheap, deep = self.pages()
        merged = pc.merged_readings([cheap], [deep])[0]
        self.assertEqual([r.source for r in merged.words[1].readings], ["latin", "bin", "deep"])
        self.assertEqual(merged.words[0].readings, None)

    def test_the_seconds_of_a_set_of_readers_are_their_share_of_the_cell_that_runs_them_summed(self):
        cost = pc.estimated_cost(SPEED_WITH_READERS, "line", frozenset({"latin", "wide"}), 8, 8)
        self.assertEqual((cost["mean_s"], cost["p95_s"], cost["estimate"]), (1.0, 1.5, True))
        both = pc.estimated_cost(SPEED_WITH_READERS, "line", frozenset({"latin", "wide", "deep"}), 8, 8)
        self.assertEqual((both["mean_s"], both["p95_s"]), (2.5, 3.5))
        self.assertEqual(pc.estimated_cost(SPEED_WITH_READERS, "line", frozenset({"deep"}), 8, 8)["mean_s"], 1.5)
        self.assertIsNone(pc.estimated_cost(SPEED_WITH_READERS, "line", frozenset({"deep"}), 4, 8))
        self.assertIsNone(pc.estimated_cost(SPEED_WITH_READERS, "zone", frozenset({"latin"}), 8, 8))

    def test_the_seconds_of_a_set_come_from_each_reader_alone_scaled_to_the_units_of_the_budget(self):
        per_reader = {
            "latin": {"page_seconds": {"line": {"mean": 0.5, "p95": 1.0}}},
            "wide": {"page_seconds": {"line": {"mean": 0.8, "p95": 1.5}}},
        }
        speed = {"cells": {"L": {"budgets": {"4": {"units_per_page_mean": 4.0}, "16": {"units_per_page_mean": 10.0}}}}}
        cost = pc.measured_cost(per_reader, speed, "line", frozenset({"latin", "wide"}), 4)
        self.assertAlmostEqual(cost["mean_s"], 0.4 * 1.3)
        self.assertAlmostEqual(cost["p95_s"], 0.4 * 2.5)
        self.assertAlmostEqual(
            pc.measured_cost(per_reader, speed, "line", frozenset({"latin", "wide"}), 16)["mean_s"], 1.3
        )
        self.assertIsNone(pc.measured_cost(per_reader, speed, "line", frozenset({"deep"}), 4))
        self.assertIsNone(pc.measured_cost(per_reader, speed, "line", frozenset({"latin"}), 8))
        self.assertIsNone(pc.measured_cost(per_reader, speed, "zone", frozenset({"latin"}), 4))
        broken = {**per_reader, "wide": {"page_seconds": {"line": {"mean": 0.8}}}}  # no p95: no measure, not a crash
        self.assertIsNone(pc.measured_cost(broken, speed, "line", frozenset({"latin", "wide"}), 4))

    def test_the_ablation_prefers_the_measured_seconds_of_each_reader_to_the_share_estimate(self):
        cheap, deep = self.pages()
        base = pc.without_readings(cheap)
        arrays = pc.wrong_rows([base]).array()
        per_reader = {
            r: {"page_seconds": {"line": {"mean": 0.1, "p95": 0.2}}} for r in ("latin", "bin", "clahe", "wide", "deep")
        }
        speed = {**SPEED_WITH_READERS}
        speed["cells"] = {
            **speed["cells"],
            "L": {
                **speed["cells"]["L"],
                "budgets": {
                    "8": {"mean": 2.0, "p95": 3.0, "units_per_page_mean": 5.0},
                    "16": {"mean": 3.0, "p95": 4.0, "units_per_page_mean": 10.0},
                },
            },
        }
        out = pc.readers_ablation(
            pc.reach_vector([base]), arrays, {"L": [cheap], "LD": [deep]}, speed, {"line": 16}, 20, per_reader
        )
        cost = out["line"]["latin+wide"]["8"]["estimated_seconds"]
        self.assertEqual(cost["source"], "per_reader")
        self.assertAlmostEqual(cost["mean_s"], 0.5 * 0.2)

    def test_the_sets_are_the_ones_the_decisions_name(self):
        self.assertEqual(
            {k: set(v) for k, v in pc.SUBSETS.items()},
            {
                "latin": {"latin"},
                "latin+wide": {"latin", "wide"},
                "latin+wide+clahe": {"latin", "wide", "clahe"},
                "all_cheap": {"latin", "bin", "clahe", "wide"},
                "deep": {"deep"},
                "latin+wide+deep": {"latin", "wide", "deep"},
                "all_cheap+deep": {"latin", "bin", "clahe", "wide", "deep"},
            },
        )

    def test_a_cost_fits_r6_only_when_both_the_mean_and_the_p95_do(self):
        self.assertTrue(pc.fits_r6({"mean_s": 1.6, "p95_s": 3.5}))
        self.assertFalse(pc.fits_r6({"mean_s": 1.61, "p95_s": 3.0}))
        self.assertFalse(pc.fits_r6({"mean_s": 1.0, "p95_s": 3.51}))
        self.assertIsNone(pc.fits_r6(None))

    def test_each_set_of_readers_adds_only_what_its_own_readings_reach_with_its_harm_and_seconds(self):
        cheap, deep = self.pages()
        base = pc.without_readings(cheap)
        arrays = pc.wrong_rows([base]).array()
        by_cell = {"L": [cheap], "LD": [deep]}
        out = pc.readers_ablation(pc.reach_vector([base]), arrays, by_cell, SPEED_WITH_READERS, {"line": 8}, 20)
        table = out["line"]
        self.assertEqual(table["latin"]["8"]["gain"], 1.0)
        self.assertEqual(table["deep"]["8"]["gain"], 0.0)
        self.assertEqual(table["latin+wide+deep"]["8"]["gain"], 1.0)
        self.assertEqual(table["latin"]["8"]["estimated_seconds"]["mean_s"], 0.5)
        self.assertTrue(table["latin"]["8"]["fits_R6_estimated"])
        self.assertFalse(table["all_cheap+deep"]["8"]["fits_R6_estimated"])  # 2.0 + 1.5 = 3.5 s over the mean cap
        self.assertEqual(set(table["latin"]), {"4", "8"})  # budgets over what the engine stores are left out
        self.assertNotIn("zone", out)

    def test_the_bad_cheap_reader_shows_up_in_the_net100_of_the_set_that_holds_it(self):
        cheap = page_with([Reading("latin", "obra", rank=0), Reading("bin", "casx", rank=0)])
        # the second reading of the right word `casa` by bin is wrong: it is contradicted
        words = list(cheap.words)
        words[2] = dataclasses.replace(words[2], readings=(Reading("bin", "casx", rank=0),))
        cheap = dataclasses.replace(cheap, words=tuple(words), _context=None)
        deep = page_with([Reading("deep", "obra", rank=0)])
        base = pc.without_readings(cheap)
        arrays = pc.wrong_rows([base]).array()
        out = pc.readers_ablation(
            pc.reach_vector([base]), arrays, {"L": [cheap], "LD": [deep]}, SPEED_WITH_READERS, {"line": 8}, 20
        )
        self.assertGreater(out["line"]["latin"]["8"]["net100"], out["line"]["all_cheap"]["8"]["net100"])


class PilotRunTest(unittest.TestCase):
    """The whole analysis over a small pilot written to disk: first readings, pass 2 files and the manifest."""

    def build(self, folder: Path, cells, engine=None):
        rng = random.Random(3)
        entries = []
        for d in range(8):
            rec = record(f"d{d}", 1, rng, errors=0.3, words=30)
            if engine:
                rec["ocr"]["engine"] = engine
            target = folder / "pages" / "train" / f"d{d}"
            target.mkdir(parents=True)
            (target / "1.json").write_text(json.dumps(rec), encoding="utf-8")
            entries.append({"set": "train", "doc_id": f"d{d}", "page": 1, "band": "medium", "wrong": 1})
            first = reading_from_dict(rec["ocr"]["reading"])
            truth = [w["text"] for w in rec["truth"]["words"]]
            for cell in cells:
                words = tuple(
                    dataclasses.replace(w, readings=(Reading("deep", t, rank=0),)) if w.text != t and cell == "Z" else w
                    for w, t in zip(first.words, truth, strict=True)
                )
                page = dataclasses.replace(first, words=words, rereads={0: ("deep",)})
                (target / f"1.rr2-{cell}.json").write_text(json.dumps(rereads_to_dict(page, "k")), encoding="utf-8")
        manifest = folder / "manifest.json"
        manifest.write_text(json.dumps({"pilot": entries}), encoding="utf-8")
        return manifest

    def args(self, folder, manifest, cells, **kw):
        import argparse

        arms = folder / "arms.json"
        arms.write_text(
            json.dumps(
                {
                    "cells": {
                        c: {"seconds_per_page_mean": 1.0, "seconds_per_page_p95": 2.0, "peak_rss_mb": 900}
                        for c in cells
                    }
                }
            )
        )
        speed = folder / "speed.json"
        speed.write_text(arms.read_text(encoding="utf-8"))
        bakeoff = folder / "bakeoff.json"
        licence = {"licence": "Apache-2.0", "url": "https://example.org/m.onnx", "sha256": "ab" * 32}
        bakeoff.write_text(
            json.dumps({"models": {n: licence for n in (*pc.CHEAP_MODELS, *pc.DEEP_MODELS)}}), encoding="utf-8"
        )
        return argparse.Namespace(
            bakeoff=bakeoff,
            data=folder,
            manifest=manifest,
            cells=list(cells),
            line_budgets=[4],
            zone_budgets=[5],
            arms=arms,
            speed=speed,
            resamples=50,
            **kw,
        )

    def test_the_cell_that_reads_the_right_text_gains_and_wins_and_the_one_that_reads_nothing_gains_nothing(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = self.build(folder, ("L", "Z"))
            result = pc.run(self.args(folder, manifest, ("L", "Z")))
        self.assertEqual(result["pages"], 8)
        self.assertEqual(result["missing"], {})
        self.assertEqual(result["cells"]["L"]["budgets"]["4"]["gain"], 0.0)
        z = result["cells"]["Z"]["budgets"]["1"]
        self.assertGreater(z["gain"], 0.0)
        self.assertEqual(z["harm"]["right_words_contradicted"], 0)
        self.assertEqual((z["cost_speed"]["mean_s"], z["cost_pilot"]["mean_s"]), (1.0, 1.0))
        self.assertEqual(result["choice"]["cell"], "Z")
        self.assertEqual(result["cells"]["Z"]["readers"], ["deep"])
        gate = result["gate_p"]
        self.assertEqual((gate["option"], gate["cell"], gate["budget"]), ("chosen", "Z", 1))
        self.assertGreater(gate["n_plain_wrong_words"], 0)
        self.assertEqual(len(gate["recall_ci95"]), 2)
        self.assertIn(gate["decision"], ("pass", "between_80_and_85", "stop"))
        self.assertEqual(result["cells"]["Z"]["collapsed"], {"5": 1})
        self.assertIn("unreached_plain_wrong_words_by_cause", gate)

    def test_the_rule_runs_under_the_official_cost_of_the_speed_set_and_under_the_pilots_beside_it(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = self.build(folder, ("Z",))
            args = self.args(folder, manifest, ("Z",))
            args.speed.write_text(
                json.dumps(
                    {"cells": {"Z": {"pass2_seconds_mean": 9.9, "pass2_seconds_p95": 9.9, "peak_reader_rss_mb": 900}}}
                )
            )
            result = pc.run(args)
        z = result["cells"]["Z"]["budgets"]["1"]
        self.assertEqual((z["cost_speed"]["mean_s"], z["cost_pilot"]["mean_s"]), (9.9, 1.0))
        self.assertEqual(
            result["choice"]["decision"], "options_for_maintainer"
        )  # the official cost is over the time budget
        self.assertEqual(result["choice_pilot_costs"]["cell"], "Z")
        self.assertEqual(result["gate_p"]["option"], "best_gain_not_chosen")
        self.assertEqual(result["gate_p_pilot_costs"]["option"], "chosen")

    def ablation_run(self, per_reader):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = self.build(folder, ("L", "LD"))
            args = self.args(folder, manifest, ("L", "LD"))
            args.line_budgets = [4, 8, 16]
            cheap = {"mean": 1.0, "p95": 2.0, "units_per_page_mean": 4.0}
            speed = {
                "cells": {
                    c: {
                        "peak_reader_rss_mb": 900,
                        "readers": {
                            r: {"mean": 1.0} for r in (("latin", "bin", "clahe", "wide") if c == "L" else ("deep",))
                        },
                        "budgets": {
                            "4": cheap,
                            "8": {**cheap, "units_per_page_mean": 8.0},
                            "16": {**cheap, "units_per_page_mean": 16.0},
                        },
                    }
                    for c in ("L", "LD")
                }
            }
            args.speed.write_text(json.dumps(speed))
            arms = json.loads(args.arms.read_text())
            if per_reader:
                arms["per_reader"] = {
                    r: {"page_seconds": {"line": {"mean": 0.1, "p95": 0.2}}}
                    for r in ("latin", "bin", "clahe", "wide", "deep")
                }
            args.arms.write_text(json.dumps(arms))
            return pc.run(args)["readers_ablation"]["line"]["latin"]["4"]["estimated_seconds"]

    def test_the_run_reads_the_per_reader_seconds_of_the_pilots_cost_file_and_falls_back_without_them(self):
        measured = self.ablation_run(per_reader=True)
        self.assertEqual(measured["source"], "per_reader")
        self.assertAlmostEqual(measured["mean_s"], 0.1 * 4.0 / 16.0)
        fallback = self.ablation_run(per_reader=False)
        self.assertNotIn("source", fallback)

    def test_the_units_the_engine_stores_come_from_its_key_and_the_ones_seen_are_reported_beside(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = self.build(folder, ("Z",), engine="e1|u:0.9:0.9:16/40:3/3")
            result = pc.run(self.args(folder, manifest, ("Z",)))
        z = result["cells"]["Z"]
        self.assertEqual((z["units_stored"], z["units_observed"]), (3, 1))
        self.assertEqual(z["collapsed"], {"5": 3})
        self.assertEqual(list(z["budgets"]), ["3"])

    def test_the_engine_limit_of_a_unit_is_read_from_the_page_key(self):
        self.assertEqual(pc.engine_unit_limit("e1|u:0.9:0.9:16/40:16/40", "line"), 16)
        self.assertEqual(pc.engine_unit_limit("e1|u:0.9:0.9:16/40:16/40", "zone"), 40)
        self.assertEqual(pc.engine_unit_limit("e1|u:0.9:0.9:16/40:8/20", "line"), 8)
        self.assertEqual(pc.engine_unit_limit("e1|u:0.9:0.9:16/40:8/20", "zone"), 20)
        self.assertIsNone(pc.engine_unit_limit("e1", "line"))
        self.assertIsNone(pc.engine_unit_limit(None, "zone"))

    def test_a_cell_without_its_pass_2_is_skipped_and_counted(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = self.build(folder, ("Z",))
            result = pc.run(self.args(folder, manifest, ("L", "Z")))
        self.assertEqual(result["missing"], {"no_pass2_L": 8})
        self.assertIn("skipped", result["cells"]["L"])
        self.assertIn("Z", result["cells"])

    def test_an_empty_pilot_says_so(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = folder / "manifest.json"
            manifest.write_text(json.dumps({"pilot": []}))
            result = pc.run(self.args(folder, manifest, ("L",)))
        self.assertEqual(result["pages"], 0)
        self.assertIn("note", result)


if __name__ == "__main__":
    unittest.main()
