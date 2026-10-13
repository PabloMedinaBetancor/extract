"""The paired comparison of two runs by document and the lazy page records."""

import argparse
import json
import tempfile
import unittest
from pathlib import Path

from bench.guard import cells_paired, dataset


class PairedTest(unittest.TestCase):
    def test_the_same_documents_are_resampled_in_both_runs(self):
        # B is A minus one in every document: the difference is the same whatever the documents drawn
        a = {"net": [3, 5, 2, 9, 4, 6] * 10}
        b = {"net": [2, 4, 1, 8, 3, 5] * 10}
        out = cells_paired.paired(a, b, "net", 400)
        self.assertEqual(out["difference"], 60.0)
        self.assertTrue(out["excludes_zero"])
        self.assertAlmostEqual(out["se"], 0.0)

    def test_a_difference_that_changes_sign_across_documents_includes_zero(self):
        a = {"net": [5, 0, 5, 0, 5, 0, 5, 0]}
        b = {"net": [0, 5, 0, 5, 0, 5, 0, 5]}
        out = cells_paired.paired(a, b, "net", 400)
        self.assertEqual(out["difference"], 0.0)
        self.assertFalse(out["excludes_zero"])

    def test_the_rule_of_the_runs_is_the_one_of_the_choice(self):
        self.assertEqual(cells_paired.CONFIGS["none"], ([], None))
        self.assertEqual(cells_paired.CONFIGS["latin+wide+deep"][0], ["L", "Z"])


class CapsTest(unittest.TestCase):
    def test_each_reader_is_hidden_from_its_own_rank_and_a_rankless_reading_stays(self):
        import dataclasses

        from bench.guard import pairs
        from extract.types import Reading, Word

        def reading(source, rank):
            return Reading(source, "x", None, None, rank=rank)

        word = Word(
            "w", 0, "x", (0, 0, 1, 1), readings=tuple([reading("latin", 3), reading("deep", 3), reading("wide", None)])
        )
        out = pairs.cap_by_reader([word, dataclasses.replace(word, readings=None)], {"latin": 4, "deep": 3})[0]
        self.assertEqual(sorted(r.source for r in out.readings), ["latin", "wide"])
        only = pairs.cap_by_reader([dataclasses.replace(word, readings=(reading("deep", 5),))], {"deep": 5})[0]
        self.assertIsNone(only.readings)
        self.assertEqual(cells_paired.parse_config("latin+wide+deep@4/5")[2], {"latin": 4, "wide": 4, "deep": 5})


def synthetic_run(name, more, safe_fixed, safe_applied, documents=("a", "b", "c", "d"), broken=(0, 0, 0, 0)):
    """A run as `run_config` writes it: per document, the fixed and broken of `more` and the fixed/applied of `safe`."""

    def rule(fixed, broken, applied):
        return {"by_document": {"fixed": list(fixed), "broken": list(broken), "applied": list(applied)}}

    def learner(shift):
        return {
            "thresholds": {},
            "more": rule(
                [m + shift for m in more], list(broken), [m + shift + b for m, b in zip(more, broken, strict=True)]
            ),
            "safe": rule(safe_fixed, [0] * 4, safe_applied),
        }

    return {
        "config": name,
        "documents": list(documents),
        "sets": ["train"],
        "wrong_words": 200,
        "sites": 10,
        "reachable_sites": 8,
        "seconds": 1.0,
        "learners": {"lightgbm": learner(0), "logistic": learner(-1)},
    }


class CompareTest(unittest.TestCase):
    def runs(self):
        return {
            "none": synthetic_run("none", [4, 4, 4, 4], [0, 0, 0, 0], [0, 0, 0, 0]),
            "latin": synthetic_run("latin", [5, 5, 5, 5], [0, 0, 0, 0], [0, 0, 0, 0]),
            "latin+wide": synthetic_run("latin+wide", [5, 6, 5, 6], [1, 1, 1, 1], [1, 1, 1, 2]),
        }

    def test_the_differences_per_100_wrong_and_the_safe_counts_are_exact(self):
        out = cells_paired.compare(self.runs(), 200)
        pair = out["paired"]["lightgbm"]["latin+wide - none"]
        self.assertEqual(pair["more_net"]["difference"], 6.0)  # (1 + 2 + 1 + 2) fixed more
        self.assertAlmostEqual(pair["more_net_per_100_wrong"], 3.0)  # 6 of 200 wrong words
        self.assertEqual(pair["safe_fixed"]["difference"], 4.0)
        self.assertEqual(pair["safe_applied"]["difference"], 5.0)
        self.assertTrue(pair["more_net"]["excludes_zero"])
        self.assertEqual(out["paired"]["lightgbm"]["latin+wide - latin"]["more_net"]["difference"], 2.0)

    def test_the_comparisons_have_the_direction_of_their_name(self):
        out = cells_paired.compare(self.runs(), 200)["paired"]["lightgbm"]
        self.assertEqual(out["latin - none"]["more_net"]["difference"], 4.0)
        self.assertNotIn("none - latin", out)

    def test_the_learner_pairs_compare_lightgbm_with_the_logistic_of_the_same_run(self):
        out = cells_paired.compare(self.runs(), 200)["learner_pairs"]["latin"]
        self.assertEqual(out["more_net"]["difference"], 4.0)  # one more fixed word in each of the four documents

    def test_runs_of_other_documents_are_refused(self):
        runs = self.runs()
        runs["wide"] = synthetic_run("wide", [1, 1, 1, 1], [0] * 4, [0] * 4, documents=("a", "b", "c", "x"))
        with self.assertRaises(ValueError):
            cells_paired.compare(runs, 50)

    def test_per_document_counts_what_each_rule_applies_fixes_and_breaks(self):
        import numpy as np

        from bench.guard import operating

        choice = type(
            "C", (), {"outcome": np.array([operating.FIXED, operating.BROKEN, operating.FIXED, operating.STILL_WRONG])}
        )()
        table = type("T", (), {"doc": np.array(["a", "a", "b", "b"], dtype=object)})()
        out = cells_paired.per_document(table, choice, np.array([True, True, True, False]), ["a", "b", "c"])
        self.assertEqual(out, {"applied": [2, 1, 0], "fixed": [1, 1, 0], "broken": [1, 0, 0]})

    def test_the_six_configs_read_their_own_cells_and_readers(self):
        c = cells_paired.CONFIGS
        self.assertEqual(c["none"], ([], None))
        self.assertEqual(c["latin"], (["L"], ("latin",)))
        self.assertEqual(c["wide"], (["L"], ("wide",)))
        self.assertEqual(c["latin+wide"], (["L"], ("latin", "wide")))
        self.assertEqual(c["deep"], (["Z"], ("deep",)))
        self.assertEqual(c["latin+wide+deep"], (["L", "Z"], ("latin", "wide", "deep")))

    def test_the_default_sets_are_the_spanish_ones(self):
        args = cells_paired.build_parser().parse_args(["run"])
        self.assertEqual(args.sets, ["train", "val", "test"])
        self.assertNotIn("dev5en", args.sets)

    def test_more_counts_what_it_breaks_against_what_it_fixes(self):
        runs = {
            "none": synthetic_run("none", [4, 4, 4, 4], [0] * 4, [0] * 4),
            "latin": synthetic_run("latin", [6, 6, 6, 6], [0] * 4, [0] * 4, broken=(1, 1, 1, 1)),
        }
        pair = cells_paired.compare(runs, 100)["paired"]["lightgbm"]["latin - none"]
        self.assertEqual(pair["more_net"]["difference"], 4.0)  # (6 - 1) - 4 in each of the four documents

    def test_the_runs_of_other_sets_are_refused(self):
        a, b = synthetic_run("a", [1] * 4, [0] * 4, [0] * 4), synthetic_run("b", [1] * 4, [0] * 4, [0] * 4)
        b["sets"] = ["dev5en"]
        with self.assertRaises(ValueError):
            cells_paired.compare({"a": a, "b": b}, 50)

    def test_safe_edge_marks_a_count_just_over_the_minimum(self):
        self.assertFalse(cells_paired.safe_edge(0))
        self.assertTrue(cells_paired.safe_edge(304))
        self.assertTrue(cells_paired.safe_edge(329))
        self.assertFalse(cells_paired.safe_edge(400))


class WiringTest(unittest.TestCase):
    def test_the_caps_reach_the_words_of_a_page_read_from_its_cells(self):
        import dataclasses
        from unittest import mock

        from bench.guard import pairs
        from extract.types import Reading, Word

        def word(*readings):
            return Word("w", 0, "x", (0, 0, 1, 1), readings=tuple(readings) or None)

        @dataclasses.dataclass(frozen=True)
        class Page:
            words: tuple

        first = Page((word(), word()))
        second = Page(
            (
                word(Reading("latin", "x", None, None, rank=1), Reading("deep", "x", None, None, rank=1)),
                word(Reading("latin", "x", None, None, rank=6)),
            )
        )
        with mock.patch("bench.run_pass2.load_cell", return_value=second):
            out = pairs.with_cells(first, Path("page.json"), ["L"], None, {"latin": 4, "deep": 1})
        self.assertEqual([r.source for r in out.words[0].readings], ["latin"])  # deep of rank 1 is under a cap of 1
        self.assertIsNone(out.words[1].readings)  # latin of rank 6 is over a cap of 4
        with mock.patch("bench.run_pass2.load_cell", return_value=second):
            out = pairs.with_cells(first, Path("page.json"), ["L"], None, None)
        self.assertEqual(len(out.words[0].readings), 2)


class IterRecordsTest(unittest.TestCase):
    def test_records_come_one_at_a_time_and_the_same_as_the_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "pages" / "train" / "doc"
            folder.mkdir(parents=True)
            (folder / "1.json").write_text(json.dumps({"ocr": {"reading": {}}, "n": 1}), encoding="utf-8")
            (folder / "2.json").write_text(json.dumps({"n": 2}), encoding="utf-8")
            lazy = dataset.iter_page_records(Path(tmp), "train")
            self.assertFalse(isinstance(lazy, list))
            self.assertEqual([r["n"] for _, r in lazy], [1])
            self.assertEqual([r["n"] for _, r in dataset.page_records(Path(tmp), "train")], [1])


if __name__ == "__main__":
    unittest.main()


class ThoroughTest(unittest.TestCase):
    def test_the_thorough_runs_load_both_cells_with_the_spec_of_the_mode(self):
        from bench.guard import pairs
        from extract.guard import readings

        parser = argparse.ArgumentParser()
        pairs.add_data_arguments(parser)
        args = parser.parse_args(["--thorough"])
        cells, spec = pairs.reading_cells(args)
        self.assertEqual(cells, ["L", "Z"])
        self.assertEqual(spec.as_meta(), readings.product_spec(True).as_meta())
        self.assertEqual(args.spec.as_meta(), spec.as_meta())
        with self.assertRaises(ValueError):
            pairs.reading_cells(parser.parse_args(["--thorough", "--cells", "L"]))


class FinalPairedTest(unittest.TestCase):
    def counts(self, fixed, broken, safe_fixed=(0, 0, 0, 0)):
        return {
            "more": {
                "fixed": list(fixed),
                "broken": list(broken),
                "applied": [f + b for f, b in zip(fixed, broken, strict=True)],
            },
            "safe": {"fixed": list(safe_fixed), "broken": [0] * 4, "applied": list(safe_fixed)},
        }

    def block(self, fixed, broken, safe_fixed=(0, 0, 0, 0)):
        zero = (0, 0)
        return {
            "documents": ["a", "b", "c", "d"],
            "language": {"a": "es", "b": "es", "c": "en", "d": "en"},
            "total": self.counts(fixed, broken, safe_fixed),
            "by_language": {
                "es": self.counts(fixed[:2] + zero, broken[:2] + zero, tuple(safe_fixed[:2]) + zero),
                "en": self.counts(zero + fixed[2:], zero + broken[2:], zero + tuple(safe_fixed[2:])),
            },
        }

    def reports(self):
        # thorough: sensed and label differ (a document the sensor moves), safe fixes in two documents, broken > 0
        thorough = {
            "by_document": {
                "sensed": self.block((6, 7, 8, 9), (1, 0, 2, 0), (3, 0, 1, 0)),
                "label": self.block((6, 7, 9, 9), (1, 0, 1, 0), (3, 0, 1, 0)),
            }
        }
        default = {
            "by_document": {
                "sensed": self.block((4, 4, 4, 4), (0, 1, 0, 1)),
                "label": self.block((4, 4, 4, 4), (0, 1, 0, 1)),
            }
        }
        return thorough, default

    def test_the_net_is_fixed_minus_broken_and_the_direction_is_thorough_minus_default(self):
        from bench.guard import final_paired

        out = final_paired.compare(*self.reports(), 200)["sensed"]
        # thorough nets 5, 7, 6, 9 = 27; default nets 4, 3, 4, 3 = 14
        self.assertEqual(out["total"]["more_net"]["difference"], 13.0)
        self.assertEqual(out["total"]["more_net_each"], {"thorough": 27, "default": 14})
        self.assertEqual(out["language_es"]["more_net"]["difference"], 5.0)  # (5 + 7) - (4 + 3)
        self.assertEqual(out["language_en"]["more_net"]["difference"], 8.0)  # (6 + 9) - (4 + 3)
        self.assertEqual(out["total"]["safe_fixed"]["difference"], 4.0)
        self.assertEqual(out["total"]["more_applied"]["difference"], (7 + 7 + 10 + 9) - (4 + 5 + 4 + 5))

    def test_the_label_partition_is_compared_apart_from_the_sensed_one(self):
        from bench.guard import final_paired

        out = final_paired.compare(*self.reports(), 200)
        self.assertEqual(sorted(out), ["label", "resamples", "sensed"])
        self.assertEqual(
            out["label"]["total"]["more_net"]["difference"], 15.0
        )  # thorough 5 + 7 + 8 + 9 = 29 against 14
        self.assertNotEqual(
            out["label"]["total"]["more_net"]["difference"], out["sensed"]["total"]["more_net"]["difference"]
        )

    def test_the_documents_of_a_label_are_compared_on_all_their_pages(self):
        from bench.guard import final_paired

        out = final_paired.compare(*self.reports(), 200)["sensed"]
        self.assertEqual(out["documents_labelled_en"]["documents"], 2)
        self.assertEqual(out["documents_labelled_en"]["more_net"]["difference"], 8.0)
        self.assertEqual(out["documents_labelled_es"]["more_net_each"], {"thorough": 12, "default": 7})

    def test_reports_of_other_documents_are_refused(self):
        from bench.guard import final_paired

        thorough, default = self.reports()
        default["by_document"]["sensed"]["documents"] = ["a", "b", "c", "x"]
        with self.assertRaises(ValueError):
            final_paired.compare(thorough, default, 50)


class SampleSizeTest(unittest.TestCase):
    def test_n_of_a_known_set_of_documents(self):
        import numpy as np

        from bench.guard import sample_size

        # four documents of 100 truth words: nets 2, 4, 6, 8 -> rates 2, 4, 6, 8 (SD 2.582), pooled effect 5
        out = sample_size.rates(np.array([2.0, 4.0, 6.0, 8.0]), np.array([100.0] * 4))
        self.assertAlmostEqual(out["effect_per_100"], 5.0)
        self.assertAlmostEqual(out["sd_between_documents"], 2.5819889, places=5)
        self.assertAlmostEqual(out["half_width"], 1.25)
        self.assertEqual(out["n"], int(np.ceil((1.96 * 2.5819889 / 1.25) ** 2)))  # 17
        self.assertEqual(out["documents"], 4)

    def test_documents_without_truth_words_and_a_non_positive_effect_give_no_n(self):
        import numpy as np

        from bench.guard import sample_size

        out = sample_size.rates(np.array([1.0, 3.0, 9.0]), np.array([100.0, 0.0, 100.0]))
        self.assertEqual(out["documents"], 2)
        self.assertEqual(sample_size.rates(np.array([-1.0, -3.0]), np.array([10.0, 10.0]))["n"], None)
