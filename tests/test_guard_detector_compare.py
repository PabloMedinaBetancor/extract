import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from bench.guard import dataset, detector_compare, metrics, train
from extract.guard.model import GuardModel
from tests.test_guard_run_validation import rows

RESAMPLES = 60
ROOT = Path(__file__).resolve().parents[1]


class DetectorCompareTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = rows(60, 1)
        cls.validation = rows(30, 2)
        cls.test = rows(40, 5)
        cls.good, _, _ = train.fit_model(cls.train, cls.validation, {"leaves": 7, "trees": 100})
        canary = rows(60, 1)
        canary.y = np.random.default_rng(0).permutation(canary.y)  # the same rows with the labels shuffled: no signal
        cls.blind, _, _ = train.fit_model(canary, cls.validation, {"leaves": 7, "trees": 20})

    def test_G31_the_reference_file_is_the_frozen_v4_detector_and_loads(self):
        path = ROOT / "models" / "guard-v4.model"
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), detector_compare.V4_SHA256)
        self.assertIsInstance(detector_compare.load_reference(path), GuardModel)
        listed = (ROOT / "models" / "SHA256SUMS").read_text(encoding="utf-8")
        self.assertIn(f"{detector_compare.V4_SHA256}  guard-v4.model", listed)

    def test_G31_a_reference_that_is_not_the_registered_file_is_refused(self):
        with tempfile.TemporaryDirectory() as folder:
            other = Path(folder) / "guard-v4.model"
            self.good.save(other)
            with self.assertRaises(ValueError):
                detector_compare.load_reference(other)

    def test_G31_the_recall_at_five_percent_is_the_rank_matched_one_computed_by_hand(self):
        report = detector_compare.compare_rows(self.test, self.blind, {"new": self.good}, RESAMPLES)
        for name, model in (("v4", self.blind), ("new", self.good)):
            expected = metrics.recall_at_share(self.test.y, model.raw(self.test.x), 0.05)
            self.assertAlmostEqual(report[name]["recall"]["value"], expected)
        loss = report["v4"]["recall"]["value"] - report["new"]["recall"]["value"]
        self.assertAlmostEqual(report["new"]["g31"]["loss"], loss)
        self.assertAlmostEqual(report["new"]["minus_v4_recall"]["value"], -loss)

    def test_G31_the_same_detector_loses_nothing_and_the_difference_is_exactly_zero(self):
        report = detector_compare.compare_rows(self.test, self.good, {"new": self.good}, RESAMPLES)
        block = report["new"]
        self.assertEqual(block["g31"]["loss"], 0.0)
        self.assertTrue(block["g31"]["met"] and block["g31"]["interval_clears"])
        self.assertEqual((block["minus_v4_recall"]["low"], block["minus_v4_recall"]["high"]), (0.0, 0.0))
        self.assertFalse(block["minus_v4_recall"]["excludes_zero"])

    def test_G31_a_detector_that_loses_more_than_a_point_fails_the_verdict(self):
        report = detector_compare.compare_rows(self.test, self.good, {"new": self.blind}, RESAMPLES)
        self.assertGreater(report["new"]["g31"]["loss"], 0.01)
        self.assertFalse(report["new"]["g31"]["met"])
        self.assertFalse(report["new"]["g31"]["interval_clears"])

    def test_G31_the_verdict_is_the_point_value_against_a_point_and_the_interval_is_apart(self):
        far = {"low": -0.05, "high": 0.0, "value": -0.0099, "se": 0.01}
        with mock.patch("bench.guard.metrics.cluster_bootstrap") as bootstrap:
            names = ("recall", "precision", "ap", "shipped_recall", "shipped_flagged_share")
            summary = {
                f"{m}.{k}": {"value": 0.5, "low": 0.4, "high": 0.6, "se": 0.1} for m in ("v4", "new") for k in names
            }
            summary["v4.recall"] = {"value": 0.80, "low": 0.7, "high": 0.9, "se": 0.1}
            summary["new.recall"] = {"value": 0.7901, "low": 0.7, "high": 0.9, "se": 0.1}
            summary["new.minus_v4.recall"] = far
            bootstrap.return_value = summary
            block = detector_compare.compare_rows(self.test, self.good, {"new": self.good}, RESAMPLES)["new"]
        self.assertTrue(block["g31"]["met"])  # lost 0.0099 <= 0.01
        self.assertFalse(block["g31"]["interval_clears"])  # but the interval goes below -0.01
        summary["new.recall"] = {"value": 0.7899, "low": 0.7, "high": 0.9, "se": 0.1}
        with mock.patch("bench.guard.metrics.cluster_bootstrap", return_value=summary):
            block = detector_compare.compare_rows(self.test, self.good, {"new": self.good}, RESAMPLES)["new"]
        self.assertFalse(block["g31"]["met"])  # lost 0.0101

    def test_G31_every_candidate_gets_its_own_block_and_the_shipped_thresholds_are_reported(self):
        report = detector_compare.compare_rows(self.test, self.blind, {"a": self.good, "b": self.blind}, RESAMPLES)
        self.assertEqual(set(report) - {"rows", "wrong", "documents"}, {"v4", "a", "b"})
        for name in ("v4", "a", "b"):
            self.assertGreater(report[name]["shipped_flagged_share"]["value"], 0.0)
            self.assertIn("shipped_recall", report[name])

    def test_G31_the_shipped_figures_are_those_of_the_thresholds_of_the_budget_with_the_number_one_for_digits(self):
        from extract.guard import signals

        report = detector_compare.compare_rows(self.test, self.blind, {"new": self.good}, RESAMPLES)
        digits = self.test.x[:, signals.INDEX["has_digit"]] == 1.0
        self.assertTrue(digits.any())
        raw = self.good.raw(self.test.x)
        t_word, t_number = self.good.thresholds(0.05)
        flagged = self.good.probability(raw) >= np.where(digits, t_number, t_word)
        self.assertAlmostEqual(report["new"]["shipped_flagged_share"]["value"], float(flagged.mean()))
        self.assertAlmostEqual(
            report["new"]["shipped_recall"]["value"], metrics.recall_precision(self.test.y, flagged)[0]
        )

    def test_G31_page_table_adds_up_to_the_recall_and_the_flagged_share_of_the_group(self):
        table = detector_compare.page_table(self.test, self.good)
        raw = self.good.raw(self.test.x)
        self.assertEqual(sum(p["words"] for p in table), len(self.test.y))
        self.assertEqual(sum(p["wrong_words"] for p in table), int(self.test.y.sum()))
        self.assertEqual(sum(p["flagged"] for p in table), int(round(0.05 * len(raw))))
        self.assertAlmostEqual(
            sum(p["wrong_flagged"] for p in table) / sum(p["wrong_words"] for p in table),
            metrics.recall_at_share(self.test.y, raw, 0.05),
        )
        self.assertEqual(len({p["page"] for p in table}), len(table))
        self.assertTrue(all(p["page"].startswith(p["doc"] + "/") for p in table))

    def test_G43_the_pooled_group_is_only_there_when_asked_for(self):
        records = [(Path(f"{i}.json"), {"lang": "es" if i % 2 else "en", "ocr": {}}) for i in range(4)]
        with mock.patch(
            "bench.guard.dataset.rows_from_records", side_effect=lambda items, languages: self.real_rows(2, 3)
        ):
            self.assertEqual(list(detector_compare.rows_by_group(iter(records), lambda r: r["lang"])), ["en", "es"])

    def test_G31_precision_average_precision_and_documents_are_those_of_the_rows(self):
        report = detector_compare.compare_rows(self.test, self.blind, {"new": self.good}, RESAMPLES)
        self.assertEqual(report["documents"], len(set(self.test.group.tolist())))
        self.assertEqual(report["rows"], len(self.test.y))
        self.assertEqual(report["wrong"], int(self.test.y.sum()))
        for name, model in (("v4", self.blind), ("new", self.good)):
            raw = model.raw(self.test.x)
            self.assertAlmostEqual(
                report[name]["precision"]["value"], metrics.precision_at_share(self.test.y, raw, 0.05)
            )
            self.assertAlmostEqual(report[name]["ap"]["value"], metrics.average_precision(self.test.y, raw))
        self.assertNotAlmostEqual(report["v4"]["ap"]["value"], report["new"]["ap"]["value"])
        self.assertNotAlmostEqual(report["v4"]["precision"]["value"], report["new"]["precision"]["value"])

    def test_G31_a_difference_whose_interval_is_all_below_zero_excludes_zero_and_one_above_does_too(self):
        worse = detector_compare.compare_rows(self.test, self.good, {"new": self.blind}, RESAMPLES)["new"]
        self.assertLess(worse["minus_v4_recall"]["high"], 0)
        self.assertTrue(worse["minus_v4_recall"]["excludes_zero"])
        better = detector_compare.compare_rows(self.test, self.blind, {"new": self.good}, RESAMPLES)["new"]
        self.assertGreater(better["minus_v4_recall"]["low"], 0)
        self.assertTrue(better["minus_v4_recall"]["excludes_zero"])

    def test_G31_main_groups_by_the_language_of_the_set_scores_both_models_and_passes_resamples_and_sets(self):
        import json

        made = {
            "val": [(Path("v.json"), {"set": "val", "ocr": {}})],
            "dev5en": [(Path("e.json"), {"set": "dev5en", "ocr": {}})],
        }
        asked = []

        def pages(data, name):
            asked.append(name)
            return iter(made[name])

        def fake(items, languages=("es", "en")):
            return self.real_rows(20, 4 if items[0][1]["set"] == "val" else 6)

        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / "out.json"
            with (
                mock.patch("bench.guard.dataset.iter_page_records", side_effect=pages),
                mock.patch("bench.guard.dataset.rows_from_records", side_effect=fake),
                mock.patch.object(detector_compare, "compare", wraps=detector_compare.compare) as wrapped,
            ):
                detector_compare.main(
                    ["--data", folder, "--sets", "val", "dev5en", "--resamples", "7", "--out", str(out)]
                )
            report = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(asked, ["val", "dev5en"])
        self.assertEqual(wrapped.call_args.kwargs["resamples"], 7)
        self.assertEqual(set(wrapped.call_args.args[0]), {"es", "en"})
        self.assertEqual(set(wrapped.call_args.args[1]), {"default", "thorough"})
        self.assertEqual(set(report["groups"]), {"es", "en"})
        self.assertEqual(report["sets"], ["val", "dev5en"])
        shipped = {k: GuardModel.load(ROOT / "models" / k).model_id() for k in ("guard.model", "guard-thorough.model")}
        self.assertEqual(
            report["candidates"], {"default": shipped["guard.model"], "thorough": shipped["guard-thorough.model"]}
        )
        self.assertEqual(report["reference"]["sha256"], detector_compare.V4_SHA256)

    def test_G31_main_reads_val_and_the_english_development_set_by_default(self):
        asked = []
        with tempfile.TemporaryDirectory() as folder:
            with (
                mock.patch(
                    "bench.guard.dataset.iter_page_records",
                    side_effect=lambda data, name: asked.append(name) or iter(()),
                ),
                mock.patch.object(detector_compare, "compare", return_value={"groups": {}}),
            ):
                detector_compare.main(["--data", folder, "--out", str(Path(folder) / "o.json")])
        self.assertEqual(asked, ["val", "dev5en"])

    def test_G31_compare_reports_each_group_and_a_group_without_wrong_words_is_not_scored(self):
        clean = rows(10, 7)
        clean.y = np.zeros_like(clean.y)
        with mock.patch.object(detector_compare, "load_reference", return_value=self.blind):
            report = detector_compare.compare(
                {"es": self.test, "en": clean}, {"default": self.good}, resamples=RESAMPLES
            )
        self.assertIn("g31", report["groups"]["es"]["default"])
        self.assertEqual(report["groups"]["en"]["wrong"], 0)
        self.assertIn("note", report["groups"]["en"])
        self.assertEqual(report["reference"]["sha256"], detector_compare.V4_SHA256)
        self.assertEqual(report["candidates"]["default"], self.good.model_id())

    @staticmethod
    def real_rows(n_documents, seed):
        r = rows(n_documents, seed)
        empty = np.empty(len(r.y), dtype=object)
        return dataset.Rows(
            r.x, r.y, r.group, r.band, r.kind, empty, r.page, r.text, r.truth, r.box, empty, counts={"pages": 1}
        )

    def test_G31_rows_by_group_splits_by_the_key_adds_all_and_takes_one_page_at_a_time(self):
        records = [(Path(f"{i}.json"), {"lang": "es" if i % 3 else "en", "ocr": {}}) for i in range(6)]
        made = {i: self.real_rows(2, 10 + i) for i in range(6)}
        seen = []

        def fake(items, languages=("es", "en")):
            self.assertEqual(len(items), 1)
            index = int(items[0][0].stem)
            seen.append(index)
            return made[index]

        with mock.patch("bench.guard.dataset.rows_from_records", side_effect=fake):
            groups = detector_compare.rows_by_group(iter(records), lambda r: r["lang"], pooled=True)
        self.assertEqual(list(groups), ["en", "es", "all"])
        sizes = {i: len(made[i].y) for i in made}
        self.assertEqual(len(groups["en"].y), sizes[0] + sizes[3])
        self.assertEqual(len(groups["es"].y), sum(sizes[i] for i in (1, 2, 4, 5)))
        self.assertEqual(len(groups["all"].y), sum(sizes.values()))
        self.assertEqual(sorted(seen), list(range(6)))
        self.assertEqual(groups["all"].x.shape[1], made[0].x.shape[1])
        self.assertEqual(groups["all"].counts, {"pages": 6})


if __name__ == "__main__":
    unittest.main()
