import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from bench.guard import metrics, run_validation, train
from extract.guard import model
from extract.guard.signals import INDEX
from tests.test_guard_train import synthetic


def rows(n_documents=30, seed=3):
    base = synthetic(n_documents, seed)
    n = len(base.y)
    rng = np.random.default_rng(seed)
    digits = rng.random(n) < 0.15
    base.x[:, INDEX["has_digit"]] = digits.astype(np.float32)
    base.kind = np.where(base.y == 1, np.where(rng.random(n) < 0.5, "substitution", "merge"), "ok").astype(object)
    base.page = np.array([f"{g}/1" for g in base.group], dtype=object)
    base.text = np.array(["obra"] * n, dtype=object)
    base.truth = np.array(["obras"] * n, dtype=object)
    base.box = np.tile(np.array([10, 10, 80, 30], dtype=np.float32), (n, 1))
    return base


class RunValidationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train = rows(60, 1)
        cls.validation = rows(30, 2)
        cls.guard, cls.raw, cls.probability = train.fit_model(cls.train, cls.validation, {"leaves": 7, "trees": 100})

    def test_G5_the_comparison_has_every_baseline_and_a_paired_difference_to_the_best(self):
        scores = {"detector": self.raw, **run_validation.baseline_scores(self.train, self.validation)}
        report = run_validation.compare(self.validation, scores, resamples=50)
        for name in (
            "detector",
            "lowest_mean_confidence",
            "lowest_min_confidence",
            "outside_dictionary_first",
            "logistic_regression",
        ):
            self.assertEqual(set(report[name]), {"recall", "precision", "average_precision"})
            self.assertLess(report[name]["recall"]["low"], report[name]["recall"]["high"])
        difference = report["detector_minus_best_baseline"]
        self.assertIn(difference["baseline"], scores)
        self.assertIn("excludes_zero", difference)
        self.assertEqual(difference["measure"], "recall at the review budget")
        self.assertIn("excludes_zero", report["detector_minus_best_baseline_ap"])

    def test_G5_misses_are_listed_by_kind_and_confidence_decile(self):
        flagged = metrics.flag_top(self.raw, 0.05)
        misses = run_validation.misses_by_kind(self.validation, flagged)
        self.assertEqual(set(misses["by_kind"]), {"substitution", "merge"})
        self.assertEqual(len(misses["by_confidence_decile"]), 10)
        total = sum(entry["wrong"] for entry in misses["by_kind"].values())
        self.assertEqual(total, int(self.validation.y.sum()))

    def test_G6_numbers_are_judged_at_a_third_of_the_word_threshold(self):
        report = run_validation.numbers_recall(self.validation, self.guard)
        self.assertGreater(report["wrong_digit_tokens"], 0)
        self.assertLessEqual(report["recall_at_0.03"], report["recall_at_0.08"])
        self.assertGreater(report["recall_at_0.05"], 0.0)

    def test_G6_numbers_report_how_much_each_budget_flags_with_both_thresholds(self):
        report = run_validation.numbers_recall(self.validation, self.guard)
        for budget in model.BUDGETS:
            share = report[f"flagged_share_all_words_at_{budget:.2f}"]
            self.assertGreaterEqual(share, budget - 0.005)
            self.assertGreater(report[f"flagged_share_digit_tokens_at_{budget:.2f}"], 0.0)
        self.assertGreaterEqual(report["flagged_share_all_words_at_0.05"], report["flagged_share_all_words_at_0.03"])

    def test_G3_the_audit_writes_sheets_with_sixty_wrong_and_sixty_right_crops(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "data"
            for doc in set(self.validation.group):
                target = data / "pages" / "val" / doc
                target.mkdir(parents=True)
                cv2.imwrite(str(target / "1.png"), np.full((120, 200), 255, np.uint8))
            written = run_validation.write_audit(self.validation, Path(folder) / "audit", data)
            sheets = sorted((Path(folder) / "audit").glob("sheet-*.png"))
        self.assertEqual(written, 120)
        self.assertEqual(len(sheets), run_validation.AUDIT_SHEETS)


if __name__ == "__main__":
    unittest.main()
