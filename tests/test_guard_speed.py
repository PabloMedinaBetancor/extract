import tempfile
import unittest
from pathlib import Path

from bench.guard import speed, train
from extract.guard import checker
from tests.test_guard_guard import page
from tests.test_guard_train import synthetic


class SpeedHelpersTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        model, _, _ = train.fit_model(synthetic(60, 1), synthetic(30, 2), {"leaves": 7, "trees": 100})
        cls.guard = checker.Guard(model)
        cls.path = Path(tempfile.mkdtemp()) / "guard.model"
        model.save(cls.path)

    def test_G10_times_are_scaled_to_a_page_of_600_words(self):
        self.assertEqual(speed.per_600(10.0, 300), 20.0)
        self.assertEqual(speed.per_600(10.0, 600), 10.0)

    def test_G10_the_page_measure_reports_total_signals_and_scoring_per_page(self):
        readings = [page(["obra", "plano", "casa", "xqzvbn"] * 100, conf=0.8) for _ in range(3)]
        report = speed.measure_pages(self.guard, readings)
        self.assertEqual(report["pages"], 3)
        for key in ("page_ms", "signals_ms", "scoring_ms"):
            self.assertGreater(report[key]["median"], 0)
            self.assertGreaterEqual(report[key]["max"], report[key]["median"])
            self.assertLessEqual(report[key]["median"], report[key]["p95"])
            self.assertLessEqual(report[key]["p95"], report[key]["max"])
        self.assertLess(report["scoring_ms"]["median"], report["page_ms"]["median"])

    def test_G10_loading_reports_seconds_and_added_memory_and_returns_a_working_guard(self):
        load = speed.measure_load(self.path)
        self.assertGreater(load["load_seconds"], 0)
        self.assertIsNotNone(load["guard"].score_page(page(["obra"]), ("es",), "es", 0.05))

    def test_G10_the_verdict_compares_each_figure_with_its_limit(self):
        pages = {"page_ms": {"median": 120.0}, "scoring_ms": {"median": 12.0}}
        load = {"load_seconds": 1.0, "added_mb": 100.0}
        self.assertEqual(
            speed.verdict(pages, load), {"page_ms_ok": True, "scoring_ms_ok": False, "load_ok": True, "memory_ok": True}
        )


if __name__ == "__main__":
    unittest.main()
