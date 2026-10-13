import json
import unittest
from pathlib import Path

from extract.engine.ocr import DEFAULT_TUNING
from extract.engine.pool import MAX_PROCESSES, plan_workers

RESULTS = json.loads((Path(__file__).resolve().parents[1] / "bench" / "results" / "engine-speed.json").read_text())
MEASURED = {entry["name"]: entry for entry in RESULTS["optimizations"]}


class SpeedResultsTest(unittest.TestCase):
    def test_O13_every_optimization_has_its_seconds_a_page_before_and_after(self):
        self.assertEqual(RESULTS["pages"], 20)
        self.assertEqual(
            set(MEASURED), {"procs_2x2", "det_buckets", "rec_buckets", "pdf_open_once", "engine_loaded_once"}
        )
        for entry in MEASURED.values():
            self.assertGreater(entry["seconds_per_page_before"], 0, entry["name"])
            self.assertGreater(entry["seconds_per_page_after"], 0, entry["name"])

    def test_O13_only_what_saves_5_percent_without_losing_accuracy_is_on(self):
        for name in ("procs_2x2", "det_buckets", "rec_buckets"):
            entry = MEASURED[name]
            kept = (
                entry["saving"] >= 0.05
                and entry["digital_word_accuracy_after"] >= entry["digital_word_accuracy_before"]
            )
            self.assertEqual(entry["on_by_default"], kept, name)
            saving = 1 - entry["seconds_per_page_after"] / entry["seconds_per_page_before"]
            self.assertAlmostEqual(entry["saving"], saving, delta=0.001, msg=name)

    def test_O13_the_defaults_in_the_code_are_the_measured_ones(self):
        self.assertEqual(DEFAULT_TUNING.det_buckets, MEASURED["det_buckets"]["on_by_default"])
        self.assertEqual(DEFAULT_TUNING.rec_buckets, MEASURED["rec_buckets"]["on_by_default"])
        plan = plan_workers(10000, 4, "openvino")
        two_by_two = MEASURED["procs_2x2"]["on_by_default"]
        self.assertEqual((plan.processes, plan.threads), (2, 2) if two_by_two else (1, 4))
        self.assertEqual(MAX_PROCESSES, 2 if two_by_two else 1)

    def test_O13_the_measured_pages_are_the_same_for_every_run(self):
        self.assertEqual(len(RESULTS["sha256_of_pages_pdf"]), 64)


if __name__ == "__main__":
    unittest.main()
