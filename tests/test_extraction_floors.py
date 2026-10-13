import json
import os
import unittest
from pathlib import Path

import pytest

from bench import measure_extraction

FLOORS = json.loads((Path(__file__).parents[1] / "bench" / "results" / "extraction-floors.json").read_text())


@pytest.mark.corpus
class ExtractionFloorsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        data = os.environ.get("EXTRACT_BENCH_DATA")
        if not data or not Path(data, "pages", "val").is_dir():
            raise AssertionError("EXTRACT_BENCH_DATA must point to the corpus with the validation pages read")
        cls.report = measure_extraction.measure(Path(data), "val")

    def test_F1_to_F3_the_readings_of_each_damage_band_stay_above_their_floors(self):
        self.assertEqual(set(self.report), {"light", "medium", "heavy"})
        for band, floors in FLOORS.items():
            if band == "note":
                continue
            got = self.report[band]
            measured = {
                "f1_recall": got["f1_recall"],
                "f2_mean": got["f2_mean"],
                "row_count": got["f3"]["row_count"],
                "row_amount": got["f3"]["row_amount"],
                "row_quantity_and_price": got["f3"]["row_quantity_and_price"],
                "min_checked": got["checked"],
            }
            for name, floor in floors.items():
                with self.subTest(band=band, metric=name):
                    self.assertGreaterEqual(measured[name], floor)

    def test_N6_no_number_of_the_validation_readings_is_checked_and_wrong(self):
        for band, got in self.report.items():
            with self.subTest(band=band):
                self.assertEqual(got["false_checked"], [])
                self.assertGreater(got["checked"], 0)


if __name__ == "__main__":
    unittest.main()
