"""The default of the second reading and its cell are one packaged file, read everywhere."""

import json
import tempfile
import unittest
from pathlib import Path

from extract import second_reading
from extract.engine import reread
from extract.options import Options
from extract.ui import server

GOOD = {
    "default": True,
    "unit": "zone",
    "readers": ["deep"],
    "reader": "deep",
    "seconds_per_page": 1.6,
    "fixes_more_pct": 4.0,
}


class DecisionFileTest(unittest.TestCase):
    def test_the_packaged_file_is_valid_and_the_places_that_read_it_agree(self):
        data = second_reading.load()
        self.assertEqual(Options().second_reading, data["default"])
        self.assertEqual((reread.DEFAULT_UNIT, reread.DEFAULT_READERS), (data["unit"], tuple(data["readers"])))
        self.assertEqual(server.SECOND_READING_COST, second_reading.cost())
        self.assertEqual(set(second_reading.cost()), set(second_reading.COST_KEYS))

    def test_the_panel_gets_the_thorough_cost_from_the_same_file(self):
        self.assertEqual(server.THOROUGH_COST, second_reading.thorough_cost())
        self.assertIn("seconds_p95", server.THOROUGH_COST)
        self.assertIn("reader_ram_mb", server.THOROUGH_COST)

    def test_a_document_read_in_the_thorough_mode_says_so_in_its_fixes_block(self):
        from extract import fixes
        from extract.types import DocumentData

        for mode, thorough in (("thorough", True), (True, False), (False, False)):
            data = DocumentData.__new__(DocumentData)
            object.__setattr__(data, "pages", [])
            object.__setattr__(data, "options", {"second_reading": mode})
            block = fixes.summary(data)
            self.assertEqual((block["thorough"], block["second_reading"]), (thorough, bool(mode)))

    def test_until_v5_a26_is_written_the_default_is_off_and_the_panel_has_no_cost(self):
        self.assertFalse(second_reading.default_on())
        self.assertIsNone(second_reading.cost()["reader"])

    def test_a_decision_that_is_on_must_carry_the_cost_the_panel_shows(self):
        self.assertEqual(second_reading.validate(dict(GOOD)), GOOD)
        for key in ("reader", "seconds_per_page", "fixes_more_pct"):
            with self.assertRaises(ValueError, msg=key):
                second_reading.validate({**GOOD, key: None})

    def test_a_bad_file_is_refused(self):
        for bad in (
            {**GOOD, "default": "yes"},
            {**GOOD, "unit": "page"},
            {**GOOD, "readers": []},
            {**GOOD, "readers": ["deep", "deep"]},
            {**GOOD, "readers": ["tesseract"]},
            {**GOOD, "seconds_per_page": -1},
            {**GOOD, "fixes_more_pct": True},
            {**GOOD, "reader": 3},
            {k: v for k, v in GOOD.items() if k != "reader"},
        ):
            with self.assertRaises(ValueError, msg=str(bad)):
                second_reading.validate(bad)

    def test_off_with_no_cost_is_allowed_and_a_file_is_loaded_from_disk(self):
        off = {**GOOD, "default": False, "reader": None, "seconds_per_page": None, "fixes_more_pct": None}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "d.json"
            path.write_text(json.dumps(off))
            self.assertEqual(second_reading.load(str(path))["unit"], "zone")


if __name__ == "__main__":
    unittest.main()
