import unittest
from decimal import Decimal

from extract.options import Options
from extract.types import NumberResult, PageReading, Word, page_of, word_id


class OptionsTest(unittest.TestCase):
    def test_X1_defaults_are_the_documented_ones(self):
        options = Options()
        self.assertEqual(options.languages, ("es", "en"))
        self.assertEqual(options.review, "balanced")
        self.assertEqual(options.recycle_pages, 120)
        self.assertEqual((options.correct, options.second_reading), ("off", False))

    def test_C6_the_correction_modes_are_off_safe_and_more(self):
        for mode in ("off", "safe", "more"):
            self.assertEqual(Options(correct=mode).correct, mode)
        self.assertTrue(Options(second_reading=True).second_reading)

    def test_X1_bad_values_are_refused(self):
        for bad in (
            {"languages": ("fr",)},
            {"languages": ()},
            {"review": "all"},
            {"recycle_pages": 3},
            {"processes": 0},
            {"backend": "gpu"},
            {"correct": "auto"},
            {"correct": None},
            {"second_reading": "yes"},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                Options(**bad)


class TypesTest(unittest.TestCase):
    def test_X1_word_ids_round_trip_their_page(self):
        self.assertEqual(word_id(3, 17), "p3w17")
        self.assertEqual(page_of("p3w17"), 3)
        with self.assertRaises(ValueError):
            page_of("w17")

    def test_X1_a_new_word_is_not_shown_as_fine(self):
        self.assertEqual(Word("p1w0", 1, "a", (0, 0, 1, 1)).status, "unscored")

    def test_X1_a_new_word_carries_none_of_the_v5_fields(self):
        word = Word("p1w0", 1, "a", (0, 0, 1, 1))
        for name in ("char_alts", "masked", "readings", "fix", "absorbed_by", "guard_status"):
            self.assertIsNone(getattr(word, name))

    def test_X1_boxes_convert_from_pixels_to_points(self):
        page = PageReading(1, 200, 400, 72.0, 144.0, 200, ())
        self.assertEqual(page.box_to_points((100, 200, 200, 400)), (36.0, 72.0, 72.0, 144.0))

    def test_X1_number_results_keep_exact_decimals(self):
        result = NumberResult(("p1w1",), "1.234,56", Decimal("1234.56"), "amount", "unverified")
        self.assertEqual(str(result.value), "1234.56")


if __name__ == "__main__":
    unittest.main()
