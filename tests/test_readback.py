import unittest

from bench import readback
from extract.engine.cache import reading_to_dict
from extract.types import PageReading, Word


def record(read: list[str], truth: list[str], patches=()) -> dict:
    """A page record whose words stand on a line, 10 px wide each; `read` is what the engine said."""
    boxes = [(i * 12.0, 0.0, i * 12.0 + 10.0, 10.0) for i in range(max(len(read), len(truth)))]
    words = tuple(Word(f"p1w{i}", 1, text, boxes[i]) for i, text in enumerate(read))
    page = PageReading(1, 100, 20, 100.0, 20.0, 72, words, status="ocr")
    return {
        "doc_id": "cX",
        "page": 1,
        "ocr": {"reading": reading_to_dict(page)},
        "truth": {"words": [{"id": f"t{i}", "text": t, "box": list(boxes[i])} for i, t in enumerate(truth)]},
        "ungrounded": list(patches),
    }


class ReadbackTest(unittest.TestCase):
    def test_a_page_read_exactly_is_all_ok(self):
        found = readback.measure([record(["alpha", "beta", "gamma"], ["alpha", "beta", "gamma"])])
        self.assertEqual((found["ok"], found["ocr_words"], found["missed"]), (3, 3, 0))
        self.assertEqual(found["ok_share"], 1.0)
        self.assertTrue(found["meets_floor"])

    def test_a_wrong_word_and_a_missed_one_are_counted_apart(self):
        found = readback.measure([record(["alpha", "betx"], ["alpha", "beta", "gamma"])])
        self.assertEqual((found["ok"], found["missed"], found["truth_words"]), (1, 1, 3))
        self.assertAlmostEqual(found["ok_share"], 0.5)
        self.assertFalse(found["meets_floor"])

    def test_the_share_is_over_the_words_not_on_ink_the_layer_does_not_describe(self):
        read = ["alpha", "beta", "LOGO"]
        found = readback.measure([record(read, ["alpha", "beta"], patches=[(24.0, 0.0, 34.0, 10.0)])])
        self.assertEqual((found["ungrounded"], found["ocr_words"], found["ok"]), (1, 3, 2))
        self.assertEqual(found["ok_share"], 1.0)

    def test_the_worst_pages_come_first(self):
        good, bad = record(["alpha", "beta"], ["alpha", "beta"]), record(["alpha", "betx"], ["alpha", "beta"])
        bad["page"] = 2
        found = readback.measure([good, bad])
        self.assertEqual(found["worst_pages"][0]["page"], "cX/2")


if __name__ == "__main__":
    unittest.main()
