import json
import tempfile
import unittest
from pathlib import Path

from extract.engine import ocr
from extract.engine.cache import PageCache, file_sha256, page_key, reading_from_dict, reading_to_dict
from extract.types import PageReading, Reading, Word


def sample_reading(page=2):
    words = (
        Word("p2w0", page, "Total", (10.5, 20.0, 80.0, 44.0), 0.9, (0.9, 0.8, 1.0, 1.0, 0.7), 0, 0),
        Word("p2w1", page, "1.234,56", (90.0, 20.0, 200.0, 44.0), 0.95, (0.95,) * 8, 0, 1),
    )
    return PageReading(page, 1654, 2339, 595.2, 841.9, 200, words, "ocr", None, None, 0.925)


def second_reading_page():
    alts = (
        (("T", 0.8), ("I", 0.1)),
        (("o", 0.99),),
    )
    readings = (
        Reading("latin", "To", 0.9, (0.9, 0.95), ()),
        Reading("bin", None, None, None, ()),
        Reading("medium", "T o", 0.8, (0.8, 0.7, 0.9), ("p2w0",), 3, True),
    )
    words = (
        Word(
            "p2w0",
            2,
            "To",
            (1.0, 2.0, 30.0, 12.0),
            0.9,
            (0.8, 0.99),
            0,
            0,
            char_alts=alts,
            masked="Io",
            readings=readings,
        ),
        Word("p2w1", 2, "x", (40.0, 2.0, 50.0, 12.0), 0.5, None, 0, 1),
    )
    return PageReading(
        2,
        100,
        100,
        72.0,
        72.0,
        72,
        words,
        "ocr",
        None,
        None,
        0.7,
        1,
        {0: ("latin", "bin", "medium")},
        {"lines_read": 1},
    )


class SecondReadingCacheTest(unittest.TestCase):
    def test_R8_alternatives_readings_and_the_page_report_survive_the_cache_and_json(self):
        page = second_reading_page()
        self.assertEqual(reading_from_dict(json.loads(json.dumps(reading_to_dict(page)))), page)

    def test_R8_a_page_read_by_v4_loads_without_the_new_fields(self):
        v4 = reading_to_dict(sample_reading())
        v4["words"] = [row[:7] for row in v4["words"]]
        del v4["rereads"], v4["reread_report"]
        loaded = reading_from_dict(v4)
        self.assertEqual(loaded, sample_reading())
        self.assertTrue(all(w.char_alts is None and w.readings is None for w in loaded.words))

    def test_R8_a_page_cached_without_the_second_reading_is_not_served_for_it_and_the_reverse(self):
        cache = PageCache(tempfile.mkdtemp())
        plain, with_rr = page_key("ab" * 32, 2, 200), page_key("ab" * 32, 2, 200, "rr1:latin:0.5:0.9:16")
        self.assertNotEqual(plain, with_rr)
        cache.put("ab" * 32, 2, plain, sample_reading())
        self.assertIsNone(cache.get("ab" * 32, 2, with_rr))
        cache.put("ab" * 32, 2, with_rr, second_reading_page())
        self.assertIsNone(cache.get("ab" * 32, 2, plain))
        self.assertEqual(cache.get("ab" * 32, 2, with_rr), second_reading_page())

    def test_R8_each_setting_of_the_second_reading_changes_the_key(self):
        keys = {
            page_key("ab" * 32, 2, 200, f"rr1:{r}:{a}:{b}:{c}")
            for r, a, b, c in [
                ("latin", 0.5, 0.9, 16),
                ("latin+bin", 0.5, 0.9, 16),
                ("latin", 0.6, 0.9, 16),
                ("latin", 0.5, 0.8, 16),
                ("latin", 0.5, 0.9, 8),
            ]
        }
        self.assertEqual(len(keys), 5)

    def test_R8_the_key_names_the_version_of_the_alternatives(self):
        self.assertIn("|alts2", ocr.fingerprint())


class PageCacheTest(unittest.TestCase):
    def setUp(self):
        self.cache = PageCache(tempfile.mkdtemp())

    def test_O7_a_reading_comes_back_equal(self):
        reading = sample_reading()
        key = page_key("ab" * 32, 2, 200)
        self.cache.put("ab" * 32, 2, key, reading)
        self.assertEqual(self.cache.get("ab" * 32, 2, key), reading)

    def test_O7_a_changed_setting_is_a_miss(self):
        self.cache.put("ab" * 32, 2, page_key("ab" * 32, 2, 200), sample_reading())
        self.assertIsNone(self.cache.get("ab" * 32, 2, page_key("ab" * 32, 2, 150)))
        self.assertIsNone(self.cache.get("ab" * 32, 2, page_key("cd" * 32, 2, 200)))

    def test_O7_a_changed_engine_is_a_miss(self):
        key = page_key("ab" * 32, 2, 200)
        original = ocr.rapidocr_version
        ocr.rapidocr_version = lambda: "0.0.0"
        try:
            other = page_key("ab" * 32, 2, 200)
        finally:
            ocr.rapidocr_version = original
        self.assertNotEqual(key, other)

    def test_O7_the_key_does_not_depend_on_the_guard(self):
        self.assertNotIn("guard", ocr.fingerprint())

    def test_O7_a_torn_file_is_dropped_and_read_again(self):
        key = page_key("ab" * 32, 2, 200)
        self.cache.put("ab" * 32, 2, key, sample_reading())
        path = next(self.cache.root.rglob("p0002.json"))
        path.write_text('{"key": "', encoding="utf-8")
        self.assertIsNone(self.cache.get("ab" * 32, 2, key))
        self.assertFalse(path.exists())

    def test_O7_nothing_is_left_under_a_temporary_name(self):
        self.cache.put("ab" * 32, 2, "k", sample_reading())
        self.assertEqual([p.name for p in self.cache.root.rglob("*.part")], [])

    def test_O7_the_file_is_plain_json(self):
        data = reading_to_dict(sample_reading())
        self.assertEqual(reading_from_dict(json.loads(json.dumps(data))), sample_reading())

    def test_O10_the_sha256_of_a_file_is_the_one_of_its_bytes(self):
        path = Path(tempfile.mkdtemp()) / "a.bin"
        path.write_bytes(b"abc")
        self.assertEqual(file_sha256(path), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad")


if __name__ == "__main__":
    unittest.main()
