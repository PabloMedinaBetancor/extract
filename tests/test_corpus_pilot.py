import unittest
from types import SimpleNamespace

from bench.corpus.pilot import band_report, page_stats


def reading(words):
    return SimpleNamespace(words=[SimpleNamespace(id=f"p1w{i}", text=t, box=b) for i, (t, b) in enumerate(words)])


def record(words):
    return {"truth": {"words": [{"id": f"t{i}", "text": t, "box": list(b)} for i, (t, b) in enumerate(words)]}}


class PilotTest(unittest.TestCase):
    def test_G3_page_stats_count_wrong_missed_and_boundary_words(self):
        truth = [
            ("uno", (0, 0, 30, 10)),
            ("dos", (40, 0, 70, 10)),
            ("tres", (80, 0, 110, 10)),
            ("cuatro", (120, 0, 170, 10)),
        ]
        read = [("uno", (0, 0, 30, 10)), ("dox", (40, 0, 70, 10)), ("cuatro", (120, 0, 170, 10))]
        stats = page_stats(record(truth), reading(read))
        self.assertEqual((stats["ocr_words"], stats["wrong"], stats["missed"]), (3, 1, 1))
        self.assertEqual(stats["kinds"], {"ok": 2, "substitution": 1})

    def test_G3_the_band_report_pools_words_and_checks_the_target(self):
        stats = {"ocr_words": 100, "truth_words": 100, "wrong": 2, "missed": 0, "boundary_only": 0, "kinds": {"ok": 98}}
        report = band_report([("light", stats), ("light", {**stats, "wrong": 1}), ("heavy", {**stats, "wrong": 30})])
        self.assertEqual(report["light"]["wrong_pct"], 1.5)
        self.assertTrue(report["light"]["on_target"])
        self.assertFalse(report["heavy"]["on_target"])


if __name__ == "__main__":
    unittest.main()
