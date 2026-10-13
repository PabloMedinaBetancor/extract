import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from bench.guard import readback
from tests.test_guard_dataset import page_record, word


class ReadbackTest(unittest.TestCase):
    def record(self, texts, truth_texts, ungrounded=()):
        read = [word(i, t, i * 60) for i, t in enumerate(texts)]
        truth = [(t, (i * 60, 0, i * 60 + 8.0 * len(t), 12)) for i, t in enumerate(truth_texts)]
        return page_record("cA", 1, "clean", read, truth, ungrounded)

    def test_G3_a_faithful_read_is_all_ok_and_meets_the_floor(self):
        report = readback.readback([self.record(["uno", "dos", "tres"], ["uno", "dos", "tres"])])
        self.assertEqual((report["ok"], report["ok_share"]), (3, 1.0))
        self.assertTrue(report["meets_floor"])

    def test_G3_a_misread_word_lowers_the_share_and_can_miss_the_floor(self):
        report = readback.readback([self.record(["uno", "dox", "tres"], ["uno", "dos", "tres"])])
        self.assertAlmostEqual(report["ok_share"], 2 / 3)
        self.assertFalse(report["meets_floor"])
        self.assertEqual(report["worst_pages"][0]["page"], "cA/1")

    def test_G3_ungrounded_words_are_not_counted_against_the_read(self):
        record = self.record(["uno", "logotipo"], ["uno"], ungrounded=[(55, 0, 140, 14)])
        report = readback.readback([record])
        self.assertEqual((report["ungrounded"], report["ok_share"]), (1, 1.0))

    def test_G3_missed_and_boundary_cases_are_reported_apart(self):
        record = self.record(["uno"], ["uno", "perdida"])
        self.assertEqual(readback.readback([record])["missed"], 1)


class AuditTest(unittest.TestCase):
    def test_G3_the_sample_has_equal_wrong_and_right_words_and_is_reproducible(self):
        y = np.array([1] * 100 + [0] * 500, dtype=np.uint8)
        rows = SimpleNamespace(y=y)
        first = readback.audit_sample(rows, per_class=60, seed=3)
        self.assertEqual(first, readback.audit_sample(rows, per_class=60, seed=3))
        self.assertEqual(int(y[first].sum()), 60)
        self.assertEqual(len(first), 120)

    def test_G3_few_wrong_words_give_a_smaller_sample_not_an_error(self):
        rows = SimpleNamespace(y=np.array([1] * 5 + [0] * 500, dtype=np.uint8))
        self.assertEqual(len(readback.audit_sample(rows, per_class=60)), 65)

    def test_G3_a_sheet_holds_the_crops_with_both_texts(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            target = data / "pages" / "val" / "cA"
            target.mkdir(parents=True)
            cv2.imwrite(str(target / "1.png"), np.full((200, 300), 255, np.uint8))
            entries = [
                {"set": "val", "doc_id": "cA", "page": 1, "box": [10, 10, 80, 30], "ocr": "obra", "truth": "obras"}
            ] * 4
            readback.crop_sheet(entries, data / "out" / "sheet.png", data)
            sheet = cv2.imread(str(data / "out" / "sheet.png"), cv2.IMREAD_GRAYSCALE)
        self.assertEqual(sheet.shape, (220, 1260))
        self.assertLess(int(sheet.min()), 130)

    def test_G3_a_word_whose_box_lies_outside_the_page_still_gets_a_cell(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            target = data / "pages" / "val" / "cA"
            target.mkdir(parents=True)
            cv2.imwrite(str(target / "1.png"), np.full((200, 300), 255, np.uint8))
            entry = {"set": "val", "doc_id": "cA", "page": 1, "box": [400, 300, 420, 320], "ocr": "x", "truth": ""}
            readback.crop_sheet([entry], data / "out" / "sheet.png", data)
            self.assertTrue((data / "out" / "sheet.png").is_file())


if __name__ == "__main__":
    unittest.main()
