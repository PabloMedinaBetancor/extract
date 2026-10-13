import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from bench.corpus import sealed
from bench.guard import dataset
from extract.engine.cache import reading_to_dict
from extract.guard import signals
from extract.types import PageReading, Word


def word(i, text, x, conf=0.95, y=0):
    return Word(f"p1w{i}", 1, text, (x, y, x + 8.0 * len(text), y + 12.0), conf, tuple([conf] * len(text)), 0, i)


def page_record(doc, number, band, read_words, truth_words, ungrounded=(), status="ocr"):
    reading = PageReading(number, 800, 600, 400.0, 300.0, 144, tuple(read_words), status)
    return {
        "doc_id": doc,
        "page": number,
        "band": band,
        "truth": {"words": [{"id": f"t{i}", "text": t, "box": list(b)} for i, (t, b) in enumerate(truth_words)]},
        "ungrounded": [list(b) for b in ungrounded],
        "ocr": {"engine": "e1", "reading": reading_to_dict(reading)},
    }


class DatasetTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.data = Path(folder.name)

    def write(self, set_name, record):
        target = self.data / "pages" / set_name / record["doc_id"]
        target.mkdir(parents=True, exist_ok=True)
        (target / f"{record['page']}.json").write_text(json.dumps(record), encoding="utf-8")

    def simple(self, doc="cA", number=1, set_name="train", wrong=False):
        read = [word(0, "obra", 0), word(1, "plano" if not wrong else "plana", 50), word(2, "casa", 110)]
        truth = [("obra", (0, 0, 32, 12)), ("plano", (50, 0, 90, 12)), ("casa", (110, 0, 142, 12))]
        self.write(set_name, page_record(doc, number, "light", read, truth))

    def test_G4_the_sealed_sets_are_refused_by_the_page_reader(self):
        for name in ("test5", "unseen5"):
            with self.assertRaises(sealed.SealedTestError):
                dataset.page_records(self.data, name)

    def test_V5B3_the_v4_test_sets_are_development_data_and_open_to_the_page_reader(self):
        for name in ("test", "unseen"):
            self.assertEqual(dataset.page_records(self.data, name), [])

    def test_G3_rows_carry_signals_labels_document_and_band(self):
        self.simple(wrong=True)
        rows = dataset.load_rows(self.data, "train")
        self.assertEqual(rows.x.shape, (3, len(signals.COLUMNS)))
        self.assertEqual(rows.y.tolist(), [0, 1, 0])
        self.assertEqual(set(rows.group), {"cA"})
        self.assertEqual(set(rows.band), {"light"})
        self.assertEqual(rows.kind.tolist(), ["ok", "substitution", "ok"])
        self.assertEqual(rows.detail[1], "letters")
        self.assertEqual((rows.text[1], rows.truth[1]), ("plana", "plano"))
        self.assertEqual(rows.box.shape, (3, 4))
        self.assertEqual(rows.chars[1], (0.95,) * 5)

    def test_G3_pages_without_a_reading_are_not_rows(self):
        self.simple(doc="cA")
        record = page_record("cB", 1, "light", [], [])
        record["ocr"] = None
        self.write("train", record)
        self.assertEqual(set(dataset.load_rows(self.data, "train").group), {"cA"})

    def test_G3_ungrounded_words_are_left_out_and_counted_while_invented_ones_stay(self):
        read = [word(0, "obra", 0), word(1, "Gobierno", 100), word(2, "fantasma", 300)]
        truth = [("obra", (0, 0, 32, 12))]
        self.write("train", page_record("cA", 1, "light", read, truth, ungrounded=[(95, 0, 170, 14)]))
        rows = dataset.load_rows(self.data, "train")
        self.assertEqual(rows.kind.tolist(), ["ok", "invented"])
        self.assertEqual(rows.y.tolist(), [0, 1])
        self.assertEqual(rows.counts["ungrounded"], 1)

    def test_G3_unreadable_pages_are_counted_and_have_no_rows(self):
        self.simple(doc="cA")
        record = page_record("cB", 1, "heavy", [word(0, "x", 0)], [("x", (0, 0, 8, 12))], status="unreadable")
        self.write("train", record)
        rows = dataset.load_rows(self.data, "train")
        self.assertEqual(set(rows.group), {"cA"})
        self.assertEqual(rows.counts["unreadable_pages"], 1)
        self.assertEqual(rows.counts["pages"], 2)

    def test_G3_missed_words_are_counted_and_never_rows(self):
        read = [word(0, "obra", 0)]
        truth = [("obra", (0, 0, 32, 12)), ("perdida", (50, 0, 106, 12))]
        self.write("train", page_record("cA", 1, "light", read, truth))
        rows = dataset.load_rows(self.data, "train")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.counts["missed"], 1)

    def test_G4_pages_read_by_another_engine_are_refused_when_one_is_named(self):
        self.simple()
        with self.assertRaises(ValueError):
            dataset.load_rows(self.data, "train", engine="other")
        self.assertEqual(len(dataset.load_rows(self.data, "train", engine="e1")), 3)

    def test_G4_the_signals_do_not_depend_on_the_truth(self):
        self.simple(doc="cA")
        first = dataset.load_rows(self.data, "train")
        for path in (self.data / "pages" / "train").glob("*/*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            for entry in record["truth"]["words"]:
                entry["text"] = "basura"
            path.write_text(json.dumps(record), encoding="utf-8")
        second = dataset.load_rows(self.data, "train")
        self.assertEqual(first.x.tobytes(), second.x.tobytes())
        self.assertNotEqual(first.y.tolist(), second.y.tolist())

    def test_G3_an_empty_set_gives_no_rows_and_a_matrix_of_the_right_width(self):
        rows = dataset.load_rows(self.data, "val")
        self.assertEqual(rows.x.shape, (0, len(signals.COLUMNS)))
        self.assertEqual(len(rows), 0)

    def test_G3_subset_keeps_every_column_aligned(self):
        self.simple(doc="cA", wrong=True)
        rows = dataset.load_rows(self.data, "train")
        wrong = rows.subset(rows.y == 1)
        self.assertEqual(len(wrong), 1)
        self.assertEqual((wrong.kind[0], wrong.page[0]), ("substitution", "cA/1"))
        self.assertEqual(wrong.x.shape[0], 1)
        self.assertTrue(np.array_equal(wrong.x[0], rows.x[1], equal_nan=True))


if __name__ == "__main__":
    unittest.main()
