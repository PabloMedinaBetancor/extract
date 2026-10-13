import json
import random
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from bench.guard import run_validation
from extract.engine.cache import reading_to_dict
from extract.guard import dictionary
from extract.guard.model import GuardModel
from extract.types import PageReading, Word

CONFUSED = {"c": "e", "a": "o", "u": "v", "n": "h", "i": "l", "t": "f", "e": "c", "o": "a"}


def write_page(data, set_name, doc, number, band, rng, words_per_page=80):
    book = dictionary.load("es")
    pool = [w for w in book.words[200:20000] if len(w) >= 4]
    read, truth = [], []
    for i in range(words_per_page):
        right = rng.choice(pool)
        text, conf = right, rng.uniform(0.85, 0.99)
        if rng.random() < 0.12:
            position = rng.randrange(len(right))
            text = right[:position] + CONFUSED.get(right[position], "x") + right[position + 1 :]
            conf = rng.uniform(0.4, 0.8)
        x = (i % 10) * 90.0
        y = (i // 10) * 20.0
        read.append(
            Word(
                f"p{number}w{i}",
                number,
                text,
                (x, y, x + 7.0 * len(text), y + 14.0),
                conf,
                tuple([conf] * len(text)),
                i // 10,
                i % 10,
            )
        )
        truth.append({"id": f"t{i}", "text": right, "box": [x, y, x + 7.0 * len(right), y + 14.0]})
    reading = PageReading(number, 900, 300, 450.0, 150.0, 144, tuple(read))
    record = {
        "schema": 1,
        "set": set_name,
        "doc_id": doc,
        "page": number,
        "band": band,
        "truth": {"words": truth},
        "ungrounded": [],
        "ocr": {"engine": "e1", "reading": reading_to_dict(reading)},
    }
    folder = data / "pages" / set_name / doc
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{number}.json").write_text(json.dumps(record), encoding="utf-8")
    cv2.imwrite(str(folder / f"{number}.png"), np.full((300, 900), 255, np.uint8))


class StudyEndToEndTest(unittest.TestCase):
    def test_G5_the_whole_study_runs_on_a_small_corpus_and_writes_a_loadable_model_and_a_report(self):
        rng = random.Random(1)
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder) / "data"
            for set_name, documents, bands in (
                ("train", 24, ("light", "medium")),
                ("val", 24, ("light", "medium")),
                ("clean", 4, ("clean",)),
            ):
                for d in range(documents):
                    for page in (1, 2):
                        write_page(data, set_name, f"{set_name}{d}", page, rng.choice(bands), rng)
            out, model_path = Path(folder) / "report.json", Path(folder) / "guard.model"
            run_validation.main(
                [
                    "--data-dir",
                    str(data),
                    "--out",
                    str(out),
                    "--model",
                    str(model_path),
                    "--audit-dir",
                    str(Path(folder) / "audit"),
                    "--resamples",
                    "30",
                    "--engine",
                    "e1",
                ]
            )
            report = json.loads(out.read_text(encoding="utf-8"))
            guard = GuardModel.load(model_path)
            sheets = sorted((Path(folder) / "audit").glob("sheet-*.png"))
        self.assertEqual(report["rows"]["train"]["words"], 24 * 2 * 80)
        self.assertTrue(report["g3_readback"]["ok_share"] > 0.8)
        self.assertEqual(report["grid"]["chosen"]["trees"] in (100, 200), True)
        self.assertIn("detector_minus_best_baseline", report["comparison_at_5_percent"])
        self.assertGreater(report["fixer"]["candidate_recall"]["candidate_recall"], 0.8)
        self.assertTrue(report["permutation_canary"]["passes"], report["permutation_canary"])
        self.assertEqual(
            set(report["ablation"]),
            {"none", "without_engine", "without_dictionary", "without_shape", "without_layout", "without_context"},
        )
        self.assertEqual([c["fraction"] for c in report["learning_curve"]], [0.25, 0.5, 1.0])
        self.assertTrue(guard.meta["calibrated"])
        self.assertIsNotNone(guard.ranker)
        self.assertEqual(len(sheets), run_validation.AUDIT_SHEETS)


if __name__ == "__main__":
    unittest.main()
