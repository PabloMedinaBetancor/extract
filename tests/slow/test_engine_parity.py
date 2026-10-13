import json
import os
import random
import unittest
from pathlib import Path

import cv2
import pytest

from extract.engine.ocr import fingerprint, load_engine, read_page

PAGES = 20
SETS = ("train", "clean5")  # the sets that may still hold readings of the v4 engine
SEED = 5


def v4_fingerprint():
    return fingerprint().removesuffix("|alts2")


def cached_v4_pages():
    """Pages of the corpus that the v4 engine read and cached, found through `EXTRACT_BENCH_DATA`. The v5 runs rewrite the
    cache of the sets they read (`train`, `val`, `test`... now hold v5 readings with the second-reading suffix), so the
    v4 caches that are left are the ones of the sets no v5 run touched: `clean5` (30 pages)."""
    found = []
    for name in SETS:
        root = Path(os.environ["EXTRACT_BENCH_DATA"]) / "pages" / name
        for path in sorted(root.glob("*/*.json")):
            ocr = json.loads(path.read_text(encoding="utf-8")).get("ocr")
            if ocr and ocr["engine"] == v4_fingerprint() and ocr["backend"] == "openvino":
                found.append((path, ocr["reading"]))
    random.Random(SEED).shuffle(found)
    return found[:PAGES]


@pytest.mark.corpus
class ParityWithV4Test(unittest.TestCase):
    def test_R1_the_first_reading_of_cached_v4_pages_is_the_same_words_boxes_and_confidences(self):
        engine = load_engine(threads=2, backend="openvino")
        pages = cached_v4_pages()
        self.assertEqual(len(pages), PAGES)
        for path, cached in pages:
            with self.subTest(page=str(path.relative_to(path.parents[2]))):
                image = cv2.imread(str(path.with_suffix(".png")))
                reading = read_page(
                    image, engine, cached["number"], cached["dpi"], (cached["width_pt"], cached["height_pt"])
                )
                self.assertEqual(len(reading.words), len(cached["words"]))
                for word, (wid, text, box, conf, chars, line, idx) in zip(reading.words, cached["words"], strict=True):
                    self.assertEqual(
                        (word.id, word.text, list(word.box), word.line_id, word.idx_in_line),
                        (wid, text, box, line, idx),
                    )
                    self.assertEqual(word.conf, conf)
                    self.assertEqual(None if word.char_conf is None else list(word.char_conf), chars)
                self.assertEqual(reading.status, cached["status"])


if __name__ == "__main__":
    unittest.main()
