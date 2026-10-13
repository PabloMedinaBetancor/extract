import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from extract.engine import ocr
from extract.engine.ocr import CharConf, load_engine, pair_confidences, read_page
from tests.support import SANS, SERIF, SPANISH_LINES, text_page_image


class FakeEngine:
    def __init__(self, lines):
        self.lines = lines

    def read_words(self, image):
        return self.lines


def fake_lines(count, conf):
    words = [(f"word{i}", conf, (10.0 * i, 0.0, 10.0 * i + 8, 10.0)) for i in range(count)]
    return [words[i : i + 5] for i in range(0, count, 5)]


class PageStatusTest(unittest.TestCase):
    blank = np.full((100, 100), 255, np.uint8)

    def test_O5_fewer_than_25_words_is_unreadable_and_keeps_its_text(self):
        reading = read_page(self.blank, FakeEngine(fake_lines(24, 0.99)))
        self.assertEqual((reading.status, reading.reason), ("unreadable", "page.reason.few_words"))
        self.assertEqual(len(reading.words), 24)

    def test_O5_low_mean_confidence_is_unreadable(self):
        reading = read_page(self.blank, FakeEngine(fake_lines(40, 0.60)))
        self.assertEqual((reading.status, reading.reason), ("unreadable", "page.reason.low_confidence"))

    def test_O5_a_readable_page_has_no_reason(self):
        reading = read_page(self.blank, FakeEngine(fake_lines(25, 0.9)))
        self.assertEqual((reading.status, reading.reason), ("ocr", None))
        self.assertAlmostEqual(reading.mean_conf, 0.9)

    def test_O5_words_of_a_page_without_text_are_none(self):
        reading = read_page(self.blank, FakeEngine([]))
        self.assertEqual((reading.status, reading.words, reading.mean_conf), ("unreadable", (), None))

    def test_O5_ids_and_lines_follow_the_reading(self):
        reading = read_page(self.blank, FakeEngine(fake_lines(12, 0.9)), number=7)
        self.assertEqual(reading.words[0].id, "p7w0")
        self.assertEqual([w.line_id for w in reading.words[3:7]], [0, 0, 1, 1])
        self.assertEqual([w.idx_in_line for w in reading.words[3:7]], [3, 4, 0, 1])


class UnalignedWordsTest(unittest.TestCase):
    blank = np.full((100, 100), 255, np.uint8)

    @staticmethod
    def conf(value, aligned):
        confidence = CharConf(value)
        confidence.chars, confidence.aligned = (value,) * 5, aligned
        return confidence

    def lines(self):
        box = (0.0, 0.0, 10.0, 10.0)
        return [[("aligned", self.conf(0.9, True), box), ("lost", self.conf(0.8, False), box), ("plain", 0.7, box)]]

    def test_O6_a_word_that_could_not_be_aligned_has_no_character_confidences_and_is_counted(self):
        reading = read_page(self.blank, FakeEngine(self.lines()))
        by_text = {w.text: w for w in reading.words}
        self.assertEqual(by_text["aligned"].char_conf, (0.9,) * 5)
        self.assertIsNone(by_text["lost"].char_conf)
        self.assertEqual(by_text["lost"].conf, 0.8)
        self.assertIsNone(by_text["plain"].char_conf)
        self.assertEqual(reading.unaligned_words, 1)

    def test_O6_the_count_is_kept_by_the_cache_the_json_and_the_pool_log(self):
        from extract.engine.cache import reading_from_dict, reading_to_dict

        reading = read_page(self.blank, FakeEngine(self.lines()))
        self.assertEqual(reading_from_dict(reading_to_dict(reading)), reading)
        self.assertEqual(reading_to_dict(reading)["unaligned_words"], 1)


class PairConfidencesTest(unittest.TestCase):
    def test_O6_each_word_gets_the_confidences_of_its_own_characters(self):
        text = "Total invoice 12.5"
        confs = [0.01 * (i + 1) for i in range(len(text))]
        paired = pair_confidences(text, ["Total", "invoice", "12.5"], confs)
        self.assertEqual([p.chars for p in paired], [tuple(confs[0:5]), tuple(confs[6:13]), tuple(confs[14:18])])
        self.assertAlmostEqual(float(paired[1]), sum(confs[6:13]) / 7)
        self.assertTrue(all(p.aligned for p in paired))

    def test_O6_misaligned_text_falls_back_to_the_line_mean_and_says_so(self):
        paired = pair_confidences("ab cd", ["ab", "cd"], [0.2, 0.4])
        self.assertAlmostEqual(float(paired[0]), 0.3)
        self.assertAlmostEqual(float(paired[1]), 0.3)
        self.assertFalse(any(p.aligned for p in paired))

    def test_O6_a_word_split_by_double_spaces_still_aligns(self):
        paired = pair_confidences("a  bc", ["a", "bc"], [0.1, 0.9, 0.9, 0.4, 0.6])
        self.assertEqual([float(p) for p in paired], [0.1, 0.5])


class EngineTest(unittest.TestCase):
    """Runs the real models (onnxruntime, two threads)."""

    @classmethod
    def setUpClass(cls):
        cls.engine = load_engine(threads=2, backend="onnxruntime")
        cls.tmp = Path(tempfile.mkdtemp())

    def test_R2_every_word_of_real_pages_has_alternatives_of_the_length_of_its_text(self):
        latin = ocr.build_latin_set(self.engine.ocr.text_rec.postprocess_op.character)
        pages = [
            text_page_image(SPANISH_LINES, SANS, 26, 1400, 900),
            text_page_image(SPANISH_LINES[:5], SERIF, 30, 1400, 700),
            text_page_image([line.upper() for line in SPANISH_LINES[:6]], SANS, 28, 1500, 800),
            text_page_image(list(reversed(SPANISH_LINES)), SERIF, 24, 1500, 850),
            text_page_image(SPANISH_LINES[2:], SANS, 22, 1300, 800, spacing=1.9),
        ]
        checked = 0
        for page in pages:
            for word in read_page(page, self.engine).words:
                if word.char_conf is None:
                    self.assertIsNone(word.char_alts)
                    continue
                self.assertEqual(len(word.char_alts), len(word.text), word.text)
                for char, conf, entries in zip(word.text, word.char_conf, word.char_alts, strict=True):
                    checked += 1
                    self.assertEqual(entries[0], (char, conf))
                    self.assertLessEqual(len(entries), 4)
                    self.assertTrue(all(c in latin.chars for c, _ in entries[1:]), entries)
                    self.assertEqual(sorted(p for _, p in entries[1:3]), [p for _, p in entries[1:3]][::-1])
        self.assertGreater(checked, 500)

    def test_R3_masked_never_changes_the_text_and_is_none_on_plain_latin_pages(self):
        reading = read_page(text_page_image(SPANISH_LINES, SANS, 26, 1400, 900), self.engine)
        self.assertTrue(all(w.masked is None or w.masked != w.text for w in reading.words))
        self.assertGreater(len(reading.words), 30)
        self.assertEqual([w.masked for w in reading.words], [None] * len(reading.words))

    def test_O1_a_path_reads_like_the_array(self):
        page = text_page_image(SPANISH_LINES[:4], SANS, 26, 1300, 500)
        path = self.tmp / "path.png"
        cv2.imwrite(str(path), page)
        from_path = read_page(path, self.engine)
        from_array = read_page(cv2.imread(str(path)), self.engine)
        self.assertEqual(from_path, from_array)
        self.assertGreater(len(from_path.words), 25)

    def test_O5_a_blank_page_is_unreadable(self):
        reading = read_page(np.full((600, 800), 255, np.uint8), self.engine)
        self.assertEqual((reading.status, reading.reason), ("unreadable", "page.reason.few_words"))

    def test_O6_the_real_decoder_confidences_end_up_on_the_right_words(self):
        from rapidocr.ch_ppocr_rec import utils as rec_utils

        captured = {}
        original = rec_utils.CTCLabelDecode.decode

        def tag_positions(self, *args, **kwargs):
            lines, infos = original(self, *args, **kwargs)
            for (text, _), info in zip(lines, infos, strict=True):
                info.confs = [round(0.001 * (i + 1), 4) for i in range(len(info.confs))]
                captured[text] = list(info.confs)
            return lines, infos

        image = np.full((300, 1500, 3), 255, np.uint8)
        cv2.putText(image, "Total invoice amount 1234.56 paid", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
        cv2.putText(image, "second line shows other words", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
        rec_utils.CTCLabelDecode.decode = tag_positions
        try:
            reading = read_page(image, self.engine)
        finally:
            rec_utils.CTCLabelDecode.decode = original
        self.assertGreaterEqual(len(reading.words), 9)
        rapidocr_would_say = []
        for word in reading.words:
            line = next(text for text in captured if word.text in text)
            confs = captured[line]
            start = line.index(word.text)
            own = confs[start : start + len(word.text)]
            self.assertEqual(word.char_conf, tuple(own), word.text)
            self.assertAlmostEqual(word.conf, sum(own) / len(own))
            rapidocr_would_say.append(confs[word.idx_in_line])
        self.assertNotEqual([round(w.conf, 4) for w in reading.words], [round(c, 4) for c in rapidocr_would_say])

    def test_O6_the_confidence_of_each_character_survives_in_a_real_page(self):
        reading = read_page(text_page_image(SPANISH_LINES, SANS, 26, 1400, 900), self.engine)
        for word in reading.words:
            self.assertEqual(len(word.char_conf), len(word.text), word.text)
            self.assertTrue(all(0.0 <= c <= 1.0 for c in word.char_conf))
        self.assertIsInstance(ocr.pair_confidences("a", ["a"], [0.5])[0], CharConf)


if __name__ == "__main__":
    unittest.main()
