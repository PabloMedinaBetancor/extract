import tempfile
import unittest
from pathlib import Path

import numpy as np
import pypdfium2 as pdfium

from extract.engine.native import is_native, read_native
from extract.engine.render import render_page
from tests.support import (
    SPANISH_LINES,
    SetLexicon,
    broken_layer_pdf,
    digital_pdf,
    rotated_cropped_pdf,
)

SPANISH_WORDS = {w.lower().strip(".,:") for line in SPANISH_LINES for w in line.split() if len(w) > 2}


def native_reading(path, lexicon=None, dpi=144):
    document = pdfium.PdfDocument(str(path))
    page = document[0]
    try:
        return read_native(page, 1, lexicon, ("es", "en"), dpi), render_page(page, dpi)
    finally:
        page.close()
        document.close()


class NativeTextTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_O3_a_legible_layer_is_used_as_it_is(self):
        reading, _ = native_reading(digital_pdf(self.tmp / "d.pdf", [SPANISH_LINES]), SetLexicon(SPANISH_WORDS))
        self.assertEqual(reading.status, "native")
        text = " ".join(w.text for w in reading.words)
        self.assertEqual(text, " ".join(" ".join(SPANISH_LINES).split()))
        self.assertTrue(all(w.status == "native" and w.conf is None and w.char_conf is None for w in reading.words))
        self.assertEqual(reading.words[0].line_id, 0)
        self.assertEqual(reading.words[-1].line_id, len(SPANISH_LINES) - 1)

    def test_O3_native_words_come_with_boxes_in_reading_order(self):
        reading, _ = native_reading(digital_pdf(self.tmp / "d.pdf", [SPANISH_LINES]))
        line_tops = {}
        for word in reading.words:
            line_tops.setdefault(word.line_id, []).append(word.box[1])
        means = [sum(v) / len(v) for _, v in sorted(line_tops.items())]
        self.assertEqual(means, sorted(means))
        self.assertEqual(len(means), len(SPANISH_LINES))
        for word in reading.words:
            left, top, right, bottom = word.box
            self.assertTrue(0 <= left < right <= reading.width_px and 0 <= top < bottom <= reading.height_px)

    def test_O3_a_broken_layer_goes_to_ocr(self):
        path = broken_layer_pdf(self.tmp / "broken.pdf")
        document = pdfium.PdfDocument(str(path))
        page = document[0]
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        textpage.close()
        page.close()
        document.close()
        self.assertGreaterEqual(len(text.replace(" ", "")), 80, "the fixture must carry a layer, or it proves nothing")
        reading, _ = native_reading(path, SetLexicon(SPANISH_WORDS))
        self.assertIsNone(reading)

    def test_O3_a_page_with_almost_no_text_is_not_native(self):
        reading, _ = native_reading(digital_pdf(self.tmp / "short.pdf", [["Only a few words here"]]))
        self.assertIsNone(reading)

    def test_O3_a_text_needs_80_characters_to_count_as_native(self):
        self.assertFalse(is_native("short text", None))
        self.assertTrue(is_native(" ".join(SPANISH_LINES), None))

    def test_O3_boxes_land_on_the_ink_of_a_rotated_and_cropped_page(self):
        for rotation in (0, 90, 180, 270):
            with self.subTest(rotation=rotation):
                path = rotated_cropped_pdf(self.tmp / f"r{rotation}.pdf", rotation)
                reading, image = native_reading(path)
                self.assertIsNotNone(reading)
                self.assertGreaterEqual(len(reading.words), 100)
                ink = image < 128
                covered = np.zeros_like(ink)
                for word in reading.words:
                    x0, y0, x1, y1 = (round(v) for v in word.box)
                    covered[max(y0 - 3, 0) : y1 + 3, max(x0 - 3, 0) : x1 + 3] = True
                    self.assertGreater(ink[y0:y1, x0:x1].mean(), 0.02, word.text)
                self.assertLess(ink[~covered].sum() / ink.sum(), 0.02)


if __name__ == "__main__":
    unittest.main()
