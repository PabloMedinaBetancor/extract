import subprocess
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pikepdf
import pypdfium2 as pdfium

from extract.engine.native import read_native
from extract.engine.render import render_page
from extract.outputs.pdf import add_text_layer, display_to_user, encode_winansi, text_width
from tests.support import SAMPLE, SANS, SPANISH_LINES, grid_reading, scanned_pdf, text_page_image

WORDS = SAMPLE["pdf_words"]


def pdftotext(path, *flags):
    return subprocess.run(["pdftotext", *flags, str(path), "-"], capture_output=True, text=True, check=True).stdout


def pdfium_words(path, page_index=0):
    document = pdfium.PdfDocument(str(path))
    page = document[page_index]
    textpage = page.get_textpage()
    try:
        return textpage.get_text_range().split()
    finally:
        textpage.close()
        page.close()
        document.close()


def render(path, page_index=0, dpi=100):
    document = pdfium.PdfDocument(str(path))
    page = document[page_index]
    try:
        return render_page(page, dpi)
    finally:
        page.close()
        document.close()


def space_slack(word):
    """The layer sets each word with a space after it, inside the box (B21), so the glyphs a viewer reports end this share of
    the box before its right edge."""
    data = word.text.encode("latin-1", "replace")
    return (word.box[2] - word.box[0]) * text_width(b" ", 1000) / text_width(data + b" ", 1000)


class SearchablePdfTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        pages = [text_page_image(SPANISH_LINES, SANS, 24, 1240, 1754) for _ in range(2)]
        self.source = scanned_pdf(self.tmp / "scan.pdf", pages)
        self.readings = [grid_reading(1, WORDS * 3), grid_reading(2, [w.upper() for w in WORDS] * 2)]
        self.out = self.tmp / "out.pdf"
        self.replaced = add_text_layer(self.source, self.readings, self.out)

    def test_U2_the_words_come_back_in_order_with_accents_enye_and_euro(self):
        expected = WORDS * 3
        self.assertEqual(pdftotext(self.out, "-f", "1", "-l", "1", "-raw").split(), expected)
        self.assertEqual(pdfium_words(self.out, 0), expected)
        self.assertEqual(pdfium_words(self.out, 1), [w.upper() for w in WORDS] * 2)

    def test_U2_the_original_page_streams_are_unchanged(self):
        with pikepdf.open(self.source) as before, pikepdf.open(self.out) as after:
            for original, written in zip(before.pages, after.pages, strict=True):
                contents = list(written.obj["/Contents"])
                self.assertEqual([c.read_bytes() for c in contents[1:-1]], [original.obj["/Contents"].read_bytes()])

    def test_U2_the_page_renders_pixel_equal_to_the_original(self):
        for index in (0, 1):
            self.assertTrue(np.array_equal(render(self.source, index), render(self.out, index)))

    def test_U2_the_file_grows_by_at_most_10_percent_and_2_kb_a_page(self):
        original, grown = self.source.stat().st_size, self.out.stat().st_size
        self.assertLessEqual(grown, original * 1.10 + 2048 * 2)

    def test_U2_a_page_of_600_words_stays_within_the_growth_limit(self):
        texts = [f"palabra{n}" for n in range(600)]
        image = text_page_image(SPANISH_LINES * 4, SANS, 18, 1240, 1754, margin=60, spacing=1.3)
        source = scanned_pdf(self.tmp / "dense.pdf", [image])
        out = self.tmp / "dense-out.pdf"
        add_text_layer(source, [grid_reading(1, texts, per_row=10)], out)
        self.assertLessEqual(out.stat().st_size, source.stat().st_size * 1.10 + 2048)

    def test_U2_characters_outside_winansi_are_counted_and_written_as_question_marks(self):
        self.assertEqual(encode_winansi("a中b"), (b"a?b", 1))
        out = self.tmp / "cjk.pdf"
        replaced = add_text_layer(self.source, [grid_reading(1, ["中文", "ok"] * 20)], out)
        self.assertEqual(replaced, 40)
        self.assertEqual(pdfium_words(out)[:2], ["??", "ok"])

    def test_U2_native_pages_get_no_layer(self):
        native = grid_reading(1, WORDS)
        native = type(native)(**{**native.__dict__, "status": "native"})
        out = self.tmp / "native.pdf"
        add_text_layer(self.source, [native], out)
        self.assertEqual(pdfium_words(out), [])

    def test_U2_words_land_where_their_boxes_say_on_rotated_and_cropped_pages(self):
        crop = (40.0, 60.0, 500.0, 700.0)
        for rotation in (0, 90, 180, 270):
            with self.subTest(rotation=rotation):
                path = self.tmp / f"r{rotation}.pdf"
                source = scanned_pdf(
                    self.tmp / f"scan-r{rotation}.pdf", [text_page_image(SPANISH_LINES, SANS, 24, 1240, 1754)]
                )
                with pikepdf.open(source) as pdf:
                    pdf.pages[0].Rotate = rotation
                    pdf.pages[0].CropBox = pikepdf.Array(crop)
                    pdf.save(path)
                displayed = (460.0, 640.0) if rotation in (0, 180) else (640.0, 460.0)
                reading = grid_reading(
                    1, ["alpha", "beta", "gamma", "delta"] * 12, dpi=100, size_pt=displayed, per_row=6
                )
                out = self.tmp / f"r{rotation}-out.pdf"
                add_text_layer(path, [reading], out)
                document = pdfium.PdfDocument(str(out))
                page = document[0]
                try:
                    found = read_native(page, 1, None, ("en",), 100)
                finally:
                    page.close()
                    document.close()
                self.assertIsNotNone(found)
                self.assertEqual([w.text for w in found.words], [w.text for w in reading.words])
                for read, wanted in zip(found.words, reading.words, strict=True):
                    self.assertAlmostEqual(read.box[0], wanted.box[0], delta=3.0, msg=wanted.text)
                    self.assertAlmostEqual(read.box[2], wanted.box[2], delta=3.0 + space_slack(wanted), msg=wanted.text)
                    got_mid, want_mid = (read.box[1] + read.box[3]) / 2, (wanted.box[1] + wanted.box[3]) / 2
                    self.assertAlmostEqual(got_mid, want_mid, delta=12.0, msg=wanted.text)

    def test_U2_a_page_whose_content_ends_unbalanced_does_not_move_the_layer(self):
        for tail in (b"\nq 0.5 0 0 0.5 100 100 cm\n", b"\nq 0 0 10 10 re W n\n"):
            with self.subTest(tail=tail):
                source = scanned_pdf(self.tmp / "open.pdf", [text_page_image(SPANISH_LINES, SANS, 24, 1240, 1754)])
                unbalanced = self.tmp / "unbalanced.pdf"
                with pikepdf.open(source) as pdf:
                    page = pdf.pages[0]
                    stream = page.obj["/Contents"]
                    stream.write(stream.read_bytes() + tail)
                    pdf.save(unbalanced)
                reading = grid_reading(
                    1, ["alpha", "beta", "gamma", "delta"] * 12, dpi=100, size_pt=(595.28, 841.89), per_row=6
                )
                out = self.tmp / "unbalanced-out.pdf"
                add_text_layer(unbalanced, [reading], out)
                document = pdfium.PdfDocument(str(out))
                page = document[0]
                try:
                    found = read_native(page, 1, None, ("en",), 100)
                finally:
                    page.close()
                    document.close()
                self.assertEqual([w.text for w in found.words], [w.text for w in reading.words])
                for read, wanted in zip(found.words, reading.words, strict=True):
                    self.assertAlmostEqual(read.box[0], wanted.box[0], delta=3.0)
                    self.assertAlmostEqual(read.box[2], wanted.box[2], delta=3.0 + space_slack(wanted))

    def test_U2_the_matrix_of_each_rotation_maps_the_display_corners_to_the_crop_box(self):
        source = scanned_pdf(self.tmp / "m.pdf", [text_page_image(SPANISH_LINES, SANS, 24, 1240, 1754)])
        for rotation, corner in ((0, (40, 60)), (90, (500, 60)), (180, (500, 700)), (270, (40, 700))):
            with pikepdf.open(source) as pdf:
                pdf.pages[0].Rotate = rotation
                pdf.pages[0].CropBox = pikepdf.Array([40, 60, 500, 700])
                a, b, c, d, e, f = display_to_user(pdf.pages[0].obj)
            self.assertEqual((e, f), corner, rotation)


if __name__ == "__main__":
    unittest.main()
