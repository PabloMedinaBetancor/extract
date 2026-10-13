import tempfile
import unittest
from pathlib import Path

import pypdfium2 as pdfium
from PIL import Image
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from extract.engine.render import page_dpi, render_page
from tests.support import SPANISH_LINES, digital_pdf, scanned_pdf, text_page_image

A4_WIDTH_INCHES = A4[0] / 72


def open_page(path):
    document = pdfium.PdfDocument(str(path))
    return document, document[0]


class RenderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def dpi_of(self, path):
        document, page = open_page(path)
        try:
            return page_dpi(page)
        finally:
            page.close()
            document.close()

    def scan_with_width(self, pixels):
        image = text_page_image(SPANISH_LINES, width=pixels, height=round(pixels * A4[1] / A4[0]), size=pixels // 60)
        return scanned_pdf(self.tmp / f"scan{pixels}.pdf", [image])

    def test_O2_a_scan_is_rendered_at_its_own_resolution(self):
        self.assertEqual(self.dpi_of(self.scan_with_width(1700)), round(1700 / A4_WIDTH_INCHES))

    def test_O2_resolution_is_clamped_to_100_and_300_dpi(self):
        self.assertEqual(self.dpi_of(self.scan_with_width(500)), 100)
        # 300 dpi on A4 would be 3,508 px tall: the 3,000 px cap lowers it further.
        self.assertEqual(self.dpi_of(self.scan_with_width(3000)), int(3000 * 72 / A4[1]))

    def test_O2_a_page_without_images_is_rendered_at_200_dpi(self):
        self.assertEqual(self.dpi_of(digital_pdf(self.tmp / "d.pdf", [SPANISH_LINES])), 200)

    def test_O2_a_small_logo_does_not_pass_for_a_scan(self):
        path = self.tmp / "logo.pdf"
        pdf = canvas.Canvas(str(path), pagesize=A4)
        logo = Image.new("L", (120, 40), 30)
        pdf.drawImage(ImageReader(logo), 50, 780, width=60, height=20)
        pdf.drawString(50, 700, "Some text")
        pdf.save()
        self.assertEqual(self.dpi_of(path), 200)

    def test_O2_the_longest_side_never_goes_over_3000_px(self):
        path = self.tmp / "plan.pdf"
        plan = (1684, 2384)
        pdf = canvas.Canvas(str(path), pagesize=plan)
        pdf.drawImage(ImageReader(Image.new("L", (1500, 2120), 200)), 0, 0, width=plan[0], height=plan[1])
        pdf.save()
        document, page = open_page(path)
        try:
            dpi = page_dpi(page)
            array = render_page(page, dpi)
        finally:
            page.close()
            document.close()
        self.assertLessEqual(max(array.shape), 3000)
        self.assertLess(dpi, 100)

    def test_O2_render_gives_a_grey_array_of_the_expected_size(self):
        document, page = open_page(digital_pdf(self.tmp / "d.pdf", [SPANISH_LINES]))
        try:
            array = render_page(page, 144)
        finally:
            page.close()
            document.close()
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.dtype.name, "uint8")
        self.assertAlmostEqual(array.shape[1], A4[0] * 2, delta=1)
        self.assertLess(array.min(), 100)


if __name__ == "__main__":
    unittest.main()
