from __future__ import annotations

import numpy as np
import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw

MIN_DPI = 100
MAX_DPI = 300
DEFAULT_DPI = 200
MAX_SIDE_PX = 3000
SCAN_COVERAGE = 0.5


def scan_pixels(page: pdfium.PdfPage) -> int:
    """Width in pixels of the widest image that covers at least half of the page, or 0 when there is none."""
    width_pt, height_pt = page.get_size()
    widest = 0
    for obj in page.get_objects(filter=[pdfium_raw.FPDF_PAGEOBJ_IMAGE]):
        left, bottom, right, top = obj.get_bounds()
        if (right - left) * (top - bottom) >= SCAN_COVERAGE * width_pt * height_pt:
            widest = max(widest, obj.get_px_size()[0])
    return widest


def scan_dpi(page: pdfium.PdfPage) -> int | None:
    """The resolution the scan of the page really has (its pixels over the width in inches), before `page_dpi` raises
    it to 100; None when no image covers half of the page. Only for telling the user: nothing reads it."""
    scan = scan_pixels(page)
    return round(scan / (page.get_size()[0] / 72)) if scan else None


def page_dpi(page: pdfium.PdfPage) -> int:
    """The resolution to render a page at: that of its scan (the widest image that covers at least half the page,
    pixels over the page width in inches), clamped to 100-300 dpi, or 200 dpi when the page has no such image. The
    longest side never goes over 3,000 px."""
    width_pt, height_pt = page.get_size()
    scan = scan_pixels(page)
    dpi = round(scan / (width_pt / 72)) if scan else DEFAULT_DPI
    dpi = min(max(dpi, MIN_DPI), MAX_DPI)
    return min(dpi, int(MAX_SIDE_PX * 72 / max(width_pt, height_pt)))


def render_page(page: pdfium.PdfPage, dpi: int) -> np.ndarray:
    """The page as a grey (height, width) uint8 array. The bitmap is closed before returning."""
    bitmap = page.render(scale=dpi / 72, grayscale=True)
    try:
        array = bitmap.to_numpy()
        return np.array(array[:, :, 0] if array.ndim == 3 else array)
    finally:
        bitmap.close()
