import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from bench.corpus.overlay import draw_overlay


class OverlayTest(unittest.TestCase):
    def test_G3_the_overlay_marks_the_truth_boxes_in_red_and_shrinks_big_pages(self):
        with tempfile.TemporaryDirectory() as folder:
            image_path = Path(folder) / "page.png"
            cv2.imwrite(str(image_path), np.full((3600, 2400), 255, np.uint8))
            record = {"truth": {"words": [{"box": [100, 200, 300, 260]}]}}
            draw_overlay(image_path, record, Path(folder) / "out" / "page.png", max_side=1800)
            shown = cv2.imread(str(Path(folder) / "out" / "page.png"))
        self.assertEqual(shown.shape[:2], (1800, 1200))
        blue, green, red = (shown[:, :, i].astype(int) for i in range(3))
        reds = (red > blue + 40) & (red > green + 40)
        self.assertGreater(reds.sum(), 50)
        top = int(np.argwhere(reds)[:, 0].min())
        self.assertTrue(98 <= top <= 102, top)


if __name__ == "__main__":
    unittest.main()
