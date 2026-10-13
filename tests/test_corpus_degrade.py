import unittest
from dataclasses import replace

import cv2
import numpy as np

from bench.corpus.degrade import BANDS, apply_damage, draw_damage, map_box


def rotation_only(seed):
    return replace(draw_damage("heavy", seed), specks=0, noise_sigma=0.0, show_through=0.0, blur_sigma=0.0)


def page_with_block(width=600, height=800, box=(200, 300, 380, 340)):
    image = np.full((height, width), 255, np.uint8)
    x0, y0, x1, y1 = box
    image[y0:y1, x0:x1] = 0
    return image


class DrawTest(unittest.TestCase):
    def test_G3_every_parameter_stays_inside_its_band(self):
        for band in BANDS:
            for seed in range(40):
                damage = draw_damage(band, seed).as_dict()
                for name, (low, high) in BANDS[band].items():
                    value = abs(damage[name]) if name == "angle" else damage[name]
                    with self.subTest(band=band, name=name):
                        self.assertTrue(low <= value <= high, value)

    def test_G4_same_seed_gives_same_damage_and_pixels(self):
        damage = draw_damage("heavy", 11)
        self.assertEqual(damage, draw_damage("heavy", 11))
        image = page_with_block()
        first, matrix_a = apply_damage(image, damage)
        second, matrix_b = apply_damage(image, damage)
        self.assertTrue(np.array_equal(first, second) and np.array_equal(matrix_a, matrix_b))
        self.assertFalse(np.array_equal(first, apply_damage(image, draw_damage("heavy", 12))[0]))

    def test_G3_the_unseen_band_never_rotates_and_adds_shadow(self):
        for seed in range(30):
            damage = draw_damage("unseen", seed)
            self.assertEqual(damage.angle, 0.0)
            self.assertGreater(damage.shadow, 0)
            self.assertGreaterEqual(damage.streaks, 1)


class GeometryTest(unittest.TestCase):
    def test_G3_the_mapped_box_follows_the_rotated_ink(self):
        damage = rotation_only(4)
        box_px = (200, 300, 380, 340)
        damaged, matrix = apply_damage(page_with_block(box=box_px), damage)
        x0, y0, x1, y1 = map_box(tuple(v * 72 / damage.dpi for v in box_px), damage.dpi, matrix)
        ys, xs = np.nonzero(damaged < 100)
        self.assertTrue(x0 - 6 <= xs.mean() <= x1 + 6 and y0 - 6 <= ys.mean() <= y1 + 6)
        self.assertGreaterEqual(abs(damage.angle), 1.0)

    def test_G3_both_ends_of_a_long_bar_stay_inside_their_mapped_boxes(self):
        damage = rotation_only(9)
        damaged, matrix = apply_damage(page_with_block(box=(100, 100, 140, 700)), damage)
        to_pt = 72 / damage.dpi
        top = map_box((100 * to_pt, 100 * to_pt, 140 * to_pt, 110 * to_pt), damage.dpi, matrix)
        bottom = map_box((100 * to_pt, 690 * to_pt, 140 * to_pt, 700 * to_pt), damage.dpi, matrix)
        ys, xs = np.nonzero(damaged[: int(top[3]) + 1] < 100)
        self.assertTrue(top[0] - 3 <= xs.mean() <= top[2] + 3)
        ys, xs = np.nonzero(damaged[int(bottom[1]) :] < 100)
        self.assertTrue(bottom[0] - 3 <= xs.mean() <= bottom[2] + 3)

    def test_G3_input_must_be_grey_uint8(self):
        with self.assertRaises(ValueError):
            apply_damage(np.zeros((10, 10, 3), np.uint8), draw_damage("light", 1))
        with self.assertRaises(ValueError):
            apply_damage(np.zeros((10, 10), np.float32), draw_damage("light", 1))

    def test_G3_heavy_damage_changes_the_page_more_than_light(self):
        base = np.full((400, 400), 255, np.uint8)
        cv2.putText(base, "Boletin oficial", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 2)
        spread = {}
        for band in ("light", "heavy"):
            diffs = [
                np.abs(apply_damage(base, draw_damage(band, seed))[0].astype(int) - base).mean() for seed in range(8)
            ]
            spread[band] = np.mean(diffs)
        self.assertGreater(spread["heavy"], spread["light"])
        self.assertGreater(spread["light"], 0)


if __name__ == "__main__":
    unittest.main()
