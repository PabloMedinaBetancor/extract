"""Turns a clean page render into a scan-like page, in three bands of damage.

A page gets one band and one seed; every parameter is drawn from the band's range by that
seed. The only geometric change is a small rotation, returned as a matrix so that the truth
boxes can follow it. A fourth band, `unseen`, is never used for training: it adds an uneven
shadow and scanner streaks and leaves the geometry alone.
"""

from dataclasses import asdict, dataclass

import cv2
import numpy as np

from bench.corpus.split import stable_rng, stable_unit

BAND_SHARES = (("light", 0.50), ("medium", 0.35), ("heavy", 0.15))

# (low, high) per parameter. Tuned once on pilot pages so that the engine's word error rate
# falls near 1-2 %, 3-6 % and 8-15 % per band (bench/corpus/pilot.py).
BANDS: dict[str, dict[str, tuple[float, float]]] = {
    "light": {
        "dpi": (170, 200),
        "blur_sigma": (0.0, 0.4),
        "noise_sigma": (0.0, 2.0),
        "jpeg_quality": (85, 95),
        "angle": (0.0, 0.3),
        "contrast": (0.95, 1.0),
        "shift": (0.0, 5.0),
        "show_through": (0.0, 0.0),
        "specks": (0, 0),
        "shadow": (0.0, 0.0),
        "streaks": (0, 0),
    },
    "medium": {
        "dpi": (120, 150),
        "blur_sigma": (0.5, 0.9),
        "noise_sigma": (3.0, 8.0),
        "jpeg_quality": (50, 75),
        "angle": (0.3, 1.0),
        "contrast": (0.80, 0.92),
        "shift": (5.0, 20.0),
        "show_through": (0.0, 0.0),
        "specks": (0, 40),
        "shadow": (0.0, 0.0),
        "streaks": (0, 0),
    },
    "heavy": {
        "dpi": (110, 130),
        "blur_sigma": (0.8, 1.1),
        "noise_sigma": (8.0, 14.0),
        "jpeg_quality": (30, 50),
        "angle": (1.0, 2.0),
        "contrast": (0.75, 0.88),
        "shift": (10.0, 30.0),
        "show_through": (0.08, 0.20),
        "specks": (60, 200),
        "shadow": (0.0, 0.0),
        "streaks": (0, 0),
    },
}
BANDS["unseen"] = {
    **BANDS["medium"],
    "angle": (0.0, 0.0),
    "shadow": (0.25, 0.50),
    "streaks": (1, 3),
}

_INTEGER_PARAMS = ("jpeg_quality", "specks", "streaks")


@dataclass(frozen=True)
class Damage:
    band: str
    seed: int
    dpi: float
    angle: float
    blur_sigma: float
    noise_sigma: float
    jpeg_quality: int
    contrast: float
    shift: float
    show_through: float
    specks: int
    shadow: float
    streaks: int

    def as_dict(self) -> dict:
        return asdict(self)


def draw_band(page_id: str, seed: int) -> str:
    """The band of a page, in the proportions of BAND_SHARES."""
    point = stable_unit(seed, "band", page_id)
    edge = 0.0
    for band, share in BAND_SHARES:
        edge += share
        if point < edge:
            return band
    return BAND_SHARES[-1][0]


def draw_damage(band: str, seed: int, ranges: dict | None = None) -> Damage:
    """Draw every parameter of a band from the page's seed."""
    spec = (ranges or BANDS)[band]
    rng = stable_rng(seed, "damage", band)
    values = {}
    for name, (low, high) in spec.items():
        value = rng.uniform(low, high) if high > low else low
        values[name] = int(round(value)) if name in _INTEGER_PARAMS else float(value)
    if band != "unseen" and rng.random() < 0.5:
        values["angle"] = -values["angle"]
    return Damage(band=band, seed=seed, **values)


def apply_damage(image: np.ndarray, damage: Damage) -> tuple[np.ndarray, np.ndarray]:
    """Damage a grey uint8 page rendered at `damage.dpi`.

    Returns the damaged image and the 2x3 matrix that maps points of the clean render to
    points of the damaged one.
    """
    if image.ndim != 2 or image.dtype != np.uint8:
        raise ValueError("expected a 2-D uint8 grey image")
    rng = stable_rng(damage.seed, "pixels", damage.band)
    height, width = image.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), damage.angle, 1.0)
    out = image
    if damage.angle:
        out = cv2.warpAffine(out, matrix, (width, height), flags=cv2.INTER_LINEAR, borderValue=255)
    if damage.blur_sigma > 0:
        out = cv2.GaussianBlur(out, (0, 0), damage.blur_sigma)
    out = out.astype(np.float32)
    out = (out - 128.0) * damage.contrast + 128.0 - damage.shift
    if damage.show_through > 0:
        ghost = cv2.GaussianBlur(cv2.flip(image, 1), (0, 0), 2.5).astype(np.float32)
        out *= 1.0 - damage.show_through * (255.0 - ghost) / 255.0
    if damage.shadow > 0:
        out *= _shadow(height, width, damage.shadow, rng)
    if damage.noise_sigma > 0:
        out += rng.normal(0.0, damage.noise_sigma, out.shape).astype(np.float32)
    out = np.clip(out, 0, 255).astype(np.uint8)
    _add_specks(out, damage.specks, rng)
    _add_streaks(out, damage.streaks, rng)
    return _jpeg_round_trip(out, damage.jpeg_quality), matrix


def map_box(box_pt: tuple[float, float, float, float], dpi: float, matrix: np.ndarray) -> tuple[float, ...]:
    """Move a box from PDF points to the pixels of the damaged image."""
    x0, y0, x1, y1 = (v * dpi / 72.0 for v in box_pt)
    corners = np.array([[x0, y0, 1.0], [x1, y0, 1.0], [x0, y1, 1.0], [x1, y1, 1.0]])
    moved = corners @ matrix.T
    return (
        float(moved[:, 0].min()),
        float(moved[:, 1].min()),
        float(moved[:, 0].max()),
        float(moved[:, 1].max()),
    )


def _shadow(height: int, width: int, strength: float, rng: np.random.Generator) -> np.ndarray:
    theta = rng.uniform(0, 2 * np.pi)
    ys, xs = np.mgrid[0:height, 0:width].astype(np.float32)
    along = (xs - width / 2) * np.cos(theta) + (ys - height / 2) * np.sin(theta)
    reach = 0.5 * (abs(width * np.cos(theta)) + abs(height * np.sin(theta)))
    ramp = np.clip((along / reach + 1.0) / 2.0, 0.0, 1.0)
    return 1.0 - strength * ramp**1.5


def _add_specks(image: np.ndarray, count: int, rng: np.random.Generator) -> None:
    height, width = image.shape
    for _ in range(count):
        centre = (int(rng.integers(0, width)), int(rng.integers(0, height)))
        cv2.circle(image, centre, int(rng.integers(1, 3)), int(rng.integers(40, 130)), -1)


def _add_streaks(image: np.ndarray, count: int, rng: np.random.Generator) -> None:
    height, width = image.shape
    rows = np.arange(height, dtype=np.float32)
    for _ in range(count):
        x = int(rng.integers(0, width - 6))
        thickness = int(rng.integers(1, 4))
        depth = rng.uniform(40, 90)
        fade = 0.6 + 0.4 * np.sin(rows / rng.uniform(80, 300) + rng.uniform(0, 6.28))
        band = image[:, x : x + thickness].astype(np.float32) - (depth * fade)[:, None]
        image[:, x : x + thickness] = np.clip(band, 0, 255).astype(np.uint8)


def _jpeg_round_trip(image: np.ndarray, quality: int) -> np.ndarray:
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
