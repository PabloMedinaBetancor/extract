"""Pass 1 of the second reading, the cutting: the crop of each unit the readers of pass 2 may read, taken from the image
the first reading saw, so every reader sees what the first one saw.

A `line` crop is the line cut along its detection quadrilateral with RapidOCR's own `get_rotate_crop_image`. A `zone`
crop is the horizontal slice of that crop from the left edge of the zone's first word to the right edge of its last,
widened by `ZONE_PAD` line heights on each side. A line whose quadrilateral could not be matched to its text (see
`Engine.read_lines`), or whose crop was turned by RapidOCR, is cut from the union of the unit's word boxes instead
with `UNION_PAD_PX` of padding; the page report counts which of the two was used.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np

from ..types import Word
from .reread import RereadConfig, Unit

ZONE_PAD = 0.25
UNION_PAD_PX = 6
MIN_SLICE_PX = 8
MAX_UPRIGHT_RATIO = 1.5


def encode(crop: np.ndarray) -> bytes:
    """PNG, one channel when the crop is grey (most scans): about a third of the bytes."""
    if crop.ndim == 3 and np.array_equal(crop[:, :, 0], crop[:, :, 1]) and np.array_equal(crop[:, :, 0], crop[:, :, 2]):
        crop = crop[:, :, 0]
    ok, data = cv2.imencode(".png", crop, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise ValueError("a crop could not be encoded")
    return data.tobytes()


def decode(data: bytes) -> np.ndarray:
    crop = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if crop is None:
        raise ValueError("a crop could not be decoded")
    return crop


def _as_bgr(image: np.ndarray) -> np.ndarray:
    return np.stack([image] * 3, axis=-1) if image.ndim == 2 else image


def quad_crop(image: np.ndarray, quad: np.ndarray) -> np.ndarray | None:
    """The line along its quadrilateral, or None when RapidOCR would have turned it (a vertical box), which the zone
    slicing cannot follow."""
    from rapidocr.utils.process_img import get_rotate_crop_image

    width = max(np.linalg.norm(quad[0] - quad[1]), np.linalg.norm(quad[2] - quad[3]))
    height = max(np.linalg.norm(quad[0] - quad[3]), np.linalg.norm(quad[1] - quad[2]))
    if width < 2 or height < 2 or height / width >= MAX_UPRIGHT_RATIO:
        return None
    return get_rotate_crop_image(image, quad.astype(np.float32))


def union_crop(image: np.ndarray, boxes: Sequence[Sequence[float]], pad_x: float, pad_y: float) -> np.ndarray:
    height, width = image.shape[:2]
    x0 = max(0, int(min(b[0] for b in boxes) - pad_x))
    y0 = max(0, int(min(b[1] for b in boxes) - pad_y))
    x1 = min(width, int(np.ceil(max(b[2] for b in boxes) + pad_x)))
    y1 = min(height, int(np.ceil(max(b[3] for b in boxes) + pad_y)))
    return image[y0 : max(y1, y0 + 1), x0 : max(x1, x0 + 1)]


def zone_slice(line_crop: np.ndarray, quad: np.ndarray, boxes: Sequence[Sequence[float]]) -> np.ndarray | None:
    """The part of a line crop between the first and the last box, with the context of `ZONE_PAD` line heights.
    A page x is mapped to the crop by projecting the box's mid-height point on the quadrilateral's top edge."""
    edge = quad[1] - quad[0]
    span = float(edge @ edge)
    if span <= 0:
        return None
    crop_h, crop_w = line_crop.shape[:2]

    def column(x: float, y: float) -> float:
        return float((np.array([x, y], dtype=np.float32) - quad[0]) @ edge) / span * crop_w

    left = min(column(b[0], (b[1] + b[3]) / 2) for b in boxes)
    right = max(column(b[2], (b[1] + b[3]) / 2) for b in boxes)
    pad = ZONE_PAD * crop_h
    x0, x1 = max(0, int(left - pad)), min(crop_w, int(np.ceil(right + pad)))
    if x1 - x0 < MIN_SLICE_PX:
        return None
    return line_crop[:, x0:x1]


def cut_unit_crops(
    image: np.ndarray, line_reads: Sequence, words: Sequence[Word], config: RereadConfig
) -> tuple[dict[str, bytes], dict[str, int]]:
    """The crops of the first `study_units` units of each kind, as `{"line<rank>": png, "zone<rank>": png}`, and the
    report counts `crops_quad`, `crops_union`, `units_line`, `units_zone`."""
    image = _as_bgr(image)
    report = {"crops_quad": 0, "crops_union": 0}
    crops: dict[str, bytes] = {}
    line_crops: dict[int, np.ndarray | None] = {}

    def of_line(line_id: int) -> np.ndarray | None:
        if line_id not in line_crops:
            quad = line_reads[line_id].quad
            line_crops[line_id] = quad_crop(image, quad) if quad is not None else None
        return line_crops[line_id]

    for kind in ("line", "zone"):
        units = config.units(words, kind)
        report[f"units_{kind}"] = len(units)
        for unit in units:
            crop = _crop_of(image, line_reads, words, unit, of_line)
            if crop is None or crop[0].size == 0:
                continue
            crops[f"{kind}{unit.rank}"] = encode(crop[0])
            report["crops_quad" if crop[1] else "crops_union"] += 1
    return crops, report


def _crop_of(image, line_reads, words: Sequence[Word], unit: Unit, of_line) -> tuple[np.ndarray, bool] | None:
    boxes = [words[i].box for i in unit.indices]
    line_crop = of_line(unit.line_id)
    if line_crop is not None:
        if unit.kind == "line":
            return line_crop, True
        sliced = zone_slice(line_crop, line_reads[unit.line_id].quad, boxes)
        if sliced is not None:
            return sliced, True
    height = max(b[3] - b[1] for b in boxes)
    pad_x = max(UNION_PAD_PX, ZONE_PAD * height) if unit.kind == "zone" else UNION_PAD_PX
    return union_crop(image, boxes, pad_x, UNION_PAD_PX), False


def crop_keys(crops: Mapping[str, object], unit: str) -> list[int]:
    """The ranks that have a stored crop for a unit kind."""
    return sorted(int(k[len(unit) :]) for k in crops if k.startswith(unit) and k[len(unit) :].isdigit())
