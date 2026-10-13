"""The recognisers of the second reading. Each reads a crop (a line or a zone, cut by `crops.py`) and gives the text
it sees and the recogniser's score for the crop; `second.py` aligns that text to the first reading's words.

- `latin`: PP-OCRv5 Latin mobile on the crop. Independent errors, on disk beside the first model.
- `bin`: the first recogniser (PP-OCRv6 small) on the crop binarised (adaptive Gaussian, block 31).
- `clahe`: the first recogniser on the crop contrast-equalised (CLAHE 2.0, 8 x 8).
- `wide`: the first recogniser on the crop at 1.12 x its width (shifts the CTC alignment).
- `small`: the first recogniser on the crop as it is (not part of a cell: the control of the crop round trip, R10).
- `deep`: the best single recogniser of the bake-off (PP-OCRv6 medium, `bench/results/reader-bakeoff.json`).

Greedy decoding and fixed threads: the same crops give the same readings. Crops are read in batches of one page.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np

from ..errors import EngineLoadError
from . import ocr
from .modelfiles import DETECTION, RECOGNITION, find_model

LATIN_FILE = "latin_PP-OCRv5_rec_mobile.onnx"
DEEP_FILE = "PP-OCRv6_rec_medium.onnx"
# Pilot: bin and clahe hurt (14,8 % and 4,7 % of the right words contradicted, no gain) and stay available by name
CHEAP = ("latin", "wide")
ALL = ("latin", "bin", "clahe", "wide", "deep")
MODEL_OF = {
    "small": RECOGNITION,
    "latin": LATIN_FILE,
    "bin": RECOGNITION,
    "clahe": RECOGNITION,
    "wide": RECOGNITION,
    "deep": DEEP_FILE,
}
BIN_BLOCK = 31
BIN_C = 10
CLAHE_CLIP = 2.0
CLAHE_GRID = (8, 8)
WIDE_FACTOR = 1.12


def prepare(name: str, crop: np.ndarray) -> np.ndarray:
    """What the reader `name` is shown: `bin`, `clahe` and `wide` act on the crop, never on the page."""
    if name in ("small", "latin", "deep"):
        return crop
    if name == "wide":
        height, width = crop.shape[:2]
        return cv2.resize(crop, (max(1, round(width * WIDE_FACTOR)), height), interpolation=cv2.INTER_LINEAR)
    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if name == "bin":
        out = cv2.adaptiveThreshold(grey, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, BIN_BLOCK, BIN_C)
    elif name == "clahe":
        out = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_GRID).apply(grey)
    else:
        raise ValueError(f"unknown reader: {name!r}")
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


class ReaderSet:
    """The recognisers of a cell, loaded on first use: one RapidOCR recogniser for each distinct model file."""

    def __init__(self, names: Sequence[str], threads: int = 2, backend: str = "auto") -> None:
        unknown = [n for n in names if n not in MODEL_OF]
        if unknown:
            raise ValueError(f"unknown readers: {unknown}")
        self.names = tuple(names)
        self.threads = threads
        self.backend = backend
        self._models: dict[str, object] = {}
        self._sha: dict[str, str] = {}

    def paths(self) -> dict[str, Path]:
        return {name: find_model(MODEL_OF[name]) for name in self.names}

    def key(self) -> str:
        """`name:sha256-prefix` of every reader, so a new model or a new reader repeats pass 2 only."""
        parts = []
        for name, path in self.paths().items():
            if path.name not in self._sha:
                self._sha[path.name] = file_sha256(path)
            parts.append(f"{name}:{self._sha[path.name][:12]}")
        return "+".join(parts)

    def _model(self, name: str):
        file = MODEL_OF[name]
        if file not in self._models:
            self._models[file] = self._load(find_model(file))
        return self._models[file]

    def _load(self, path: Path):
        try:
            from rapidocr import EngineType, RapidOCR
        except ImportError as error:
            raise EngineLoadError(f"rapidocr is not installed: {error}") from error
        ocr._patch_rapidocr()
        ocr._tuning = ocr.Tuning(det_buckets=False, rec_buckets=True)
        use_openvino = self.backend == "openvino" or (
            self.backend == "auto" and ocr.free_memory_mb() > ocr.OPENVINO_MIN_FREE_MB
        )
        params = {"Rec.model_path": str(path), "Det.model_path": str(find_model(DETECTION))}
        if use_openvino:
            params |= {
                "Rec.engine_type": EngineType.OPENVINO,
                "EngineConfig.openvino.inference_num_threads": self.threads,
                "EngineConfig.openvino.num_streams": 1,
                "EngineConfig.openvino.enable_cpu_pinning": False,
            }
        else:
            params |= {
                "EngineConfig.onnxruntime.intra_op_num_threads": self.threads,
                "EngineConfig.onnxruntime.inter_op_num_threads": 1,
            }
        model = RapidOCR(params=params)
        ocr._quiet()
        return model

    def read_many(self, name: str, crops: Sequence[np.ndarray]) -> list[tuple[str, float | None]]:
        """(text, score) of each crop; the score is the recogniser's for the whole crop."""
        from rapidocr.ch_ppocr_rec.typings import TextRecInput

        if not crops:
            return []
        model = self._model(name)
        result = model.text_rec(TextRecInput(img=[prepare(name, c) for c in crops]))
        texts, scores = list(result.txts or ()), list(result.scores or ())
        if len(texts) != len(crops):
            return [("", None)] * len(crops)
        return [(str(t), float(s)) for t, s in zip(texts, scores, strict=True)]

    def close(self) -> None:
        self._models.clear()


ReaderFactory = Callable[..., ReaderSet]
