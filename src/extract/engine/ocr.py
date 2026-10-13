"""RapidOCR PP-OCRv6 small, set up as measured on scanned planning documents: detection at 1,280 px for real, no
orientation classifier, text score 0.3, OpenVINO when memory allows. Each word keeps the confidence of every one of
its own characters."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib.metadata
import logging
import os
import string
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np
import psutil

from ..errors import EngineLoadError
from ..types import PageReading, Word, word_id
from .modelfiles import DETECTION, RECOGNITION, find_model
from .quiet_openvino import import_openvino

DETECTION_SIDE = 1280
MAX_SIDE = 3000
TEXT_SCORE = 0.3
OPENVINO_MIN_FREE_MB = 2000
MIN_WORDS = 25
MIN_MEAN_CONF = 0.65

DET_BUCKET_PX = 128
REC_BUCKET_PX = 160
REC_HEIGHT_PX = 48

ALT_TOP = 3
ALT_FLOOR = 1e-3
COUNTERPART_FLOOR = 1e-4
# The rule for what may be named as an alternative or kept in the masked reading: any symbol that appears in written
# Spanish, Catalan or English as an invoice or an official document has it (letters with their marks, digits,
# punctuation, typographic quotes and dashes, the signs of units, money and numbering), not only letters.
LATIN_EXTRA = "áéíóúÁÉÍÓÚñÑüÜ¿¡€«»ºªàèìòùÀÈÌÒÙâêîôûÂÊÎÔÛäëïöÄËÏÖãõÃÕçÇŀ·ÿœŒæÆßøØåÅ‘’‚“”„‹›–—−…‐‑²³¹°•±×÷‰§¶©®™½¼¾£¥¢µ№"
ACCENT_COUNTERPARTS = {
    **{plain: (marked,) for plain, marked in zip("aeiouAEIOUnN", "áéíóúÁÉÍÓÚñÑ", strict=True)},
    "u": ("ú", "ü"),
    "U": ("Ú", "Ü"),
    **{marked: (plain,) for plain, marked in zip("aeiouAEIOUnNuU", "áéíóúÁÉÍÓÚñÑüÜ", strict=True)},
}

_patched = False
_alternatives_on = True


@dataclass(frozen=True)
class Tuning:
    """Speed settings measured in `bench/results/engine-speed.json`. Each one makes the models see a few fixed input
    shapes instead of one per image, so OpenVINO keeps a handful of compiled shapes."""

    det_buckets: bool = False
    rec_buckets: bool = False


DEFAULT_TUNING = Tuning()
_tuning = DEFAULT_TUNING


class CharConf(float):
    """A word's mean confidence that also carries the confidence of each of its characters, their alternatives and
    the masked reading of the word. It travels through RapidOCR's word results in place of the plain float."""

    chars: tuple[float, ...]
    aligned: bool
    alts: tuple[tuple[tuple[str, float], ...], ...] | None = None
    masked: str | None = None


@dataclass(frozen=True)
class LatinSet:
    """The classes of the recogniser's character list that the alternatives may name: ASCII letters, digits and
    punctuation, Spanish accents and `ñ ü ¿ ¡ € « » º ª`. `ids` are their class indices, `column` maps a class to its
    position in `ids` (-1 outside the set) and `counterparts` holds, for each member, the columns of the same letter
    with or without its mark (-1 pads). `keep_ids` adds the blank and the space for the masked decode."""

    ids: np.ndarray
    chars: tuple[str, ...]
    column: np.ndarray
    counterparts: np.ndarray
    keep_ids: np.ndarray


def build_latin_set(character: Sequence[str]) -> LatinSet:
    """`character` is the decoder's list, blank first and the space last, so a class index is a position in it."""
    allowed = set(string.ascii_letters + string.digits + string.punctuation + LATIN_EXTRA)
    position = {char: i for i, char in enumerate(character) if len(char) == 1}
    chars = tuple(sorted(c for c in allowed if c in position))
    ids = np.array([position[c] for c in chars], dtype=np.int64)
    column = np.full(len(character), -1, dtype=np.int64)
    column[ids] = np.arange(len(ids))
    counterparts = np.full((len(ids), 2), -1, dtype=np.int64)
    for i, char in enumerate(chars):
        for j, other in enumerate(o for o in ACCENT_COUNTERPARTS.get(char, ()) if o in position):
            counterparts[i, j] = column[position[other]]
    space = position.get(" ")
    keep = [0] + ([space] if space is not None else []) + list(ids)
    return LatinSet(ids, chars, column, counterparts, np.array(sorted(set(keep)), dtype=np.int64))


def line_alternatives(probs: np.ndarray, best: np.ndarray, character: Sequence[str], latin: LatinSet):
    """Per emitted character of one line, its alternatives, and the Latin-only decode of the line.

    `probs` is the line's (steps, classes) softmax and `best` its argmax per step, which the decoder has already
    computed. The emitted characters are those of the greedy decode (a class whose run starts where the argmax
    changes, the blank dropped). The alternatives are read at the peak step of the character's run: the top classes
    of the Latin set, the emitted one first; classes under `ALT_FLOOR` are dropped and the accent counterpart of a
    vowel, `n` or `u` is added from `COUNTERPART_FLOOR` up. The masked text decodes the same steps with only the Latin
    set, the blank and the space allowed; None when it is the line's own text.
    """
    start = np.ones(len(best), dtype=bool)
    start[1:] = best[1:] != best[:-1]
    first = np.flatnonzero(start & (best != 0))
    if len(first) == 0:
        return [], None
    run_starts = np.flatnonzero(start)
    end_of = dict(zip(run_starts.tolist(), np.append(run_starts[1:], len(best)).tolist(), strict=True))
    classes = best[first]
    peaks = np.array(
        [s + int(np.argmax(probs[s : end_of[s], c])) for s, c in zip(first.tolist(), classes.tolist(), strict=True)]
    )
    emitted_p = probs[peaks, classes]
    # Reading the Latin columns of a row is a scattered memory access, so it is done only where the emitted class
    # leaves enough mass for an alternative or a counterpart.
    rows = np.flatnonzero(1.0 - emitted_p >= COUNTERPART_FLOOR)
    extra: dict[int, list[tuple[str, float]]] = {}
    if len(rows):
        latin_probs = probs[np.ix_(peaks[rows], latin.ids)]
        others = np.where(latin.ids[None, :] == classes[rows, None], 0.0, latin_probs)
        order = np.argsort(-others, axis=1)[:, :ALT_TOP]
        emitted_col = latin.column[classes[rows]]
        pairs = latin.counterparts[emitted_col]
        pair_p = np.where(
            (pairs >= 0) & (emitted_col[:, None] >= 0), latin_probs[np.arange(len(rows))[:, None], pairs.clip(0)], 0.0
        )
        for k, row in enumerate(rows.tolist()):
            taken, found = set(), []
            for col in order[k].tolist():
                if len(found) > ALT_TOP - 2 or others[k, col] < ALT_FLOOR:
                    break
                taken.add(col)
                found.append((character[int(latin.ids[col])], float(others[k, col])))
            for col, p in zip(pairs[k].tolist(), pair_p[k].tolist(), strict=True):
                if col >= 0 and col not in taken and p >= COUNTERPART_FLOOR:
                    found.append((character[int(latin.ids[col])], p))
            extra[row] = found
    alts = [((character[cls], float(emitted_p[row])), *extra.get(row, ())) for row, cls in enumerate(classes.tolist())]
    return alts, _masked_text(probs, best, character, latin)


def _masked_text(probs: np.ndarray, best: np.ndarray, character: Sequence[str], latin: LatinSet) -> str | None:
    """The line decoded with the Latin set only, or None when the recogniser never left it (then the two decodes
    are the same and the columns are not read)."""
    if np.isin(best, latin.keep_ids).all():
        return None
    restricted = latin.keep_ids[probs[:, latin.keep_ids].argmax(axis=1)]
    keep = np.ones(len(restricted), dtype=bool)
    keep[1:] = restricted[1:] != restricted[:-1]
    keep &= restricted != 0
    return "".join(character[c] for c in restricted[keep].tolist())


def pair_confidences(
    text: str,
    words: Sequence[str],
    confs: Sequence[float],
    alts: Sequence[tuple[tuple[str, float], ...]] | None = None,
    masked_line: str | None = None,
) -> list[CharConf]:
    """Confidence of each word from the characters that really belong to it.

    `confs` holds one value per character of the line `text` (spaces included) and the words are, in order, its runs
    of non-space characters. RapidOCR itself hands word k the confidence of character k of the line. When the
    characters do not line up, every word gets the line mean for each of its characters and `aligned` is False, and
    has no alternatives. `alts` is parallel to `confs`; the first entry of each keeps the value of `confs`, so a word's
    first alternative is its character confidence. `masked_line` is the Latin-only decode of the line: a word carries
    it when the line has the same words in both.
    """
    line_mean = sum(confs) / len(confs) if confs else 0.0

    def fallback() -> list[CharConf]:
        out = []
        for word in words:
            conf = CharConf(line_mean)
            conf.chars, conf.aligned = (line_mean,) * len(word), False
            out.append(conf)
        return out

    if len(confs) != len(text):
        return fallback()
    own_alts = alts if alts is not None and len(alts) == len(text) else None
    masked_words = masked_line.split() if masked_line is not None else None
    if masked_words is not None and len(masked_words) != len(text.split()):
        masked_words = None
    positions = [i for i, ch in enumerate(text) if not ch.isspace()]
    out, taken = [], 0
    for word in words:
        slots = positions[taken : taken + len(word)]
        taken += len(word)
        if len(slots) != len(word) or any(text[i] != ch for i, ch in zip(slots, word, strict=True)):
            return fallback()
        chars = tuple(float(confs[i]) for i in slots)
        conf = CharConf(sum(chars) / len(chars) if chars else line_mean)
        conf.chars, conf.aligned = chars, True
        if own_alts is not None and all(own_alts[i] and own_alts[i][0][0] == text[i] for i in slots):
            conf.alts = tuple(((own_alts[i][0][0], float(confs[i])), *own_alts[i][1:]) for i in slots)
        out.append(conf)
    if taken != len(positions):
        return fallback()
    if masked_words is not None and len(masked_words) == len(out):
        line_words = text.split()
        for conf, plain, masked in zip(out, line_words, masked_words, strict=True):
            conf.masked = masked if masked != plain else None
    return out


def _patch_rapidocr() -> None:
    global _patched
    if _patched:
        return
    import_openvino()  # the first import of OpenVINO in this process, with its usage reporting off
    from rapidocr.cal_rec_boxes import main as boxes
    from rapidocr.ch_ppocr_det import main as detector
    from rapidocr.ch_ppocr_det.utils import DetPreProcess

    # With limit_type "max" RapidOCR ignores the requested side and detects at 960, 1,500 or 2,000 px by image size.
    detector.TextDetector.get_preprocess = lambda self, max_wh: DetPreProcess(
        DETECTION_SIDE, "max", self.mean, self.std
    )

    _patch_shape_buckets()
    _patch_runtime_cache()
    original = boxes.CalRecBoxes.cal_ocr_word_box

    def cal_ocr_word_box(self, rec_txt, bbox, word_info, return_single_char_box=False):
        contents, word_boxes, confs = original(self, rec_txt, bbox, word_info, return_single_char_box)
        paired = pair_confidences(
            rec_txt,
            contents,
            list(confs or ()),
            getattr(word_info, "alts", None),
            getattr(word_info, "masked_line", None),
        )
        return contents, word_boxes, paired

    boxes.CalRecBoxes.cal_ocr_word_box = cal_ocr_word_box
    _patched = True


# 32 kernels per model: a worker stays near 1.25 GB over 120 pages (the default kept growing to 1.7 GB), for ~3 % more time.
CPU_RUNTIME_CACHE = int(os.environ.get("EXTRACT_CPU_CACHE", "32")) or None


def _patch_runtime_cache() -> None:
    """Bound the OpenVINO CPU plugin's per-shape cache (its default keeps 5,000 entries, so a worker's memory grows with
    every new page size). It only decides how many compiled kernels stay in memory, never what a model returns."""
    from rapidocr.inference_engine.openvino.device_config import CPUConfig

    plain_config = CPUConfig.get_config

    def get_config(self):
        config = plain_config(self)
        if CPU_RUNTIME_CACHE is not None:
            config["CPU_RUNTIME_CACHE_CAPACITY"] = str(CPU_RUNTIME_CACHE)
        return config

    CPUConfig.get_config = get_config


def _patch_shape_buckets() -> None:
    import math

    import cv2 as cv
    from rapidocr.ch_ppocr_det.utils import DetPreProcess
    from rapidocr.ch_ppocr_rec import main as recognizer
    from rapidocr.ch_ppocr_rec import utils as rec_utils

    plain_resize = DetPreProcess.resize

    def resize(self, img):
        if not _tuning.det_buckets:
            return plain_resize(self, img)
        height, width = img.shape[:2]
        ratio = min(1.0, self.limit_side_len / max(height, width))
        new_h = max(DET_BUCKET_PX, round(height * ratio / DET_BUCKET_PX) * DET_BUCKET_PX)
        new_w = max(DET_BUCKET_PX, round(width * ratio / DET_BUCKET_PX) * DET_BUCKET_PX)
        return cv.resize(img, (new_w, new_h))

    DetPreProcess.resize = resize

    def bucket_ratio(max_wh_ratio):
        if not _tuning.rec_buckets:
            return max_wh_ratio
        width = math.ceil(max_wh_ratio * REC_HEIGHT_PX / REC_BUCKET_PX) * REC_BUCKET_PX
        return width / REC_HEIGHT_PX

    plain_norm = recognizer.TextRecognizer.resize_norm_img
    recognizer.TextRecognizer.resize_norm_img = lambda self, img, max_wh_ratio: plain_norm(
        self, img, bucket_ratio(max_wh_ratio)
    )
    plain_decode = rec_utils.CTCLabelDecode.__call__

    def decode(self, preds, return_word_box=False, **kwargs):
        if "max_wh_ratio" in kwargs:
            kwargs["max_wh_ratio"] = bucket_ratio(kwargs["max_wh_ratio"])
        if not (_alternatives_on and return_word_box):
            return plain_decode(self, preds, return_word_box, **kwargs)
        # The same decode as RapidOCR's, with the argmax kept: the alternatives need it and computing it twice over
        # 18,710 classes would cost more than they do.
        best = preds.argmax(axis=2)
        prob = np.take_along_axis(preds, best[:, :, None], axis=2)[:, :, 0]
        lines, infos = self.decode(
            best,
            prob,
            return_word_box,
            kwargs.get("wh_ratio_list", (1.0,)),
            kwargs.get("max_wh_ratio", 1.0),
            remove_duplicate=True,
        )
        latin = _latin_set(self)
        for row, info in enumerate(infos):
            info.alts, info.masked_line = line_alternatives(preds[row], best[row], self.character, latin)
        return lines, infos

    rec_utils.CTCLabelDecode.__call__ = decode


_latin_sets: dict[str, LatinSet] = {}


def _latin_set(decoder) -> LatinSet:
    """The Latin set of a decoder's character list, kept by the list's content (not by the object: a recogniser that
    is loaded and freed can leave its address to another with a different list)."""
    key = hashlib.blake2b("\x00".join(decoder.character).encode(), digest_size=16).hexdigest()
    if key not in _latin_sets:
        _latin_sets[key] = build_latin_set(decoder.character)
    return _latin_sets[key]


def free_memory_mb() -> int:
    return int(psutil.virtual_memory().available // (1024 * 1024))


def rapidocr_version() -> str:
    try:
        return importlib.metadata.version("rapidocr")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def fingerprint() -> str:
    """Everything that changes the words an engine reads. Threads and the backend are left out on purpose: they do
    not change the models, and a page read by either is a valid reading."""
    return f"rapidocr-{rapidocr_version()}|PP-OCRv6-small|det{DETECTION_SIDE}|max{MAX_SIDE}|score{TEXT_SCORE}|nocls|v1|alts2"


@dataclass
class Engine:
    ocr: Any
    backend: str
    threads: int
    fallback_reason: str | None = None

    def read_words(self, image: np.ndarray) -> list[list[tuple[str, CharConf, tuple[float, float, float, float]]]]:
        """Lines of (text, confidence, box) for a grey or BGR image."""
        return [line.words for line in self.read_lines(image)]

    def read_lines(self, image: np.ndarray) -> list[LineRead]:
        """The same lines with the detection quadrilateral of each, when it can be matched to the line: RapidOCR keeps
        a line's quadrilateral and text always but its word results only when they are not empty, so the two lists can
        drift apart, and a quadrilateral is handed over only when its text is the line's words joined."""
        result = self.ocr(_as_bgr(image), return_word_box=True)
        if not getattr(result, "txts", None):
            return []
        boxes, texts = getattr(result, "boxes", None), result.txts
        lines = []
        for raw, line in enumerate(getattr(result, "word_results", None) or ()):
            words = []
            for item in line or ():
                if not isinstance(item, (tuple, list)) or len(item) < 3 or not item[2] or not str(item[0]).strip():
                    continue
                xs, ys = [p[0] for p in item[2]], [p[1] for p in item[2]]
                words.append(
                    (str(item[0]).strip(), item[1], (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))))
                )
            if not words:
                continue
            quad = None
            joined = " ".join(w[0] for w in words)
            if boxes is not None and raw < len(boxes) and raw < len(texts) and " ".join(texts[raw].split()) == joined:
                quad = np.asarray(boxes[raw], dtype=np.float32)
            lines.append(LineRead(words, quad))
        return lines


@dataclass
class LineRead:
    """One line of a reading: its words as (text, confidence, box) and the detection quadrilateral (4 x 2, float32) the
    line was found in, or None when it could not be matched to the line's text."""

    words: list
    quad: np.ndarray | None = None


def _as_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    return image


def load_engine(threads: int = 2, backend: str = "auto", tuning: Tuning | None = None) -> Engine:
    """Load the models. `auto` takes OpenVINO when more than 2 GB are free and falls back to onnxruntime when it is
    short of memory or cannot start; the reason is kept in `fallback_reason` for the caller to report."""
    try:
        from rapidocr import EngineType, RapidOCR
    except ImportError as error:
        raise EngineLoadError(f"rapidocr is not installed: {error}") from error
    global _tuning
    _tuning = tuning or DEFAULT_TUNING
    _patch_rapidocr()
    _quiet()
    base = {
        "Det.model_path": str(find_model(DETECTION)),
        "Rec.model_path": str(find_model(RECOGNITION)),
        "Global.max_side_len": MAX_SIDE,
        "Global.use_cls": False,
        "Global.text_score": TEXT_SCORE,
        "Det.limit_type": "max",
        "Det.limit_side_len": DETECTION_SIDE,
    }
    fallback_reason = None
    if backend in ("auto", "openvino"):
        free = free_memory_mb()
        if backend == "auto" and free <= OPENVINO_MIN_FREE_MB:
            fallback_reason = f"only {free} MB free, OpenVINO needs more than {OPENVINO_MIN_FREE_MB}"
        else:
            try:
                ocr = RapidOCR(
                    params={
                        **base,
                        "Det.engine_type": EngineType.OPENVINO,
                        "Rec.engine_type": EngineType.OPENVINO,
                        "EngineConfig.openvino.inference_num_threads": threads,
                        "EngineConfig.openvino.num_streams": 1,
                        "EngineConfig.openvino.enable_cpu_pinning": False,
                    }
                )
                _quiet()
                return Engine(ocr, "openvino", threads)
            except Exception as error:  # the optional backend may fail in many ways; the reason is kept and reported
                if backend == "openvino":
                    raise EngineLoadError(f"OpenVINO did not start: {error}") from error
                fallback_reason = f"OpenVINO did not start: {error}"
    ocr = RapidOCR(
        params={
            **base,
            "EngineConfig.onnxruntime.intra_op_num_threads": threads,
            "EngineConfig.onnxruntime.inter_op_num_threads": 1,
        }
    )
    _quiet()
    return Engine(ocr, "onnxruntime", threads, fallback_reason)


def _quiet() -> None:
    logging.getLogger("RapidOCR").setLevel(logging.ERROR)


_default_engine: Engine | None = None


def default_engine() -> Engine:
    """The engine of this process, loaded on first use with the threads its physical cores allow."""
    global _default_engine
    if _default_engine is None:
        _default_engine = load_engine(threads=psutil.cpu_count(logical=False) or 2)
    return _default_engine


def read_page(
    image: np.ndarray | str | Path,
    engine: Engine | None = None,
    number: int = 1,
    dpi: int = 200,
    size_pt: tuple[float, float] | None = None,
) -> PageReading:
    """OCR one rendered page, given as an array or the path of an image. The status is `unreadable` (text kept) when
    fewer than 25 words came out or their mean confidence is under 0.65."""
    return read_page_with_crops(image, engine, number, dpi, size_pt, None)[0]


def read_page_with_crops(
    image: np.ndarray | str | Path,
    engine: Engine | None = None,
    number: int = 1,
    dpi: int = 200,
    size_pt: tuple[float, float] | None = None,
    study=None,
) -> tuple[PageReading, dict[str, bytes]]:
    """Pass 1 of the second reading: `read_page` plus, when `study` (a `RereadConfig`) is given and the page is read
    as OCR, the PNG crops of the units that pass 2 may read, keyed `line<rank>` and `zone<rank>` (see `crops.py`). The
    page's `reread_report` counts the crops cut from a detection quadrilateral and those cut from the union of word
    boxes. The first reading is the same with or without `study`."""
    if isinstance(image, (str, Path)):
        loaded = cv2.imread(str(image))
        if loaded is None:
            raise ValueError(f"cannot read the image {image}")
        image = loaded
    engine = engine or default_engine()
    height, width = image.shape[:2]
    words: list[Word] = []
    unaligned = 0
    quads = hasattr(engine, "read_lines") and study is not None
    line_reads = engine.read_lines(image) if quads else [LineRead(line) for line in engine.read_words(image)]
    for line_id, line_read in enumerate(line_reads):
        line = line_read.words
        for idx, (text, conf, box) in enumerate(line):
            own = isinstance(conf, CharConf) and conf.aligned
            unaligned += isinstance(conf, CharConf) and not conf.aligned
            words.append(
                Word(
                    id=word_id(number, len(words)),
                    page=number,
                    text=text,
                    box=box,
                    conf=float(conf),
                    char_conf=tuple(conf.chars) if own else None,
                    char_alts=conf.alts if own else None,
                    masked=conf.masked if own else None,
                    line_id=line_id,
                    idx_in_line=idx,
                )
            )
    mean_conf = sum(w.conf for w in words) / len(words) if words else None
    status, reason = "ocr", None
    if len(words) < MIN_WORDS:
        status, reason = "unreadable", "page.reason.few_words"
    elif mean_conf < MIN_MEAN_CONF:
        status, reason = "unreadable", "page.reason.low_confidence"
    width_pt, height_pt = size_pt if size_pt else (width * 72 / dpi, height * 72 / dpi)
    reading = PageReading(
        number, width, height, width_pt, height_pt, dpi, tuple(words), status, reason, None, mean_conf, unaligned
    )
    if study is None or status != "ocr":
        return reading, {}
    from .crops import cut_unit_crops

    crops, report = cut_unit_crops(image, line_reads, reading.words, study)
    return dataclasses.replace(reading, reread_report=report), crops
