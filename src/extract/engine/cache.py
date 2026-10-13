from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from ..types import PageReading, Reading, Word
from . import ocr

CACHE_FORMAT = 2


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def page_key(pdf_sha256: str, page: int, dpi: int, second_reading: str = "") -> str:
    """The fingerprint of one engine reading: the PDF, the page, the render resolution and the engine settings, with the
    settings of the second reading (`RereadConfig.key()`) when it is on. The guard model is not part of it, so a new
    model never invalidates the OCR."""
    text = f"{CACHE_FORMAT}|{pdf_sha256}|{page}|{dpi}|{ocr.fingerprint()}"
    if second_reading:
        text += f"|{second_reading}"
    return hashlib.sha256(text.encode()).hexdigest()


class PageCache:
    """One JSON file per page under `<work_dir>/pages/<pdf sha256>/`, written under a temporary name and renamed."""

    def __init__(self, work_dir: str | Path) -> None:
        self.root = Path(work_dir) / "pages"

    def _path(self, pdf_sha256: str, page: int) -> Path:
        return self.root / pdf_sha256 / f"p{page:04d}.json"

    def get(self, pdf_sha256: str, page: int, key: str) -> PageReading | None:
        path = self._path(pdf_sha256, page)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return None
        if data.get("key") != key:
            return None
        return reading_from_dict(data["reading"])

    def put(self, pdf_sha256: str, page: int, key: str, reading: PageReading, meta: dict | None = None) -> None:
        """Store a reading. `meta` (the backend and the OCR library version it was read with) is kept in the file for
        audit and is not part of the key."""
        path = self._path(pdf_sha256, page)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}.part")
        partial.write_text(
            json.dumps({"key": key, **(meta or {}), "reading": reading_to_dict(reading)}), encoding="utf-8"
        )
        os.replace(partial, path)

    def put_crops(self, pdf_sha256: str, page: int, key: str, crops: dict[str, bytes]) -> None:
        """The crops of pass 1 next to the page's file, under the page's key (`pNNNN.crops.npz`: one byte array of PNG
        data for each `line<rank>` and `zone<rank>`)."""
        write_crops_file(crops_path(self, pdf_sha256, page), key, crops)

    def get_crops(self, pdf_sha256: str, page: int, key: str) -> dict[str, bytes] | None:
        return read_crops_file(crops_path(self, pdf_sha256, page), key)

    def put_rereads(self, pdf_sha256: str, page: int, key: str, reading: PageReading) -> None:
        """The pass 2 result of a page, cached apart from the first reading (`pNNNN.rr2.json`) under the key of pass 1
        plus `rr2:<readers with their sha256>:<unit>`: the words' readings by position and the page's report."""
        path = rereads_path(self, pdf_sha256, page)
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.{os.getpid()}.part")
        partial.write_text(json.dumps(rereads_to_dict(reading, key)), encoding="utf-8")
        os.replace(partial, path)

    def get_rereads(self, pdf_sha256: str, page: int, key: str, first: PageReading) -> PageReading | None:
        """`first` (the page of pass 1) with the cached readings of pass 2 put on its words, or None."""
        path = rereads_path(self, pdf_sha256, page)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return None
        except (OSError, ValueError):
            path.unlink(missing_ok=True)
            return None
        return apply_rereads(first, data, key)


def rereads_to_dict(reading: PageReading, key: str) -> dict:
    """The pass 2 result of a page: the words' readings by position, which lines were read by which readers, and the
    page's report counts."""
    return {
        "key": key,
        "readings": [_readings_to_list(w.readings) for w in reading.words],
        "rereads": {str(line): list(readers) for line, readers in reading.rereads.items()},
        "report": dict(reading.reread_report),
    }


def apply_rereads(first: PageReading, data: dict, key: str) -> PageReading | None:
    """`first` (the page of pass 1) with the readings of `data` put on its words, or None when the key or the number
    of words does not match."""
    if data.get("key") != key or len(data.get("readings", ())) != len(first.words):
        return None
    words = tuple(
        dataclasses.replace(w, readings=_readings_from_list(r))
        for w, r in zip(first.words, data["readings"], strict=True)
    )
    return dataclasses.replace(
        first,
        words=words,
        rereads={int(line): tuple(readers) for line, readers in data["rereads"].items()},
        reread_report={**first.reread_report, **data["report"]},
    )


def write_crops_file(path: Path, key: str, crops: dict[str, bytes]) -> None:
    """Crops of pass 1 in one `.npz` (an array of PNG bytes for each `line<rank>` and `zone<rank>`), under a key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    with open(partial, "wb") as handle:
        np.savez(handle, __key__=np.array(key), **{k: np.frombuffer(v, np.uint8) for k, v in crops.items()})
    os.replace(partial, path)


def read_crops_file(path: Path, key: str) -> dict[str, bytes] | None:
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["__key__"]) != key:
                return None
            return {name: data[name].tobytes() for name in data.files if name != "__key__"}
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError):
        path.unlink(missing_ok=True)
        return None


def crops_path(cache: PageCache, pdf_sha256: str, page: int) -> Path:
    return cache._path(pdf_sha256, page).with_suffix(".crops.npz")


def rereads_path(cache: PageCache, pdf_sha256: str, page: int) -> Path:
    return cache._path(pdf_sha256, page).with_suffix(".rr2.json")


def reading_to_dict(reading: PageReading) -> dict:
    return {
        "number": reading.number,
        "width_px": reading.width_px,
        "height_px": reading.height_px,
        "width_pt": reading.width_pt,
        "height_pt": reading.height_pt,
        "dpi": reading.dpi,
        "status": reading.status,
        "reason": reading.reason,
        "error": reading.error,
        "mean_conf": reading.mean_conf,
        "unaligned_words": reading.unaligned_words,
        "words": [
            [
                w.id,
                w.text,
                list(w.box),
                w.conf,
                list(w.char_conf) if w.char_conf is not None else None,
                w.line_id,
                w.idx_in_line,
                _alts_to_list(w.char_alts),
                w.masked,
                _readings_to_list(w.readings),
            ]
            for w in reading.words
        ],
        "rereads": {str(line): list(readers) for line, readers in reading.rereads.items()},
        "reread_report": dict(reading.reread_report),
    }


def _alts_to_list(alts):
    return None if alts is None else [[[char, p] for char, p in entries] for entries in alts]


def _alts_from_list(alts):
    return None if alts is None else tuple(tuple((char, p) for char, p in entries) for entries in alts)


def _readings_to_list(readings):
    if readings is None:
        return None
    return [
        [
            r.source,
            r.text,
            r.conf,
            list(r.char_conf) if r.char_conf is not None else None,
            list(r.group),
            r.rank,
            r.context,
        ]
        for r in readings
    ]


def _readings_from_list(readings):
    if readings is None:
        return None
    return tuple(
        Reading(
            source,
            text,
            conf,
            tuple(chars) if chars is not None else None,
            tuple(group),
            rank,
            bool(rest[0]) if rest else False,
        )
        for source, text, conf, chars, group, rank, *rest in readings
    )


def reading_from_dict(data: dict) -> PageReading:
    words = tuple(
        Word(
            id=wid,
            page=data["number"],
            text=text,
            box=tuple(box),
            conf=conf,
            char_conf=tuple(chars) if chars is not None else None,
            line_id=line,
            idx_in_line=idx,
            char_alts=_alts_from_list(extra[0]) if extra else None,
            masked=extra[1] if extra else None,
            readings=_readings_from_list(extra[2]) if extra else None,
        )
        for wid, text, box, conf, chars, line, idx, *extra in data["words"]
    )
    return PageReading(
        data["number"],
        data["width_px"],
        data["height_px"],
        data["width_pt"],
        data["height_pt"],
        data["dpi"],
        words,
        data["status"],
        data["reason"],
        data["error"],
        data["mean_conf"],
        data["unaligned_words"],
        {int(line): tuple(readers) for line, readers in data.get("rereads", {}).items()},
        dict(data.get("reread_report", {})),
    )
