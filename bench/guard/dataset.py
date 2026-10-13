"""Turns pages that carry an OCR reading into rows for the detector: signals, label, document, band.

A row is one OCR word. Words that sit on ink the text layer does not describe (`ungrounded`) are left out of
every set and counted. Pages the engine called `unreadable` are left out too, because the guard does not run on
them, and are counted so a recall is never reported without them.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bench.corpus import sealed
from bench.corpus.labels import TruthWord
from bench.guard import artifacts
from extract.engine.cache import reading_from_dict
from extract.guard import signals


@dataclass
class Rows:
    x: np.ndarray
    y: np.ndarray
    group: np.ndarray
    band: np.ndarray
    kind: np.ndarray
    detail: np.ndarray
    page: np.ndarray
    text: np.ndarray
    truth: np.ndarray
    box: np.ndarray
    chars: np.ndarray
    counts: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.y)

    def subset(self, mask: np.ndarray) -> "Rows":
        return Rows(
            self.x[mask],
            self.y[mask],
            self.group[mask],
            self.band[mask],
            self.kind[mask],
            self.detail[mask],
            self.page[mask],
            self.text[mask],
            self.truth[mask],
            self.box[mask],
            self.chars[mask],
            self.counts,
        )


def _object_array(items) -> np.ndarray:
    """A 1-D object array holding the items (tuples must not be unpacked into a 2-D array)."""
    out = np.empty(len(items), dtype=object)
    for i, item in enumerate(items):
        out[i] = item
    return out


def page_records(data_dir: Path, set_name: str) -> list[tuple[Path, dict]]:
    """The page records of one set that already carry an OCR reading, in a fixed order. The sealed sets are refused."""
    return list(iter_page_records(data_dir, set_name))


def iter_page_records(data_dir: Path, set_name: str):
    """`page_records` one record at a time: a run over a whole set keeps one parsed page in memory instead of all of them
    (a page record is several MB of dicts; the list of a set is the peak of a protocol run on a small PC). The sealed sets
    are refused at the call, not at the first record."""
    sealed.refuse_sealed_set(set_name)
    paths = sorted((data_dir / "pages" / set_name).glob("*/*.json"))

    def records():
        for path in paths:
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("ocr"):
                yield path, record

    return records()


def rows_of_page(
    record: dict, languages=("es", "en"), locale: str = "auto", mark_artifacts: bool = True
) -> tuple[dict, dict]:
    """The signal matrix, labels and counts of one read page."""
    reading = reading_from_dict(record["ocr"]["reading"])
    counts = {"pages": 1, "unreadable_pages": 0, "ocr_words": len(reading.words), "ungrounded": 0, "missed": 0}
    if reading.status != "ocr":
        counts["unreadable_pages"] = 1
        return None, counts
    truth = [TruthWord(w["id"], w["text"], tuple(w["box"])) for w in record["truth"]["words"]]
    read = [(w.id, w.text, w.box) for w in reading.words]
    alignment, found = artifacts.align(read, truth, record.get("ungrounded", []), mark_artifacts)
    counts["artifacts"] = sum(found.values())
    counts["missed"] = len(alignment.missed)
    counts["truth_words"] = len(truth)
    if locale == "auto":
        from extract import checks

        locale = checks.detect_locale(reading.words)
    matrix = signals.extract(reading.words, languages, locale)
    keep = [i for i, label in enumerate(alignment.labels) if label.wrong is not None]
    counts["ungrounded"] = len(alignment.labels) - len(keep)
    page = {
        "x": matrix[keep],
        "y": np.array([int(alignment.labels[i].wrong) for i in keep], dtype=np.uint8),
        "kind": np.array([alignment.labels[i].kind for i in keep], dtype=object),
        "detail": np.array([alignment.labels[i].detail or "" for i in keep], dtype=object),
        "text": np.array([reading.words[i].text for i in keep], dtype=object),
        "truth": np.array([alignment.labels[i].truth_text or "" for i in keep], dtype=object),
        "box": np.array([reading.words[i].box for i in keep], dtype=np.float32).reshape(-1, 4),
        "chars": _object_array([reading.words[i].char_conf for i in keep]),
        "boundary_only": alignment.boundary_only,
    }
    counts["boundary_only"] = alignment.boundary_only
    return page, counts


def load_rows(data_dir: Path, set_name: str, languages=("es", "en"), engine: str | None = None) -> Rows:
    """All the rows of a set. With `engine`, pages read by another engine fingerprint are refused."""
    return rows_from_records(page_records(data_dir, set_name), languages, engine)


def rows_from_records(records, languages=("es", "en"), engine: str | None = None) -> Rows:
    """The rows of (json path, record) pairs; a record without an OCR reading is left out."""
    parts, totals = [], {}
    for path, record in records:
        if not record.get("ocr"):
            continue
        if engine is not None and record["ocr"].get("engine") != engine:
            raise ValueError(f"{path} was read by another engine than {engine}")
        page, counts = rows_of_page(record, languages)
        for name, value in counts.items():
            totals[name] = totals.get(name, 0) + value
        if page is None or not len(page["y"]):
            continue
        n = len(page["y"])
        parts.append(
            (
                page["x"],
                page["y"],
                np.full(n, record["doc_id"], dtype=object),
                np.full(n, record["band"], dtype=object),
                page["kind"],
                page["detail"],
                np.full(n, f"{record['doc_id']}/{record['page']}", dtype=object),
                page["text"],
                page["truth"],
                page["box"],
                page["chars"],
            )
        )
    if not parts:
        empty = np.empty(0, dtype=object)
        return Rows(
            np.empty((0, len(signals.COLUMNS)), np.float32),
            np.empty(0, np.uint8),
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            empty,
            np.empty((0, 4), np.float32),
            empty,
            totals,
        )
    columns = [np.concatenate(column) for column in zip(*parts, strict=True)]
    return Rows(*columns, counts=totals)
