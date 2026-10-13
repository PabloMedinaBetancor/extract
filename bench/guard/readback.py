"""G3: how well undamaged pages are read back against their own truth, and the sheets for the label audit.

A clean page read at 97 % or more of its words equal to the truth says the filters let faithful text layers
through and the labeller aligns words. The audit sheets show crops with both texts so that a person can say
whether a label is right; they carry document text, so they are written outside git.
"""

import json
from pathlib import Path

import cv2
import numpy as np

from bench.corpus.labels import TruthWord
from bench.guard import artifacts
from extract.engine.cache import reading_from_dict

READBACK_FLOOR = 0.97
SHEET_COLUMNS = 3
CROP_PAD = 8


def readback(records: list[dict]) -> dict:
    """Counts over the undamaged pages: words equal to their truth, boundary mismatches, invented, ungrounded, missed."""
    totals = {
        "pages": 0,
        "ocr_words": 0,
        "ok": 0,
        "boundary_only": 0,
        "invented": 0,
        "ungrounded": 0,
        "missed": 0,
        "truth_words": 0,
    }
    worst = []
    for record in records:
        reading = reading_from_dict(record["ocr"]["reading"])
        truth = [TruthWord(w["id"], w["text"], tuple(w["box"])) for w in record["truth"]["words"]]
        alignment, _ = artifacts.align(
            [(w.id, w.text, w.box) for w in reading.words], truth, record.get("ungrounded", [])
        )
        kinds = alignment.kinds
        scored = len(reading.words) - kinds.get("ungrounded", 0) - kinds.get("artifact", 0)
        totals["pages"] += 1
        totals["ocr_words"] += len(reading.words)
        totals["ok"] += kinds.get("ok", 0)
        totals["boundary_only"] += alignment.boundary_only
        totals["invented"] += kinds.get("invented", 0)
        totals["ungrounded"] += kinds.get("ungrounded", 0) + kinds.get("artifact", 0)
        totals["missed"] += len(alignment.missed)
        totals["truth_words"] += len(truth)
        worst.append((kinds.get("ok", 0) / max(1, scored), f"{record['doc_id']}/{record['page']}"))
    scored = totals["ocr_words"] - totals["ungrounded"]
    report = {
        **totals,
        "ok_share": totals["ok"] / max(1, scored),
        "boundary_mismatch_share": totals["boundary_only"] / max(1, scored),
        "missed_share": totals["missed"] / max(1, totals["truth_words"]),
        "floor": READBACK_FLOOR,
        "worst_pages": [{"page": page, "ok_share": round(share, 3)} for share, page in sorted(worst)[:5]],
    }
    report["meets_floor"] = report["ok_share"] >= READBACK_FLOOR
    return report


def audit_sample(rows, per_class: int = 60, seed: int = 1) -> list[int]:
    """Row indices: `per_class` words labelled wrong and as many labelled right, drawn at random."""
    rng = np.random.default_rng(seed)
    picks = []
    for label in (1, 0):
        pool = np.nonzero(rows.y == label)[0]
        picks += rng.choice(pool, size=min(per_class, len(pool)), replace=False).tolist()
    return picks


def crop_sheet(entries: list[dict], out_path: Path, data_dir: Path) -> None:
    """One picture: each entry is a crop of a word with the OCR text and the truth text under it."""
    cells = []
    for entry in entries:
        image = cv2.imread(
            str(data_dir / "pages" / entry["set"] / entry["doc_id"] / f"{entry['page']}.png"), cv2.IMREAD_GRAYSCALE
        )
        x0, y0, x1, y1 = (int(v) for v in entry["box"])
        height, width = image.shape
        crop = image[
            max(0, y0 - CROP_PAD) : min(height, y1 + CROP_PAD), max(0, x0 - CROP_PAD) : min(width, x1 + CROP_PAD)
        ]
        if not crop.size:
            crop = np.full((12, 12), 200, np.uint8)
        cell = np.full((110, 420), 255, np.uint8)
        scale = min(1.0, 400 / max(1, crop.shape[1]), 56 / max(1, crop.shape[0]))
        crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cell[4 : 4 + crop.shape[0], 8 : 8 + crop.shape[1]] = crop
        cv2.putText(cell, f"ocr:   {entry['ocr']}"[:48], (8, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
        cv2.putText(cell, f"truth: {entry['truth']}"[:48], (8, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.45, 0, 1)
        cv2.rectangle(cell, (0, 0), (419, 109), 120, 1)
        cells.append(cell)
    rows = [
        np.hstack(
            cells[i : i + SHEET_COLUMNS]
            + [np.full((110, 420), 255, np.uint8)] * (SHEET_COLUMNS - len(cells[i : i + SHEET_COLUMNS]))
        )
        for i in range(0, len(cells), SHEET_COLUMNS)
    ]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.vstack(rows))


def load_clean(data_dir: Path) -> list[dict]:
    paths = sorted((data_dir / "pages" / "clean").glob("*/*.json"))
    return [r for r in (json.loads(p.read_text(encoding="utf-8")) for p in paths) if r.get("ocr")]
