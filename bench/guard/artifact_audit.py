"""Audit sheets for the text-layer artifact rule (from a review of the labels): a seeded sample of the components marked
`layer_glue` and `layer_duplicate`, each as a crop of the page with the OCR words boxed, so a reader can say whether the
ink matches the OCR (the artifact is real) or the layer (the OCR made the error and the rule hides it).

    python3 -s -m bench.guard.artifact_audit --out DIR --per-reason 30 --seed 1
"""

import argparse
import json
import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

from bench.corpus.labels import TruthWord
from bench.guard import accounting, artifacts, dataset
from bench.guard.pairs import reading_from_dict

REASONS = ("layer_glue", "layer_duplicate")
PER_SHEET = 6


def candidates(data: Path, sets) -> dict:
    found = {r: [] for r in REASONS}
    for set_name in sets:
        for path, record in dataset.page_records(data, set_name):
            reading = reading_from_dict(record["ocr"]["reading"])
            if reading.status != "ocr" or not reading.words:
                continue
            truth = [TruthWord(w["id"], w["text"], tuple(w["box"])) for w in record["truth"]["words"]]
            read = [(w.id, w.text, w.box) for w in reading.words]
            alignment, _ = artifacts.align(read, truth, record.get("ungrounded", []), mark_artifacts=False)
            texts = [w.text for w in reading.words]
            for component in accounting.components_of(texts, alignment.labels):
                why = artifacts.reason(component, texts)
                if why in REASONS:
                    found[why].append(
                        (path, record["doc_id"], record["page"], component, [reading.words[i] for i in component.words])
                    )
    return found


def crop(path: Path, words, label: str) -> np.ndarray:
    image = cv2.imread(str(path.with_suffix(".png")))
    left = int(min(w.box[0] for w in words))
    right = int(max(w.box[2] for w in words))
    top = int(min(w.box[1] for w in words))
    bottom = int(max(w.box[3] for w in words))
    height = max(bottom - top, 20)
    x0, x1 = max(left - 4 * height, 0), min(right + 4 * height, image.shape[1])
    y0, y1 = max(top - height, 0), min(bottom + height, image.shape[0])
    view = image[y0:y1, x0:x1].copy()
    for w in words:
        cv2.rectangle(
            view, (int(w.box[0]) - x0, int(w.box[1]) - y0), (int(w.box[2]) - x0, int(w.box[3]) - y0), (0, 0, 255), 1
        )
    scale = max(1.0, 700 / view.shape[1])
    view = cv2.resize(view, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    banner = np.full((26, view.shape[1], 3), 255, np.uint8)
    cv2.putText(banner, label, (4, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return np.vstack([banner, view])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sets", nargs="+", default=["train", "val"])
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--per-reason", type=int, default=30)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    found = candidates(args.data, args.sets)
    index = {}
    for why in REASONS:
        pool = found[why]
        sample = rng.sample(pool, min(args.per_reason, len(pool)))
        index[why] = {"population": len(pool), "sample": []}
        for sheet in range(0, len(sample), PER_SHEET):
            tiles = []
            for k, (path, doc, page, component, words) in enumerate(sample[sheet : sheet + PER_SHEET]):
                number = sheet + k + 1
                label = f"{why} #{number}  OCR: {' | '.join(w.text for w in words)}  LAYER: {component.truth}"
                tiles.append(crop(path, words, label))
                index[why]["sample"].append(
                    {"n": number, "doc": doc, "page": page, "ocr": [w.text for w in words], "layer": component.truth}
                )
            width = max(t.shape[1] for t in tiles)
            tiles = [np.pad(t, ((0, 4), (0, width - t.shape[1]), (0, 0)), constant_values=255) for t in tiles]
            cv2.imwrite(str(args.out / f"{why}-{sheet // PER_SHEET + 1}.png"), np.vstack(tiles))
    (args.out / "index.json").write_text(json.dumps(index, indent=1, ensure_ascii=False), encoding="utf-8")
    print({w: index[w]["population"] for w in REASONS})
    return 0


if __name__ == "__main__":
    sys.exit(main())
