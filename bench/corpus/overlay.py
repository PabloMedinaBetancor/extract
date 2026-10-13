"""Draws the truth boxes over a page image so a person can check that they sit on the words.

The pictures show document text, so they are written outside git (a work folder), never to bench/results.

    python -m bench.corpus.overlay --set clean --count 10 --out work/overlays
"""

import argparse
import json
import os
from pathlib import Path

import cv2


def draw_overlay(image_path: Path, record: dict, out_path: Path, max_side: int = 1800) -> None:
    image = cv2.cvtColor(cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE), cv2.COLOR_GRAY2BGR)
    for word in record["truth"]["words"]:
        x0, y0, x1, y1 = (int(round(v)) for v in word["box"])
        cv2.rectangle(image, (x0, y0), (x1, y1), (0, 0, 220), 1)
    scale = max_side / max(image.shape[:2])
    if scale < 1:
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), image)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--set", default="clean")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted((args.data_dir / "pages" / args.set).glob("*/*.json"))
    step = max(1, len(paths) // args.count)
    for path in paths[::step][: args.count]:
        record = json.loads(path.read_text(encoding="utf-8"))
        draw_overlay(path.with_suffix(".png"), record, args.out / f"{record['doc_id']}-{record['page']}.png")


if __name__ == "__main__":
    main()
