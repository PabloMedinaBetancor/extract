"""Pilot of the damage bands: read the pilot pages with the engine and report the word error rate per band.

The ranges of `degrade.BANDS` are tuned once with this, so that the rates fall near 1-2 %, 3-6 % and 8-15 %.
The pilot pages belong to no other set. Only statistics are written, never page text.

    python -m bench.corpus.pilot [--render] [--out bench/results/pilot-bands.json]
"""

import argparse
import json
import os
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from bench.corpus import labels
from bench.corpus.labels import TruthWord

TARGETS = {"light": (1.0, 2.0), "medium": (3.0, 6.0), "heavy": (8.0, 15.0)}


def page_stats(record: dict, reading) -> dict:
    """Label one read page against its truth and count what went wrong."""
    truth = [TruthWord(w["id"], w["text"], tuple(w["box"])) for w in record["truth"]["words"]]
    read = [(w.id, w.text, w.box) for w in reading.words]
    alignment = labels.label_words(read, truth)
    return {
        "ocr_words": len(read),
        "truth_words": len(truth),
        "wrong": sum(label.wrong for label in alignment.labels),
        "missed": len(alignment.missed),
        "boundary_only": alignment.boundary_only,
        "kinds": dict(alignment.kinds),
    }


def band_report(rows: list[tuple[str, dict]]) -> dict:
    """Pool the per-page stats by band: pooled error rate, spread across pages, shapes of the errors."""
    report = {}
    by_band = defaultdict(list)
    for band, stats in rows:
        by_band[band].append(stats)
    for band, pages in by_band.items():
        words = sum(p["ocr_words"] for p in pages)
        rates = [100.0 * p["wrong"] / p["ocr_words"] for p in pages if p["ocr_words"]]
        kinds = Counter()
        for p in pages:
            kinds.update(p["kinds"])
        report[band] = {
            "pages": len(pages),
            "ocr_words": words,
            "wrong_pct": round(100.0 * sum(p["wrong"] for p in pages) / max(1, words), 2),
            "page_wrong_pct_median": round(statistics.median(rates), 2) if rates else None,
            "page_wrong_pct_min_max": [round(min(rates), 2), round(max(rates), 2)] if rates else None,
            "missed_pct": round(
                100.0 * sum(p["missed"] for p in pages) / max(1, sum(p["truth_words"] for p in pages)), 2
            ),
            "kinds": dict(kinds),
            "per_page": {
                p["page"]: round(100.0 * p["wrong"] / max(1, p["ocr_words"]), 1) for p in pages if "page" in p
            },
        }
        if band in TARGETS:
            low, high = TARGETS[band]
            report[band]["target_pct"] = [low, high]
            report[band]["on_target"] = low <= report[band]["wrong_pct"] <= high
    return report


def main() -> None:
    from bench.corpus import catalog
    from extract.engine.ocr import default_engine, read_page

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--out", type=Path, default=Path("bench/results/pilot-bands.json"))
    parser.add_argument("--render", action="store_true", help="render the pilot pages again with the current ranges")
    parser.add_argument("--dump", type=Path, help="keep the OCR words of every page here, for a look at the errors")
    args = parser.parse_args()
    if args.render:
        catalog.render_selected(
            args.data_dir,
            Path("bench/corpus/split.json"),
            catalog.default_catalog_root(),
            sets=["pilot"],
        )
    engine = default_engine()
    rows = []
    for path in sorted((args.data_dir / "pages" / "pilot").glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        reading = read_page(path.with_suffix(".png"), engine, record["page"], round(record["image"]["dpi"]))
        rows.append((record["band"], page_stats(record, reading)))
        rows[-1][1]["page"] = f"{record['doc_id']}/{record['page']}"
        if args.dump:
            args.dump.mkdir(parents=True, exist_ok=True)
            words = [[w.text, list(w.box), list(w.char_conf or [])] for w in reading.words]
            (args.dump / f"{record['doc_id']}_{record['page']}.json").write_text(json.dumps(words), encoding="utf-8")
    report = band_report(rows)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
