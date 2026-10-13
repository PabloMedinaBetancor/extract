"""The undamaged read-back: how many of the words the engine reads from a CLEAN render of a page equal their
truth. It tells that the truth (the PDF text layer) and the reading agree when nothing damages the page, so that the
errors measured on damaged pages are the damage's and not the truth's.

    python -m bench.run_corpus --set clean5en                 # reads the pages (one memory-capped scope)
    python -m bench.readback --set clean5en --out bench/results/readback-v5en.json

Shares: `ok_share` is the OCR words equal to their truth over the OCR words that are not on ink the layer does not
describe; the floor is 97 %. A page with no reading is a failure of the run, not a page to skip.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from bench.corpus.labels import TruthWord
from bench.guard import artifacts
from extract.engine.cache import reading_from_dict

FLOOR = 0.97
WORST = 5


def page_counts(record: dict) -> dict:
    """Counts of one page record: the reading is aligned to the truth and each OCR word is `ok`, `ungrounded`,
    `invented` or wrong, each truth word `missed` or not; `boundary_only` are the words of a merged or split group whose
    letters are all right."""
    reading = reading_from_dict(record["ocr"]["reading"])
    truth = [TruthWord(w["id"], w["text"], tuple(w["box"])) for w in record["truth"]["words"]]
    patches = [tuple(box) for box in record.get("ungrounded", [])]
    alignment, _ = artifacts.align([(w.id, w.text, w.box) for w in reading.words], truth, patches)
    kinds = alignment.kinds
    return {
        "ocr_words": len(alignment.labels),
        "ok": kinds["ok"],
        "boundary_only": alignment.boundary_only,
        "invented": kinds["invented"],
        "ungrounded": kinds["ungrounded"],
        "missed": len(alignment.missed),
        "truth_words": len(truth),
    }


def measure(records: list[dict], floor: float = FLOOR) -> dict:
    total = {k: 0 for k in ("ocr_words", "ok", "boundary_only", "invented", "ungrounded", "missed", "truth_words")}
    pages = []
    for record in records:
        counts = page_counts(record)
        for key, value in counts.items():
            total[key] += value
        grounded = counts["ocr_words"] - counts["ungrounded"]
        pages.append((counts["ok"] / grounded if grounded else 0.0, f"{record['doc_id']}/{record['page']}"))
    grounded = total["ocr_words"] - total["ungrounded"]
    ok_share = total["ok"] / grounded
    return {
        "pages": len(records),
        **total,
        "ok_share": ok_share,
        "boundary_mismatch_share": total["boundary_only"] / grounded,
        "missed_share": total["missed"] / total["truth_words"],
        "floor": floor,
        "meets_floor": ok_share >= floor,
        "worst_pages": [{"page": name, "ok_share": round(share, 3)} for share, name in sorted(pages)[:WORST]],
    }


def records_of(data_dir: Path, set_name: str) -> list[dict]:
    paths = sorted(p for p in (data_dir / "pages" / set_name).glob("*/*.json") if p.stem.isdigit())
    records = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    missing = [f"{r['doc_id']}/{r['page']}" for r in records if "reading" not in (r.get("ocr") or {})]
    if missing:
        raise SystemExit(f"{len(missing)} pages of {set_name} have no reading yet, for example {missing[0]}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", required=True, dest="set_name")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = measure(records_of(args.data_dir, args.set_name))
    text = json.dumps(report, indent=1)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
