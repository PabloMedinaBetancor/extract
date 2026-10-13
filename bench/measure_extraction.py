"""F1-F3 and the number rules on the engine's readings of the validation invoices, by damage band.

    python -m bench.measure_extraction [--data DIR] [--split val] [--out bench/results/extraction-val.json]

Reads the page JSONs written by `bench.run_corpus` (never the sealed test) and, for each band, reports invoice
detection, field and line accuracy (`bench.f_eval`) and what `checks.run` did with the numbers of the page: how many
came out `checked`, and how many of those are not what the sidecar says (a false `checked`, which must be zero).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

from bench import f_eval, inject
from extract import checks
from extract.engine.cache import reading_from_dict
from extract.fields import extract_invoice


def records(data: Path, split: str) -> list[dict]:
    paths = sorted((data / "pages" / split).glob("*/*.json"))
    found = [json.loads(p.read_text(encoding="utf-8")) for p in paths]
    return [r for r in found if r.get("source") == "invoice" and r.get("ocr")]


def _plain(text: str) -> str:
    """A printed number without its spaces, case, sentence punctuation and percent sign, to compare read and printed."""
    return inject.identifier_of(text).rstrip("%")


def printed_rates(truth: dict) -> list[str]:
    """The rates of the invoice: they are printed with their percent sign and are not in the list of numbers."""
    rates = [pair["rate"] for pair in truth["vat"]]
    if truth["withholding"]:
        rates.append(truth["withholding"]["rate"])
    return rates


def checked_counts(item: f_eval.Item) -> tuple[int, list[tuple[str, str]]]:
    """How many numbers of the page came out checked, and which of them are not a number the invoice prints."""
    truth = item.truth
    printed = {_plain(entry["text"]) for entry in truth["numbers"]} | {_plain(rate) for rate in printed_rates(truth)}
    results = checks.run(item.page.words, extract_invoice(item.page), truth["language"], suggest=False)
    checked = [r for r in results if r.status == "checked"]
    wrong = [(truth["id"], r.text) for r in checked if _plain(r.text) not in printed]
    return len(checked), wrong


def measure(data: Path, split: str) -> dict:
    invoices = data / "invoices"
    by_band: dict[str, list[f_eval.Item]] = defaultdict(list)
    for record in records(data, split):
        truth = json.loads((invoices / record["set"] / f"{record['doc_id']}.json").read_text(encoding="utf-8"))
        by_band[record["band"]].append(f_eval.Item(reading_from_dict(record["ocr"]["reading"]), truth))
    report = {}
    for band, items in sorted(by_band.items()):
        counts = [checked_counts(item) for item in items]
        evaluation = f_eval.evaluate(items)
        report[band] = {
            "pages": len(items),
            "f1_recall": evaluation["f1"]["recall"],
            "f2": evaluation["f2"],
            "f2_mean": sum(evaluation["f2"].values()) / len(evaluation["f2"]),
            "f3": evaluation["f3"],
            "checked": sum(c for c, _ in counts),
            "false_checked": [w for _, wrong in counts for w in wrong],
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--split", default="val", choices=("train", "val"))
    parser.add_argument("--out", type=Path, default=Path("bench/results/extraction-val.json"))
    args = parser.parse_args()
    report = measure(args.data, args.split)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({band: {k: v for k, v in r.items() if k != "f2"} for band, r in report.items()}, indent=1))


if __name__ == "__main__":
    main()
