"""The one read of the sealed test set: the guard's numbers, F1-F3 and N6, with the model and the code frozen first.

    python -m bench.read_test freeze      # once, when nothing is left to change: writes models/FROZEN.sha256
    python -m bench.read_test dry-run     # the same code on validation, written under /tmp
    python -m bench.read_test read        # the read of the test; a second one needs a written reason

`read` refuses to start if a frozen file changed, or if the images or the truth of a sealed page differ from the
record `bench.corpus.sealed.freeze_record` wrote. Only then does it open the pages for metrics, which is logged in
`bench/results/test_reads.json`. After that no threshold, model or rule moves: what comes out is published as it is.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from bench import f_eval, measure_extraction
from bench.corpus import sealed
from bench.guard import baselines, dataset, final
from extract.guard.checker import Guard

FROZEN = Path("models/FROZEN.sha256")
FREEZE_RECORD = Path("bench/results/test-freeze.json")
OUT = Path("bench/results")
SLICES = ("pages", "invoices", "unseen")
FROZEN_GLOBS = (
    "models/guard.model",
    "src/extract/guard/data/*",
    "src/extract/guard/*.py",
    "src/extract/checks.py",
    "bench/guard/*.py",
    "bench/corpus/labels.py",
    "bench/corpus/split.json",
    "bench/SOURCES.csv",
    "bench/corpus/sealed.py",
    "bench/read_test.py",
)


class FrozenError(RuntimeError):
    pass


def frozen_files(root: Path) -> list[Path]:
    found = {path for pattern in FROZEN_GLOBS for path in root.glob(pattern) if path.is_file()}
    return sorted(found, key=lambda path: path.relative_to(root).as_posix())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root: Path, engine: str) -> int:
    """Write the hashes of the model, the dictionaries, the split and the code, and the engine they were built for."""
    target = root / FROZEN
    if target.exists():
        raise FrozenError(f"{FROZEN} exists: the frozen set is written once")
    lines = [f"{digest(path)}  {path.relative_to(root).as_posix()}" for path in frozen_files(root)]
    target.write_text(f"# engine {engine}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def verify(root: Path) -> str:
    """The engine of the freeze, after checking that every listed file is as it was and no listed glob has a new one."""
    target = root / FROZEN
    if not target.is_file():
        raise FrozenError(f"{FROZEN} is missing: freeze before reading the test")
    listed, engine = {}, None
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.startswith("# engine "):
            engine = line.removeprefix("# engine ")
            continue
        hashed, name = line.split("  ", 1)
        listed[name] = hashed
    changed = [name for name, hashed in listed.items() if not (root / name).is_file() or digest(root / name) != hashed]
    added = [name for name in (path.relative_to(root).as_posix() for path in frozen_files(root)) if name not in listed]
    if changed or added:
        raise FrozenError(f"changed since the freeze: {changed}; not in the freeze: {added}")
    if engine is None:
        raise FrozenError(f"{FROZEN} does not name the engine")
    return engine


def split_records(pages) -> dict[str, list]:
    """The sealed pages by slice: catalogue pages of the test, invoice pages of the test, and the unseen damage."""
    slices = {name: [] for name in SLICES}
    for path, record in pages:
        if record["set"] == "unseen":
            slices["unseen"].append((path, record))
        elif record.get("source") == "invoice":
            slices["invoices"].append((path, record))
        else:
            slices["pages"].append((path, record))
    return slices


def engines_of(records) -> set[str]:
    return {record["ocr"]["engine"] for _, record in records if record.get("ocr")}


def analyse(slices: dict[str, list], data: Path, resamples: int, engine: str | None = None, seed: int = 1) -> dict:
    """Every number of the read, from the records of each slice. Works on any set of read pages."""
    train_records = dataset.page_records(data, "train")
    validation_records = dataset.page_records(data, "val")
    seen = engines_of(train_records) | engines_of(validation_records)
    for records in slices.values():
        seen |= engines_of(records)
    if len(seen) != 1 or (engine is not None and seen != {engine}):
        raise FrozenError(f"the pages were read by {sorted(seen)}, not by the frozen engine {engine}")
    train_rows = dataset.rows_from_records(train_records)
    validation_rows = dataset.rows_from_records(validation_records)
    logistic = baselines.fit_logistic(train_rows.x, train_rows.y, validation_rows.x, validation_rows.y)
    guard = Guard.load(Path("models/guard.model")).model
    report = {"engine": next(iter(seen)), "model": digest(Path("models/guard.model")), "slices": {}}
    for name, records in slices.items():
        rows = dataset.rows_from_records(records)
        scores = {
            "lowest_mean_confidence": baselines.lowest_mean_confidence(rows.x),
            "lowest_minimum_confidence": baselines.lowest_minimum_confidence(rows.x),
            "outside_dictionary_first": baselines.outside_the_dictionary_first(rows.x),
            "logistic_regression": logistic.score(rows.x),
        }
        report["slices"][name] = final.evaluate(rows, guard, scores, resamples, seed)
    return report


def extraction(records, data: Path) -> dict:
    """F1-F3 over the test pages, and N6: the numbers the rules called `checked` that are not what the invoice prints.

    N6 compares the text of each checked number with the numbers the invoice prints, not with the true value of the
    same word (an OCR reading has no aligned true word), so a misread that equals another printed number of the
    same invoice would not count. The pages are not told which are unreadable, which makes the count of checked
    numbers a little larger and the criterion harder, not looser.
    """
    items = f_eval.items_from_records(records, data / "invoices")
    bands = [record["band"] for _, record in records if (record.get("ocr") or {}).get("reading")]
    report = f_eval.evaluate(items)
    by_band, wrong = {}, []
    for item, band in zip(items, bands, strict=True):
        if item.truth is None:
            continue
        count, found = measure_extraction.checked_counts(item)
        entry = by_band.setdefault(band, {"invoices": 0, "checked": 0, "false_checked": 0})
        entry["invoices"] += 1
        entry["checked"] += count
        entry["false_checked"] += len(found)
        wrong += found
    checked = sum(entry["checked"] for entry in by_band.values())
    report["n6"] = {
        "invoices": sum(entry["invoices"] for entry in by_band.values()),
        "checked": checked,
        "false_checked": len(wrong),
        "false_checked_numbers": wrong,
        "rate_upper_bound_95": 3 / checked if checked and not wrong else None,
        "by_band": by_band,
    }
    injection = Path("bench/results/injection.json")
    if injection.is_file():
        report["injection"] = json.loads(injection.read_text(encoding="utf-8"))
    return report


def flat_metrics(report: dict, extraction_report: dict) -> dict:
    """The figures the numbers gate reads, under the names of the report."""

    def figure(block: dict, scale: float = 1.0) -> dict:
        return {key: block[key] * scale for key in ("value", "low", "high")}

    pages, invoices = report["slices"]["pages"], report["slices"]["invoices"]
    flat = {
        "guard.pages.recall_at_review_5pct": figure(pages["at_review_5pct"]["detector"]["recall"]),
        "guard.pages.precision_at_review_5pct": figure(pages["at_review_5pct"]["detector"]["precision"]),
        "guard.invoices.recall_at_review_5pct": figure(invoices["at_review_5pct"]["detector"]["recall"]),
        "guard.baseline_conf.recall_at_review_5pct": figure(
            pages["at_review_5pct"]["lowest_mean_confidence"]["recall"]
        ),
        "guard.baseline_logreg.recall_at_review_5pct": figure(pages["at_review_5pct"]["logistic_regression"]["recall"]),
    }
    for band in ("light", "medium", "heavy"):
        if band in pages["calibration"]["ece"]:
            flat[f"guard.calibration.ece_{band}"] = figure(pages["calibration"]["ece"][band])
    fixer = pages["fixer"]
    if "top1_all" in fixer:
        flat["guard.fixer.top1_pct"] = figure(fixer["top1_all"], 100.0)
        flat["guard.fixer.top1_given_candidate_pct"] = figure(fixer["top1_given_candidate"], 100.0)
    if "precision" in fixer.get("shown", {}):
        flat["guard.fixer.suggestion_precision_pct"] = figure(fixer["shown"]["precision"], 100.0)
    speed = Path("bench/results/guard-speed.json")
    if speed.is_file():
        flat["guard.speed.ms_per_page"] = {"value": json.loads(speed.read_text(encoding="utf-8"))["page_ms"]["median"]}
    flat["extract.synth.false_checked_numbers"] = {"value": extraction_report["n6"]["false_checked"]}
    return flat


def write(out: Path, report: dict, extraction_report: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    report["metrics"] = flat_metrics(report, extraction_report)
    (out / "guard-test.json").write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")
    (out / "extraction-test.json").write_text(
        json.dumps(extraction_report, indent=1, default=float) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["freeze", "dry-run", "read"])
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--resamples", type=int, default=final.RESAMPLES)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    root = Path.cwd()
    if args.action == "freeze":
        engines = engines_of(dataset.page_records(args.data, "train"))
        if len(engines) != 1:
            raise FrozenError(f"the training pages were read by {sorted(engines)}")
        print(f"{freeze(root, next(iter(engines)))} files frozen in {FROZEN}")
        return
    if args.action == "dry-run":
        records = dataset.page_records(args.data, "val")
        slices = {"pages": [r for r in records if r[1].get("source") != "invoice"], "invoices": [], "unseen": []}
        slices["invoices"] = [r for r in records if r[1].get("source") == "invoice"]
        slices["unseen"] = slices["pages"][:20]
        out = args.out or Path("/tmp/read_test_dry")
        report = analyse(slices, args.data, args.resamples)
        write(out, report, extraction(records, args.data))
        print(f"dry run written to {out}")
        return
    engine = verify(root)
    previous = json.loads((root / FREEZE_RECORD).read_text(encoding="utf-8"))[-1]
    parts = sealed.fingerprint_parts(
        sealed.read_test_pages(args.data, "ocr", sealed.V4_READS_PATH, sets=sealed.V4_SETS)
    )
    if (parts["images"], parts["truth"]) != (previous["images"], previous["truth"]):
        raise FrozenError("the images or the truth of the sealed pages differ from the freeze record")
    pages = sealed.read_test_pages(args.data, "metrics", sealed.V4_READS_PATH, sets=sealed.V4_SETS)
    report = analyse(split_records(pages), args.data, args.resamples, engine)
    tests = [(path, record) for path, record in pages if record["set"] == "test"]
    write(args.out or root / OUT, report, extraction(tests, args.data))
    print(json.dumps(report["metrics"], indent=1, default=float))


if __name__ == "__main__":
    try:
        main()
    except FrozenError as error:
        sys.exit(str(error))
