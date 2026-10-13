"""Pass 2 of the second reading over the development corpus: read the crops that `run_corpus --second-reading` stored
with the readers of one cell and keep each page's readings beside it.

    python -m bench.run_pass2 --set train --cell L --budget 16

The cells of the pilot (gate P) are L = (line, cheap readers), Z = (zone, deep), and the crossed ones
LD = (line, deep) and ZC = (zone, cheap). Page `<n>.json` gets `<n>.rr2-<cell>.json`, a file that holds the key of the
page's first reading and of the cell's readers (so a changed model or reader repeats only this) and the readings of the
words; `load_cell` gives the page back with them. The seconds of each page and the peak memory go to
`bench/results/reader-arms.json` with the cell, the readers and their sha256.

Run it inside one memory-capped scope, like `run_corpus`. It is one process, which is the
reader process of the product without its queue: the memory it reports is the memory the reader process needs.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import statistics
import time
from pathlib import Path

from bench.run_corpus import pages_of, pilot_pages
from extract.engine.cache import apply_rereads, read_crops_file, reading_from_dict, rereads_to_dict
from extract.engine.crops import decode
from extract.engine.readers import CHEAP, ReaderSet
from extract.engine.reread import RereadConfig
from extract.engine.second import second_pass

CELLS = {
    "L": ("line", CHEAP),
    "Z": ("zone", ("deep",)),
    "LD": ("line", ("deep",)),
    "ZC": ("zone", CHEAP),
}
RESULT = Path("bench/results/reader-arms.json")


def cell_path(page_json: Path, cell: str) -> Path:
    return page_json.with_name(f"{page_json.stem}.rr2-{cell}.json")


def cell_key(first_key: str, readers: ReaderSet, unit: str) -> str:
    return f"{first_key}|rr2:{readers.key()}:{unit}"


def load_cell(page_json: Path, cell: str, readers_key: str | None = None):
    """The page of the corpus with the readings of `cell`, or None when pass 2 has not read it (or read it with other
    readers than `readers_key`, when given)."""
    ocr = json.loads(page_json.read_text(encoding="utf-8")).get("ocr")
    path = cell_path(page_json, cell)
    if not ocr or "reading" not in ocr or not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if readers_key is not None and data.get("key") != f"{ocr['engine']}|rr2:{readers_key}":
        return None
    first = reading_from_dict(ocr["reading"])
    return apply_rereads(first, data, data["key"])


def study_of(engine_key: str) -> RereadConfig:
    """The `RereadConfig` that a corpus page was read with, from the `u:` part of its engine key."""
    part = next(p for p in engine_key.split("|") if p.startswith("u:"))
    tau_char, tau_word, _, study = part[2:].split(":")
    lines, _, zones = study.partition("/")
    return RereadConfig(float(tau_char), float(tau_word), int(lines), int(zones or lines))


def unit_seconds(page, crops, readers, unit: str, names, config: RereadConfig) -> dict:
    """What the page costs pass 2 in a form that adds up (the pilot's `pages` entries): `fixed_s` is what does not
    depend on the budget (choosing the units and building the page, no crop read) and `unit_s[k]` the seconds the
    readers of the cell take on the unit of rank k alone (decoding its crop included). A budget b costs
    `fixed_s + sum(unit_s[:b])`."""
    began = time.perf_counter()
    second_pass(page, {}, readers, unit, config)
    fixed = time.perf_counter() - began
    each = []
    for k in range(len(config.units(page.words, unit))):
        data = crops.get(f"{unit}{k}")
        if data is None:
            break
        began = time.perf_counter()
        image = decode(data)
        for name in names:
            readers.read_many(name, [image])
        each.append(time.perf_counter() - began)
    return {"fixed_s": round(fixed, 4), "unit_s": [round(t, 4) for t in each]}


def per_reader(args: argparse.Namespace) -> int:
    """Seconds of ONE reader on every stored crop of the pages, by unit kind (`--per-reader NAME`): each unit alone, the
    decoding of its crop included. Mean and p95 per unit and per page (the units a page stores, at most 16 lines and 40
    zones), to `cells`-independent `per_reader.NAME` in the result (the ablation of the readers needs their prices)."""
    readers = ReaderSet((args.per_reader,), threads=args.threads, backend="openvino")
    pages = pilot_pages(args.manifest, args.data) if args.pilot else pages_of(args.manifest, args.set, args.data)
    unit_times: dict[str, list[float]] = {"line": [], "zone": []}
    page_times: dict[str, list[float]] = {"line": [], "zone": []}
    warmed = False
    for _, path in pages:
        ocr = json.loads(path.read_text(encoding="utf-8")).get("ocr")
        if not ocr or "reading" not in ocr or "|u:" not in ocr.get("engine", "") or ocr["reading"]["status"] != "ocr":
            continue
        config = study_of(ocr["engine"])
        crops = read_crops_file(path.with_suffix(".crops.npz"), config.key())
        if crops is None:
            continue
        first = reading_from_dict(ocr["reading"])
        for unit in unit_times:
            total = 0.0
            for k in range(len(config.units(first.words, unit))):
                data = crops.get(f"{unit}{k}")
                if data is None:
                    break
                began = time.perf_counter()
                readers.read_many(args.per_reader, [decode(data)])
                took = time.perf_counter() - began
                if not warmed:  # the first call compiles the model
                    warmed = True
                    began = time.perf_counter()
                    readers.read_many(args.per_reader, [decode(data)])
                    took = time.perf_counter() - began
                unit_times[unit].append(took)
                total += took
            page_times[unit].append(total)

    def stats(values: list[float]) -> dict:
        ordered = sorted(values)
        return {
            "mean": round(statistics.fmean(values), 4),
            "p95": round(ordered[int(0.95 * (len(ordered) - 1))], 4),
            "n": len(values),
        }

    record = {
        "sha256": readers.key(),
        "unit_seconds": {u: stats(v) for u, v in unit_times.items() if v},
        "page_seconds": {u: stats(v) for u, v in page_times.items() if v},
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "threads": args.threads,
    }
    print(json.dumps({args.per_reader: record}), flush=True)
    results = json.loads(args.result.read_text()) if args.result.is_file() else {"cells": {}}
    results.setdefault("per_reader", {})[args.per_reader] = record
    args.result.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


def run(args: argparse.Namespace) -> int:
    unit, names = CELLS[args.cell]
    readers = ReaderSet(names, threads=args.threads, backend="openvino")
    reader_key = readers.key()
    seconds: list[float] = []
    per_page: list[dict] = []
    done = skipped = 0
    pages = pilot_pages(args.manifest, args.data) if args.pilot else pages_of(args.manifest, args.set, args.data)
    warmed = False
    for _, path in pages:
        record = json.loads(path.read_text(encoding="utf-8"))
        ocr = record.get("ocr")
        if not ocr or "reading" not in ocr or "|u:" not in ocr.get("engine", ""):
            skipped += 1
            continue
        if ocr["reading"]["status"] != "ocr":
            continue
        key = f"{ocr['engine']}|rr2:{reader_key}:{unit}"
        target = cell_path(path, args.cell)
        if not args.units and target.is_file() and json.loads(target.read_text(encoding="utf-8")).get("key") == key:
            continue  # a stopped run resumes where it was
        config = study_of(ocr["engine"])
        crops = read_crops_file(path.with_suffix(".crops.npz"), config.key())
        if crops is None:
            skipped += 1
            continue
        first = reading_from_dict(ocr["reading"])
        if not warmed:  # loading and compiling the models is not the cost of a page
            second_pass(first, crops, readers, unit, config)
            warmed = True
        began = time.perf_counter()
        page = second_pass(first, crops, readers, unit, config)
        seconds.append(time.perf_counter() - began)
        if args.units:
            per_page.append(
                {
                    "id": path.relative_to(args.data).as_posix(),
                    **unit_seconds(first, crops, readers, unit, names, config),
                }
            )
        if not (target.is_file() and json.loads(target.read_text(encoding="utf-8")).get("key") == key):
            partial = target.with_name(f"{target.name}.{os.getpid()}.part")
            partial.write_text(json.dumps(rereads_to_dict(page, key)), encoding="utf-8")
            os.replace(partial, target)
        done += 1
        if args.limit and done >= args.limit:
            break
    summary = {
        "cell": args.cell,
        "unit": unit,
        "readers": reader_key,
        "sets": args.set,
        "pages": done,
        "skipped_without_pass_1": skipped,
        "seconds_per_page_mean": round(statistics.fmean(seconds), 3) if seconds else None,
        "seconds_per_page_p95": round(sorted(seconds)[int(0.95 * (len(seconds) - 1))], 3) if seconds else None,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "threads": args.threads,
    }
    if args.units:
        summary["per_page"] = per_page
    print(json.dumps({k: v for k, v in summary.items() if k != "per_page"}), flush=True)
    results = json.loads(args.result.read_text()) if args.result.is_file() else {"cells": {}}
    results["cells"][args.cell] = summary
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--pilot", action="store_true", help="the 60 pages of the pilot (gate P) instead of --set")
    parser.add_argument(
        "--units", action="store_true", help="also time every unit alone: `per_page` for reader-arms.json"
    )
    parser.add_argument("--cell", choices=sorted(CELLS))
    parser.add_argument("--per-reader", metavar="NAME", help="time one reader on every stored crop instead of a cell")
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--manifest", type=Path, default=Path("bench/corpus/split-v5.json"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--result", type=Path, default=RESULT)
    args = parser.parse_args()
    if not args.set and not args.pilot:
        parser.error("give --set or --pilot")
    if args.per_reader:
        raise SystemExit(per_reader(args))
    if not args.cell:
        parser.error("give --cell or --per-reader")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
