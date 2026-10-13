"""The thorough second reading as the product runs it: the pages of the pilot go through the real reader process
(`ReaderProcess` with `settings_for("thorough")`: latin + wide + deep loaded together, 16 lines then 40 zones), one page at
a time, and the run measures what the panel and R7 promise: seconds a page (mean and p95; the first page, that loads the
models, apart), the peak memory of the reader process, how often it was replaced and capped, and the parity of its
output with the L and Z cells that `run_pass2` stored (the readings of L and of Z put together, word by word).

    python -m bench.run_thorough --pilot --data <data> [--threads 4] [--limit N] [--result bench/results/thorough-run.json]

Run it inside one memory-capped scope, on a quiet machine. The pages must have their pass 1
(`run_corpus --second-reading`) with the units of the thorough mode (16 lines, 40 zones) and the L and Z cells (for the
parity); a page without any of them is counted and left out."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import threading
import time
from pathlib import Path

from bench.run_corpus import pages_of, pilot_pages
from bench.run_pass2 import load_cell, study_of
from extract import second_reading
from extract.engine import readerproc
from extract.engine.cache import read_crops_file, reading_from_dict
from extract.engine.readerproc import ReaderProcess, settings_for

RESULT = Path("bench/results/thorough-run.json")
PAGE_TIMEOUT_S = 120.0


def expected_readings(path: Path):
    """The page as L then Z leave it, from the stored cells: every word with the readings of the lines' readers (from L)
    and then the zones' reader (from Z), in that order. A cell stored with more readers than the mode uses (the pilot's L
    was read with four before the pilot) is cut to the mode's own."""
    (_, line_readers), (_, zone_readers) = second_reading.thorough_passes()
    lines, zones = load_cell(path, "L"), load_cell(path, "Z")
    if lines is None or zones is None:
        return None
    return tuple(
        tuple(r for r in a.readings or () if r.source in line_readers)
        + tuple(r for r in b.readings or () if r.source in zone_readers)
        for a, b in zip(lines.words, zones.words, strict=True)
    )


def differences(out, expected) -> list[dict]:
    """The words whose readings are not those of the two cells put together (none when the output is the same)."""
    found = []
    for index, (word, want) in enumerate(zip(out.words, expected, strict=True)):
        got = tuple(word.readings or ())
        if got != want:
            found.append(
                {"word": index, "got": [(r.source, r.text) for r in got], "want": [(r.source, r.text) for r in want]}
            )
    return found


def entries_of(pages, block: dict, left_out: dict):
    """(id, first reading, crops, expected readings) of every page that has what the run needs; the others are counted in
    `left_out` (no pass 1, other units than the mode's, no L or Z cell)."""
    for _, path in pages:
        ocr = json.loads(path.read_text(encoding="utf-8")).get("ocr")
        if not ocr or "reading" not in ocr or "|u:" not in ocr.get("engine", "") or ocr["reading"]["status"] != "ocr":
            left_out["no_pass_1"] += 1
            continue
        config = study_of(ocr["engine"])
        if (config.study_units, config.study_zones) != (block["lines"], block["zones"]):
            left_out["other_units"] += 1
            continue
        crops = read_crops_file(path.with_suffix(".crops.npz"), config.key())
        want = expected_readings(path)
        if crops is None or want is None:
            left_out["no_cells"] += 1
            continue
        yield f"{path.parent.name}/{path.stem}", reading_from_dict(ocr["reading"]), crops, want


def measure(entries, process, *, clock=time.perf_counter, limit: int = 0, timeout: float = PAGE_TIMEOUT_S) -> dict:
    """Send the entries one by one through `process` and measure them: seconds a page (the first, that loads the models,
    apart), the memory of the process every 50 ms (its peak, and the peak of each page), and the parity of each answer
    with the expected readings. A page the process does not answer in `timeout` seconds is listed with its error and left
    out of the counts."""
    peaks: dict = {"mb": 0, "by_page": {}}
    current: dict = {"id": None}
    stop = threading.Event()

    def watch() -> None:
        while not stop.wait(0.05):
            mb = process._rss_mb()
            peaks["mb"] = max(peaks["mb"], mb)
            if current["id"] is not None:
                peaks["by_page"][current["id"]] = max(peaks["by_page"].get(current["id"], 0), mb)

    threading.Thread(target=watch, daemon=True).start()
    seconds: list[float] = []
    per_page: list[dict] = []
    compared = equal = 0
    different: list[dict] = []
    first_seconds = None
    for name, first, crops, want in entries:
        current["id"] = name
        began = clock()
        job, back = process.submit(first, crops)
        answered = [(job, back)] if back is not None else []
        while not answered and clock() - began < timeout:
            answered = [pair for pair in process.collect_pending(0.5) if pair[0] == job]
        took = clock() - began
        current["id"] = None
        if not answered:
            per_page.append({"id": name, "seconds": round(took, 3), "error": "no answer"})
            continue
        out = answered[0][1]
        found = differences(out, want) if not out.reread_report.get("reader_failed") else [{"reader_failed": True}]
        compared += 1
        equal += not found
        different += [{"page": name, **d} for d in found[:3]]
        if first_seconds is None:
            first_seconds = took
        else:
            seconds.append(took)
        per_page.append(
            {
                "id": name,
                "seconds": round(took, 3),
                "units_read": out.reread_report.get("units_read"),
                "equal": not found,
            }
        )
        if limit and compared >= limit:
            break
    process.drain()
    stop.set()
    ordered = sorted(seconds)
    return {
        "pages": compared,
        "first_page_seconds": None if first_seconds is None else round(first_seconds, 3),
        "seconds_per_page_mean": round(statistics.fmean(seconds), 3) if seconds else None,
        "seconds_per_page_p95": round(ordered[int(0.95 * (len(ordered) - 1))], 3) if seconds else None,
        "seconds_per_page_max": round(ordered[-1], 3) if seconds else None,
        "reader_rss_peak_mb": peaks["mb"],
        "heaviest_pages_mb": dict(sorted(peaks["by_page"].items(), key=lambda kv: -kv[1])[:5]),
        "parity": {
            "compared": compared,
            "equal": equal,
            "different_pages": compared - equal,
            "first_differences": different[:10],
        },
        "per_page": per_page,
    }


def run(args: argparse.Namespace) -> int:
    pages = pilot_pages(args.manifest, args.data) if args.pilot else pages_of(args.manifest, args.set, args.data)
    events: list[tuple[str, dict]] = []
    settings = settings_for("thorough", threads=args.threads, backend="openvino")
    if args.rss_cap:
        settings = dataclasses.replace(settings, rss_cap_mb=args.rss_cap)
    process = ReaderProcess(settings, log=lambda code, **params: events.append((code, params)))
    if not process.start():
        print(json.dumps({"error": "the reader process did not start: not enough free memory", "events": events}))
        return 1
    left_out = {"no_pass_1": 0, "other_units": 0, "no_cells": 0}
    measured = measure(entries_of(pages, second_reading.thorough(), left_out), process, limit=args.limit)
    process.stop()
    summary = {
        "mode": "thorough",
        "passes": [list(p[:1]) + [list(p[1])] for p in settings.passes],
        "left_out": left_out,
        "threads": args.threads,
        "rss_trigger_mb": settings.rss_trigger_mb,
        "rss_cap_used_mb": settings.rss_cap_mb or readerproc.RSS_CAP_MB,
        "replaced": process.replaced,
        "events": {code: sum(1 for c, _ in events if c == code) for code in sorted({c for c, _ in events})},
        **measured,
    }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_page"}), flush=True)
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--manifest", type=Path, default=Path("bench/corpus/split-v5.json"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--rss-cap", type=int, default=0, help="MB: replace the hard cap of the reader process (a measurement)"
    )
    parser.add_argument("--result", type=Path, default=RESULT)
    args = parser.parse_args()
    if not args.set and not args.pilot:
        parser.error("give --set or --pilot")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
