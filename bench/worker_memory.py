"""Memory of one OCR worker over many pages, with and without the second reading's pass 1 (R7, the 1.4 GB recycle limit).

    python -m bench.worker_memory --mode plain  --pilot --data <data> --rounds 2
    python -m bench.worker_memory --mode study  --pilot --data <data> --rounds 2
    python -m bench.worker_memory --mode study --trim ...    # malloc_trim(0) after every page

One process per run, so the peaks of two modes are never mixed. The worker's own steps are the ones measured: the engine
is loaded once, each page is read with `read_page_with_crops` and (in `study`) its crops are written like the worker
does. The result (`bench/results/worker-memory.json`, one record per mode) has the resident size after the engine
load, after each page, the peak, the growth in MB per page over the second half of the run and the page that held the
peak.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import resource
import statistics
import time
from pathlib import Path

import psutil

from bench.run_corpus import pages_of, pilot_pages, task_of
from extract.engine.cache import write_crops_file
from extract.engine.ocr import DEFAULT_TUNING, load_engine, read_page_with_crops
from extract.engine.reread import RereadConfig

RESULT = Path("bench/results/worker-memory.json")


def _relative(path: Path) -> str:
    """The page's path from its set folder on (`pages/<set>/<doc>/<n>.json`): no machine paths in the saved result."""
    parts = Path(path).parts
    return "/".join(parts[parts.index("pages") :]) if "pages" in parts else Path(path).name


def rss_mb() -> int:
    return psutil.Process().memory_info().rss // (1024 * 1024)


def trim() -> None:
    ctypes.CDLL("libc.so.6").malloc_trim(0)


def run(args: argparse.Namespace) -> int:
    pages = pilot_pages(args.manifest, args.data) if args.pilot else pages_of(args.manifest, args.set, args.data)
    study = RereadConfig() if args.mode == "study" else None
    engine = load_engine(args.threads, "openvino", DEFAULT_TUNING)
    base = rss_mb()
    series, peaks, slow = [], [], []
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    for _ in range(args.rounds):
        for _, path in pages:
            task = task_of(path, study)
            began = time.perf_counter()
            reading, crops = read_page_with_crops(task.image, engine, task.number, task.dpi, task.size_pt, study)
            if study is not None and reading.status == "ocr":
                write_crops_file(scratch / "crops.npz", task.key, crops)
            slow.append(time.perf_counter() - began)
            if args.trim:
                trim()
            series.append(rss_mb())
            peaks.append(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
            del reading, crops
    half = series[len(series) // 2 :]
    xs = list(range(len(half)))
    mx, my = statistics.fmean(xs), statistics.fmean(half)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, half, strict=True)) / max(1.0, sum((x - mx) ** 2 for x in xs))
    top = max(range(len(peaks)), key=lambda i: peaks[i] - (peaks[i - 1] if i else base))
    record = {
        "mode": args.mode + ("+trim" if args.trim else ""),
        "pages_read": len(series),
        "rss_after_engine_load_mb": base,
        "rss_after_each_page_mb": series,
        "peak_rss_mb": peaks[-1],
        "rss_final_mb": series[-1],
        "growth_mb_per_page_second_half": round(slope, 3),
        "page_that_raised_the_peak_most": {"index": top, "path": _relative(pages[top % len(pages)][1])},
        "seconds_per_page_mean": round(statistics.fmean(slow), 3),
        "threads": args.threads,
    }
    print(json.dumps({k: v for k, v in record.items() if k != "rss_after_each_page_mb"}), flush=True)
    results = json.loads(args.result.read_text()) if args.result.is_file() else {}
    results[record["mode"]] = record
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=("plain", "study"), required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--manifest", type=Path, default=Path("bench/corpus/split-v5.json"))
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--trim", action="store_true")
    parser.add_argument("--scratch", default="/tmp/ocrapp-worker-memory")
    parser.add_argument("--result", type=Path, default=RESULT)
    args = parser.parse_args()
    if not args.set and not args.pilot:
        parser.error("give --set or --pilot")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
