"""What the guard costs per page on real pages, with cold caches, in CPU time of this process.

Limits of the design: the whole guard (signals, scoring, reasons, suggestions) in at most 150 ms for a page of 600
words, scoring alone in at most 10 ms, loading in at most 3 s and adding at most 250 MB.

    python -m bench.guard.speed [--pages 20]
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np
import psutil

from bench.guard import dataset
from extract.engine.cache import reading_from_dict
from extract.guard import dictionary, signals
from extract.guard.checker import Guard

WORDS_PER_PAGE = 600
LIMITS = {"page_ms": 150.0, "scoring_ms": 10.0, "load_seconds": 3.0, "added_mb": 250.0}
MIN_WORDS = 300


def per_600(milliseconds: float, words: int) -> float:
    return milliseconds * WORDS_PER_PAGE / words


def measure_pages(guard: Guard, readings, languages=("es", "en"), locale="es") -> dict:
    """Total, signals-only and scoring-only CPU milliseconds per page, scaled to 600 words, with the caches cleared first."""
    totals, signal_times, scoring_times = [], [], []
    for reading in readings:
        for language in languages:
            dictionary.load(language).clear_caches()
        started = time.process_time()
        guard.score_page(reading, languages, locale, 0.05)
        totals.append(per_600(1000 * (time.process_time() - started), len(reading.words)))
        for language in languages:
            dictionary.load(language).clear_caches()
        started = time.process_time()
        matrix = signals.extract(reading.words, languages, locale)
        signal_times.append(per_600(1000 * (time.process_time() - started), len(reading.words)))
        started = time.process_time()
        guard.model.raw(matrix)
        scoring_times.append(per_600(1000 * (time.process_time() - started), len(reading.words)))
    return {
        "pages": len(totals),
        "page_ms": _summary(totals),
        "signals_ms": _summary(signal_times),
        "scoring_ms": _summary(scoring_times),
    }


def _summary(values: list[float]) -> dict:
    return {
        "median": statistics.median(values),
        "p95": float(np.percentile(values, 95)),
        "max": max(values),
    }


def measure_load(model_path: Path, languages=("es", "en")) -> dict:
    """Seconds to load the model and both dictionaries, and the resident memory they add."""
    process = psutil.Process()
    dictionary.load.cache_clear()
    before = process.memory_info().rss
    started = time.perf_counter()
    guard = Guard.load(model_path)
    for language in languages:
        dictionary.load(language)
    seconds = time.perf_counter() - started
    return {"load_seconds": seconds, "added_mb": (process.memory_info().rss - before) / 1e6, "guard": guard}


def verdict(pages: dict, load: dict) -> dict:
    return {
        "page_ms_ok": pages["page_ms"]["median"] <= LIMITS["page_ms"],
        "scoring_ms_ok": pages["scoring_ms"]["median"] <= LIMITS["scoring_ms"],
        "load_ok": load["load_seconds"] <= LIMITS["load_seconds"],
        "memory_ok": load["added_mb"] <= LIMITS["added_mb"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--model", type=Path, default=Path("models/guard.model"))
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("bench/results/guard-speed.json"))
    args = parser.parse_args()
    load = measure_load(args.model)
    readings = []
    for _, record in dataset.page_records(args.data_dir, "val"):
        reading = reading_from_dict(record["ocr"]["reading"])
        if reading.status == "ocr" and len(reading.words) >= MIN_WORDS:
            readings.append(reading)
    chosen = [readings[i] for i in np.random.default_rng(1).permutation(len(readings))[: args.pages]]
    pages = measure_pages(load["guard"], chosen)
    report = {
        "limits": LIMITS,
        "load_seconds": load["load_seconds"],
        "added_mb": load["added_mb"],
        **pages,
        **verdict(pages, load),
    }
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
