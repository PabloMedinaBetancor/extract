"""What candidates and features cost per page on real pages, with cold caches, in CPU time of this process.

The detector marks the words that get candidates, at most `SHARE` of the page; the page is timed at that cap, so
the figure is the worst case the design allows. Times are scaled to a page of 600 words.

    python3 -s -m bench.guard.candidate_speed [--pages 20]
"""

import argparse
import json
import os
import statistics
import time
from pathlib import Path

import numpy as np

from bench.guard import dataset
from bench.guard.speed import MIN_WORDS, per_600
from extract.engine.cache import reading_from_dict
from extract.guard import candidates, dictionary, features, signals
from extract.guard.checker import Guard

LANGUAGES = ("es", "en")
SHARES = (0.05, 0.10, 0.20)


def time_page(guard: Guard, reading, share: float) -> dict:
    """Milliseconds, scaled to 600 words, for the context, the candidates and the features of one page."""
    for language in LANGUAGES:
        dictionary.load(language).clear_caches()
    words = reading.words
    matrix = signals.extract(words, LANGUAGES, "es")
    raw = guard.model.raw(matrix)
    chosen = np.argsort(-raw)[: int(share * len(words))]
    started = time.process_time()
    fragments = set(np.nonzero(matrix[:, signals.INDEX["hyphen_fragment"]] == 1)[0].tolist())
    ctx = candidates.PageContext(words, LANGUAGES, fragments)
    after_context = time.process_time()
    listed = {int(i): candidates.generate(ctx, int(i)) for i in chosen if ctx.is_site(int(i))}
    after_candidates = time.process_time()
    rows = sum(len(features.feature_rows(ctx, i, found, matrix[i])) for i, found in listed.items())
    done = time.process_time()
    scale = lambda seconds: per_600(1000 * seconds, len(words))  # noqa: E731
    return {
        "sites": len(listed),
        "rows": rows,
        "context_ms": scale(after_context - started),
        "candidates_ms": scale(after_candidates - after_context),
        "features_ms": scale(done - after_candidates),
        "total_ms": scale(done - started),
    }


def summary(values) -> dict:
    return {
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "p95": float(np.percentile(values, 95)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--model", type=Path, default=Path("models/guard.model"))
    parser.add_argument("--pages", type=int, default=20)
    parser.add_argument("--out", type=Path, default=Path("bench/results/candidate-speed.json"))
    args = parser.parse_args()
    guard = Guard.load(args.model)
    readings = []
    for _, record in dataset.page_records(args.data_dir, "val"):
        reading = reading_from_dict(record["ocr"]["reading"])
        if reading.status == "ocr" and len(reading.words) >= MIN_WORDS:
            readings.append(reading)
    chosen = [readings[i] for i in np.random.default_rng(1).permutation(len(readings))[: args.pages]]
    report = {"pages": len(chosen), "shares": {}}
    for share in SHARES:
        timed = [time_page(guard, reading, share) for reading in chosen]
        report["shares"][str(share)] = {
            name: summary([t[name] for t in timed])
            for name in ("total_ms", "context_ms", "candidates_ms", "features_ms")
        } | {"sites_per_page": statistics.fmean(t["sites"] for t in timed)}
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
