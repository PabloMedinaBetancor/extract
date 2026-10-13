"""What searching the dictionary three edits away would add to the pool, and what it costs.

The symmetric-delete index finds words within two edits. This replaces the first source by an exhaustive search
within three, ordered by weighted distance, with a larger cap, and measures the recall it reaches (as written) and
the seconds per site, on the wrong plain one-to-one words of the cached development pages.

    python3 -s -m bench.guard.pool_radius --sets train val --out bench/results/pool-radius.json
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from bench.guard import pool_recall
from extract.guard import dictionary, lexicon
from extract.guard.edits import weighted_distance
from extract.guard.forms import apply_case, split_word

RADIUS = 3
CAPS = (20, 24)


def within_radius(book: dictionary.Dictionary, core: str) -> list[tuple[str, float]]:
    """(word, log frequency) of every listed word within `RADIUS` edits of the letters, by exhaustive search."""
    lengths = book._lengths.astype(np.int16)
    near = np.nonzero(np.abs(lengths - len(core)) <= RADIUS)[0]
    distances = process.cdist(
        [core], [book.words[i] for i in near.tolist()], scorer=Levenshtein.distance, score_cutoff=RADIUS, dtype=np.uint8
    )[0]
    return [
        (book.words[i], float(book.log_freq[i])) for i, d in zip(near.tolist(), distances, strict=True) if d <= RADIUS
    ]


def radius_texts(books, token: str, cap: int) -> list[str]:
    prefix, core, suffix = split_word(token)
    found: dict[str, float] = {}
    for book in books:
        for word, frequency in within_radius(book, core.casefold()):
            found[word] = max(found.get(word, 0.0), frequency)
    ordered = sorted(found, key=lambda w: (weighted_distance(core, w), -found[w], w))[:cap]
    return [prefix + apply_case(word, core) + suffix for word in ordered]


def measure(data_dir: Path, set_name: str) -> dict:
    cases, _ = pool_recall.load_cases(data_dir, [set_name])
    sites = [s for s in pool_recall.wrong_sites(cases) if s.site]
    books = [dictionary.load(language) for language in lexicon.available(pool_recall.LANGUAGES)]
    others = ("fold", "punct", "page")
    out = {"sites": len(sites)}
    base = [pool_recall.reached(pool_recall._union(s, ("dictionary", *others)), s.truth)[0] for s in sites]
    out["two_edits_index"] = round(sum(base) / len(sites), 4)
    for cap in CAPS:
        started = time.process_time()
        texts = [radius_texts(books, s.token, cap) for s in sites]
        seconds = time.process_time() - started
        hits = [
            pool_recall.reached([*t, *pool_recall._union(s, others)], s.truth)[0]
            for s, t in zip(sites, texts, strict=True)
        ]
        out[f"three_edits_cap_{cap}"] = {
            "written": round(sum(hits) / len(sites), 4),
            "gain_points": round(100 * (sum(hits) - sum(base)) / len(sites), 2),
            "ms_per_site": round(1000 * seconds / len(sites), 2),
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sets", nargs="+", default=["train", "val"])
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--out", type=Path, default=Path("bench/results/pool-radius.json"))
    args = parser.parse_args(argv)
    result = {name: measure(args.data, name) for name in args.sets}
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
