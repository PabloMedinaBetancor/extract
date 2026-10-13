"""Downloads the public documents of the benchmark corpus listed in bench/SOURCES.csv and checks each one's sha256.

    python -m bench.fetch_corpus --dry-run             # what would be fetched and how long it would take
    python -m bench.fetch_corpus --limit 3             # three documents, to try it
    python -m bench.fetch_corpus                       # all of them, about 50 minutes

The files go to `<out>/catalog/originals/<doc_id>.pdf`, with a `catalog.db` beside them that `bench.corpus.catalog`
reads: point EXTRACT_CATALOG at that `originals` folder. The catalogue is one SQLite table, `sources(id, url,
original_path, original_sha256, state, kind)`; only the rows with `state = 'current'` are used.

The documents come from public administrations and are fetched slowly, so as not to load a site: 8 to 12 seconds between
two requests to the same site, at most 150 requests an hour and 800 a day to each site. The program says what it is in
its `User-Agent` (`extract-bench-fetch`). The record of requests is kept in `<out>/.fetch-state.json`, so a stopped run
resumes where it was and keeps the limits across runs. A site that answers 403 or 429 is left alone for the rest of the
run.

A document whose hash differs from the one in SOURCES.csv has changed since the corpus was made: it is counted,
reported, and not kept, and the exit status is 1.
"""

import argparse
import csv
import hashlib
import json
import os
import random
import sqlite3
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

GAP_SECONDS = (8.0, 12.0)
PER_HOUR = 150
PER_DAY = 800
STOP_STATUSES = (403, 429)
STATE_NAME = ".fetch-state.json"
ORIGINALS = Path("catalog") / "originals"
CATALOGUE_TYPE = "regulation"


class Clock:
    """The real clock; the tests give another one with the same two methods."""

    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


def site_of(url: str) -> str:
    return urlparse(url).netloc.lower()


def read_sources(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Fetcher:
    def __init__(self, out: Path, clock: Clock | None = None, rng: random.Random | None = None):
        self.out = out
        self.clock = clock or Clock()
        self.rng = rng or random.Random()
        self.state_path = out / STATE_NAME
        self.requests: dict[str, list[float]] = defaultdict(list)
        if self.state_path.is_file():
            self.requests.update(json.loads(self.state_path.read_text(encoding="utf-8")))
        self.next_gap: dict[str, float] = {}

    def save_state(self) -> None:
        self.out.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.requests), encoding="utf-8")

    def ready_at(self, site: str) -> float:
        """The earliest moment the next request to `site` respects the gap, the hourly cap and the daily cap."""
        times = sorted(self.requests[site])
        ready = 0.0
        if times:
            ready = times[-1] + self.next_gap.setdefault(site, self.rng.uniform(*GAP_SECONDS))
        if len(times) >= PER_HOUR:
            ready = max(ready, times[-PER_HOUR] + 3600)
        if len(times) >= PER_DAY:
            ready = max(ready, times[-PER_DAY] + 86400)
        return ready

    def record(self, site: str) -> None:
        self.requests[site].append(self.clock.now())
        self.requests[site] = [t for t in self.requests[site] if t > self.clock.now() - 86400]
        self.next_gap.pop(site, None)
        self.save_state()

    def pending(self, rows: list[dict]) -> list[dict]:
        folder = self.out / ORIGINALS
        return [
            r
            for r in rows
            if not (folder / f"{r['doc_id']}.pdf").is_file() or sha256_of(folder / f"{r['doc_id']}.pdf") != r["sha256"]
        ]

    def run(self, rows: list[dict], limit: int | None = None) -> dict:
        todo = self.pending(rows)[:limit]
        queues: dict[str, list[dict]] = defaultdict(list)
        for row in todo:
            queues[site_of(row["source_url"])].append(row)
        report = {
            "fetched": 0,
            "changed": [],
            "failed": [],
            "skipped_sites": [],
            "already": len(rows) - len(self.pending(rows)),
        }
        while queues:
            site = min(queues, key=self.ready_at)
            wait = self.ready_at(site) - self.clock.now()
            if wait > 0:
                self.clock.sleep(wait)
            row = queues[site].pop(0)
            if not queues[site]:
                del queues[site]
            outcome = self.fetch_one(row)
            if outcome == "ok":
                report["fetched"] += 1
            elif outcome == "changed":
                report["changed"].append(row["doc_id"])
            elif outcome == "blocked":
                report["skipped_sites"].append(site)
                report["failed"] += [row["doc_id"]] + [r["doc_id"] for r in queues.pop(site, [])]
            else:
                report["failed"].append(row["doc_id"])
        return report

    def fetch_one(self, row: dict) -> str:
        site = site_of(row["source_url"])
        target = self.out / ORIGINALS / f"{row['doc_id']}.pdf"
        request = urllib.request.Request(row["source_url"], headers={"User-Agent": "extract-bench-fetch"})
        self.record(site)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            print(f"{row['doc_id']}: HTTP {error.code} from {site}", file=sys.stderr)
            return "blocked" if error.code in STOP_STATUSES else "failed"
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"{row['doc_id']}: {error}", file=sys.stderr)
            return "failed"
        if hashlib.sha256(body).hexdigest() != row["sha256"]:
            print(
                f"{row['doc_id']}: changed since the corpus was made ({row['source_url']}); not kept", file=sys.stderr
            )
            return "changed"
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(body)
        partial.replace(target)
        return "ok"


def write_catalogue(out: Path, rows: list[dict]) -> int:
    """Writes `<out>/catalog/catalog.db` in the shape `bench.corpus.catalog` reads, for the documents that are here, so
    that `EXTRACT_CATALOG=<out>/catalog/originals python -m bench.corpus.catalog render` runs unchanged."""
    here = [r for r in rows if (out / ORIGINALS / f"{r['doc_id']}.pdf").is_file()]
    database = out / "catalog" / "catalog.db"
    database.unlink(missing_ok=True)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "CREATE TABLE sources (id INTEGER PRIMARY KEY, url TEXT, original_path TEXT, original_sha256 TEXT, state TEXT, kind TEXT)"
        )
        connection.executemany(
            "INSERT INTO sources VALUES (?, ?, ?, ?, 'current', ?)",
            [
                (int(r["catalog_id"]), r["source_url"], f"originals/{r['doc_id']}.pdf", r["sha256"], CATALOGUE_TYPE)
                for r in here
            ],
        )
    connection.close()
    return len(here)


def plan(rows: list[dict], fetcher: Fetcher) -> str:
    todo = fetcher.pending(rows)
    sites = defaultdict(int)
    for row in todo:
        sites[site_of(row["source_url"])] += 1
    lines = [f"{len(todo)} of {len(rows)} documents to fetch from {len(sites)} sites"]
    lines += [f"  {count:4d}  {site}" for site, count in sorted(sites.items(), key=lambda item: -item[1])]
    busiest = max(sites.values(), default=0)
    mean_gap = sum(GAP_SECONDS) / 2
    lines.append(
        f"at about {mean_gap:.0f} s between requests to a site, the busiest site alone needs {busiest * mean_gap / 60:.0f} minutes"
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", type=Path, default=Path("bench/SOURCES.csv"))
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--limit", type=int, help="fetch at most this many documents")
    parser.add_argument("--dry-run", action="store_true", help="print the plan and fetch nothing")
    args = parser.parse_args()
    rows = read_sources(args.sources)
    fetcher = Fetcher(args.out)
    if args.dry_run:
        print(plan(rows, fetcher))
        return
    report = fetcher.run(rows, args.limit)
    print(f"{write_catalogue(args.out, rows)} documents in {args.out / 'catalog'}")
    print(
        f"{report['fetched']} fetched, {report['already']} already there, {len(report['changed'])} changed upstream, {len(report['failed'])} failed"
    )
    if report["changed"] or report["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
