"""Read the corpus with the engine and store each reading in the page's own JSON, ready for the guard to learn from.

    python -m bench.run_corpus --set train --set val --set clean

Pages come from the manifest (`bench/corpus/split-v5.json`) and are read from `<data>/pages/<set>/<doc_id>/<n>.png`. The
`ocr` field of the page's JSON holds the serialised `PageReading`, the engine fingerprint, the backend and the time,
written under a temporary name and renamed. A page that already has a reading with the same fingerprint is skipped,
so a stopped run resumes where it was and a changed engine rereads everything.

The pages go through the product's own `OcrPool` (the one `BatchReader` uses), so the log of one run, `job.log` in
`<data>/runs/<stamp>/`, is the evidence for O9: `bench/endurance.py` reads it and the result goes to
`bench/results/endurance.json`. The backend is fixed to OpenVINO, so a run never mixes two backends.

Run it only inside one memory-capped scope, for example:

    systemd-run --user --scope --unit=run-corpus -p MemoryHigh=2500M -p MemoryMax=3G python -m bench.run_corpus ...

The sealed test T5 (`--set test5`: its catalogue pages, its invoices and its unseen-damage slice) is read through
`bench.corpus.sealed` with the purpose "ocr": this script never compares a reading with the truth and prints nothing
about quality. `--documents N` reads only the first N catalogue documents of T5's frozen list (the invoices and the
unseen-damage slice are always read). The v4 test (`test`, `unseen`) is development data and an ordinary set.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from bench import endurance
from extract.engine.cache import reading_to_dict
from extract.engine.pool import CHUNK_PAGES, RECYCLE_PAGES
from extract.engine.reread import RereadConfig
from extract.engine.worker import Chunk, PageTask

SETS = ("clean", "clean5", "train", "val", "test", "unseen", "test5", "dev5en", "test5en", "clean5en")
BACKEND = "openvino"
MAX_SCOPE_MB = 4096
MAX_UNALIGNED = 0.005


def scope_limit_mb() -> int | None:
    """The memory cap of the cgroup this process runs in, or None when there is none."""
    try:
        relative = next(line for line in Path("/proc/self/cgroup").read_text().splitlines() if line.startswith("0::"))[
            3:
        ]
        limit = (Path("/sys/fs/cgroup") / relative.lstrip("/") / "memory.max").read_text().strip()
    except (OSError, StopIteration):
        return None
    return None if limit == "max" else int(limit) // (1024 * 1024)


def pages_of(manifest: Path, sets: list[str], data: Path, documents: int | None = None) -> list[tuple[str, Path]]:
    """(set, json path) for every manifest page of the chosen sets; the sealed test goes through its only reader."""
    entries = json.loads(manifest.read_text(encoding="utf-8"))["pages"]
    pages = []
    for name in sets:
        if name == "test5":
            from bench.corpus import sealed

            late = {
                e["doc_id"] for e in entries if e["role"] == "test5" and e.get("position", 0) >= (documents or 10**9)
            }
            pages += [
                (record["set"], path)
                for path, record in sealed.read_test_pages(data, "ocr")
                if record["doc_id"] not in late
            ]
            continue
        if name == "test5en":
            from bench.corpus import sealed

            late = {
                e["doc_id"] for e in entries if e["role"] == "test5en" and e.get("position", 0) >= (documents or 10**9)
            }
            pages += [
                (record["set"], path)
                for path, record in sealed.read_test_pages(data, "ocr", sealed.EN_READS_PATH, sets=sealed.EN_TEST_SETS)
                if record["doc_id"] not in late
            ]
            continue
        wanted = sorted((e["doc_id"], e["page"]) for e in entries if e["set"] == name)
        pages += [(name, data / "pages" / name / doc / f"{n}.json") for doc, n in wanted]
    return pages


def pilot_pages(manifest: Path, data: Path) -> list[tuple[str, Path]]:
    """(set, json path) of the 60 pages of the pilot (gate P), in the manifest's order; their sets are train, val and
    test, so this is how a run reads them without reading those sets whole."""
    entries = json.loads(manifest.read_text(encoding="utf-8"))["pilot"]
    return [(e["set"], data / "pages" / e["set"] / e["doc_id"] / f"{e['page']}.json") for e in entries]


def pending(pages: list[tuple[str, Path]], fingerprint: str) -> list[tuple[str, Path]]:
    todo = []
    for name, path in pages:
        ocr = json.loads(path.read_text(encoding="utf-8")).get("ocr")
        if not ocr or ocr.get("engine") != fingerprint or ocr.get("backend") != BACKEND or "reading" not in ocr:
            todo.append((name, path))
    return todo


def write_reading(path: Path, ocr: dict) -> None:
    record = json.loads(path.read_text(encoding="utf-8"))
    record["ocr"] = ocr
    partial = path.with_name(f"{path.name}.{os.getpid()}.part")
    partial.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    os.replace(partial, path)


def task_of(path: Path, study=None) -> PageTask:
    """With `study` (a `RereadConfig`) the worker also cuts the crops of pass 1 and writes them beside the image as
    `<n>.crops.npz`, under the key `study.key()`."""
    record = json.loads(path.read_text(encoding="utf-8"))
    size = (record["truth"]["width_pt"], record["truth"]["height_pt"])
    key = study.key() if study is not None else ""
    return PageTask(int(record["page"]), round(record["image"]["dpi"]), key, str(path.with_suffix(".png")), size, study)


def chunks_of_document(tasks: list[PageTask], first_id: int = 0) -> list[Chunk]:
    """Up to four pages of one document per chunk. The image of each page travels in its task, so the chunk needs
    neither a PDF path nor a hash."""
    ordered = sorted(tasks, key=lambda t: t.number)
    return [
        Chunk(first_id + i, "", "", tuple(ordered[start : start + CHUNK_PAGES]))
        for i, start in enumerate(range(0, len(ordered), CHUNK_PAGES))
    ]


def by_document(todo: list[tuple[str, Path]]) -> list[list[Path]]:
    groups: dict[Path, list[Path]] = {}
    for _, path in todo:
        groups.setdefault(path.parent, []).append(path)
    return list(groups.values())


def backends_of(entries: list[dict]) -> dict[int, str]:
    return {e["pid"]: e["backend"] for e in entries if e["code"] == "worker.ready"}


def recycled_by(entries: list[dict]) -> dict[str, int]:
    """How many workers left because they had read their pages and how many because their memory passed the limit."""
    reasons: dict[str, int] = {}
    for entry in entries:
        if entry["code"] == "worker.recycled":
            reasons[entry.get("reason", "unknown")] = reasons.get(entry.get("reason", "unknown"), 0) + 1
    return reasons


def unaligned_share(readings: list) -> float:
    """The share of words the engine could not give their own character confidences (the criterion: at most 0.5 %)."""
    words = sum(len(r.words) for r in readings)
    return sum(r.unaligned_words for r in readings) / words if words else 0.0


def read_pages(
    todo: list[tuple[str, Path]], reader, fingerprint: str, on_done=None, write: bool = True, study=None
) -> tuple[list, list[str]]:
    """Read the pages with the product's pool, one document at a time, and write each reading as it arrives.
    Returns the readings and the pages that came out unreadable or could not be read."""
    readings, failed = [], []
    for paths in by_document(todo):
        paths_by_number = {int(json.loads(p.read_text(encoding="utf-8"))["page"]): p for p in paths}
        tasks = [task_of(p, study) for p in paths]

        def on_page(reading, pid, seconds, paths_by_number=paths_by_number):
            path = paths_by_number[reading.number]
            if reading.error:
                failed.append(f"{path}: {reading.error}")
                return
            backend = backends_of(reader.log.entries).get(pid)
            ocr = {
                "engine": fingerprint,
                "backend": backend,
                "seconds": round(seconds, 3),
                "worker_pid": pid,
                "reading": reading_to_dict(reading),
            }
            if write:
                write_reading(path, ocr)
            readings.append(reading)
            if on_done:
                on_done(len(readings))

        if not reader.pool.run(chunks_of_document(tasks), on_page):
            raise RuntimeError("the pool stopped before the run ended")
    return readings, failed


def run(args: argparse.Namespace) -> int:
    limit = scope_limit_mb()
    if not args.unscoped and (limit is None or limit > MAX_SCOPE_MB):
        print(f"refusing to run: no memory cap of at most {MAX_SCOPE_MB} MB on this process ({limit}); use systemd-run")
        return 2
    from extract.engine.batch import BatchReader
    from extract.engine.ocr import fingerprint
    from extract.options import Options

    study = RereadConfig() if args.second_reading else None
    engine_key = fingerprint() + (f"|{study.key()}" if study else "")
    pages = (
        pilot_pages(args.manifest, args.data)
        if args.pilot
        else pages_of(args.manifest, args.set, args.data, args.documents)
    )
    todo = pages if args.no_write else pending(pages, engine_key)
    if args.limit:
        todo = todo[: args.limit]
    print(
        f"{len(pages)} pages in {'the pilot' if args.pilot else args.set}, {len(todo)} to read, scope cap {limit} MB",
        flush=True,
    )
    if not todo:
        return 0
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    work_dir = args.data / "runs" / stamp
    options = Options(
        work_dir=work_dir, backend=BACKEND, recycle_pages=args.recycle, processes=args.processes, threads=args.threads
    )
    began = time.monotonic()
    reader = BatchReader(options, factory=args.factory)
    try:
        readings, failed = read_pages(
            todo,
            reader,
            engine_key,
            lambda n: print(n, flush=True) if n % 20 == 0 else 0,
            write=not args.no_write,
            study=study,
        )
    finally:
        reader.close()
    share = unaligned_share(readings)
    result = endurance.evaluate(reader.log.entries)
    result.update(
        sets=args.set,
        run=stamp,
        wall_seconds=round(time.monotonic() - began),
        recycle_pages=args.recycle,
        recycled_by=recycled_by(reader.log.entries),
        failed=failed,
        unaligned_share=round(share, 5),
        unaligned_within_limit=share <= MAX_UNALIGNED,
        backends=sorted(set(backends_of(reader.log.entries).values())),
    )
    append_run(args.endurance, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("peak_rss_mb_by_worker", "failed")}), flush=True)
    return 1 if failed else 0


def append_run(path: Path, result: dict) -> None:
    """endurance.json keeps every run with its recycle setting and says whether some run met O9 (400 pages in a row,
    three recycles, resident memory and speed within the limits): a run with a lower recycle after one that failed
    must not hide the failure."""
    data = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {"runs": []}
    if "runs" not in data:
        data = {"runs": [data]}
    data["runs"].append(result)
    data["o9_met"] = any(run["passed"] for run in data["runs"])
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    os.replace(partial, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", action="append", choices=SETS, default=[])
    parser.add_argument("--pilot", action="store_true", help="the 60 pages of the pilot (gate P) instead of --set")
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--manifest", type=Path, default=Path("bench/corpus/split-v5.json"))
    parser.add_argument("--documents", type=int, help="with --set test5: the first N catalogue documents of T5's list")
    parser.add_argument("--endurance", type=Path, default=Path("bench/results/endurance.json"))
    parser.add_argument(
        "--recycle", type=int, default=RECYCLE_PAGES, help="pages per worker; memory can recycle sooner"
    )
    parser.add_argument("--processes", type=int, default=1, help="workers; 1 until the pool reads its own cgroup")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--factory", default="extract.engine.ocr:load_engine")
    parser.add_argument(
        "--no-write", action="store_true", help="read every page again and keep nothing: the endurance run for O9"
    )
    parser.add_argument(
        "--second-reading",
        action="store_true",
        help="pass 1 of the second reading: also cut and keep the crops of the units (`<n>.crops.npz`); pass 2 is "
        "`bench/run_pass2.py`",
    )
    parser.add_argument("--unscoped", action="store_true", help="allow running without a memory cap (tests only)")
    args = parser.parse_args()
    if not args.set and not args.pilot:
        parser.error("give --set or --pilot")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
