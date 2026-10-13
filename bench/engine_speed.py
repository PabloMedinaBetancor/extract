"""Speed of the OCR engine settings on the same 20 pages, before and after each one (criterion O13).

    python3 bench/engine_speed.py prepare          # pick 20 pages from the catalog and build the page PDF
    python3 bench/engine_speed.py run VARIANT...   # baseline procs_2x2 det_buckets rec_buckets
    python3 bench/engine_speed.py reopen           # time of opening the PDF once per page against once per document
    python3 bench/engine_speed.py report           # write bench/results/engine-speed.json

The second reading (v5, criterion R6) on the same pages, per cell of the pilot (L, Z, LD, ZC):

    python3 bench/engine_speed.py alts             # the cost of the character alternatives and the masked reading
    python3 bench/engine_speed.py cell L Z         # end to end with the second reading: pass 1, the reader process
    python3 bench/engine_speed.py pass2 L Z        # seconds a page of pass 2 alone, by reader and by cell, and its memory
    python3 bench/engine_speed.py v5report         # write bench/results/reader-speed.json

The pages are cut from catalog documents that are not part of the repository; only their ids and SHA-256 are kept.
Everything is written under `$EXTRACT_BENCH_DATA/engine-speed/`, except the report.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sqlite3
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path

import psutil
import pypdfium2 as pdfium
from PIL import Image
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from extract.engine.batch import BatchReader
from extract.engine.ocr import DEFAULT_TUNING, Tuning
from extract.engine.render import page_dpi, render_page, scan_pixels
from extract.options import Options

SEED = 11
SCANNED_PAGES = 12
DIGITAL_PAGES = 8
DATA = Path(os.environ["EXTRACT_BENCH_DATA"]) / "engine-speed"
RESULTS = Path(__file__).resolve().parent / "results" / "engine-speed.json"
VARIANTS = {
    "baseline": dict(processes=1, threads=4, tuning=Tuning()),
    "procs_2x2": dict(processes=2, threads=2, tuning=Tuning()),
    "det_buckets": dict(processes=1, threads=4, tuning=Tuning(det_buckets=True)),
    "rec_buckets": dict(processes=1, threads=4, tuning=Tuning(rec_buckets=True)),
}
MIN_SAVING = 0.05


def catalog_pdfs() -> list[Path]:
    catalog_dir = Path(os.environ["EXTRACT_CATALOG"]).parent
    con = sqlite3.connect(f"file:{catalog_dir / 'catalog.db'}?mode=ro", uri=True)
    rows = con.execute(
        "select original_path from sources where lower(original_path) like '%.pdf' and state = 'current' order by id"
    ).fetchall()
    con.close()
    return [catalog_dir / row[0] for row in rows]


def prepare() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    chosen = random.Random(SEED).sample(catalog_pdfs(), 120)
    scanned, digital, sources = [], [], []
    for path in chosen:
        if len(scanned) >= SCANNED_PAGES and len(digital) >= DIGITAL_PAGES:
            break
        try:
            document = pdfium.PdfDocument(str(path))
        except pdfium.PdfiumError:
            continue
        number = random.Random(f"{SEED}{path.name}").randrange(len(document))
        page = document[number]
        try:
            is_scan = scan_pixels(page) > 0
            textpage = page.get_textpage()
            text = textpage.get_text_range()
            textpage.close()
            wanted = scanned if is_scan else digital
            limit = SCANNED_PAGES if is_scan else DIGITAL_PAGES
            if len(wanted) >= limit or (not is_scan and len(text) < 1500):
                continue
            dpi = page_dpi(page)
            image = render_page(page, dpi)
        finally:
            page.close()
            document.close()
        wanted.append((image, dpi, None if is_scan else text.split()))
        sources.append({"catalog_file": path.name, "page": number + 1, "kind": "scanned" if is_scan else "digital"})
    pages = scanned + digital
    pdf = canvas.Canvas(str(DATA / "pages.pdf"))
    for image, dpi, _ in pages:
        height, width = image.shape
        pdf.setPageSize((width / dpi * 72, height / dpi * 72))
        pdf.drawImage(ImageReader(Image.fromarray(image)), 0, 0, width / dpi * 72, height / dpi * 72)
        pdf.showPage()
    pdf.save()
    (DATA / "truth.json").write_text(json.dumps([words for _, _, words in pages]))
    (DATA / "sources.json").write_text(json.dumps(sources, indent=1))
    print(f"{len(scanned)} scanned and {len(digital)} digital pages")


class PeakMemory(threading.Thread):
    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.peak_mb = 0
        self.stop = threading.Event()

    def run(self) -> None:
        me = psutil.Process()
        while not self.stop.wait(0.5):
            sizes = []
            for child in me.children(recursive=True):
                try:
                    sizes.append(child.memory_info().rss // (1024 * 1024))
                except psutil.NoSuchProcess:
                    continue
            self.peak_mb = max(self.peak_mb, max(sizes, default=0))


def run_variant(name: str) -> None:
    setup = VARIANTS[name.split(":")[0]]
    work = Path(tempfile.mkdtemp(prefix="engine-speed-"))
    options = Options(work_dir=work, processes=setup["processes"], threads=setup["threads"], backend="openvino")
    reader = BatchReader(options, tuning=setup["tuning"])
    ready, original_log = [], reader.pool.log

    def log(code, **params):
        if code == "worker.ready":
            ready.append((time.perf_counter(), params["load_seconds"]))
        original_log(code, **params)

    reader.pool.log = log
    memory = PeakMemory()
    memory.start()
    try:
        document = reader.read(DATA / "pages.pdf")
        finished = time.perf_counter()
    finally:
        memory.stop.set()
        reader.close()
    started = max(t for t, _ in ready[: setup["processes"]])
    pages = len(document.pages)
    record = {
        "variant": name.split(":")[0],
        **{k: v for k, v in setup.items() if k != "tuning"},
        "tuning": setup["tuning"].__dict__,
        "pages": pages,
        "seconds_per_page": round((finished - started) / pages, 3),
        "load_seconds": round(max(s for _, s in ready), 1),
        "peak_worker_rss_mb": memory.peak_mb,
        "backends": sorted({e["backend"] for e in reader.log.entries if e["code"] == "worker.ready"}),
        "unreadable": list(document.unreadable_pages),
        "words": [[w.text for w in page.words] for page in document.pages],
    }
    (DATA / "runs").mkdir(exist_ok=True)
    (DATA / "runs" / f"{name.replace(chr(58), chr(95))}.json").write_text(json.dumps(record))
    print(name, record["seconds_per_page"], "s/page, load", record["load_seconds"], "s, peak", memory.peak_mb, "MB")


def reopen() -> None:
    """Render 20 consecutive pages of the longest source document with the PDF opened for every page and once."""
    sources = json.loads((DATA / "sources.json").read_text())
    names = {path.name: path for path in catalog_pdfs()}
    biggest = max(
        (names[s["catalog_file"]] for s in sources if s["catalog_file"] in names), key=lambda p: p.stat().st_size
    )
    document = pdfium.PdfDocument(str(biggest))
    count = min(20, len(document))
    document.close()

    def render_pages(reopen_each: bool) -> float:
        began = time.perf_counter()
        held = None if reopen_each else pdfium.PdfDocument(str(biggest))
        for number in range(count):
            document = pdfium.PdfDocument(str(biggest)) if reopen_each else held
            page = document[number]
            render_page(page, 150)
            page.close()
            if reopen_each:
                document.close()
        if held is not None:
            held.close()
        return time.perf_counter() - began

    render_pages(False)
    reopened, once = render_pages(True), render_pages(False)
    record = {
        "pages": count,
        "megabytes": round(biggest.stat().st_size / 1e6),
        "reopen_each_page": reopened,
        "open_once": once,
    }
    (DATA / "reopen.json").write_text(json.dumps(record))
    print(
        "reopen each page",
        round(reopened / count, 3),
        "s; open once",
        round(once / count, 3),
        "s;",
        record["megabytes"],
        "MB",
    )


def word_accuracy(read: list[str], truth: list[str]) -> float:
    matched = sum((Counter(read) & Counter(truth)).values())
    return matched / len(truth)


def agreement(words: list[list[str]], reference: list[list[str]]) -> float:
    matched = total = 0
    for a, b in zip(words, reference, strict=True):
        matched += sum((Counter(a) & Counter(b)).values())
        total += max(len(a), len(b))
    return matched / total


def report() -> None:
    truth = json.loads((DATA / "truth.json").read_text())
    repeats: dict[str, list[dict]] = {}
    for path in sorted((DATA / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        repeats.setdefault(run["variant"], []).append(run)
    runs = {
        name: {
            **group[0],
            "seconds_per_page": round(sum(r["seconds_per_page"] for r in group) / len(group), 3),
            "all_seconds_per_page": [r["seconds_per_page"] for r in group],
            "peak_worker_rss_mb": max(r["peak_worker_rss_mb"] for r in group),
        }
        for name, group in repeats.items()
    }
    base = runs["baseline"]

    def accuracy(run):
        pairs = [(w, t) for w, t in zip(run["words"], truth, strict=True) if t]
        return round(sum(word_accuracy(w, t) * len(t) for w, t in pairs) / sum(len(t) for _, t in pairs), 4)

    entries = []
    for name, label in (
        ("procs_2x2", "two processes of two threads against one of four"),
        ("det_buckets", "detection input sides in steps of 128 px"),
        ("rec_buckets", "recognition widths in steps of 160 px"),
    ):
        if name not in runs:
            continue
        after = runs[name]
        saving = 1 - after["seconds_per_page"] / base["seconds_per_page"]
        entry = {
            "name": name,
            "what": label,
            "seconds_per_page_before": base["seconds_per_page"],
            "seconds_per_page_after": after["seconds_per_page"],
            "runs_before": base["all_seconds_per_page"],
            "runs_after": after["all_seconds_per_page"],
            "saving": round(saving, 3),
            "digital_word_accuracy_before": accuracy(base),
            "digital_word_accuracy_after": accuracy(after),
            "agreement_with_baseline": round(agreement(after["words"], base["words"]), 4),
            "peak_worker_rss_mb_before": base["peak_worker_rss_mb"],
            "peak_worker_rss_mb_after": after["peak_worker_rss_mb"],
        }
        entry["on_by_default"] = (
            saving >= MIN_SAVING and entry["digital_word_accuracy_after"] >= entry["digital_word_accuracy_before"]
        )
        entries.append(entry)
    reopen_path = DATA / "reopen.json"
    if reopen_path.exists():
        re = json.loads(reopen_path.read_text())
        before, after = re["reopen_each_page"] / re["pages"], re["open_once"] / re["pages"]
        entries.append(
            {
                "name": "pdf_open_once",
                "what": "a PDF opened once per document in a worker instead of once per page",
                "seconds_per_page_before": round(before, 3),
                "seconds_per_page_after": round(after, 3),
                "saving": round(1 - after / before, 3),
                "note": "rendering only, 20 consecutive pages of the largest source document; a fraction of a percent "
                "of the seconds a page, kept because it is the simpler design and costs nothing",
            }
        )
    page = base["seconds_per_page"]
    entries.append(
        {
            "name": "engine_loaded_once",
            "what": "the engine loaded once per process and kept until the recycle instead of once per page",
            "seconds_per_page_before": round(page + base["load_seconds"], 3),
            "seconds_per_page_after": page,
            "saving": round(base["load_seconds"] / (page + base["load_seconds"]), 3),
            "note": "derived from the measured load time, not a separate run",
        }
    )
    summary = {
        "pages": base["pages"],
        "backend": base["backends"],
        "sha256_of_pages_pdf": hashlib.sha256((DATA / "pages.pdf").read_bytes()).hexdigest(),
        "baseline": {
            k: base[k] for k in ("processes", "threads", "seconds_per_page", "load_seconds", "peak_worker_rss_mb")
        },
        "default_tuning": DEFAULT_TUNING.__dict__,
        "optimizations": entries,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(summary, indent=1) + "\n")
    for entry in entries:
        print(entry["name"], entry["saving"], entry.get("on_by_default"))


SPEED_RESULT = Path(__file__).resolve().parent / "results" / "reader-speed.json"
R6_MEAN_S, R6_P95_S, R6_END_TO_END, R6_ALTS_MS = 1.6, 3.5, 1.4, 60


def _merge(section: str, key: str, value: dict) -> None:
    DATA.mkdir(exist_ok=True)
    path = DATA / "v5-speed.json"
    current = json.loads(path.read_text()) if path.exists() else {}
    current.setdefault(section, {})[key] = value
    path.write_text(json.dumps(current, indent=1))


def alts_cost() -> None:
    """R6: the alternatives and `masked` at most 60 ms a page. The same rendered pages are read with them on and off,
    alternating, and the difference of the medians is the cost."""
    import statistics

    from extract.engine import ocr as ocr_module
    from extract.engine.ocr import load_engine, read_page

    document = pdfium.PdfDocument(str(DATA / "pages.pdf"))
    images = []
    for index in range(len(document)):
        page = document[index]
        images.append((render_page(page, page_dpi(page)), page.get_size()))
        page.close()
    engine = load_engine(threads=4, backend="openvino")
    read_page(images[0][0], engine)
    times = {True: [], False: []}
    for _ in range(3):
        for on in (False, True):
            ocr_module._alternatives_on = on
            for image, size in images:
                began = time.perf_counter()
                read_page(image, engine, 1, 200, size)
                times[on].append(time.perf_counter() - began)
    ocr_module._alternatives_on = True
    median = {on: statistics.median(t) for on, t in times.items()}
    mean = {on: statistics.fmean(t) for on, t in times.items()}
    record = {
        "pages": len(images),
        "median_seconds_off": round(median[False], 3),
        "median_seconds_on": round(median[True], 3),
        "mean_seconds_off": round(mean[False], 3),
        "mean_seconds_on": round(mean[True], 3),
        "cost_ms_by_mean": round(1000 * (mean[True] - mean[False]), 1),
        "limit_ms": R6_ALTS_MS,
        "load_average": os.getloadavg()[0],
    }
    _merge("alts", "cost", record)
    print(record)


def cell_e2e(cell: str) -> None:
    from bench.run_pass2 import CELLS
    from extract.engine.readerproc import ReaderSettings

    unit, names = CELLS[cell]
    work = DATA / f"v5-work-{cell}"
    shutil.rmtree(work, ignore_errors=True)
    options = Options(work_dir=work, processes=1, threads=2, backend="openvino", second_reading=True)
    reader = BatchReader(options, reader_settings=ReaderSettings(unit=unit, readers=names, threads=2))
    ready = []
    original = reader.pool.log

    def log(code, **params):
        if code == "worker.ready":
            ready.append(time.perf_counter())
        original(code, **params)

    reader.pool.log = log
    try:
        began = time.perf_counter()
        document = reader.read(DATA / "pages.pdf")
        finished = time.perf_counter()
    finally:
        reader.close()
    pages = len(document.pages)
    start = max(ready) if ready else began
    record = {
        "cell": cell,
        "pages": pages,
        "seconds_per_page": round((finished - start) / pages, 3),
        "reader_events": [e["code"] for e in reader.log.entries if e["code"].startswith("reader.")],
        "pages_with_readings": sum(any(w.readings for w in p.words) for p in document.pages),
        "load_average": os.getloadavg()[0],
    }
    _merge("e2e", cell, record)
    print(record)


BUDGETS = {"line": (4, 8, 16), "zone": (5, 10, 20, 40)}


def pass2_times(cell: str) -> None:
    """Seconds a page of pass 2 alone for each reader of the cell and for the cell, from the crops that `cell_e2e` kept
    in the page cache, and the memory of this one process (the reader process needs about that)."""
    import resource
    import statistics

    from bench.run_pass2 import CELLS
    from extract.engine.cache import PageCache, file_sha256, page_key
    from extract.engine.crops import decode
    from extract.engine.readers import ReaderSet
    from extract.engine.reread import RereadConfig
    from extract.engine.second import second_pass

    unit, names = CELLS[cell]
    # the crossed cells read the crops pass 1 kept for their unit: line crops sit with L, zone crops with Z
    config, cache = RereadConfig(), PageCache(DATA / ("v5-work-L" if unit == "line" else "v5-work-Z"))
    sha = file_sha256(DATA / "pages.pdf")
    document = pdfium.PdfDocument(str(DATA / "pages.pdf"))
    pages = []
    for number in range(1, len(document) + 1):
        page = document[number - 1]
        dpi = page_dpi(page)
        page.close()
        key = page_key(sha, number, dpi, config.key())
        reading, crops = cache.get(sha, number, key), cache.get_crops(sha, number, key)
        if reading is not None and crops and reading.status == "ocr":
            pages.append((reading, crops))
    readers = ReaderSet(names, threads=2, backend="openvino")
    for reading, crops in pages[:2]:  # warm up: loading and compiling are not the cost of a page
        second_pass(reading, crops, readers, unit, config)
    by_reader, whole = {}, []
    for name in names:
        seconds = []
        for reading, crops in pages:
            began = time.perf_counter()
            second_pass(reading, crops, readers, unit, config, names=(name,))
            seconds.append(time.perf_counter() - began)
        by_reader[name] = seconds
    for reading, crops in pages:
        began = time.perf_counter()
        second_pass(reading, crops, readers, unit, config)
        whole.append(time.perf_counter() - began)

    def stats(seconds):
        ordered = sorted(seconds)
        return {"mean": round(statistics.fmean(seconds), 3), "p95": round(ordered[int(0.95 * (len(ordered) - 1))], 3)}

    budgets = {}
    for budget in BUDGETS[unit]:  # the page cost of a budget: the units of rank < budget only (the crops are all there)
        smaller = (
            RereadConfig(config.tau_char, config.tau_word, study_units=budget, study_zones=config.study_zones)
            if unit == "line"
            else RereadConfig(config.tau_char, config.tau_word, study_units=config.study_units, study_zones=budget)
        )
        seconds = []
        for reading, crops in pages:
            began = time.perf_counter()
            second_pass(reading, crops, readers, unit, smaller)
            seconds.append(time.perf_counter() - began)
        budgets[str(budget)] = {
            **stats(seconds),
            "units_per_page_mean": round(statistics.fmean(len(smaller.units(r.words, unit)) for r, _ in pages), 1),
        }

    units = [len(config.units(r.words, unit)) for r, _ in pages]
    crop_pixels = [
        decode(c[f"{unit}{k}"]).size
        for r, c in pages
        for k in range(len(config.units(r.words, unit)))
        if f"{unit}{k}" in c
    ]
    record = {
        "cell": cell,
        "unit": unit,
        "readers": list(names),
        "pages": len(pages),
        "cell_seconds": stats(whole),
        "budgets": budgets,
        "reader_seconds": {name: stats(seconds) for name, seconds in by_reader.items()},
        "units_per_page_mean": round(statistics.fmean(units), 1) if units else None,
        "crop_pixels_mean": round(statistics.fmean(crop_pixels)) if crop_pixels else None,
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
        "load_average": os.getloadavg()[0],
    }
    _merge("pass2", cell, record)
    print(record)


def v5report() -> None:
    data = json.loads((DATA / "v5-speed.json").read_text())
    base = json.loads(RESULTS.read_text())["baseline"]["seconds_per_page"]
    cells = {}
    for cell in sorted(set(data.get("e2e", {})) | set(data.get("pass2", {}))):
        e2e = data.get("e2e", {}).get(cell) or data.get("e2e", {}).get(cell[0])  # LD and ZC read L's and Z's crops
        pass2 = data.get("pass2", {}).get(cell, {})
        mean = pass2.get("cell_seconds", {}).get("mean")
        p95 = pass2.get("cell_seconds", {}).get("p95")
        ratio = e2e["seconds_per_page"] / base if e2e else None
        cells[cell] = {
            "pass2_seconds_mean": mean,
            "pass2_seconds_p95": p95,
            "pass2_within_cap": mean is not None and mean <= R6_MEAN_S and p95 <= R6_P95_S,
            "end_to_end_seconds_per_page": e2e["seconds_per_page"] if e2e else None,
            "end_to_end_ratio_to_v4": round(ratio, 2) if ratio is not None else None,
            "end_to_end_within_cap": ratio <= R6_END_TO_END if ratio is not None else None,
            "peak_reader_rss_mb": pass2.get("peak_rss_mb"),
            "readers": pass2.get("reader_seconds"),
            "units_per_page_mean": pass2.get("units_per_page_mean"),
            "budgets": pass2.get("budgets"),
        }
    summary = {
        "pages": 20,
        "v4_baseline_seconds_per_page_this_session": base,
        "caps": {"pass2_mean_s": R6_MEAN_S, "pass2_p95_s": R6_P95_S, "end_to_end_ratio": R6_END_TO_END},
        "alternatives": data.get("alts", {}).get("cost"),
        "cells": cells,
        "note": "figures taken on a loaded machine are a first sample; the quiet-machine run is the one that decides",
    }
    SPEED_RESULT.write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    command, *rest = sys.argv[1:] or ["report"]
    if command == "alts":
        alts_cost()
    elif command == "cell":
        for cell in rest:
            cell_e2e(cell)
    elif command == "pass2":
        for cell in rest:
            pass2_times(cell)
    elif command == "v5report":
        v5report()
    elif command == "prepare":
        prepare()
    elif command == "run":
        for variant in rest:
            run_variant(variant)
    elif command == "reopen":
        reopen()
    else:
        report()
