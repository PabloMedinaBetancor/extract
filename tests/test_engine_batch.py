import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

import psutil
import pypdfium2 as pdfium

from extract import events
from extract.engine import ocr
from extract.engine.batch import BatchReader
from extract.engine.cache import PageCache, file_sha256
from extract.engine.pool import (
    CHUNK_PAGES,
    LOW_MEMORY_MB,
    RECYCLE_PAGES,
    OcrPool,
    PoolPlan,
    cgroup_headroom_mb,
    chunks_of,
    memory_status,
    plan_workers,
    rss_limit_mb,
)
from extract.engine.worker import Chunk, PageReader, PageTask
from extract.errors import Cancelled, EngineLoadError, PasswordProtected, SourceChanged, UnsupportedFile
from extract.options import Options
from tests.fake_engine import FakeEngine
from tests.support import SPANISH_LINES, broken_page_pdf, digital_pdf, scanned_pdf, text_page_image

FAKE = "tests.fake_engine:load_engine"


def scan_pdf(directory, pages, name="scan.pdf"):
    images = [text_page_image(SPANISH_LINES[: 3 + n % 3], width=827, height=1169, size=14) for n in range(pages)]
    return scanned_pdf(Path(directory) / name, images)


class Flag:
    def __init__(self):
        self._set = threading.Event()

    def set(self):
        self._set.set()

    def is_set(self):
        return self._set.is_set()


class WorkerPlanTest(unittest.TestCase):
    def test_O7_workers_are_sized_by_free_memory_and_cores(self):
        cases = [
            (2500, 4, "openvino", 1),
            (4200, 4, "openvino", 2),
            (10000, 4, "openvino", 2),
            (900, 4, "openvino", 1),
            (2000, 4, "onnxruntime", 1),
            (3000, 4, "onnxruntime", 2),
            (9000, 2, "onnxruntime", 2),
        ]
        for free, cores, backend, expected in cases:
            with self.subTest(free=free, backend=backend):
                plan = plan_workers(free, cores, backend)
                self.assertEqual(plan.processes, expected)
                self.assertEqual(plan.threads, max(1, cores // expected))

    def test_O7_auto_picks_openvino_only_with_memory_for_it(self):
        self.assertEqual(plan_workers(3000, 4).backend, "openvino")
        self.assertEqual(plan_workers(1900, 4).backend, "onnxruntime")

    def test_O8_the_recycle_length_is_one_constant_of_120_pages(self):
        self.assertEqual(RECYCLE_PAGES, 120)
        self.assertEqual(Options().recycle_pages, RECYCLE_PAGES)

    def test_O7_chunks_hold_four_consecutive_pages_of_one_pdf(self):
        tasks = [PageTask(n, 200, "k") for n in (1, 2, 3, 4, 5, 6, 9, 10)]
        chunks = chunks_of("a.pdf", "sha", tasks)
        self.assertEqual([[t.number for t in c.pages] for c in chunks], [[1, 2, 3, 4], [5, 6], [9, 10]])
        self.assertEqual(CHUNK_PAGES, 4)


class CgroupMemoryTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.proc = self.tmp / "cgroup"
        self.proc.write_text("0::/user.slice/run-test.scope\n")
        self.folder = self.tmp / "fs" / "user.slice" / "run-test.scope"
        self.folder.mkdir(parents=True)

    def limits(self, maximum="max", high="max", current=0):
        (self.folder / "memory.max").write_text(f"{maximum}\n")
        (self.folder / "memory.high").write_text(f"{high}\n")
        (self.folder / "memory.current").write_text(f"{current}\n")

    def headroom(self):
        return cgroup_headroom_mb(str(self.proc), str(self.tmp / "fs"))

    def test_O12_the_headroom_is_the_nearer_limit_minus_what_the_cgroup_uses(self):
        mib = 1024 * 1024
        self.limits(maximum=3000 * mib, high=2500 * mib, current=300 * mib)
        self.assertEqual(self.headroom(), 2200)
        self.limits(maximum=2000 * mib, high=2500 * mib, current=300 * mib)
        self.assertEqual(self.headroom(), 1700)
        self.limits(maximum=2000 * mib, current=2500 * mib)
        self.assertEqual(self.headroom(), 0)

    def test_O12_a_cgroup_without_limits_or_files_gives_none(self):
        self.limits()
        self.assertIsNone(self.headroom())
        self.assertIsNone(cgroup_headroom_mb(str(self.tmp / "missing"), str(self.tmp / "fs")))
        self.proc.write_text("12:memory:/old/v1\n")
        self.assertIsNone(self.headroom())

    def test_O12_the_free_memory_is_the_lesser_of_the_system_and_the_cgroup_and_says_which(self):
        status = memory_status()
        self.assertEqual(status["free_mb"], min(status["system_mb"], status["cgroup_mb"] or status["system_mb"]))
        self.limits(maximum=1000 * 1024 * 1024, current=0)
        capped = memory_status(str(self.proc), str(self.tmp / "fs"))
        self.assertEqual((capped["free_mb"], capped["limit"], capped["cgroup_mb"]), (1000, "cgroup", 1000))

    def test_O12_the_log_names_the_limit_that_decided_the_workers(self):
        log = []
        pool = OcrPool(
            self.tmp / "work",
            factory=FAKE,
            free_memory=lambda: 2200,
            cores=4,
            log=lambda code, **params: log.append((code, params)),
        )
        try:
            pool._decide()
        finally:
            pool.close()
        decision = next(params for code, params in log if code == "ram.workers")
        self.assertEqual((decision["limit"], decision["processes"], decision["threads"]), ("injected", 1, 4))

    def test_O12_two_workers_only_when_both_budgets_and_the_reserve_fit_in_the_cap(self):
        self.assertEqual(plan_workers(3000 - 300, 4, "openvino").processes, 1)
        self.assertEqual(plan_workers(3000 + 800, 4, "openvino").processes, 2)


class PdfiumObjectsTest(unittest.TestCase):
    """O8: after each page no pdfium page, text page or bitmap is left open."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.open_objects = set()

    @contextmanager
    def counting(self):
        """Count the pdfium pages, text pages and bitmaps that are open: opened by the calls that make them and
        removed by close()."""

        def opening(original):
            def opened(this, *args, **kwargs):
                made = original(this, *args, **kwargs)
                self.open_objects.add(id(made))
                return made

            return opened

        def closing(original):
            def closed(this, *args, **kwargs):
                self.open_objects.discard(id(this))
                return original(this, *args, **kwargs)

            return closed

        with (
            mock.patch.object(pdfium.PdfDocument, "get_page", opening(pdfium.PdfDocument.get_page)),
            mock.patch.object(pdfium.PdfPage, "render", opening(pdfium.PdfPage.render)),
            mock.patch.object(pdfium.PdfPage, "get_textpage", opening(pdfium.PdfPage.get_textpage)),
            mock.patch.object(pdfium.PdfPage, "close", closing(pdfium.PdfPage.close)),
            mock.patch.object(pdfium.PdfBitmap, "close", closing(pdfium.PdfBitmap.close)),
            mock.patch.object(pdfium.PdfTextPage, "close", closing(pdfium.PdfTextPage.close)),
        ):
            yield

    def test_O8_no_pdfium_object_stays_open_after_a_page(self):
        path = scan_pdf(self.tmp, 3)
        cache = PageCache(self.tmp / "work")
        reader = PageReader(FakeEngine(1), cache)
        chunk = Chunk(1, str(path), file_sha256(path), ())
        with self.counting():
            for number in (1, 2, 3):
                reading = reader.read(chunk, PageTask(number, 100, f"key{number}"))
                self.assertEqual(reading.status, "ocr")
                self.assertEqual(self.open_objects, set(), f"page {number}")
            reader.close()

    def test_O8_the_counter_sees_a_page_that_is_left_open(self):
        path = scan_pdf(self.tmp, 1)
        with self.counting():
            document = pdfium.PdfDocument(str(path))
            page = document[0]
            self.assertEqual(len(self.open_objects), 1)
            page.close()
            document.close()
            self.assertEqual(self.open_objects, set())

    def test_O4_a_page_that_cannot_be_loaded_becomes_unreadable_with_its_error(self):
        path = broken_page_pdf(self.tmp / "broken.pdf")
        reader = PageReader(FakeEngine(1), PageCache(self.tmp / "work"))
        chunk = Chunk(1, str(path), file_sha256(path), ())
        bad = reader.read(chunk, PageTask(2, 100, "k2"))
        good = reader.read(chunk, PageTask(3, 100, "k3"))
        reader.close()
        self.assertEqual((bad.status, bad.reason), ("unreadable", "page.reason.render_failed"))
        self.assertIn("Failed to load page", bad.error)
        self.assertEqual(good.status, "ocr")


class PoolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.log = []

    def pool(self, **kwargs):
        defaults = dict(
            factory=FAKE,
            log=lambda code, **params: self.log.append((code, params)),
            free_memory=lambda: 8000,
            cores=2,
        )
        return OcrPool(self.tmp / "work", **{**defaults, **kwargs})

    def run_pages(self, pool, pages, cancel=None):
        path = scan_pdf(self.tmp, pages)
        sha = file_sha256(path)
        chunks = chunks_of(str(path), sha, [PageTask(n, 100, f"k{n}") for n in range(1, pages + 1)])
        seen = []
        finished = pool.run(chunks, lambda reading, pid, seconds: seen.append((reading.number, pid)), cancel)
        return finished, seen

    def test_O8_the_worker_is_replaced_every_recycle_pages(self):
        pool = self.pool(recycle_pages=8, processes=1, threads=1)
        try:
            finished, seen = self.run_pages(pool, 40)
        finally:
            pool.close()
        self.assertTrue(finished)
        self.assertEqual(sorted(n for n, _ in seen), list(range(1, 41)))
        pids = [pid for _, pid in sorted(seen)]
        self.assertEqual(len(set(pids)), 5)
        for pid in set(pids):
            self.assertEqual(pids.count(pid), 8)
        starts = [params for code, params in self.log if code == "worker.started"]
        self.assertEqual(len(starts), 5)

    def test_O9_a_worker_over_its_memory_limit_is_replaced_before_its_pages_are_up(self):
        pool = self.pool(recycle_pages=120, processes=1, threads=1, rss_limit_mb=1)
        try:
            finished, seen = self.run_pages(pool, 24)
        finally:
            pool.close()
        self.assertTrue(finished)
        self.assertEqual(sorted(n for n, _ in seen), list(range(1, 25)))
        recycled = [params for code, params in self.log if code == "worker.recycled"]
        self.assertTrue(recycled)
        for params in recycled:
            self.assertEqual(params["reason"], "rss")
            self.assertGreater(params["rss_mb"], 1)
            self.assertGreaterEqual(params["pages"], 8)
            self.assertLess(params["pages"], 120)

    def test_O9_a_worker_under_its_limit_is_replaced_when_its_pages_are_up_and_says_so(self):
        pool = self.pool(recycle_pages=8, processes=1, threads=1, rss_limit_mb=100000)
        try:
            self.run_pages(pool, 24)
        finally:
            pool.close()
        recycled = [params for code, params in self.log if code == "worker.recycled"]
        self.assertTrue(recycled)
        self.assertEqual({params["reason"] for params in recycled}, {"pages"})
        self.assertEqual({params["pages"] for params in recycled}, {8})

    def test_O9_a_worker_is_replaced_a_chunk_of_headroom_below_the_ceiling_and_never_below_the_floor(self):
        self.assertEqual(rss_limit_mb(PoolPlan(1, 4, "openvino", 1500, 4200)), 1400)
        self.assertEqual(rss_limit_mb(PoolPlan(2, 2, "openvino", 1500, 4200)), 1400)
        self.assertEqual(rss_limit_mb(PoolPlan(1, 4, "openvino", 1500, 2500)), 1400)
        self.assertEqual(rss_limit_mb(PoolPlan(1, 4, "openvino", 1500, 2350)), 1350)
        self.assertEqual(rss_limit_mb(PoolPlan(1, 4, "openvino", 1500, 2000)), 1300)
        self.assertEqual(rss_limit_mb(PoolPlan(2, 2, "onnxruntime", 700, 1900)), 1300)
        self.assertEqual(rss_limit_mb(PoolPlan(1, 4, "openvino", 1500, 9000), ceiling=2000), 1800)

    def test_O8_recycling_counts_pages_not_chunks(self):
        pool = self.pool(recycle_pages=8, processes=1, threads=1)
        path = scan_pdf(self.tmp, 48)
        sha = file_sha256(path)
        odd = [PageTask(n, 100, f"k{n}") for n in range(1, 49, 2)]
        chunks = chunks_of(str(path), sha, odd)
        self.assertEqual({len(c.pages) for c in chunks}, {1})
        seen = []
        try:
            pool.run(chunks, lambda reading, pid, seconds: seen.append(pid))
        finally:
            pool.close()
        self.assertEqual(len(seen), 24)
        self.assertEqual(sorted(Counter(seen).values()), [8, 8, 8])

    def test_O8_the_log_carries_the_speed_of_each_window_and_the_memory_of_every_worker(self):
        pool = self.pool(recycle_pages=40, processes=2, threads=1)
        try:
            self.run_pages(pool, 44)
        finally:
            pool.close()
        windows = [params for code, params in self.log if code == "pool.window"]
        self.assertEqual(len(windows), 2)
        self.assertEqual([w["pages_done"] for w in windows], [20, 40])
        for window in windows:
            self.assertGreater(window["seconds_per_page"], 0)
            self.assertGreater(len(window["rss_mb"]), 0)
            self.assertTrue(all(mb > 10 for mb in window["rss_mb"].values()))

    def test_O12_the_decision_is_in_the_log_with_its_numbers(self):
        pool = self.pool(free_memory=lambda: 4200, cores=4)
        try:
            self.run_pages(pool, 8)
        finally:
            pool.close()
        decision = next(params for code, params in self.log if code == "ram.workers")
        self.assertEqual(
            (decision["free_mb"], decision["budget_mb"], decision["processes"], decision["threads"]),
            (4200, 1500, 2, 2),
        )
        self.assertEqual(len([1 for code, _ in self.log if code == "worker.started"]), 2)

    def test_O12_with_2500_mb_free_there_is_one_worker(self):
        pool = self.pool(free_memory=lambda: 2500, cores=4)
        try:
            self.run_pages(pool, 16)
        finally:
            pool.close()
        self.assertEqual(len([1 for code, _ in self.log if code == "worker.started"]), 1)

    def test_O12_a_drop_under_600_mb_starts_no_replacement_until_it_recovers(self):
        state = {"free": 8000, "calls_low": 0}

        def free_memory():
            if state["free"] < LOW_MEMORY_MB:
                state["calls_low"] += 1
                if state["calls_low"] > 40:
                    state["free"] = 8000
            return state["free"]

        def log(code, **params):
            self.log.append((code, params))
            if code == "worker.started" and len([1 for c, _ in self.log if c == "worker.started"]) == 1:
                state["free"] = 300

        pool = self.pool(recycle_pages=8, processes=1, threads=1, free_memory=free_memory, log=log)
        try:
            finished, seen = self.run_pages(pool, 24)
        finally:
            pool.close()
        self.assertTrue(finished)
        self.assertEqual(len(seen), 24)
        codes = [code for code, _ in self.log]
        self.assertIn("ram.low", codes)
        self.assertIn("ram.recovered", codes)
        self.assertLess(codes.index("ram.low"), codes.index("ram.recovered"))
        low = next(params for code, params in self.log if code == "ram.low")
        self.assertEqual((low["free_mb"], low["threshold_mb"]), (300, 600))
        started_while_low = [p for c, p in self.log if c == "worker.started" and p["free_mb"] < LOW_MEMORY_MB]
        self.assertEqual(started_while_low, [])

    def test_O7_a_chunk_whose_worker_dies_twice_comes_back_unreadable_and_the_run_ends(self):
        with mock.patch.dict(os.environ, {"EXTRACT_FAKE_DIE": "1"}):
            pool = self.pool(processes=1, threads=1)
            try:
                finished, seen = self.run_pages(pool, 4)
                results = []
                path = scan_pdf(self.tmp, 2, "again.pdf")
                chunks = chunks_of(str(path), file_sha256(path), [PageTask(n, 100, f"k{n}") for n in (1, 2)])
                pool.run(chunks, lambda reading, pid, seconds: results.append(reading), None)
            finally:
                pool.close()
        self.assertTrue(finished)
        self.assertEqual(sorted(n for n, _ in seen), [1, 2, 3, 4])
        self.assertEqual([r.status for r in results], ["unreadable", "unreadable"])
        self.assertIn("died twice", results[0].error)
        self.assertEqual(len([1 for code, _ in self.log if code == "worker.died"]), 4)

    def test_O7_stop_returns_within_two_seconds_even_when_a_worker_ignores_sigterm(self):
        flag = Flag()
        with mock.patch.dict(os.environ, {"EXTRACT_FAKE_SECONDS": "0.4", "EXTRACT_FAKE_IGNORE_TERM": "1"}):
            pool = self.pool(processes=1, threads=1)
            path = scan_pdf(self.tmp, 40)
            chunks = chunks_of(str(path), file_sha256(path), [PageTask(n, 100, f"k{n}") for n in range(1, 41)])
            seen = []

            def on_page(reading, pid, seconds):
                seen.append(pid)
                if len(seen) == 2:
                    flag.set()
                    on_page.stopped_at = time.perf_counter()

            try:
                finished = pool.run(chunks, on_page, flag)
                elapsed = time.perf_counter() - on_page.stopped_at
                alive = [p for p in psutil.Process().children(recursive=True) if "spawn_main" in " ".join(p.cmdline())]
            finally:
                pool.close()
        self.assertFalse(finished)
        self.assertLess(elapsed, 2.0)
        self.assertGreater(elapsed, 0.9, "the worker ignored SIGTERM, so only the kill can have stopped it")
        self.assertEqual(alive, [])

    def test_O7_stop_returns_within_two_seconds_and_leaves_no_partial_files(self):
        flag = Flag()
        with mock.patch.dict(os.environ, {"EXTRACT_FAKE_SECONDS": "0.4"}):
            pool = self.pool(processes=2, threads=1)
            path = scan_pdf(self.tmp, 60)
            sha = file_sha256(path)
            chunks = chunks_of(str(path), sha, [PageTask(n, 100, f"k{n}") for n in range(1, 61)])
            seen = []

            def on_page(reading, pid, seconds):
                seen.append(reading.number)
                if len(seen) == 3:
                    flag.set()
                    on_page.stopped_at = time.perf_counter()

            try:
                finished = pool.run(chunks, on_page, flag)
                elapsed = time.perf_counter() - on_page.stopped_at
            finally:
                pool.close()
        self.assertFalse(finished)
        self.assertLess(elapsed, 2.0)
        self.assertLess(len(seen), 60)
        files = list((self.tmp / "work").rglob("*"))
        self.assertEqual([f for f in files if f.suffix == ".part"], [])
        cached = {int(f.stem[1:]) for f in files if f.name.startswith("p0") and f.suffix == ".json"}
        self.assertTrue(cached <= set(range(1, 61)))
        self.assertLess(len(cached), 60)


class ImageTaskTest(unittest.TestCase):
    """A task with an image is read as it is: no PDF, no render, nothing in the cache."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def png(self, name, lines=3):
        import cv2

        path = self.tmp / name
        cv2.imwrite(str(path), text_page_image(SPANISH_LINES[:lines], width=900, height=400, size=22))
        return path

    def test_O7_a_chunk_of_images_gives_what_read_page_gives_and_writes_no_cache(self):
        from extract.engine.ocr import read_page

        images = [self.png("a.png"), self.png("b.png", 2)]
        tasks = [PageTask(n, 150, "", str(p), (400.0, 200.0)) for n, p in enumerate(images, 1)]
        pool = OcrPool(self.tmp / "work", factory=FAKE, processes=1, threads=1, free_memory=lambda: 8000, cores=2)
        seen = {}
        try:
            finished = pool.run(
                [Chunk(0, "", "", tuple(tasks))], lambda reading, pid, seconds: seen.update({reading.number: reading})
            )
        finally:
            pool.close()
        self.assertTrue(finished)
        for number, image in enumerate(images, 1):
            self.assertEqual(seen[number], read_page(image, FakeEngine(1), number, 150, (400.0, 200.0)))
        self.assertFalse((self.tmp / "work" / "pages").exists())

    def test_O4_an_image_that_cannot_be_read_is_unreadable_with_its_error(self):
        reader = PageReader(FakeEngine(1), PageCache(self.tmp / "work"))
        bad = self.tmp / "bad.png"
        bad.write_bytes(b"not an image")
        reading = reader.read(Chunk(0, "", "", ()), PageTask(1, 150, "", str(bad)))
        self.assertEqual((reading.status, reading.reason), ("unreadable", "page.reason.render_failed"))
        self.assertIn("cannot read the image", reading.error)

    def test_O8_images_recycle_and_stop_like_pages(self):
        images = [self.png(f"p{n}.png") for n in range(1, 25)]
        tasks = [PageTask(n, 150, "", str(p)) for n, p in enumerate(images, 1)]
        chunks = [Chunk(n, "", "", (task,)) for n, task in enumerate(tasks)]
        pool = OcrPool(
            self.tmp / "work", factory=FAKE, recycle_pages=8, processes=1, threads=1, free_memory=lambda: 8000, cores=2
        )
        seen = []
        try:
            pool.run(chunks, lambda reading, pid, seconds: seen.append(pid))
        finally:
            pool.close()
        self.assertEqual(sorted(Counter(seen).values()), [8, 8, 8])


class BatchReaderTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.readers = []

    def tearDown(self):
        for reader in self.readers:
            reader.close()

    def reader(self, **kwargs):
        options = Options(work_dir=self.tmp / "work", processes=1, threads=1, **kwargs)
        reader = BatchReader(options, factory=FAKE, free_memory=lambda: 8000, cores=2)
        self.readers.append(reader)
        return reader

    def test_O7_a_second_read_of_the_same_file_reads_no_page_again(self):
        path = scan_pdf(self.tmp, 6)
        reader = self.reader()
        first = reader.read(path)
        self.assertEqual(reader.pool.pages_done, 6)
        second = reader.read(path)
        self.assertEqual(reader.pool.pages_done, 6)
        self.assertEqual(first.pages, second.pages)

    def test_O7_a_changed_engine_rereads_every_page(self):
        path = scan_pdf(self.tmp, 4)
        reader = self.reader()
        reader.read(path)
        with mock.patch.object(ocr, "rapidocr_version", lambda: "9.9.9"):
            reader.read(path)
        self.assertEqual(reader.pool.pages_done, 8)

    def test_O7_a_changed_file_is_read_again(self):
        path = scan_pdf(self.tmp, 3)
        reader = self.reader()
        reader.read(path)
        other = scan_pdf(self.tmp, 4, "scan2.pdf")
        reader.read(other)
        self.assertEqual(reader.pool.pages_done, 7)

    def test_O7_native_and_scanned_pages_are_told_apart_in_a_mixed_document(self):
        from tests.support import mixed_pdf

        scan = text_page_image(SPANISH_LINES, width=827, height=1169, size=14)
        path = mixed_pdf(self.tmp / "mixed.pdf", SPANISH_LINES, scan)
        document = self.reader().read(path)
        self.assertEqual([p.status for p in document.pages], ["native", "ocr"])
        self.assertEqual(document.status, "done")

    def test_O7_the_real_engine_reads_a_scan_in_a_worker_and_the_main_process_never_loads_it(self):
        pages = [text_page_image(SPANISH_LINES, width=1240, height=1754, size=24) for _ in range(2)]
        path = scanned_pdf(self.tmp / "real.pdf", pages)
        script = (
            "import json, sys\n"
            "from extract.engine.batch import BatchReader\n"
            "from extract.options import Options\n"
            f"reader = BatchReader(Options(work_dir={str(self.tmp / 'work')!r}, processes=1, threads=2, backend='onnxruntime'))\n"
            f"document = reader.read({str(path)!r})\n"
            "reader.close()\n"
            "print(json.dumps({'status': document.status, 'words': [w.text for w in document.pages[0].words],"
            " 'loaded': [m for m in ('rapidocr', 'onnxruntime', 'openvino') if m in sys.modules]}))\n"
        )
        done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        result = json.loads(done.stdout.strip().splitlines()[-1])
        self.assertEqual(result["status"], "done")
        expected = " ".join(SPANISH_LINES).split()
        self.assertGreater(sum(w in result["words"] for w in expected) / len(expected), 0.9)
        self.assertEqual(result["loaded"], [])

    def test_O4_an_unreadable_page_makes_the_document_partial_and_the_batch_goes_on(self):
        reader = self.reader()
        broken = reader.read(broken_page_pdf(self.tmp / "broken.pdf"))
        self.assertEqual(broken.status, "partial")
        self.assertEqual(broken.unreadable_pages, (2,))
        self.assertEqual(broken.pages[1].reason, "page.reason.render_failed")
        self.assertEqual([p.status for p in broken.pages if p.number != 2], ["native", "native"])
        fine = reader.read(digital_pdf(self.tmp / "fine.pdf", [SPANISH_LINES]))
        self.assertEqual(fine.status, "done")

    def test_O3_a_lexicon_that_has_no_list_for_the_languages_sends_the_page_to_ocr_with_a_warning(self):
        class NoList:
            def has(self, word, languages):
                raise LookupError("no word list")

            share = has

        options = Options(work_dir=self.tmp / "work", processes=1, threads=1)
        reader = BatchReader(options, NoList(), factory=FAKE, free_memory=lambda: 8000, cores=2)
        self.readers.append(reader)
        seen = []
        document = reader.read(digital_pdf(self.tmp / "d.pdf", [SPANISH_LINES, SPANISH_LINES]), on_event=seen.append)
        self.assertEqual([p.status for p in document.pages], ["ocr", "ocr"])
        warnings = [e for e in seen if isinstance(e, events.Warning)]
        self.assertEqual([w.code for w in warnings], ["guard.warning.no_lexicon"])

    def test_A5_a_backend_that_was_asked_for_and_is_not_the_one_started_is_refused(self):
        options = Options(work_dir=self.tmp / "work", processes=1, threads=1, backend="openvino")
        reader = BatchReader(options, factory=FAKE, free_memory=lambda: 8000, cores=2)
        self.readers.append(reader)
        with self.assertRaises(EngineLoadError):
            reader.read(scan_pdf(self.tmp, 2))

    def test_O7_the_cache_files_say_which_backend_and_library_version_read_them(self):
        reader = self.reader()
        reader.read(scan_pdf(self.tmp, 2))
        files = sorted((self.tmp / "work" / "pages").rglob("p000*.json"))
        self.assertEqual(len(files), 2)
        data = json.loads(files[0].read_text())
        self.assertEqual(data["backend"], "fake")
        self.assertEqual(data["rapidocr_version"], ocr.rapidocr_version())

    def test_A5_a_file_that_is_not_a_pdf_is_refused_with_its_code(self):
        path = self.tmp / "text.pdf"
        path.write_bytes(b"this is not a pdf at all")
        with self.assertRaises(UnsupportedFile) as caught:
            self.reader().read(path)
        self.assertEqual(caught.exception.code, "error.unsupported_file")

    def test_A5_a_password_protected_file_is_refused_with_its_code(self):
        import pikepdf

        path = self.tmp / "locked.pdf"
        with pikepdf.Pdf.new() as pdf:
            pdf.add_blank_page()
            pdf.save(path, encryption=pikepdf.Encryption(owner="o", user="u"))
        with self.assertRaises(PasswordProtected) as caught:
            self.reader().read(path)
        self.assertEqual(caught.exception.code, "error.password_protected")

    def test_O10_originals_are_only_read(self):
        path = scan_pdf(self.tmp, 3)
        path.chmod(0o444)
        before = (file_sha256(path), path.stat().st_mtime_ns)
        self.reader().read(path)
        self.assertEqual((file_sha256(path), path.stat().st_mtime_ns), before)

    def test_O10_the_read_never_touches_the_network(self):
        path = scan_pdf(self.tmp, 2)
        with (
            mock.patch.object(socket.socket, "connect", side_effect=AssertionError("network")),
            mock.patch.object(socket, "create_connection", side_effect=AssertionError("network")),
        ):
            document = self.reader().read(path)
        self.assertEqual(document.status, "done")

    def test_O7_a_file_that_changes_during_the_read_is_reported(self):
        path = scan_pdf(self.tmp, 2)
        reader = self.reader()

        def touch(event):
            if isinstance(event, events.PageRead):
                os.utime(path, ns=(1, 1))
                path.write_bytes(path.read_bytes() + b"\n%x")

        with self.assertRaises(SourceChanged):
            reader.read(path, on_event=touch)

    def test_A4_events_come_in_order_and_progress_only_grows(self):
        path = scan_pdf(self.tmp, 5)
        seen = []
        self.reader().read(path, on_event=seen.append)
        self.assertIsInstance(seen[0], events.Started)
        progress = [e.done for e in seen if isinstance(e, events.Progress)]
        self.assertEqual(progress, [1, 2, 3, 4, 5])
        self.assertEqual(len([e for e in seen if isinstance(e, events.PageRead)]), 5)

    def test_O7_cancelling_raises_and_leaves_no_output(self):
        flag = Flag()
        path = scan_pdf(self.tmp, 30)
        reader = self.reader()
        seen = []

        def handler(event):
            seen.append(event)
            if isinstance(event, events.PageRead):
                flag.set()

        with mock.patch.dict(os.environ, {"EXTRACT_FAKE_SECONDS": "0.3"}):
            began = time.perf_counter()
            with self.assertRaises(Cancelled):
                reader.read(path, on_event=handler, cancel=flag)
        self.assertLess(time.perf_counter() - began, 15)
        self.assertEqual(list((self.tmp / "work").rglob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
