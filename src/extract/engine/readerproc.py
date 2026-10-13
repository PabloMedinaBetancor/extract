"""The reader process of pass 2: one process, apart from the OCR workers, with its own resident-memory cap, a trigger
that replaces it, and a bounded queue. The workers load no extra recogniser, so their memory and recycling are the
same as without a second reading.

The parent (`ReaderProcess`) starts it only when its budget fits the free memory, hands it pages (the first reading
and its stored crops), takes the pages back with their readings, replaces it above its trigger or after a crash
without losing a page (a page it could not read comes back without readings and its units counted as skipped), and
stops it in 2 seconds. The process never touches the network and decodes greedily with fixed threads, so the same
crops give the same readings twice.
"""

from __future__ import annotations

import importlib
import multiprocessing as mp
import os
import queue
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass

import psutil

from .. import second_reading as second_reading_decision
from ..types import PageReading
from .ocr import free_memory_mb
from .orphan import next_task
from .reread import DEFAULT_READERS, DEFAULT_UNIT, RereadConfig
from .second import second_pass, second_pass_cells, without_readings

DEFAULT_FACTORY = "extract.engine.readers:ReaderSet"
RSS_TRIGGER_MB = 1300
RSS_CAP_MB = 1500
THOROUGH_BUDGET_MB = (
    1600  # free memory it needs to start (and to be replaced): its cap, so it never starts where it cannot grow
)
THOROUGH_RSS_CAP_MB = 1600  # the memory limit of the thorough mode; the thorough mode loads latin, wide and deep together (60 pilot pages peaked at 1.586 GB)
BUDGET_MB = {"openvino": 1100, "onnxruntime": 700}
QUEUE_PAGES = 4
STOP_SECONDS = 2.0
STALL_SECONDS = (
    300.0  # a reader process that sends nothing for this long, with a page waiting, is stalled and is killed
)
MIN_PAGES_FOR_RSS_RECYCLE = 4
RECYCLE_PAGES = 200


@dataclass(frozen=True)
class ReaderSettings:
    """What pass 2 reads and with what: the unit kind and the readers of the cell, the threads, the recycling triggers
    and the factory of the reader set (a `module:name` that is called with `(names, threads, backend)`)."""

    unit: str = DEFAULT_UNIT
    readers: tuple[str, ...] = DEFAULT_READERS
    config: RereadConfig = RereadConfig()
    threads: int = 2
    backend: str = "auto"
    rss_trigger_mb: int = RSS_TRIGGER_MB
    recycle_pages: int = RECYCLE_PAGES
    factory: str = DEFAULT_FACTORY
    passes: tuple[tuple[str, tuple[str, ...]], ...] | None = None  # (unit, readers) in order; the thorough mode
    rss_cap_mb: int | None = None  # the hard cap of this process; None: `RSS_CAP_MB`
    budget_mb: int | None = None  # free memory it needs to start; None: `BUDGET_MB` of the backend


THOROUGH_UNIT = "line+zone"


def settings_for(second_reading, **given) -> ReaderSettings:
    """The settings of pass 2 for an `Options.second_reading`: the packaged cell, or for `"thorough"` the lines and the
    zones in one reader process with the units pass 1 stores for them (`given` overrides the rest, e.g. the factory)."""
    if not second_reading_decision.is_thorough(second_reading):
        return ReaderSettings(**given)
    block = second_reading_decision.thorough()
    config = RereadConfig(study_units=block["lines"], study_zones=block["zones"])
    return ReaderSettings(
        **{
            "unit": THOROUGH_UNIT,
            "readers": second_reading_decision.thorough_readers(),
            "config": config,
            "passes": second_reading_decision.thorough_passes(),
            "rss_cap_mb": THOROUGH_RSS_CAP_MB,
            "budget_mb": THOROUGH_BUDGET_MB,
            **given,
        }
    )


def load_factory(spec: str):
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


def reader_main(task_queue, result_queue, settings: ReaderSettings) -> None:
    pid = os.getpid()
    try:
        readers = load_factory(settings.factory)(settings.readers, settings.threads, settings.backend)
    except Exception:
        result_queue.put(("failed", pid, traceback.format_exc()))
        return
    result_queue.put(("ready", pid))
    done = 0
    try:
        while True:
            job = next_task(task_queue)
            if job is None:
                return
            job_id, page, crops = job
            result_queue.put(("take", pid, job_id))
            try:
                if settings.passes:
                    out = second_pass_cells(page, crops, readers, settings.passes, settings.config)
                else:
                    out = second_pass(page, crops, readers, settings.unit, settings.config)
                result_queue.put(("page", pid, job_id, out))
            except Exception:
                result_queue.put(("error", pid, job_id, traceback.format_exc()))
            done += 1
            rss = psutil.Process().memory_info().rss // (1024 * 1024)
            if done >= settings.recycle_pages or (done >= MIN_PAGES_FOR_RSS_RECYCLE and rss > settings.rss_trigger_mb):
                result_queue.put(("recycle", pid, "pages" if done >= settings.recycle_pages else "rss", rss, done))
                return
    except Exception:
        result_queue.put(("crash", pid, traceback.format_exc()))
        raise


class ReaderProcess:
    """The parent's side. `submit(page, crops)` queues a page and returns its id (blocking, while it collects finished
    pages, when `QUEUE_PAGES` are already waiting); `collect(wait)` returns the finished `(id, page)` pairs;
    `drain()` waits for everything. `started` is False when the memory did not allow it: the pages then come back
    at once without readings."""

    def __init__(
        self,
        settings: ReaderSettings,
        *,
        free_memory: Callable[[], int] = free_memory_mb,
        budget_mb: int | None = None,
        log: Callable[..., None] = lambda code, **params: None,
    ) -> None:
        self.settings = settings
        self.free_memory = free_memory
        self.budget_mb = budget_mb or settings.budget_mb or BUDGET_MB["openvino"]
        self.log = log
        self._context = mp.get_context("spawn")
        self._process = None
        self._tasks = self._results = None
        self._jobs: dict[int, PageReading] = {}
        self._taken: set[int] = set()
        self._next = 0
        self.started = False
        self.replaced = 0
        self._stash: list[tuple[int, PageReading]] = []
        self._progress = time.perf_counter()  # when the process last said anything, or got a page after being idle

    def start(self) -> bool:
        free = self.free_memory()
        if free < self.budget_mb:
            self.log("reader.skipped", free_mb=free, budget_mb=self.budget_mb)
            return False
        self._tasks = self._context.Queue(QUEUE_PAGES)
        self._results = self._context.Queue()
        self._spawn()
        self.started = True
        return True

    def _spawn(self) -> None:
        self._process = self._context.Process(
            target=reader_main, args=(self._tasks, self._results, self.settings), daemon=True
        )
        self._process.start()
        self._progress = time.perf_counter()
        self.log("reader.started", pid=self._process.pid, free_mb=self.free_memory())

    def submit(self, page: PageReading, crops: dict[str, bytes]) -> tuple[int, PageReading | None]:
        """(id, None) when queued, (id, the page without readings) when there is no reader process to take it."""
        self._next += 1
        job = self._next
        if not self.started:
            return job, without_readings(page)
        if not self._jobs:
            self._progress = time.perf_counter()
        self._jobs[job] = page
        while True:
            try:
                self._tasks.put((job, page, crops), timeout=0.5)
                return job, None
            except queue.Full:
                self._stash.extend(self.collect_pending(0.0))

    def _alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    def collect_pending(self, wait: float) -> list[tuple[int, PageReading]]:
        """The jobs answered so far (waiting up to `wait` seconds for the first), and the lost ones without readings."""
        out, self._stash = self._stash, []
        if self._results is None:
            return out
        deadline = time.perf_counter() + wait
        while True:
            try:
                message = self._results.get(timeout=max(0.0, min(0.2, deadline - time.perf_counter())))
            except queue.Empty:
                message = None
            if message is not None:
                self._progress = time.perf_counter()
                kind = message[0]
                if kind == "take":
                    self._taken.add(message[2])
                elif kind in ("page", "error"):
                    job = message[2]
                    self._taken.discard(job)
                    page = self._jobs.pop(job, None)
                    if page is not None:
                        if kind == "error":
                            self.log("reader.error", job=job, error=message[3][-400:])
                        out.append((job, message[3] if kind == "page" else without_readings(page)))
                elif kind == "recycle":
                    self.log("reader.recycled", reason=message[2], rss_mb=message[3], pages=message[4])
                    self._wait_exit()
                    self.replaced += 1
                    if self._jobs and not self._respawn():
                        for job in list(self._jobs):
                            out.append((job, without_readings(self._jobs.pop(job))))
                        return out
                elif kind in ("failed", "crash"):
                    self.log("reader.failed", error=message[2][-400:])
                    self._wait_exit()
                    self.started = False
                    for job in list(self._jobs):
                        out.append((job, without_readings(self._jobs.pop(job))))
                    return out
            elif self._jobs and self._over_cap():
                self.log("reader.capped", rss_mb=self._rss_mb(), cap_mb=self._cap_mb())
                self._kill()
                out.extend(self._after_death())
            elif self._jobs and not self._alive():
                out.extend(self._after_death())
            elif self._jobs and time.perf_counter() - self._progress > STALL_SECONDS:
                self.log("reader.stalled", seconds=round(time.perf_counter() - self._progress), jobs=len(self._jobs))
                self._kill()
                out.extend(self._after_death())
            if out or time.perf_counter() >= deadline or (message is None and not self._jobs):
                return out

    def _rss_mb(self) -> int:
        try:
            return psutil.Process(self._process.pid).memory_info().rss // (1024 * 1024)
        except (psutil.Error, AttributeError, ValueError):
            return 0

    def _cap_mb(self) -> int:
        return self.settings.rss_cap_mb or RSS_CAP_MB

    def _over_cap(self) -> bool:
        """The reader process holds more than `RSS_CAP_MB` (the hard cap of its own; the trigger at `rss_trigger_mb` only
        retires it between two pages)."""
        return self._alive() and self._rss_mb() > self._cap_mb()

    def _respawn(self) -> bool:
        """A replacement process, but only if the memory the first one gave back still holds its budget; otherwise the
        reading stops (the pages that wait come back without readings) as it does at the start."""
        free = self.free_memory()
        if free < self.budget_mb:
            self.log("reader.skipped", free_mb=free, budget_mb=self.budget_mb)
            self.started = False
            return False
        self._spawn()
        return True

    def _after_death(self) -> list[tuple[int, PageReading]]:
        """The process left without saying so (a crash, a kill). A "take" message can die with it, so the lost job is
        found by elimination: what is still in the task queue goes back to a new process, and every other unanswered
        job is returned without readings."""
        self.replaced += 1
        waiting: list[tuple] = []
        while True:
            try:
                waiting.append(self._tasks.get(timeout=0.3))
            except queue.Empty:
                break
        kept = {job[0] for job in waiting if job is not None}
        out = []
        for job in list(self._jobs):
            if job not in kept:
                self.log("reader.lost", job=job)
                out.append((job, without_readings(self._jobs.pop(job))))
        self._taken.clear()
        if self._jobs:
            if not self._respawn():
                return out + [(job, without_readings(self._jobs.pop(job))) for job in list(self._jobs)]
            for job in waiting:
                if job is not None:
                    self._tasks.put(job)
        return out

    def _kill(self) -> None:
        """Ends the process whatever it is doing: asked first, then forced, each wait bounded."""
        if self._process is None:
            return
        self._process.terminate()
        self._process.join(STOP_SECONDS)
        if self._process.is_alive():
            self._process.kill()
            self._process.join(STOP_SECONDS)

    def _wait_exit(self) -> None:
        if self._process is not None:
            self._process.join(STOP_SECONDS)

    def pending(self) -> int:
        return len(self._jobs)

    def drain(self, cancel=None) -> list[tuple[int, PageReading]]:
        out: list[tuple[int, PageReading]] = []
        while self._jobs:
            if cancel is not None and cancel.is_set():
                break
            out.extend(self.collect_pending(1.0))
        return out

    def stop(self) -> None:
        """Stops the process within `STOP_SECONDS`; jobs still waiting are dropped."""
        self._jobs.clear()
        if self._process is not None and self._process.is_alive():
            try:
                self._tasks.put_nowait(None)
            except queue.Full:
                pass
            self._process.join(STOP_SECONDS)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(1.0)
        for q in (self._tasks, self._results):  # pages nobody will read must not keep the parent from exiting
            if q is not None:
                q.cancel_join_thread()
                q.close()
        self._process = None
        self.started = False
