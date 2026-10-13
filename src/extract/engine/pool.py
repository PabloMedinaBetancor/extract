"""The pool of OCR worker processes: sized from free memory and cores, recycled every `RECYCLE_PAGES` pages or sooner
when a worker's memory passes its limit, and kept within memory bounds by never starting a worker the machine cannot
hold."""

from __future__ import annotations

import multiprocessing
import os
import queue
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import psutil

from ..errors import EngineLoadError
from .ocr import DEFAULT_TUNING, Tuning
from .worker import DEFAULT_FACTORY, Chunk, WorkerSettings, failed_reading, worker_main

RECYCLE_PAGES = 120
RSS_LIMIT_MB = 1600
RSS_HEADROOM_MB = 200
RSS_FLOOR_MB = 1300
CHUNK_PAGES = 4
WORKER_BUDGET_MB = {"openvino": 1500, "onnxruntime": 700}
RESERVE_MB = 800
MAX_PROCESSES = 2
LOW_MEMORY_MB = 600
OPENVINO_MIN_FREE_MB = 2000
WINDOW_PAGES = 20
STOP_SECONDS = 1.0

LogFn = Callable[..., None]


MIB = 1024 * 1024
PROC_CGROUP = "/proc/self/cgroup"
CGROUP_ROOT = "/sys/fs/cgroup"


def _cgroup_number(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:  # no such control file: this cgroup has no such limit (or the system has no cgroup v2)
        return None
    return None if text == "max" else int(text)


def cgroup_headroom_mb(proc_file: str = PROC_CGROUP, root: str = CGROUP_ROOT) -> int | None:
    """How much more memory the process's own cgroup may use before its `memory.high` or `memory.max`, or None when it
    has no such limit (or it cannot be read). A scope started with `systemd-run -p MemoryMax=...` has one."""
    try:
        entries = Path(proc_file).read_text().splitlines()
    except OSError:
        return None
    path = next((e.split("::", 1)[1] for e in entries if e.startswith("0::")), None)
    if path is None:
        return None
    folder = Path(root) / path.strip("/")
    limits = [
        v for v in (_cgroup_number(folder / "memory.max"), _cgroup_number(folder / "memory.high")) if v is not None
    ]
    current = _cgroup_number(folder / "memory.current")
    if not limits or current is None:
        return None
    return max(0, (min(limits) - current) // MIB)


def memory_status(proc_file: str = PROC_CGROUP, root: str = CGROUP_ROOT) -> dict[str, int | str | None]:
    """The free memory the pool may count on: the system's available memory or what is left in the process's own
    cgroup, whichever is less, and which of the two it was."""
    system = int(psutil.virtual_memory().available // MIB)
    cgroup = cgroup_headroom_mb(proc_file, root)
    free = system if cgroup is None else min(system, cgroup)
    source = "cgroup" if cgroup is not None and cgroup < system else "system"
    return {"free_mb": free, "system_mb": system, "cgroup_mb": cgroup, "limit": source}


def free_memory_mb() -> int:
    return int(memory_status()["free_mb"])


@dataclass(frozen=True)
class PoolPlan:
    processes: int
    threads: int
    backend: str
    budget_mb: int
    free_mb: int


def plan_workers(free_mb: int, cores: int, backend: str = "auto") -> PoolPlan:
    """Processes and threads for the free memory and physical cores: a worker is budgeted at 1.5 GB with OpenVINO and
    0.7 GB with onnxruntime, and 800 MB are kept for everything else. Two processes at most: two of two threads
    measured faster than one of four, and more were not measured (`bench/results/engine-speed.json`)."""
    if backend == "auto":
        backend = "openvino" if free_mb > OPENVINO_MIN_FREE_MB else "onnxruntime"
    budget = WORKER_BUDGET_MB[backend]
    processes = max(1, min(cores, MAX_PROCESSES, (free_mb - RESERVE_MB) // budget))
    return PoolPlan(processes, max(1, cores // processes), backend, budget, free_mb)


def rss_limit_mb(plan: PoolPlan, ceiling: int = RSS_LIMIT_MB) -> int:
    """The memory at which a worker is replaced. The ceiling is the peak O9 allows (1.6 GB), or less when the free
    memory shared by the workers, after the reserve, is smaller. A worker is only looked at after a chunk, and one
    chunk was measured to add up to 190 MB (a large page), so it is replaced `RSS_HEADROOM_MB` below the ceiling to keep
    its peak under it; never below `RSS_FLOOR_MB`, which a worker passes in a few pages and without which it would be
    replaced as soon as it started."""
    share = (plan.free_mb - RESERVE_MB) // plan.processes
    return max(RSS_FLOOR_MB, min(ceiling, share) - RSS_HEADROOM_MB)


def chunks_of(path: str, sha256: str, pages, first_id: int = 0) -> list[Chunk]:
    """Split the pages to read into chunks of up to four consecutive pages of the same PDF."""
    chunks, current = [], []
    for task in pages:
        if current and (len(current) == CHUNK_PAGES or task.number != current[-1].number + 1):
            chunks.append(Chunk(first_id + len(chunks), path, sha256, tuple(current)))
            current = []
        current.append(task)
    if current:
        chunks.append(Chunk(first_id + len(chunks), path, sha256, tuple(current)))
    return chunks


EXIT_NO_MAIN_GUARD = 87  # the exit code of a worker that was started by a script that reads at its top level
NO_MAIN_GUARD_HINT = 'put every call to `extract.read` and `extract.Reader` under `if __name__ == "__main__":`'


def leave_if_started_by_an_unguarded_script() -> None:
    """Called by every read before it starts workers. A worker imports the script that started it again; if that
    script reads at its top level, the worker would start workers of its own. It says so on stderr and leaves with
    `EXIT_NO_MAIN_GUARD`, which the parent turns into an `EngineLoadError` instead of a quiet `partial` document."""
    if getattr(multiprocessing.current_process(), "_inheriting", False):
        sys.stderr.write(
            "Extract: an OCR worker is importing this script, and the script reads a document at its top level; "
            f"{NO_MAIN_GUARD_HINT}\n"
        )
        sys.stderr.flush()
        os._exit(EXIT_NO_MAIN_GUARD)


class _Worker:
    def __init__(self, process) -> None:
        self.process = process
        self.chunk: int | None = None
        self.ready = False


class OcrPool:
    """Workers are started with `spawn`, load the engine once and are replaced after `recycle_pages` pages. The
    workers persist between `run` calls until `close`."""

    def __init__(
        self,
        work_dir: Path | str,
        *,
        backend: str = "auto",
        recycle_pages: int = RECYCLE_PAGES,
        processes: int | None = None,
        threads: int | None = None,
        factory: str = DEFAULT_FACTORY,
        tuning: Tuning = DEFAULT_TUNING,
        free_memory: Callable[[], int] = free_memory_mb,
        cores: int | None = None,
        log: LogFn | None = None,
        poll_seconds: float = 0.05,
        rss_limit_mb: int | None = None,
    ) -> None:
        self.work_dir = str(work_dir)
        self.backend = backend
        self.recycle_pages = max(CHUNK_PAGES, recycle_pages)
        self.fixed_processes = processes
        self.fixed_threads = threads
        self.factory = factory
        self.tuning = tuning
        self.free_memory = free_memory
        self.cores = cores or psutil.cpu_count(logical=False) or 1
        self.log = log or (lambda code, **params: None)
        self.poll_seconds = poll_seconds
        self.rss_limit_mb = rss_limit_mb
        leave_if_started_by_an_unguarded_script()
        self._context = multiprocessing.get_context("spawn")
        self._workers: dict[int, _Worker] = {}
        self._plan: PoolPlan | None = None
        self._queues = None
        self._low_memory = False
        self._chunk_ids = 0
        self.pages_done = 0
        self.peak_rss_mb: dict[int, int] = {}
        self._window_start = time.perf_counter()
        self._window_seconds: list[float] = []

    def _fresh_queues(self) -> None:
        self._queues = (self._context.Queue(), self._context.Queue())

    def _decide(self) -> PoolPlan:
        if self.free_memory is free_memory_mb:
            source = memory_status()
            free = int(source["free_mb"])
        else:
            source, free = {"limit": "injected"}, self.free_memory()
        plan = plan_workers(free, self.cores, self.backend)
        if self.fixed_processes is not None:
            plan = PoolPlan(
                self.fixed_processes, self.fixed_threads or plan.threads, plan.backend, plan.budget_mb, free
            )
        elif self.fixed_threads is not None:
            plan = PoolPlan(plan.processes, self.fixed_threads, plan.backend, plan.budget_mb, free)
        self.log(
            "ram.workers",
            free_mb=free,
            budget_mb=plan.budget_mb,
            reserve_mb=RESERVE_MB,
            cores=self.cores,
            processes=plan.processes,
            threads=plan.threads,
            backend=plan.backend,
            limit=source["limit"],
            system_mb=source.get("system_mb"),
            cgroup_mb=source.get("cgroup_mb"),
        )
        return plan

    def _spawn(self) -> None:
        task_queue, result_queue = self._queues
        limit = self.rss_limit_mb if self.rss_limit_mb is not None else rss_limit_mb(self._plan)
        settings = WorkerSettings(
            self.work_dir, self._plan.threads, self._plan.backend, self.factory, self.tuning, rss_limit_mb=limit
        )
        process = self._context.Process(
            target=worker_main, args=(task_queue, result_queue, settings, self.recycle_pages), daemon=True
        )
        process.start()
        self._workers[process.pid] = _Worker(process)
        self.log("worker.started", pid=process.pid, free_mb=self.free_memory(), alive=len(self._workers))

    def _fill_workers(self, pending: int) -> None:
        """Start workers up to the plan, but only when the memory is there: under 600 MB free no worker is started
        until it recovers, and the run goes on with the ones it has."""
        wanted = min(self._plan.processes, max(1, pending))
        while len(self._workers) < wanted:
            free = self.free_memory()
            if free < LOW_MEMORY_MB:
                if not self._low_memory:
                    self._low_memory = True
                    self.log("ram.low", free_mb=free, threshold_mb=LOW_MEMORY_MB, workers=len(self._workers))
                return
            if self._low_memory:
                self._low_memory = False
                self.log("ram.recovered", free_mb=free)
            self._spawn()

    def run(self, chunks: Sequence[Chunk], on_page: Callable, cancel=None) -> bool:
        """Read the chunks, calling `on_page(reading, pid, seconds)` for each page as it arrives. Returns False if
        `cancel.is_set()` stopped it: workers are stopped within two seconds, pages in flight are dropped and
        finished pages stay in the cache."""
        if not chunks:
            return True
        if self._queues is None:
            self._fresh_queues()
        if self._plan is None:
            self._plan = self._decide()
        task_queue, result_queue = self._queues
        by_id = {}
        for chunk in chunks:
            self._chunk_ids += 1
            numbered = replace(chunk, id=self._chunk_ids)
            by_id[numbered.id] = numbered
            task_queue.put(numbered)
        retries = dict.fromkeys(by_id, 0)
        unfinished = set(by_id)
        seen_pages: set[tuple[int, int]] = set()
        while unfinished:
            if cancel is not None and cancel.is_set():
                self._stop_workers()
                return False
            self._fill_workers(len(unfinished))
            self._reap(by_id, unfinished, retries, on_page, seen_pages)
            try:
                message = result_queue.get(timeout=self.poll_seconds)
            except queue.Empty:
                continue
            kind = message[0]
            if kind == "ready":
                _, pid, backend, fallback, seconds = message
                if self.backend != "auto" and backend != self.backend:
                    self._stop_workers()
                    raise EngineLoadError(f"{self.backend} was asked for and the worker started {backend}")
                if pid in self._workers:
                    self._workers[pid].ready = True
                self.log("worker.ready", pid=pid, backend=backend, load_seconds=round(seconds, 1))
                if fallback:
                    self.log("engine.fallback", reason=fallback, pid=pid)
            elif kind == "failed":
                self.log("worker.failed", pid=message[1], error=message[2])
                self._stop_workers()
                raise EngineLoadError(f"an OCR worker could not start:\n{message[2]}")
            elif kind == "crash":
                self.log("worker.crash", pid=message[1], error=message[2])
            elif kind == "recycle":
                _, pid, reason, rss, pages = message
                self.log(
                    "worker.recycled",
                    pid=pid,
                    reason=reason,
                    pages=pages,
                    rss_mb=rss,
                    peak_rss_mb=max(self.peak_rss_mb.get(pid, 0), rss),
                    pages_done=self.pages_done,
                )
            elif kind == "take":
                if message[1] in self._workers:
                    self._workers[message[1]].chunk = message[2]
            elif kind == "page":
                _, pid, chunk_id, reading, seconds = message
                if chunk_id not in by_id or (chunk_id, reading.number) in seen_pages:
                    continue
                seen_pages.add((chunk_id, reading.number))
                self._count_page(pid, seconds, reading.unaligned_words)
                on_page(reading, pid, seconds)
            elif kind == "chunk":
                unfinished.discard(message[2])
                if message[1] in self._workers:
                    self._workers[message[1]].chunk = None
        return True

    def _reap(self, by_id, unfinished, retries, on_page, seen_pages) -> None:
        """Forget workers that left. A recycled worker leaves quietly; one that died with a chunk in hand has the
        chunk's unread pages queued once more, and a second death makes them unreadable."""
        for pid, worker in list(self._workers.items()):
            if worker.process.is_alive():
                continue
            del self._workers[pid]
            worker.process.join(0.1)
            code = worker.process.exitcode
            if code == EXIT_NO_MAIN_GUARD:
                self._stop_workers()
                raise EngineLoadError(
                    "the script that called Extract reads a document at its top level, and each OCR worker imports "
                    f"that script again; {NO_MAIN_GUARD_HINT}",
                    cause="no_main_guard",
                )
            if code == 0:
                continue  # it said why it left in a "recycle" message, logged when that arrived
            self.log("worker.died", pid=pid, exit_code=code, chunk=worker.chunk)
            held = {w.chunk for w in self._workers.values() if w.chunk is not None}
            lost = [worker.chunk] if worker.chunk is not None else [c for c in unfinished if c not in held]
            for chunk_id in lost:
                if chunk_id not in unfinished:
                    continue
                chunk = by_id[chunk_id]
                left = tuple(t for t in chunk.pages if (chunk_id, t.number) not in seen_pages)
                retries[chunk_id] += 1
                if retries[chunk_id] > 1:
                    for task in left:
                        on_page(failed_reading(task.number, f"the OCR worker died twice (exit code {code})"), pid, 0.0)
                    unfinished.discard(chunk_id)
                else:
                    retry = replace(chunk, pages=left)
                    by_id[chunk_id] = retry
                    self._queues[0].put(retry)

    def _count_page(self, pid: int, seconds: float, unaligned: int = 0) -> None:
        self.pages_done += 1
        rss = self._sample_rss(pid)
        self.log(
            "pool.page", pid=pid, seconds=round(seconds, 2), rss_mb=rss, n=self.pages_done, unaligned_words=unaligned
        )
        self._window_seconds.append(seconds)
        if len(self._window_seconds) >= WINDOW_PAGES:
            now = time.perf_counter()
            wall = now - self._window_start
            self.log(
                "pool.window",
                pages_done=self.pages_done,
                seconds_per_page=round(wall / len(self._window_seconds), 2),
                worker_seconds_per_page=round(sum(self._window_seconds) / len(self._window_seconds), 2),
                rss_mb=self.worker_rss_mb(),
            )
            self._window_start, self._window_seconds = now, []

    def _sample_rss(self, pid: int) -> int:
        try:
            rss = psutil.Process(pid).memory_info().rss // (1024 * 1024)
        except psutil.NoSuchProcess:
            return 0
        self.peak_rss_mb[pid] = max(self.peak_rss_mb.get(pid, 0), rss)
        return rss

    def worker_rss_mb(self) -> dict[int, int]:
        sizes = {}
        for pid in self._workers:
            try:
                sizes[pid] = psutil.Process(pid).memory_info().rss // (1024 * 1024)
            except psutil.NoSuchProcess:
                continue
        return sizes

    def _stop_workers(self) -> None:
        began = time.perf_counter()
        workers = list(self._workers.values())
        for worker in workers:
            if worker.process.is_alive():
                worker.process.terminate()
        deadline = began + STOP_SECONDS
        for worker in workers:
            worker.process.join(max(0.0, deadline - time.perf_counter()))
            if worker.process.is_alive():
                worker.process.kill()
                worker.process.join(0.5)
        self._workers.clear()
        self._discard_partial_files()
        self._queues = None
        self._plan = None
        self._window_start, self._window_seconds = time.perf_counter(), []

    def _discard_partial_files(self) -> None:
        for partial in (Path(self.work_dir) / "pages").rglob("*.part"):
            partial.unlink(missing_ok=True)

    def close(self) -> None:
        if self._queues is not None:
            for _ in self._workers:
                self._queues[0].put(None)
            deadline = time.perf_counter() + 5
            for worker in self._workers.values():
                worker.process.join(max(0.0, deadline - time.perf_counter()))
        self._stop_workers()
