"""The code of one OCR worker process and the reading of one page, which closes every pdfium object it opens."""

from __future__ import annotations

import importlib
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import psutil
import pypdfium2 as pdfium

from ..types import PageReading
from .cache import PageCache, write_crops_file
from .ocr import DEFAULT_TUNING, Tuning, rapidocr_version, read_page_with_crops
from .orphan import next_task, parent_gone
from .render import render_page
from .reread import RereadConfig

DEFAULT_FACTORY = "extract.engine.ocr:load_engine"
RENDER_FAILED = "page.reason.render_failed"


@dataclass(frozen=True)
class PageTask:
    number: int
    dpi: int
    key: str
    image: str | None = None
    size_pt: tuple[float, float] | None = None
    study: RereadConfig | None = None


@dataclass(frozen=True)
class Chunk:
    id: int
    path: str
    sha256: str
    pages: tuple[PageTask, ...]


@dataclass(frozen=True)
class WorkerSettings:
    work_dir: str
    threads: int
    backend: str
    factory: str = DEFAULT_FACTORY
    tuning: Tuning = DEFAULT_TUNING
    rss_limit_mb: int = 0


def failed_reading(number: int, error: str) -> PageReading:
    """A page that could not be opened, rendered or read: unreadable, with what went wrong."""
    return PageReading(number, 0, 0, 0.0, 0.0, 0, (), "unreadable", RENDER_FAILED, error)


def load_factory(spec: str):
    module, _, name = spec.partition(":")
    return getattr(importlib.import_module(module), name)


class PageReader:
    """Reads pages of one PDF at a time, keeping that document open until another one is asked for."""

    def __init__(self, engine: Any, cache: PageCache) -> None:
        self.engine = engine
        self.cache = cache
        self._path: str | None = None
        self._document: pdfium.PdfDocument | None = None

    def _open(self, path: str) -> pdfium.PdfDocument:
        if self._path != path:
            self.close()
            self._document = pdfium.PdfDocument(path)
            self._path = path
        return self._document

    def read(self, chunk: Chunk, task: PageTask) -> PageReading:
        """Read one page of the chunk's PDF, or the image file of the task: an image is read as it is, with no render and
        nothing written to the cache."""
        if task.image is not None:
            try:
                reading, crops = read_page_with_crops(
                    task.image, self.engine, task.number, task.dpi, task.size_pt, task.study
                )
                if task.study is not None and reading.status == "ocr":
                    write_crops_file(Path(task.image).with_suffix(".crops.npz"), task.key, crops)
                return reading
            except ValueError as error:
                return failed_reading(task.number, f"ValueError: {error}")
        try:
            document = self._open(chunk.path)
            page = document[task.number - 1]
            try:
                image = render_page(page, task.dpi)
                size = page.get_size()
            finally:
                page.close()
            reading, crops = read_page_with_crops(image, self.engine, task.number, task.dpi, size, task.study)
        except pdfium.PdfiumError as error:  # the page becomes unreadable with its error; the batch goes on
            return failed_reading(task.number, f"PdfiumError: {error}")
        meta = {"backend": self.engine.backend, "rapidocr_version": rapidocr_version()}
        self.cache.put(chunk.sha256, task.number, task.key, reading, meta)
        if task.study is not None and reading.status == "ocr":
            self.cache.put_crops(chunk.sha256, task.number, task.key, crops)
        return reading

    def close(self) -> None:
        if self._document is not None:
            self._document.close()
        self._document = self._path = None


MIN_PAGES_FOR_RSS_RECYCLE = 8


def worker_main(task_queue, result_queue, settings: WorkerSettings, max_pages: int) -> None:
    """Load the engine once and read chunks until a stop sentinel arrives or the worker is due for replacement: after
    `max_pages` pages, or earlier, once it has read a few pages, if its memory passes `settings.rss_limit_mb`. The
    reason is sent to the parent before the worker leaves."""
    pid = os.getpid()
    started = time.perf_counter()
    try:
        engine = load_factory(settings.factory)(settings.threads, settings.backend, settings.tuning)
    except Exception:  # reported to the parent, which decides what to do
        result_queue.put(("failed", pid, traceback.format_exc()))
        return
    result_queue.put(("ready", pid, engine.backend, engine.fallback_reason, time.perf_counter() - started))
    reader = PageReader(engine, PageCache(Path(settings.work_dir)))
    try:
        pages_read = 0
        reason = None
        while reason is None:
            chunk = next_task(task_queue)
            if chunk is None:
                break
            result_queue.put(("take", pid, chunk.id))
            for task in chunk.pages:
                if parent_gone():
                    return
                began = time.perf_counter()
                reading = reader.read(chunk, task)
                result_queue.put(("page", pid, chunk.id, reading, time.perf_counter() - began))
            pages_read += len(chunk.pages)
            result_queue.put(("chunk", pid, chunk.id))
            rss_mb = psutil.Process().memory_info().rss // (1024 * 1024)
            if pages_read >= max_pages:
                reason = "pages"
            elif settings.rss_limit_mb and pages_read >= MIN_PAGES_FOR_RSS_RECYCLE and rss_mb > settings.rss_limit_mb:
                reason = "rss"
        if reason is not None:
            result_queue.put(("recycle", pid, reason, rss_mb, pages_read))
    except Exception:
        result_queue.put(("crash", pid, traceback.format_exc()))
        raise
    finally:
        reader.close()
