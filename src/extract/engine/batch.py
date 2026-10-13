"""Reading whole PDFs: the main-process pre-pass (native text, cache) and the dispatch of the rest to the pool."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
import pypdfium2.raw as pdfium_raw

from .. import events
from ..errors import Cancelled, PasswordProtected, SourceChanged, UnsupportedFile
from ..options import Options
from ..types import PageReading
from . import second
from .cache import PageCache, file_sha256, page_key
from .native import Lexicon, read_native
from .ocr import DEFAULT_TUNING, Tuning
from .pool import OcrPool, chunks_of, free_memory_mb
from .readerproc import ReaderProcess, ReaderSettings, load_factory, settings_for
from .render import MIN_DPI, page_dpi, scan_dpi
from .worker import DEFAULT_FACTORY, PageTask, failed_reading

WARNING_CODES = {
    "ram.low",
    "engine.fallback",
    "worker.died",
    "worker.crash",
    "reader.skipped",
    "reader.failed",
    "page.warning.low_resolution",
}


@dataclass(frozen=True)
class DocumentReading:
    path: Path
    sha256: str
    pages: tuple[PageReading, ...]

    @property
    def unreadable_pages(self) -> tuple[int, ...]:
        return tuple(p.number for p in self.pages if p.status == "unreadable")

    @property
    def status(self) -> str:
        return "partial" if self.unreadable_pages else "done"


def resolve_work_dir(options: Options) -> Path:
    if options.work_dir is not None:
        return Path(options.work_dir)
    cache_home = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
    return Path(cache_home) / "extract"


class JobLog:
    """What the engine decided and measured, kept in memory and appended to `<work>/job.log` as JSON lines."""

    def __init__(self, work_dir: Path) -> None:
        self.entries: list[dict[str, Any]] = []
        self.path = work_dir / "job.log"

    def __call__(self, code: str, **params: Any) -> None:
        entry = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "code": code, **params}
        self.entries.append(entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")


def _forget_last_pdfium_error() -> None:
    """pdfium keeps the code of the last failed load and does not reset it on success, and a PDF with no pages fails
    without setting one: its error would then be read as the one left by the file loaded before (a locked file made
    the next empty one "password protected"). A load of bytes that are not a PDF sets the plain format error."""
    pdfium_raw.FPDF_LoadMemDocument64(b"%", 1, None)


def open_pdf(path: Path) -> pdfium.PdfDocument:
    _forget_last_pdfium_error()
    try:
        return pdfium.PdfDocument(str(path))
    except pdfium.PdfiumError as error:
        if getattr(error, "err_code", None) == pdfium_raw.FPDF_ERR_PASSWORD:
            raise PasswordProtected(f"{path.name} is password protected", file=path.name) from error
        raise UnsupportedFile(f"{path.name} is not a PDF that can be read: {error}", file=path.name) from error


class BatchReader:
    """Reads documents one after another with workers and models loaded once, until `close`."""

    def __init__(
        self,
        options: Options,
        lexicon: Lexicon | None = None,
        *,
        factory: str = DEFAULT_FACTORY,
        tuning: Tuning = DEFAULT_TUNING,
        free_memory: Callable[[], int] = free_memory_mb,
        cores: int | None = None,
        reader_settings: ReaderSettings | None = None,
    ) -> None:
        self.options = options
        self.reader_settings = reader_settings or settings_for(options.second_reading)
        self._reader: ReaderProcess | None = None
        self._readers_key: str | None = None
        self._free_memory = free_memory
        self.lexicon = lexicon
        self.work_dir = resolve_work_dir(options)
        self.cache = PageCache(self.work_dir)
        self.log = JobLog(self.work_dir)
        self.pool = OcrPool(
            self.work_dir,
            backend=options.backend,
            recycle_pages=options.recycle_pages,
            processes=options.processes,
            threads=options.threads,
            factory=factory,
            tuning=tuning,
            free_memory=free_memory,
            cores=cores,
            log=self._pool_log,
        )
        self._emit: Callable[[events.Event], None] = lambda event: None
        self._warned_lexicon = False

    def _key(self, sha256: str, number: int, dpi: int) -> str:
        """The key of pass 1; with the second reading on it carries the settings of the units (A1.3)."""
        suffix = self.reader_settings.config.key() if self.options.second_reading else ""
        return page_key(sha256, number, dpi, suffix)

    def _rr2_key(self, key: str) -> str:
        if self._readers_key is None:
            settings = self.reader_settings
            names = load_factory(settings.factory)(settings.readers, settings.threads, settings.backend)
            self._readers_key = f"rr2:{names.key()}:{settings.unit}"
        return f"{key}|{self._readers_key}"

    def _reader_process(self) -> ReaderProcess:
        if self._reader is None:
            self._reader = ReaderProcess(self.reader_settings, free_memory=self._free_memory, log=self._pool_log)
            self._reader.start()
        return self._reader

    def _pool_log(self, code: str, **params: Any) -> None:
        self.log(code, **params)
        if code in WARNING_CODES:
            self._emit(events.Warning(code, params))

    def read(self, path: str | Path, on_event=None, cancel=None, transform=None) -> DocumentReading:
        """Read one PDF. Pages with a legible text layer keep it; the rest are read from the cache or by the workers.
        An unreadable page does not stop the document, which comes out `partial`. `transform` is applied to every page
        as it comes in (the guard); the events are those of the pages, the caller ends the job."""
        path = Path(path)
        emit = on_event or (lambda event: None)
        self._emit = emit
        self._warned_lexicon = False
        before = path.stat()
        sha256 = file_sha256(path)
        doc_id = sha256[:12]
        document = open_pdf(path)
        try:
            total = len(document)
            emit(events.Started(1, total))
            readings: dict[int, PageReading] = {}
            todo: list[PageTask] = []
            done = 0

            def finish(reading: PageReading) -> None:
                nonlocal done
                if transform is not None:
                    reading = transform(reading)
                readings[reading.number] = reading
                done += 1
                doubtful = sum(1 for w in reading.words if w.status in ("doubtful", "unscored"))
                fixed = sum(1 for w in reading.words if w.fix is not None and w.corrected_by == "guard")
                emit(events.PageRead(doc_id, reading.number, reading.status, len(reading.words), doubtful, fixed))
                emit(events.Progress(doc_id, done, total))

            waiting: dict[int, tuple[int, str]] = {}

            def deliver(reading: PageReading) -> None:
                if not self.options.second_reading or reading.status != "ocr":
                    finish(reading)
                    return
                key = self._key(sha256, reading.number, reading.dpi)
                full_key = self._rr2_key(key)
                cached = self.cache.get_rereads(sha256, reading.number, full_key, reading)
                crops = None if cached is not None else self.cache.get_crops(sha256, reading.number, key)
                if cached is not None:
                    finish(cached)
                elif crops is None:
                    finish(second.without_readings(reading))
                else:
                    job, back = self._reader_process().submit(reading, crops)
                    if back is not None:
                        finish(back)
                    else:
                        waiting[job] = (reading.number, full_key)
                    settle(self._reader.collect_pending(0.0))

            def settle(results) -> None:
                for job, reading in results:
                    number, full_key = waiting.pop(job)
                    if not reading.reread_report.get("reader_failed"):
                        self.cache.put_rereads(sha256, number, full_key, reading)
                    finish(reading)

            for number in range(1, total + 1):
                self._check(cancel)
                reading, task = self._prepare(document, path, sha256, number)
                if reading is not None:
                    deliver(reading)
                else:
                    todo.append(task)
            self._read_ocr_pages(path, sha256, todo, lambda reading, pid, seconds: deliver(reading), cancel)
            if waiting:
                settle(self._reader.drain(cancel))
                self._check(cancel)
        finally:
            document.close()
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise SourceChanged(f"{path.name} changed while it was being read", file=path.name)
        return DocumentReading(path, sha256, tuple(readings[n] for n in sorted(readings)))

    def _no_lexicon(self, error: LookupError) -> None:
        if not self._warned_lexicon:
            self._warned_lexicon = True
            languages = list(self.options.languages)
            self.log("guard.no_lexicon", languages=languages, error=str(error))
            self._emit(events.Warning("guard.warning.no_lexicon", {"language": ", ".join(languages)}))

    def _check(self, cancel) -> None:
        if cancel is not None and cancel.is_set():
            raise Cancelled("the read was cancelled")

    def _prepare(self, document, path: Path, sha256: str, number: int) -> tuple[PageReading | None, PageTask | None]:
        """The page's reading if the main process can give it (text layer, cache, a page that will not open), else
        the task for a worker."""
        try:
            page = document[number - 1]
        except pdfium.PdfiumError as error:
            return failed_reading(number, f"PdfiumError: {error}"), None
        try:
            dpi = page_dpi(page)
            try:
                native = read_native(page, number, self.lexicon, self.options.languages, dpi)
            except LookupError as error:
                # no word list for the languages asked: the text layer cannot be judged, so the page is read by OCR
                self._no_lexicon(error)
                native = None
            if native is not None:
                return native, None
            real_dpi = scan_dpi(page)
            if real_dpi is not None and real_dpi < MIN_DPI:
                # the page is read at 100 dpi as it always was; the user is told that its scan has less
                self._pool_log("page.warning.low_resolution", page=number, dpi=real_dpi)
            key = self._key(sha256, number, dpi)
            cached = self.cache.get(sha256, number, key)
            study = self.reader_settings.config if self.options.second_reading else None
            return (cached, None) if cached is not None else (None, PageTask(number, dpi, key, study=study))
        except pdfium.PdfiumError as error:  # a page pdfium cannot read is reported as unreadable, the document goes on
            return failed_reading(number, f"PdfiumError: {error}"), None
        finally:
            page.close()

    def _read_ocr_pages(self, path: Path, sha256: str, todo, on_page, cancel) -> None:
        if not todo:
            return
        chunks = chunks_of(str(path), sha256, todo)
        if not self.pool.run(chunks, on_page, cancel):
            raise Cancelled("the read was cancelled")

    def close(self) -> None:
        if self._reader is not None:
            self._reader.stop()
        self.pool.close()
