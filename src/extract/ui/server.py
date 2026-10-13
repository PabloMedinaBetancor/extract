"""The local server of the interface.

Standard library only, on top of the public API `extract`. It listens on the loopback address at a random port; every
request needs the token of this run and a loopback `Host`. Texts are never sent: only i18n keys and their parameters.
"""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from collections import OrderedDict
from datetime import date
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

import extract
from extract import second_reading

PROTOCOL = 1
LONG_POLL_SECONDS = 20
# Seconds a scanned page takes. The long figure is a page full of text (measured in development); the short one is a
# page with little on it, which took 0.34 times as long in the readings of 04/10/2026 (the three sample PDFs against five
# catalogue pages, bench/results/ui-estimate-seconds.json). The screen says "about X to Y" because the real time
# depends on how much text the pages hold.
SECONDS_PER_SCANNED_PAGE = 3.3
SECONDS_PER_SCANNED_PAGE_LOW = 1.1
SECONDS_PER_DIGITAL_PAGE = 0.2
# The seconds above are those of the machine they were measured on. A page of average text took the mean of the two figures
# there; what this computer takes is learnt from its own reads (the work folder keeps it) and scales the estimate.
REFERENCE_PAGE_SECONDS = (SECONDS_PER_SCANNED_PAGE + SECONDS_PER_SCANNED_PAGE_LOW) / 2
SPEED_FACTOR_RANGE = (0.4, 8.0)
MIN_PAGES_FOR_SPEED = 3
CORRECT_MODES = ("off", "safe", "more")
# What the Import panel says about the second reading ("about X s more a page; fixes Y % more"): filled from the
# pilot's measurement (bench/results/reader-arms.json) when the reader is frozen; None until then.
SECOND_READING_COST: dict[str, Any] = second_reading.cost()
THOROUGH_COST: dict[str, Any] = second_reading.thorough_cost()
MAX_BODY = 1_000_000
STATIC = Path(__file__).with_name("static")
I18N = Path(__file__).with_name("i18n")
LANGUAGES = ("en", "es")
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
}
EVENT_NAMES = {
    "PageRead": "page.read",
    "Progress": "page.progress",
    "Warning": "job.warning",
    "Exported": "doc.exported",
}


class UiError(extract.ExtractError):
    code = "error.ui"


class NotFound(UiError):
    code = "error.not_found"


class OutsideRoot(UiError):
    code = "error.path_outside_home"


class BadRequest(UiError):
    code = "error.bad_request"


class Busy(UiError):
    code = "error.job_running"


class CannotOpen(UiError):
    code = "error.cannot_open_folder"


class CannotReadFolder(UiError):
    code = "error.cannot_read_folder"


class FileNotReadable(UiError):
    code = "error.file_not_readable"


class EditsWouldBeLost(UiError):
    code = "error.edits_would_be_lost"


class CannotWrite(UiError):
    code = "error.cannot_write"


class NameTooLong(UiError):
    code = "error.name_too_long"


class BadPattern(UiError):
    code = "error.bad_pattern"


def _cell(doc: extract.Document, cell: Any) -> dict[str, Any] | None:
    if cell is None:
        return None
    return {"text": cell.text, "word_ids": list(cell.word_ids), "page": cell.page, "status": doc.field_status(cell)}


def word_json(doc: extract.Document, word: extract.Word) -> dict[str, Any]:
    return {
        "corrected_by": word.corrected_by,
        "absorbed_by": word.absorbed_by,
        "fix": doc.fix_info(word.id),
        "id": word.id,
        "text": word.text,
        "box": list(word.box),
        "status": word.status,
        "conf": word.conf,
        "p_wrong": word.p_wrong,
        "reasons": [{"key": r.key, "params": dict(r.params)} for r in word.reasons],
        "suggestion": word.suggestion,
        "review": word.review,
        "original": word.original,
    }


def number_json(number: extract.Number) -> dict[str, Any]:
    fix = number.suggestion
    return {
        "word_ids": list(number.word_ids),
        "text": number.text,
        "value": None if number.value is None else str(number.value),
        "kind": number.kind,
        "status": number.status,
        "rules": [
            {"key": r.key, "params": dict(r.params), "result": r.result, "verifying": r.verifying} for r in number.rules
        ],
        "suggestion": None if fix is None else {"position": fix.position, "old": fix.old, "new": fix.new},
    }


def invoice_json(doc: extract.Document) -> dict[str, Any] | None:
    invoice = doc.invoice
    if invoice is None:
        return None
    return {
        "number": _cell(doc, invoice.number),
        "date": _cell(doc, invoice.date),
        "subtotal": _cell(doc, invoice.subtotal),
        "vat": [
            {"rate": _cell(doc, v.rate), "base": _cell(doc, v.base), "amount": _cell(doc, v.amount)}
            for v in invoice.vat
        ],
        "withholding_rate": _cell(doc, invoice.withholding_rate),
        "withholding": _cell(doc, invoice.withholding),
        "total": _cell(doc, invoice.total),
        "currency": invoice.currency,
        "lines": [
            {
                "description": _cell(doc, line.description),
                "quantity": _cell(doc, line.quantity),
                "unit_price": _cell(doc, line.unit_price),
                "amount": _cell(doc, line.amount),
            }
            for line in invoice.lines
        ],
    }


def open_mismatches(doc: extract.Document, reviewed: set[tuple[str, ...]]) -> list[extract.Number]:
    """The numbers whose rules fail and that nobody has accepted as they are."""
    return [n for n in doc.numbers if n.status == "mismatch" and n.word_ids and n.word_ids not in reviewed]


def pending(doc: extract.Document, reviewed: set[tuple[str, ...]] = frozenset()) -> list[dict[str, Any]]:
    """What is left to check, a number that does not add up first, then the doubtful words, the likeliest wrong first."""
    items = [
        {"id": n.word_ids[0], "page": _page_of(n.word_ids[0]), "status": "mismatch"}
        for n in open_mismatches(doc, reviewed)
    ]
    seen = {item["id"] for item in items}
    items += [{"id": w.id, "page": w.page, "status": w.status} for w in doc.doubtful() if w.id not in seen]
    return items


def _page_of(word_id: str) -> int:
    return int(re.match(r"p(\d+)w", word_id).group(1))


def doc_json(doc: extract.Document, reviewed: set[tuple[str, ...]] = frozenset()) -> dict[str, Any]:
    todo = pending(doc, reviewed)
    fixed = [{"id": w.id, "page": w.page, "status": w.status} for w in doc.fixed()]
    fixed_per_page: dict[int, int] = {}
    for item in fixed:
        fixed_per_page[item["page"]] = fixed_per_page.get(item["page"], 0) + 1
    per_page: dict[int, int] = {}
    for item in todo:
        per_page[item["page"]] = per_page.get(item["page"], 0) + 1
    return {
        "id": doc.doc_id,
        "name": doc.name,
        "status": doc.status,
        "doc_type": doc.doc_type,
        "locale": doc.locale,
        "unreadable": [
            {"page": p.number, "reason": p.reason, "error": p.error} for p in doc.pages if p.status == "unreadable"
        ],
        "pages": [
            {
                "number": p.number,
                "width": p.width_px,
                "height": p.height_px,
                "status": p.status,
                "reason": p.reason,
                "words": len(p.words),
                "todo": per_page.get(p.number, 0),
                "fixed": fixed_per_page.get(p.number, 0),
            }
            for p in doc.pages
        ],
        "doubtful": todo,
        "fixed": fixed,
        "mode": doc.mode,
        "fixes": doc.fixes,
        "notes": doc.page_notes,
        "numbers": [number_json(n) for n in doc.numbers],
        "numbers_checked": sum(1 for n in doc.numbers if n.status == "checked"),
        "invoice": invoice_json(doc),
    }


def flagged_ids(doc: extract.Document, reviewed: set[tuple[str, ...]]) -> list[str]:
    return [i for n in open_mismatches(doc, reviewed) for i in n.word_ids]


def changes(doc: extract.Document, word_id: str, reviewed: set[tuple[str, ...]]) -> dict[str, Any]:
    full = doc_json(doc, reviewed)
    return {
        "word": word_json(doc, doc.word(word_id)),
        "flagged": flagged_ids(doc, reviewed),
        "doubtful": full["doubtful"],
        "fixed": full["fixed"],
        "fixes": full["fixes"],
        "pages": full["pages"],
        "numbers": full["numbers"],
        "numbers_checked": full["numbers_checked"],
        "invoice": full["invoice"],
    }


class State:
    """Everything one run of the server holds: the token, the open documents, the running job and its events."""

    def __init__(self, root: Path, work_dir: Path, lang: str | None, reader_factory: Any = None) -> None:
        self.reader_factory = reader_factory or extract.Reader
        self.token = secrets.token_urlsafe(24)
        self.root = root.resolve()
        self.work_dir = Path(work_dir).resolve()
        self.lang = lang
        self.port = 0
        self.docs: dict[str, extract.Document] = {}
        self.sources: dict[str, Path | None] = {}  # the file each open document was read from
        self.reviewed: dict[str, set[tuple[str, ...]]] = {}
        self.order: list[str] = []
        self.events: list[dict[str, Any]] = []
        self.changed = threading.Condition()
        self.job: threading.Thread | None = None
        self.cancel: extract.CancelToken | None = None
        self.lock = threading.RLock()
        self.images: OrderedDict[tuple[str, int, float], bytes] = OrderedDict()
        self.speed_factor, self.speed_measured = self._load_speed()

    def _load_speed(self) -> tuple[float, bool]:
        try:
            factor = float(json.loads((self.work_dir / "speed.json").read_text(encoding="utf-8"))["factor"])
        except (OSError, ValueError, KeyError, TypeError):
            return 1.0, False
        low, high = SPEED_FACTOR_RANGE
        return (factor, True) if low <= factor <= high else (1.0, False)

    def learn_speed(self, entries: list[dict[str, Any]]) -> None:
        """Keep what a read says about how fast this computer is (averaged with what earlier reads said)."""
        factor = measured_speed_factor(entries)
        if factor is None:
            return
        with self.lock:
            self.speed_factor = (self.speed_factor + factor) / 2 if self.speed_measured else factor
            self.speed_measured = True
            try:
                self.work_dir.mkdir(parents=True, exist_ok=True)
                (self.work_dir / "speed.json").write_text(json.dumps({"factor": round(self.speed_factor, 3)}), "utf-8")
            except OSError:
                pass  # the estimate is only a courtesy: a folder that cannot be written does not stop a read

    def options(self, payload: dict[str, Any] | None = None) -> extract.Options:
        payload = payload or {}
        languages = tuple(payload.get("languages") or extract.Options().languages)
        try:
            return extract.Options(
                languages=languages,
                review=payload.get("review", "balanced"),
                work_dir=self.work_dir,
                correct=payload.get("correct", "off"),
                second_reading=payload.get("second_reading", second_reading.default_on()),
            )
        except ValueError as error:
            raise BadRequest(str(error)) from error

    def path(self, raw: str | None, *, must_exist: bool = True) -> Path:
        """A path typed or pasted by the user, inside the home folder: spaces and the quotes that "copy as path" adds
        around it are dropped, a relative one starts from the home folder (not from where the server was started), and
        one the system cannot even look up (too long, `~user` of nobody) is "not found", never an internal error."""
        text = clean_path(raw)
        try:
            path = Path(text or str(self.root)).expanduser()
            path = (path if path.is_absolute() else self.root / path).resolve()
            outside = path != self.root and self.root not in path.parents
            exists = outside or not must_exist or path.exists()
        except (OSError, RuntimeError) as error:
            raise NotFound(text, path=text) from error
        if outside:
            raise OutsideRoot(str(path), path=str(path))
        if not exists:
            raise NotFound(str(path), path=str(path))
        return path

    def emit(self, kind: str, /, **fields: Any) -> None:
        with self.changed:
            self.events.append({"n": len(self.events) + 1, "event": kind, "time": time.time(), **fields})
            self.changed.notify_all()

    def wait(self, after: int, seconds: float = LONG_POLL_SECONDS) -> list[dict[str, Any]]:
        with self.changed:
            self.changed.wait_for(lambda: len(self.events) > after, timeout=seconds)
            return self.events[after:]

    def add(self, doc: extract.Document, source: Path | None = None) -> None:
        with self.lock:
            if doc.doc_id not in self.docs:
                self.order.append(doc.doc_id)
            self.docs[doc.doc_id] = doc
            self.sources[doc.doc_id] = source if source is not None else doc.source
            self.reviewed[doc.doc_id] = set()  # a new reading starts with nothing accepted

    def same_content_as(self, doc: extract.Document, path: Path) -> extract.Document | None:
        """The open document that has exactly the content of `doc` but was read from another file: the interface keys
        its documents by that content, so a second copy of a file would silently take the first one's place."""
        with self.lock:
            earlier = self.docs.get(doc.doc_id)
            if earlier is not None and self.sources.get(doc.doc_id) not in (None, path):
                return earlier
        return None

    def doc(self, doc_id: str) -> extract.Document:
        try:
            return self.docs[doc_id]
        except KeyError:
            raise NotFound(doc_id) from None

    def edited_among(self, paths: list[Path]) -> list[str]:
        """The names of the files that are open as documents the user has changed (a word accepted or typed over, a
        number accepted): reading one again replaces the document and those decisions are gone."""
        edited = {i for i, doc in self.docs.items() if self.reviewed.get(i) or any(w.review for w in doc.words())}
        if not edited:
            return []  # nothing the user did is open: no file needs to be read to know it
        names = []
        for path in paths:
            try:
                digest = hashlib.sha256()
                with open(path, "rb") as handle:
                    for block in iter(lambda: handle.read(1 << 20), b""):
                        digest.update(block)
            except OSError:
                continue  # a file that cannot be read is reported by the read itself
            if digest.hexdigest()[:12] in edited:
                names.append(path.name)
        return names

    def start(self, raw_paths: list[str], payload: dict[str, Any]) -> None:
        with self.lock:
            if self.job is not None and self.job.is_alive():
                raise Busy()
            paths = [self.path(p) for p in raw_paths]
            if not paths:
                raise BadRequest("no files")
            at_risk = [] if payload.get("replace") else self.edited_among(paths)
            if at_risk:
                raise EditsWouldBeLost(", ".join(at_risk), names=at_risk)
            options = self.options(payload.get("options"))
            self.cancel = extract.CancelToken()
            self.job = threading.Thread(target=self._read, args=(paths, options, self.cancel), daemon=True)
            self.job.start()

    def _read(self, paths: list[Path], options: extract.Options, cancel: extract.CancelToken) -> None:
        infos = []
        for index, path in enumerate(paths):
            try:
                infos.append(inspect_readable(path))
            except (extract.ExtractError, UiError) as error:
                self.emit("job.error", **wire_error(error), path=str(path), index=index)
                infos.append(None)
            except Exception as error:  # the read must always end with job.finished, whatever the file system says
                self.emit(
                    "job.error",
                    code="error.internal",
                    params={"detail": f"{type(error).__name__}: {error}"},
                    path=str(path),
                    index=index,
                )
                infos.append(None)
        self.emit(
            "job.started", documents=len(paths), pages=sum(i.pages for i in infos if i), names=[p.name for p in paths]
        )
        status = "done"
        index = 0

        def forward(event: Any) -> None:
            name = type(event).__name__
            if name in EVENT_NAMES:
                body = event.to_dict()
                body.pop("event")
                self.emit(EVENT_NAMES[name], index=index, **body)

        try:
            with self.reader_factory(options) as reader:
                for index, path in enumerate(paths):
                    if infos[index] is None:
                        continue
                    self.emit("doc.begin", index=index, path=str(path), name=path.name, pages=infos[index].pages)
                    try:
                        doc = reader.read(path, forward, cancel)
                    except extract.Cancelled:
                        status = "stopped"
                        break
                    except extract.ExtractError as error:
                        self.emit("job.error", **wire_error(error), path=str(path), index=index)
                        continue
                    earlier = self.same_content_as(doc, path)
                    if earlier is None:
                        self.add(doc, path)
                    else:
                        self.emit(
                            "job.warning",
                            code="job.warning.same_content",
                            params={"name": path.name, "same_as": earlier.name},
                            index=index,
                        )
                        doc = earlier
                    self.emit(
                        "doc.read",
                        index=index,
                        doc_id=doc.doc_id,
                        name=doc.name,
                        status=doc.status,
                        doubtful=len(doc.doubtful()),
                        fixed=doc.fixes["applied"],
                        numbers_checked=sum(1 for n in doc.numbers if n.status == "checked"),
                        unreadable=list(doc.unreadable_pages),
                    )
                self.learn_speed(list(getattr(reader, "job_log", ())))
        except extract.ExtractError as error:
            self.emit("job.error", **wire_error(error))
            status = "failed"
        except Exception as error:
            if isinstance(error, OSError) and _inside(error.filename, self.work_dir):
                # the work folder (the cache and the logs) cannot be written: say which one
                self.emit("job.error", code="error.cannot_write_work", params={"path": str(self.work_dir)})
            else:
                self.emit("job.error", code="error.internal", params={"detail": f"{type(error).__name__}: {error}"})
            status = "failed"
        self.emit("job.finished", status=status)

    def page_png(self, doc_id: str, number: int, zoom: float) -> bytes:
        key = (doc_id, number, zoom)
        with self.lock:
            if key in self.images:
                self.images.move_to_end(key)
                return self.images[key]
        data = self.doc(doc_id).page_image(number, zoom)
        with self.lock:
            self.images[key] = data
            while len(self.images) > 6:
                self.images.popitem(last=False)
        return data


def measured_speed_factor(entries: list[dict[str, Any]]) -> float | None:
    """How many times slower (or faster) this computer read than the machine the estimate was measured on, from the job
    log of one read: the median seconds a worker took per page, shared among the workers; None when fewer than three
    pages went through the OCR (a cached page costs nothing and says nothing)."""
    pages = [e for e in entries if e.get("code") == "pool.page" and e.get("seconds")]
    if len(pages) < MIN_PAGES_FOR_SPEED:
        return None
    seconds = sorted(float(e["seconds"]) for e in pages)
    median = seconds[len(seconds) // 2]
    workers = max(1, len({e.get("pid") for e in pages}))
    low, high = SPEED_FACTOR_RANGE
    return min(max(median / workers / REFERENCE_PAGE_SECONDS, low), high)


def clean_path(raw: str | None) -> str:
    """The text of a folder or file box without the spaces around it or one pair of quotes around the whole path."""
    text = (raw or "").strip()
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        text = text[1:-1].strip()
    return text


def expand_pattern(pattern: str, name: str) -> str:
    stem = Path(name).stem
    result = pattern.replace("{name}", stem).replace("{date}", date.today().isoformat()).strip()
    if not result or "/" in result or "\\" in result or result.startswith("."):
        raise BadPattern("name pattern", pattern=pattern)
    return result


def _inside(filename: Any, folder: Path) -> bool:
    """Whether an OSError names `folder` or something under it."""
    if not filename:
        return False
    try:
        path = Path(os.fsdecode(filename)).resolve()
    except (TypeError, ValueError, OSError):
        return False
    return path == folder or folder in path.parents


def write_error(error: OSError, destination: Path) -> dict[str, Any]:
    """What a screen says when the files could not be written: the folder, never the name of a partial file."""
    if error.errno == errno.ENAMETOOLONG:
        return {"code": NameTooLong.code, "params": {}}
    return {"code": CannotWrite.code, "params": {"path": str(destination)}}


def wire_error(error: extract.ExtractError) -> dict[str, Any]:
    """The code and params a screen shows; a missing thorough guard model has its own text, not the generic load error."""
    if error.params.get("cause") == "thorough_model_missing":
        return {"code": "error.thorough_model_missing", "params": {"path": error.params.get("path")}}
    if isinstance(error, extract.EngineLoadError) and "detail" not in error.params:
        return {"code": error.code, "params": {**error.params, "detail": str(error)}}  # which file or list is missing
    return {"code": error.code, "params": error.params}


def open_folder(path: Path) -> None:
    """Show a folder in the file manager of the computer that runs the server. No shell is involved."""
    if sys.platform == "win32":
        os.startfile(str(path))
        return
    opener = "open" if sys.platform == "darwin" else "xdg-open"
    if shutil.which(opener) is None:
        raise CannotOpen(str(path), path=str(path))
    process = subprocess.Popen(
        [opener, str(path)], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    threading.Thread(target=process.wait, daemon=True).start()


def thorough_available() -> bool:
    """The thorough mode needs its guard model (it travels in the wheel) and the reader models (`fetch_models.py --all`)."""
    return second_reading.thorough_installed()


class Handler(BaseHTTPRequestHandler):
    server_version = "ExtractUI"
    state: State

    def log_message(self, *args: Any) -> None:
        pass

    def _send(self, status: int, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps({"v": PROTOCOL, **payload}, ensure_ascii=False, default=str).encode("utf-8")
        self._send(status, body, "application/json; charset=utf-8")

    def _error(self, error: extract.ExtractError, status: int) -> None:
        self._json({"error": wire_error(error)}, status)

    def _allowed(self, query: dict[str, list[str]]) -> bool:
        port = self.state.port
        host = self.headers.get("Host", "")
        if host not in (f"127.0.0.1:{port}", f"localhost:{port}"):
            self._json({"error": {"code": "error.forbidden_host", "params": {}}}, HTTPStatus.FORBIDDEN)
            return False
        given = self.headers.get("X-Token") or (query.get("t") or [""])[0]
        if not hmac.compare_digest(given.encode(), self.state.token.encode()):
            self._json({"error": {"code": "error.forbidden_token", "params": {}}}, HTTPStatus.FORBIDDEN)
            return False
        return True

    def _dispatch(self, method: str) -> None:
        url = urlsplit(self.path)
        query = parse_qs(url.query)
        if not self._allowed(query):
            return
        body: dict[str, Any] = {}
        if method == "POST":
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                length = -1
            if not 0 <= length <= MAX_BODY:
                return self._json({"error": {"code": "error.bad_request", "params": {}}}, HTTPStatus.BAD_REQUEST)
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw) if raw else {}
            except ValueError:
                return self._json({"error": {"code": "error.bad_request", "params": {}}}, HTTPStatus.BAD_REQUEST)
        route = unquote(url.path)
        try:
            for pattern, verb, function in ROUTES:
                match = pattern.fullmatch(route)
                if match and verb == method:
                    return function(self, match, query, body)
            if method == "GET" and route in ("/", "/index.html"):
                return self._static("index.html")
            if method == "GET" and route.startswith("/static/"):
                return self._static(route[len("/static/") :])
            raise NotFound(route)
        except NotFound as error:
            self._error(error, HTTPStatus.NOT_FOUND)
        except OutsideRoot as error:
            self._error(error, HTTPStatus.FORBIDDEN)
        except extract.ExtractError as error:
            self._error(error, HTTPStatus.BAD_REQUEST)
        except (KeyError, ValueError, TypeError) as error:
            self._json(
                {"error": {"code": "error.bad_request", "params": {"detail": str(error)}}}, HTTPStatus.BAD_REQUEST
            )
        except Exception as error:
            detail = f"{type(error).__name__}: {error}"
            self._json(
                {"error": {"code": "error.internal", "params": {"detail": detail}}}, HTTPStatus.INTERNAL_SERVER_ERROR
            )

    def do_GET(self) -> None:
        self._dispatch("GET")

    def do_POST(self) -> None:
        self._dispatch("POST")

    def _static(self, name: str) -> None:
        path = (STATIC / name).resolve()
        if STATIC.resolve() not in path.parents or not path.is_file():
            raise NotFound(name)
        content = path.read_bytes()
        if name == "index.html":
            content = content.replace(b"{{token}}", self.state.token.encode())
        self._send(200, content, TYPES.get(path.suffix, "application/octet-stream"))


def _first(query: dict[str, list[str]], name: str, default: str | None = None) -> str | None:
    return (query.get(name) or [default])[0]


def api_start(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    paths = body.get("paths", [])
    if not isinstance(paths, list) or not all(isinstance(p, str) for p in paths):
        raise BadRequest("paths must be a list of file paths")
    if not isinstance(body.get("options", {}), dict):
        raise BadRequest("options must be an object")
    h.state.start(paths, body)
    h._json({"started": True})


def api_events(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    after = int(_first(q, "after", "0"))
    h._json({"events": h.state.wait(after, min(LONG_POLL_SECONDS, float(_first(q, "wait", str(LONG_POLL_SECONDS)))))})


def api_stop(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    if h.state.cancel is not None:
        h.state.cancel.cancel()
    h._json({"stopped": True})


def api_defaults(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    state = h.state
    h._json(
        {
            "root": str(state.root),
            "work_dir": str(state.work_dir),
            "export_dir": str(state.work_dir / "extracted"),
            "seconds_per_scanned_page": SECONDS_PER_SCANNED_PAGE,
            "seconds_per_scanned_page_low": SECONDS_PER_SCANNED_PAGE_LOW,
            "seconds_per_digital_page": SECONDS_PER_DIGITAL_PAGE,
            "speed_factor": round(state.speed_factor, 3),
            "speed_measured": state.speed_measured,
            "lang": state.lang,
            "docs": state.order,
            "formats": ["pdf", "xlsx", "json", "txt"],
            "correct": "off",
            "second_reading": second_reading.default_on(),
            "modes": list(CORRECT_MODES),
            "second_reading_cost": SECOND_READING_COST,
            "second_reading_thorough_cost": THOROUGH_COST,
            "thorough_available": thorough_available(),
        }
    )


def listing(folder: Path) -> tuple[list[str], list[dict[str, Any]]]:
    """The sub-folders and the PDFs of a folder. A folder that cannot be read is one clear error; an entry that cannot be
    looked at (a link to nothing, a file that went away) is left out instead of hiding the whole folder."""
    try:
        entries = sorted(folder.iterdir(), key=lambda p: p.name.lower())
    except OSError as error:
        raise CannotReadFolder(str(folder), path=str(folder)) from error
    dirs, pdfs = [], []
    for entry in entries:
        if entry.name.startswith("."):
            continue
        try:
            if entry.is_dir():
                dirs.append(entry.name)
            elif entry.suffix.lower() == ".pdf":
                pdfs.append({"name": entry.name, "size": entry.stat().st_size})
        except OSError:
            continue
    return dirs, pdfs


def api_browse(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    state = h.state
    path = state.path(_first(q, "path"))
    folder = path if path.is_dir() else path.parent
    dirs, pdfs = listing(folder)
    parent = None if folder == state.root else str(folder.parent)
    h._json({"path": str(folder), "parent": parent, "dirs": dirs, "pdfs": pdfs})


def inspect_json(path: Path, info: Any) -> dict[str, Any]:
    return {
        "path": str(path),
        "name": path.name,
        "pages": info.pages,
        "kind": info.kind,
        "size": info.size,
        "scanned_pages": info.scanned_pages,
        "digital_pages": info.digital_pages,
    }


def inspect_readable(path: Path) -> Any:
    """`extract.inspect` with the file system's own reasons said as such: a file this account may not read is not
    "a PDF that cannot be read", and a folder or a file that went away is not an internal error."""
    if not os.access(path, os.R_OK) or not path.is_file():
        raise FileNotReadable(path.name, file=path.name)
    try:
        return extract.inspect(path)
    except OSError as error:
        raise FileNotReadable(path.name, file=path.name) from error


def api_inspect(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    path = h.state.path(_first(q, "path"))
    h._json(inspect_json(path, inspect_readable(path)))


def api_inspect_many(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    """Look at several files at once; every file that cannot be added comes back with its own name and reason."""
    files, skipped = [], []
    for raw in body.get("paths", []):
        name = Path(str(raw)).name
        try:
            path = h.state.path(str(raw))
            files.append(inspect_json(path, inspect_readable(path)))
        except (UiError, extract.ExtractError) as error:
            skipped.append({"path": str(raw), "name": name, **wire_error(error)})
    h._json({"files": files, "skipped": skipped})


def api_docs(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    h._json({"docs": [doc_json(h.state.doc(i), h.state.reviewed[i]) for i in h.state.order]})


def api_doc(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    h._json({"doc": doc_json(h.state.doc(m.group(1)), h.state.reviewed[m.group(1)])})


def api_words(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    doc = h.state.doc(m.group(1))
    number = int(m.group(2))
    if not 1 <= number <= len(doc.pages):
        raise NotFound(f"page {number}")
    page = doc.pages[number - 1]
    flagged = flagged_ids(doc, h.state.reviewed[doc.doc_id])
    h._json(
        {
            "page": number,
            "width": page.width_px,
            "height": page.height_px,
            "words": [word_json(doc, w) for w in page.words],
            "flagged": [i for i in flagged if _page_of(i) == number],
        }
    )


def api_page_png(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    zoom = min(3.0, max(0.25, float(_first(q, "zoom", "1"))))
    number = int(m.group(2))
    if not 1 <= number <= len(h.state.doc(m.group(1)).pages):
        raise NotFound(f"page {number}")
    h._send(200, h.state.page_png(m.group(1), number, zoom), "image/png")


def api_crop_png(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    h._send(200, h.state.doc(m.group(1)).word_crop(m.group(2)), "image/png")


def _act(h: Handler, body: dict, action: str) -> None:
    doc = h.state.doc(str(body["doc"]))
    word = str(body["word"])
    with h.state.lock:
        if action == "accept":
            doc.accept(word)
            for number in doc.numbers:
                if number.status == "mismatch" and word in number.word_ids:
                    h.state.reviewed[doc.doc_id].add(number.word_ids)
        elif action == "use_suggestion":
            doc.use_suggestion(word)
        elif action == "reopen":
            doc.reopen(word)
            reviewed = h.state.reviewed[doc.doc_id]
            for key in [k for k in reviewed if word in k]:
                reviewed.discard(key)
        else:
            text = str(body["text"])
            if not text.strip():
                raise BadRequest("empty text")
            doc.correct(word, text)
        h._json(changes(doc, word, h.state.reviewed[doc.doc_id]))


def _full(doc: extract.Document, reviewed: set[tuple[str, ...]], **extra: Any) -> dict[str, Any]:
    return {**changes_all(doc, reviewed), **extra}


def changes_all(doc: extract.Document, reviewed: set[tuple[str, ...]]) -> dict[str, Any]:
    full = doc_json(doc, reviewed)
    return {
        "flagged": flagged_ids(doc, reviewed),
        "doubtful": full["doubtful"],
        "fixed": full["fixed"],
        "fixes": full["fixes"],
        "mode": full["mode"],
        "pages": full["pages"],
        "numbers": full["numbers"],
        "numbers_checked": full["numbers_checked"],
        "invoice": full["invoice"],
    }


def api_undo(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    doc = h.state.doc(str(body["doc"]))
    word = str(body["word"])
    with h.state.lock:
        doc.undo(word)
        h._json(changes(doc, word, h.state.reviewed[doc.doc_id]))


def api_undo_page(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    doc = h.state.doc(str(body["doc"]))
    number = int(body["page"])
    with h.state.lock:
        undone = doc.undo_page(number)
        page = doc.pages[number - 1]
        h._json(
            _full(
                doc,
                h.state.reviewed[doc.doc_id],
                page=number,
                undone=undone,
                words=[word_json(doc, w) for w in page.words],
            )
        )


def api_mode(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    doc = h.state.doc(str(body["doc"]))
    mode = str(body["mode"])
    if mode not in CORRECT_MODES:
        raise BadRequest(f"mode must be one of {CORRECT_MODES}")
    with h.state.lock:
        doc.set_mode(mode)
        h._json(_full(doc, h.state.reviewed[doc.doc_id], withheld=doc.fixes["withheld"]))


def api_export(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    state = h.state
    ids = [str(i) for i in body.get("docs", [])]
    formats = tuple(body.get("formats") or ())
    if not ids or not formats:
        raise BadRequest("nothing to export")
    destination = state.path(body.get("destination") or str(state.work_dir / "extracted"), must_exist=False)
    pattern = str(body["pattern"]) if body.get("pattern") is not None else "{name}"
    language = body.get("language") if body.get("language") in LANGUAGES else "en"
    expand_pattern(pattern, "name.pdf")
    written, failed = {}, {}
    for doc_id in ids:
        doc = state.doc(doc_id)
        try:
            paths = doc.export(destination, formats, name=expand_pattern(pattern, doc.name), language=language)
        except extract.ExtractError as error:
            failed[doc_id] = {"code": error.code, "params": error.params}
            continue
        except OSError as error:
            failed[doc_id] = write_error(error, destination)
            continue
        written[doc_id] = [str(p) for p in paths]
        state.emit("doc.exported", doc_id=doc_id, paths=written[doc_id])
    h._json({"written": written, "failed": failed, "destination": str(destination)})


def api_open_folder(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    path = h.state.path(str(body.get("path") or ""))
    if not path.is_dir():
        raise NotFound(str(path), path=str(path))
    open_folder(path)
    h._json({"opened": str(path)})


def api_i18n(h: Handler, m: re.Match, q: dict, body: dict) -> None:
    lang = m.group(1)
    if lang not in LANGUAGES:
        raise NotFound(lang)
    h._send(
        200,
        json.dumps({"v": PROTOCOL, "strings": json.loads((I18N / f"{lang}.json").read_text("utf-8"))}).encode(),
        "application/json",
    )


ROUTES = [
    (re.compile(r"/api/read"), "POST", api_start),
    (re.compile(r"/api/events"), "GET", api_events),
    (re.compile(r"/api/stop"), "POST", api_stop),
    (re.compile(r"/api/defaults"), "GET", api_defaults),
    (re.compile(r"/api/browse"), "GET", api_browse),
    (re.compile(r"/api/inspect"), "GET", api_inspect),
    (re.compile(r"/api/inspect_many"), "POST", api_inspect_many),
    (re.compile(r"/api/docs"), "GET", api_docs),
    (re.compile(r"/api/doc/([0-9a-f]+)"), "GET", api_doc),
    (re.compile(r"/api/words/([0-9a-f]+)/(\d+)"), "GET", api_words),
    (re.compile(r"/api/page/([0-9a-f]+)/(\d+)\.png"), "GET", api_page_png),
    (re.compile(r"/api/crop/([0-9a-f]+)/(p\d+w\d+)\.png"), "GET", api_crop_png),
    (re.compile(r"/api/accept"), "POST", lambda h, m, q, b: _act(h, b, "accept")),
    (re.compile(r"/api/correct"), "POST", lambda h, m, q, b: _act(h, b, "correct")),
    (re.compile(r"/api/use_suggestion"), "POST", lambda h, m, q, b: _act(h, b, "use_suggestion")),
    (re.compile(r"/api/reopen"), "POST", lambda h, m, q, b: _act(h, b, "reopen")),
    (re.compile(r"/api/undo"), "POST", api_undo),
    (re.compile(r"/api/undo_page"), "POST", api_undo_page),
    (re.compile(r"/api/mode"), "POST", api_mode),
    (re.compile(r"/api/export"), "POST", api_export),
    (re.compile(r"/api/open_folder"), "POST", api_open_folder),
    (re.compile(r"/api/i18n/(\w+)\.json"), "GET", api_i18n),
]


def make_server(
    work_dir: Path | None = None,
    preload: list[Path] | None = None,
    root: Path | None = None,
    lang: str | None = None,
    reader_factory: Any = None,
) -> tuple[ThreadingHTTPServer, State]:
    """A server bound to a random loopback port, with `preload` saved documents already open. Not started yet."""
    state = State(root or Path.home(), work_dir or Path.home() / ".cache" / "extract", lang, reader_factory)
    for saved in preload or []:
        state.add(extract.load(saved, options=extract.Options(work_dir=state.work_dir)))
    handler = type("BoundHandler", (Handler,), {"state": state})
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    state.port = server.server_address[1]
    return server, state


def serve(
    work_dir: Path | None = None,
    preload: list[Path] | None = None,
    root: Path | None = None,
    browser: bool = True,
    lang: str | None = None,
) -> None:
    """Run the interface until interrupted: print the address and, unless `browser` is false, open it."""
    server, state = make_server(work_dir, preload, root, lang)
    url = f"http://127.0.0.1:{state.port}/?t={state.token}"
    print(url, flush=True)
    if browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
