"""Writing the outputs: never over an existing file, under a partial name until each one has been checked."""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path

import openpyxl
import pikepdf

from ..engine.cache import file_sha256
from ..errors import Cancelled, OutputExists, SourceChanged
from ..types import DocumentData
from . import jsondoc
from .excel import write_excel
from .pdf import add_text_layer, unreadable_with_text
from .text import to_text

SUFFIXES = {"pdf": ".searchable.pdf", "xlsx": ".xlsx", "json": jsondoc.NAME_SUFFIX, "txt": ".txt"}
FORMATS = tuple(SUFFIXES)
MAX_COPIES = 1000


def _copy_name(folder: Path, stem: str, suffix: str, copy: int) -> Path:
    return folder / f"{stem}{suffix}" if copy == 1 else folder / f"{stem}-{copy}{suffix}"


def first_free_copy(folder: Path, stem: str, suffixes: Sequence[str]) -> int:
    """The first copy number (1 is the plain name, 2 is `-2`...) for which none of the files exists, so that the files
    of one export share their number: `name-2.xlsx` goes with `name-2.txt`, not with `name-3.txt`."""
    for copy in range(1, MAX_COPIES):
        if not any(_copy_name(folder, stem, suffix, copy).exists() for suffix in suffixes):
            return copy
    return MAX_COPIES


def claim(partial: Path, folder: Path, stem: str, suffix: str, first: int = 1) -> Path:
    """Move the checked partial file to `<stem><suffix>`, or to `<stem>-2<suffix>`, `<stem>-3<suffix>`... if that name
    is taken, starting from copy number `first`. The move fails rather than replace, so a file that appears meanwhile
    is never overwritten."""
    for copy in range(first, MAX_COPIES):
        target = _copy_name(folder, stem, suffix, copy)
        try:
            os.link(partial, target)
        except FileExistsError:
            continue
        except OSError:
            # no hard links on this file system: the check-then-rename has a small window, never an overwrite of a
            # file that already existed when the name was chosen
            if target.exists():
                continue
            os.replace(partial, target)
            return target
        partial.unlink()
        return target
    raise OutputExists(f"{stem}{suffix} and its {MAX_COPIES} copies already exist", file=f"{stem}{suffix}")


def _check_pdf(path: Path, data: DocumentData) -> None:
    """The file opens and has as many pages as the document. That the original pages are untouched is the job of
    `add_text_layer`, which never edits their streams (tested on the pages' content)."""
    with pikepdf.open(path) as pdf:
        if len(pdf.pages) != len(data.pages):
            raise ValueError(f"the PDF has {len(pdf.pages)} pages, the document {len(data.pages)}")


def _check_json(path: Path, data: DocumentData) -> None:
    if jsondoc.load(path) != jsondoc.exported(data):
        raise ValueError("the JSON does not read back as the document")


def _check_xlsx(path: Path, data: DocumentData) -> None:
    book = openpyxl.load_workbook(path)
    if "Numbers" not in book.sheetnames or ("Fields" in book.sheetnames) != (data.invoice is not None):
        raise ValueError("the workbook does not have the expected sheets")


def _check_txt(path: Path, data: DocumentData) -> None:
    if path.read_text(encoding="utf-8") != to_text(data):
        raise ValueError("the text does not read back as written")


def export(
    data: DocumentData,
    source: Path,
    destination: str | Path,
    formats: Sequence[str] = FORMATS,
    *,
    name: str | None = None,
    language: str = "en",
    cancel=None,
    on_written: Callable[[str, Path], None] | None = None,
    on_unreadable_text: Callable[[list[int]], None] | None = None,
) -> list[Path]:
    """Write the document in each format into `destination` and return the paths. Nothing is read again: the words,
    numbers and corrections come from `data`, and the engine is never loaded. A file is written as
    `.<name>.partial-<pid>.<ext>`, checked, and only then given its name. `on_unreadable_text` gets the pages marked
    unreadable whose few words went into the PDF text layer all the same."""
    destination = Path(destination)
    if destination.is_file():
        raise OutputExists(f"{destination.name} is a file, not a folder", file=destination.name)
    unknown = [f for f in formats if f not in SUFFIXES]
    if unknown:
        raise ValueError(f"unknown formats {unknown}; choose from {FORMATS}")
    source = Path(source)
    if "pdf" in formats and file_sha256(source) != data.sha256:
        raise SourceChanged(f"{source.name} is not the file this document was read from", file=source.name)
    destination.mkdir(parents=True, exist_ok=True)
    stem = name or Path(data.source).stem
    written: list[Path] = []
    first = first_free_copy(destination, stem, [SUFFIXES[kind] for kind in formats])
    for kind in formats:
        if cancel is not None and cancel.is_set():
            raise Cancelled("the export was cancelled")
        final = destination / f"{stem}{SUFFIXES[kind]}"
        partial = destination / f".{final.stem}.partial-{os.getpid()}{final.suffix}"
        try:
            if kind == "pdf":
                add_text_layer(source, data.pages, partial)
                pages = unreadable_with_text(data.pages)
                if pages and on_unreadable_text is not None:
                    on_unreadable_text(pages)
                _check_pdf(partial, data)
            elif kind == "json":
                jsondoc.save(data, partial)
                _check_json(partial, data)
            elif kind == "xlsx":
                write_excel(data, partial, language)
                _check_xlsx(partial, data)
            else:
                partial.write_text(to_text(data), encoding="utf-8")
                _check_txt(partial, data)
            path = claim(partial, destination, stem, SUFFIXES[kind], first)
        finally:
            partial.unlink(missing_ok=True)
        written.append(path)
        if on_written is not None:
            on_written(kind, path)
    return written
