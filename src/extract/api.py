"""The public API: read PDFs, review and correct the words, export. The interface uses this and nothing else."""

from __future__ import annotations

import io
import json
import os
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import pypdfium2 as pdfium
from PIL import Image

from . import events, fixes, pipeline, second_reading
from .engine import ocr
from .engine.batch import BatchReader, open_pdf, resolve_work_dir
from .engine.cache import file_sha256
from .engine.native import NATIVE_MIN_CHARS, page_text
from .engine.pool import leave_if_started_by_an_unguarded_script
from .engine.render import render_page, scan_pixels
from .engine.worker import DEFAULT_FACTORY
from .errors import Cancelled, EngineLoadError, ModelMismatch, SourceChanged
from .options import CORRECT_MODES, Options
from .outputs import export as exporter
from .outputs import jsondoc
from .types import Cell, DocumentData, InvoiceFields, NumberResult, PageReading, Word, base_id

PDFIUM_LOCK = threading.Lock()
PAGE_NOTES = frozenset({"page.warning.low_resolution", "page.warning.sideways"})


class CancelToken:
    """A switch that stops a running `read` or `export` from another thread, or from its own event callback.

    Create one, pass it as `cancel=`, and call `cancel()` when the job should stop. A read stops within about two
    seconds, raises `Cancelled` and sends `Finished("stopped")`. Pages that were already read stay in the page cache,
    so reading the same file again starts where this read stopped. A token cannot be reset: make a new one for the
    next job.

    Example:
        ```python
        import threading

        import extract

        if __name__ == "__main__":
            token = extract.CancelToken()
            threading.Timer(0.5, token.cancel).start()
            try:
                extract.read("invoice.pdf", extract.Options(work_dir="cancel-work"), cancel=token)
            except extract.Cancelled:
                print("stopped")
        ```

        ```text
        # output
        stopped
        ```

    See also:
        `read`, `Reader.read`, `Document.export`, `Cancelled`.
    """

    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        """Ask the job that holds this token to stop.

        Safe to call from any thread, and more than once. The job notices within about two seconds; it does not
        return anything here.
        """
        self._event.set()

    def is_set(self) -> bool:
        """Return True once `cancel` has been called.

        Returns:
            Whether the token has been cancelled.
        """
        return self._event.is_set()


@dataclass(frozen=True)
class FileInfo:
    """What `inspect` learns about a PDF without reading any page with OCR.

    Attributes:
        path: The path that was inspected, as a `Path`.
        pages: Number of pages.
        kind: `digital` when every page has a text layer and no scanned image, `scanned` when none does, `mixed`
            otherwise. See `inspect` for the exact rule.
        size: Size of the file in bytes.
        scanned_pages: Pages that count as scanned and would need OCR.
        digital_pages: Pages that have a text layer long enough to be used as it is. `scanned_pages` plus
            `digital_pages` is always `pages`.
    """

    path: Path
    pages: int
    kind: str
    size: int
    scanned_pages: int
    digital_pages: int


def inspect(path: str | Path) -> FileInfo:
    """Say how many pages a PDF has and whether it is digital, scanned or mixed, without reading it.

    It opens the PDF, looks at each page's text layer and images, and closes it, with no OCR and no model. A page
    counts as digital when its text layer has at least 80 characters that are not spaces and no image covers half the
    page or more; any other page, a blank one included, counts as scanned. The quality of a text layer is judged later,
    when the document is read.

    Args:
        path: The PDF, as a string or a `Path`.

    Returns:
        A `FileInfo` with the page counts, the kind and the file size.

    Raises:
        FileNotFoundError: `path` does not exist. This is the built-in exception, not an `ExtractError`.
        UnsupportedFile: The file is not a PDF that can be opened. The error carries `file=`.
        PasswordProtected: The PDF needs a password to be opened. The error carries `file=`. A PDF that has only an
            owner password (restrictions on printing or copying) opens normally.

    Example:
        ```python
        import extract

        info = extract.inspect("scans/damaged.pdf")
        print(info.kind, info.pages, info.size)
        ```

        ```text
        # output
        scanned 3 41019
        ```

    See also:
        `FileInfo`, `read`.
    """
    path = Path(path)
    document = open_pdf(path)
    try:
        digital = 0
        total = len(document)
        for number in range(total):
            page = document[number]
            try:
                with PDFIUM_LOCK:
                    long_enough = len("".join(page_text(page).split())) >= NATIVE_MIN_CHARS
                    scan = scan_pixels(page) > 0
            finally:
                page.close()
            digital += 1 if long_enough and not scan else 0
    finally:
        document.close()
    kind = "digital" if digital == total else "scanned" if digital == 0 else "mixed"
    return FileInfo(path, total, kind, path.stat().st_size, total - digital, digital)


class Document:
    """A document that has been read: its pages and words with their status, its numbers and its invoice fields.

    You do not build a `Document`. `read` and `Reader.read` return one after reading a PDF, and `load` returns one
    from a saved reading (a `.extract.json` file).

    What you read from it (`pages`, `numbers`, `invoice`, `word`) is a snapshot of the present state. What you change
    (`correct`, `accept`, `use_suggestion`, `reopen`, `set_mode`, `undo`) makes a new snapshot and runs the number
    rules that the change can affect again. The words themselves are immutable objects: after a change, ask for the word
    again, because a `Word` you already hold keeps its old values. Every method that takes a word id, such as `"p1w13"`,
    raises `KeyError` when no word of the document has that id.

    Two documents are equal (`==`) when the JSON they would save is the same: pages, words, numbers, invoice, options and
    corrections. A document loaded from a save of another is equal to it. A `Document` is not hashable.

    Note:
        The methods that change the document hold a lock, so changes from several threads never interleave. The reading
        methods take no lock: do not read from one thread while another changes the same document. Every `correct`,
        `accept`, `use_suggestion`, `reopen`, `undo` and `set_mode` appends one line to `corrections.jsonl` in the work
        folder (`Options.work_dir`; by default `$XDG_CACHE_HOME/extract`, else `~/.cache/extract`); `undo_page` and
        `undo_all` write nothing there. A document from `load` writes there too: pass `Options(work_dir=...)` to `load`
        to choose the folder. Nothing is written for a change that raises.

    Attributes:
        source: The PDF the document was read from, as a `Path`, or `None` for a document from `load` whose PDF was
            not found. Without it you can still read, correct and save, and export to `xlsx`, `json` and `txt`;
            `page_image`, `word_crop` and the `pdf` export raise `SourceChanged`.

    Example:
        ```python
        import extract

        doc = extract.load("invoice.extract.json", options=extract.Options(work_dir="work"))
        print(doc.name, doc.status, len(doc.pages))
        for word in doc.doubtful():
            print(word.id, word.text, round(word.p_wrong, 2), word.suggestion)
        doc.accept("p1w11")
        doc.correct("p1w13", "Esquivar")
        word = doc.word("p1w13")
        print(word.text, word.status, word.review, word.original, len(doc.doubtful()))
        ```

        ```text
        # output
        invoice.pdf done 1
        p1w11 Pelmuda 0.19 None
        p1w13 Esquilmar 0.12 Esquivar
        p1w4 F-2026-0498 0.04 None
        Esquivar doubtful corrected Esquilmar 1
        ```

    See also:
        `read`, `load`, `Word`, `Page`, `Number`, `Invoice`.
    """

    def __init__(
        self,
        data: DocumentData,
        source: Path | None = None,
        *,
        languages: Sequence[str] = ("es", "en"),
        work_dir: Path | None = None,
    ) -> None:
        self._data = data
        self.source = Path(source) if source is not None else None
        self._languages = tuple(languages)
        self._work_dir = work_dir
        self._lock = threading.RLock()
        self._source_checked = False
        self._index: dict[str, tuple[int, int]] = {}
        self._reindex()

    def _reindex(self) -> None:
        self._index = {w.id: (p, i) for p, page in enumerate(self._data.pages) for i, w in enumerate(page.words)}

    def __eq__(self, other: object) -> bool:
        """Compare two documents by what they would save.

        Args:
            other: Anything. A value that is not a `Document` is never equal.

        Note:
            Equal means the saved JSON would be the same: pages, words, numbers, invoice, options and corrections. A
            document loaded from a save of another is equal to it. Because documents are mutable and compare by
            value, they cannot be dictionary keys or set members.
        """
        return isinstance(other, Document) and jsondoc.exported(self._data) == jsondoc.exported(other._data)

    __hash__ = None  # type: ignore[assignment]

    @property
    def name(self) -> str:
        """The file name of the PDF the document was read from, such as `invoice.pdf`. No folder is included."""
        return self._data.source

    @property
    def sha256(self) -> str:
        """SHA-256 of the source PDF, as 64 hexadecimal digits.

        A saved reading records it, and the methods that need the PDF use it to refuse a different file.
        """
        return self._data.sha256

    @property
    def doc_id(self) -> str:
        """The first 12 hexadecimal digits of `sha256`: the id that events and the interface use for this document."""
        return self._data.sha256[:12]

    @property
    def status(self) -> str:
        """`done`, or `partial` when at least one page could not be read (see `unreadable_pages`).

        A `partial` document is returned, not raised: check this value after every read.
        """
        return self._data.status

    @property
    def unreadable_pages(self) -> tuple[int, ...]:
        """The numbers of the pages that could not be read, 1-based, in page order. `()` when there is none.

        Each such page has the status `unreadable`, and its `Page.reason` says why. Its words are kept as `unscored`,
        and no number on it is ever `checked`.
        """
        return self._data.unreadable_pages

    @property
    def pages(self) -> tuple[PageReading, ...]:
        """The pages in order; page 1 is at index 0.

        Each `Page` holds its words with their status and corrections.
        """
        return self._data.pages

    @property
    def numbers(self) -> tuple[NumberResult, ...]:
        """Every number of the document with its status, in reading order. See `Number`."""
        return self._data.numbers

    @property
    def invoice(self) -> InvoiceFields | None:
        """The fields and lines of an invoice, or None when `doc_type` is `generic`.

        Only page 1 is read for fields. A field that was not found is `None` inside the `Invoice`, never a guess.
        """
        return self._data.invoice

    @property
    def doc_type(self) -> str:
        """`invoice` or `generic`.

        A document is an `invoice` when page 1 has an invoice title near the top and at least two totals labels, such
        as the base, the VAT and the total. Anything else is `generic`, and `Document.invoice` is then `None`.
        """
        return self._data.doc_type

    @property
    def locale(self) -> str | None:
        """`es` or `en`: how the amounts of the document are written (`1.234,56` or `1,234.56`).

        It is decided from the numbers themselves and the number rules rest on it. `Options.languages[0]` breaks a
        tie. It is `None` only for a document loaded from a file that had no locale.
        """
        return self._data.locale

    def word(self, word_id: str) -> Word:
        """Return the present state of a word, with its text after any correction.

        Args:
            word_id: The id of the word, such as `"p3w17"` (page 3, word 17: the page counts from 1 and the word from
                0, in reading order). A token id such as `"p3w17#1"` is accepted and gives the word it belongs to.

        Returns:
            The `Word`. Ask again after a change: a word you already hold is not updated.
        """
        page, index = self._index[base_id(word_id)]
        return self._data.pages[page].words[index]

    def words(self) -> Iterator[Word]:
        """Iterate over every word of every page, in reading order.

        Returns:
            An iterator of `Word`. Words that a guard fix joined into another stay in the sequence with the text `""`
            and the status `absorbed`.
        """
        for page in self._data.pages:
            yield from page.words

    def doubtful(self) -> list[Word]:
        """List the words that still need a decision, the likeliest to be wrong first.

        A word is listed when nobody has decided it (`review` is `None`) and its status is `doubtful` or `unscored`.
        The `doubtful` words come first, by falling `p_wrong`; the `unscored` words come after them, in reading order.
        A word the user accepted or corrected leaves the list, although its `status` does not change.

        `unscored` means that nobody could judge the word: there was no guard, the page was unreadable, or the guard
        had no confidence to work with. These words are in the list on purpose. To keep only the words the guard
        flagged, filter on the status.

        Returns:
            A new list of `Word`. Empty when there is nothing to decide.
        """
        pending = [w for w in self.words() if w.review is None and w.status in ("doubtful", "unscored")]
        return sorted(pending, key=lambda w: (w.status != "doubtful", -(w.p_wrong or 0.0)))

    def fixed(self) -> list[Word]:
        """List the words that carry a fix the guard applied, in reading order.

        These are the first word of a join, a segmented word, a changed word and a dropped word. With the correction
        mode `off` (the default) the list is empty.

        Returns:
            A new list of `Word`.
        """
        return [w for w in self.words() if fixes.is_applied(w)]

    def fix_info(self, word_id: str) -> dict[str, Any] | None:
        """Return the guard's proposal for a word, applied or not, as the saved JSON carries it.

        Returns:
            A dictionary with the keys `op`, `kind`, `text`, `ids`, `source`, `sources`, `p_fix`, `p_keep`, `rule`,
            `modes`, `reasons` and `applied`, or `None` when the word has no proposal. `modes` lists the correction
            modes that apply it; an empty list means the proposal is shown but never applied. See `Fix`.
        """
        return jsondoc.fix_json(self.word(word_id))

    def field_status(self, cell: Cell | None) -> str:
        """Return how far a field of the invoice can be trusted.

        Args:
            cell: A field of `Document.invoice`, such as `doc.invoice.total`, or `None` for a field the invoice does
                not have.

        Returns:
            `mismatch` if any number made of the cell's words is `mismatch`, else `checked` if one of them is
            `checked`, else `unverified`. `None` gives `unverified`.
        """
        from .outputs.fields import cell_status

        return cell_status(cell, self._data)

    def _replace_word(self, word: Word, **changes: Any) -> None:
        page, index = self._index[word.id]
        words = list(self._data.pages[page].words)
        words[index] = replace(word, **changes)
        pages = list(self._data.pages)
        pages[page] = replace(pages[page], words=tuple(words))
        self._data = replace(self._data, pages=tuple(pages))

    def _renew_numbers(self) -> None:
        by_id = {w.id: w for page in fixes.token_pages(self._data.pages) for w in page.words}
        invoice = pipeline.refresh_cells(self._data.invoice, by_id) if self._data.invoice is not None else None
        locale, numbers = pipeline.run_numbers(self._data.pages, invoice, self._languages, self._data.locale)
        self._data = replace(self._data, invoice=invoice, numbers=numbers, locale=locale)

    def _log_correction(self, word_id: str, original: str, corrected: str, action: str) -> None:
        if self._work_dir is None:
            return
        record = {
            "sha256": self._data.sha256,
            "word": word_id,
            "original": original,
            "corrected": corrected,
            "action": action,
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self._work_dir.mkdir(parents=True, exist_ok=True)
        with open(self._work_dir / "corrections.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def correct(self, word_id: str, text: str) -> None:
        """Replace the text of a word with the text you typed.

        The OCR text stays in `Word.original` (from the first correction on: a second correction does not overwrite
        it). The word gets `review="corrected"` and `corrected_by="user"`, its `suggestion` is cleared, and the number
        rules that cover the word run again. The word's `status` does not change, and it leaves `Document.doubtful`. A
        corrected number becomes `checked` only if a verifying rule passes with the new text.

        Args:
            word_id: The id of the word, such as `"p1w13"`.
            text: The new text of the word, exactly as it should appear.
        """
        with self._lock:
            word = self.word(word_id)
            original = word.original if word.original is not None else word.text
            self._replace_word(
                word,
                text=text,
                original=original,
                review="corrected",
                corrected_by="user",
                suggestion=None,
                suggestion_conf=None,
            )
            self._renew_numbers()
            self._log_correction(word_id, original, text, "correct")

    def accept(self, word_id: str) -> None:
        """Confirm that a word is right as it stands.

        Any word can be accepted, not only the doubtful ones. The word gets `review="accepted"` and leaves
        `Document.doubtful`. Its `status` stays as it was, so an accepted word can still have the status `doubtful`: test
        `review` to know whether somebody decided. Accepting never makes a number `checked`: only the rules do that.

        Note:
            Accepting a word you had corrected turns `corrected` into `accepted` and keeps the text you typed;
            `Document.reopen` then clears the decision but does not restore the OCR text. Decide each word once.
        """
        with self._lock:
            word = self.word(word_id)
            self._replace_word(word, review="accepted")
            self._log_correction(word_id, word.text, word.text, "accept")

    def use_suggestion(self, word_id: str) -> None:
        """Apply the suggestion that a word carries, as a correction by the user.

        When the word has a suggestion of its own (`Word.suggestion`, the guard's plain text change), that one is
        applied; only when it has none is the one-digit fix of its `Number` (`DigitSuggestion`) used. The change goes through
        `correct`. A digit
        suggestion exists only when exactly one single-digit change makes the failing rule pass, so a tie gives none (common
        for IBANs), and a number written in several words (an IBAN in groups of four) never gets one that this method can
        apply: correct the word that holds the wrong digit.

        It applies whatever is offered. A word suggestion can have a small `Word.suggestion_conf`, and a digit change can
        have `DigitSuggestion.verified` false: look at `Document.word_crop` and those two values before you call it.

        Raises:
            ValueError: The word has no suggestion to apply.
        """
        with self._lock:
            word = self.word(word_id)
            text = word.suggestion
            if text is None:
                fix = next(
                    (n.suggestion for n in self._data.numbers if n.word_ids == (word_id,) and n.suggestion), None
                )
                if fix is None or word.text[fix.position : fix.position + 1] != fix.old:
                    raise ValueError(f"{word_id} has no suggestion")
                text = word.text[: fix.position] + fix.new + word.text[fix.position + 1 :]
            self.correct(word_id, text)

    def reopen(self, word_id: str) -> None:
        """Take back a decision you made on a word.

        An accepted word becomes undecided and is in `Document.doubtful` again. A corrected word gets the text it was
        read with, `original` and `corrected_by` are cleared, and the number rules that cover it run again.

        Raises:
            ValueError: The word is neither accepted nor corrected, or it is a corrected word that carries a guard
                fix (use `undo` for it).
        """
        with self._lock:
            word = self.word(word_id)
            if word.review not in ("accepted", "corrected"):
                raise ValueError(f"{word_id} has not been accepted or corrected")
            if word.review == "accepted":
                self._replace_word(word, review=None)
            else:
                if word.status in ("fixed", "absorbed"):
                    raise ValueError(f"{word_id} carries a fix of the guard: use undo")
                text = word.original if word.original is not None else word.text
                self._replace_word(word, text=text, original=None, review=None, corrected_by=None)
                self._renew_numbers()
            self._log_correction(word_id, word.text, self.word(word_id).text, "reopen")

    @property
    def mode(self) -> str:
        """The correction mode the document is in: `off`, `safe` or `more`.

        It is the mode the read used, or the one last given to `set_mode`.
        """
        return str(self._data.options.get("correct", "off"))

    @property
    def page_notes(self) -> list[dict[str, Any]]:
        """What the read noticed about pages whose text may not be what is printed, though nothing was changed.

        Each note is a dictionary: `{"key": "page.warning.low_resolution", "page": 2, "dpi": 80}` for a scan under 100
        dpi (read at 100 dpi; `dpi` is its real resolution), and `{"key": "page.warning.sideways", "page": 2}` when the
        lines run up or down. The list is a copy, empty when there is nothing to say, and the JSON keeps it under
        `options.page_notes`.
        """
        return [dict(note) for note in self._data.options.get("page_notes", ())]

    @property
    def fixes(self) -> dict[str, Any]:
        """A summary of what the guard changed, as a dictionary.

        The keys are `mode`, `second_reading` (bool), `thorough` (bool), `applied` and `undone` (counts),
        `by_source` and `by_op` (counts by the source and the operation of each fix), `withheld` (the pages where the
        fixes were held back: a list of `{"page": 1, "share": 0.12}`) and `second_reading_failed_pages` (the pages that
        did not get their second reading).
        """
        with self._lock:
            return fixes.summary(self._data)

    def tokens(self, page: int) -> tuple[Word, ...]:
        """Return the words of a page as the text layer and the number rules see them.

        Absorbed words are gone and a word the guard segmented is split into tokens whose ids have the form `"p3w17#1"`.
        `Document.word` accepts a token id and gives the word it comes from.

        Args:
            page: The page number, counting from 1.

        Returns:
            The tokens of the page, in reading order, as `Word` objects.

        Raises:
            IndexError: `page` is larger than the number of pages.

        Note:
            The low end is not checked: `0` returns the last page and `-1` the one before it, as a list index does.
        """
        return fixes.tokens(self._data.pages[page - 1])

    def _rebuild(self, pages: Sequence[PageReading], mode: str, withheld: list[dict[str, Any]]) -> None:
        pages, doc_type, invoice, locale, numbers = pipeline.analyse_fixed(pages, self._languages, mode)
        options = {k: v for k, v in self._data.options.items() if k != "fixes_withheld"} | {"correct": mode}
        if withheld:
            options["fixes_withheld"] = withheld
        self._data = replace(
            self._data,
            pages=tuple(pages),
            options=options,
            doc_type=doc_type,
            invoice=invoice,
            locale=locale,
            numbers=numbers,
        )
        self._reindex()

    def set_mode(self, mode: str) -> None:
        """Change the correction mode of the document without reading it again.

        The guard's stored proposals are applied again for `mode`. Words that a user decided never move, and a page
        where the fixes would change more than 8 % of its words keeps none (the pages held back are in
        `Document.fixes["withheld"]`). `off` takes every applied fix back, so the text is that of an `off` read.

        Args:
            mode: `off`, `safe` or `more`, as for `Options.correct`. Arithmetic digit fixes are applied only in `more`.

        Raises:
            ValueError: `mode` is not one of the three.
        """
        if mode not in CORRECT_MODES:
            raise ValueError(f"mode must be one of {CORRECT_MODES}, got {mode!r}")
        with self._lock:
            withheld: list[dict[str, Any]] = []

            def note(event: events.Event) -> None:
                if isinstance(event, events.Warning):
                    withheld.append(dict(event.params))

            pages = tuple(fixes.apply_fixes(p, mode, note)[0] for p in self._data.pages)
            self._rebuild(pages, mode, withheld)
            self._log_correction("", "", mode, "mode")

    def _undo(self, number: int | None, word_id: str | None) -> int:
        count = 0
        pages = []
        for page in self._data.pages:
            if (number is not None and page.number != number) or (
                word_id is not None and page.number != self.word(word_id).page
            ):
                pages.append(page)
                continue
            page, done = fixes.undo_words(page, word_id)
            count += done
            pages.append(page)
        if count:
            self._data = replace(self._data, pages=tuple(pages))
            self._reindex()
            self._renew_numbers()
        return count

    def undo(self, word_id: str) -> None:
        """Take back the guard's fix of a word, or of the group of words it belongs to.

        Pass the id of a word that carries an applied fix, such as the owner of a join. The text, id, box and status of
        the words come back as they were read. The word counts as decided (`review="accepted"`), so a later `set_mode`
        does not apply the fix again. The proposal stays on the word and `fix_info` still shows it.

        Raises:
            ValueError: The word has no applied fix.
        """
        with self._lock:
            word = self.word(word_id)
            if not self._undo(None, word_id):
                raise ValueError(f"{word_id} has no applied fix")
            self._log_correction(word.id, word.text, self.word(word_id).text, "undo")

    def undo_page(self, number: int) -> int:
        """Take back every fix the guard applied on one page.

        Args:
            number: The page number, counting from 1.

        Returns:
            How many fixes were undone. 0 when the page has none.

        Raises:
            ValueError: The document has no such page.
        """
        with self._lock:
            if not 1 <= number <= len(self._data.pages):
                raise ValueError(f"no page {number}")
            return self._undo(number, None)

    def undo_all(self) -> int:
        """Take back every fix the guard applied in the document.

        The text is then exactly that of the `off` read. Words you corrected or accepted yourself stay as you left
        them.

        Returns:
            How many fixes were undone. 0 when there are none.
        """
        with self._lock:
            return self._undo(None, None)

    def save(self, path: str | Path) -> None:
        """Write the document as a saved reading, a `.extract.json` file, that `load` reads back.

        The file holds every page, word, number and invoice field with its status, and every correction made so far.
        Its fields are described in `docs/reference/json-format.md`. The PDF is not part of it.

        Args:
            path: The file to write. Its folder must exist. By convention the name ends in `.extract.json`.

        Raises:
            OSError: The file cannot be written (a missing folder, no permission, a full disk).

        Note:
            `save` overwrites an existing file without asking and does not check what it wrote; `export` with
            `formats=("json",)` never replaces anything and checks the file.
        """
        with self._lock:
            jsondoc.save(self._data, path)

    def export(
        self,
        destination: str | Path,
        formats: Sequence[str] = exporter.FORMATS,
        *,
        name: str | None = None,
        language: str = "en",
        on_event: Callable[[events.Event], None] | None = None,
        cancel: CancelToken | None = None,
    ) -> list[Path]:
        """Write the document as a searchable PDF, a workbook, a JSON file and a text file.

        Nothing is read again and no model is loaded, so exporting after corrections is cheap, and the files hold the
        corrections made so far. Every file is written under a hidden temporary name, checked (the PDF has the same number
        of pages, the JSON reads back equal, the workbook has its sheets, the text reads back as written) and only then
        given its final name.

        Args:
            destination: The folder to write into. It is created if it does not exist.
            formats: Which files to write, from `"pdf"` (`<name>.searchable.pdf`), `"xlsx"` (`<name>.xlsx`), `"json"`
                (`<name>.extract.json`) and `"txt"` (`<name>.txt`). The default is all four, in that order.
            name: The name shared by the files, without extension. The default is the stem of the PDF's name.
            language: `"en"` or `"es"`: the language of the reasons and rule texts in the workbook. Any other value
                makes the workbook show the message keys and their values instead of sentences.
            on_event: A function called, in the thread that called `export`, with
                `Warning("export.warning.unreadable_text_layer", ...)` when unreadable pages still get their few words into
                the PDF text layer, and with `Exported` once everything is written.
            cancel: A `CancelToken`. It is looked at between formats only, not inside one.

        Returns:
            The paths of the files written, in the order of `formats`.

        Raises:
            ValueError: A name in `formats` is not one of the four, or a file failed its check after writing.
            OutputExists: `destination` is a file, or the name and its numbered copies `name-2` to `name-999` are all
                taken.
            SourceChanged: `"pdf"` was asked for and the source PDF is missing (`Document.source` is `None`) or is not
                the file the document was read from (its hash differs). The other formats do not need the PDF.
            Cancelled: The `cancel` token was set before a format was written.

        Note:
            A taken name is never overwritten: the whole set takes the first free copy number (`invoice-2.xlsx`,
            `invoice-2.txt`, ...), so the files of one export always match. A file that fails its check or is interrupted
            is removed; formats already finished keep their names. The searchable PDF keeps the original pages and adds an
            invisible text layer in WinAnsi encoding (other characters become `?` in the layer only), and it has no owner
            password even when the source had one.

        Example:
            ```python
            import extract

            doc = extract.load("invoice.extract.json")
            paths = doc.export("export-example", ("xlsx", "txt"))
            print([path.name for path in paths])
            again = doc.export("export-example", ("xlsx", "txt"))
            print([path.name for path in again])
            ```

            ```text
            # output
            ['invoice.xlsx', 'invoice.txt']
            ['invoice-2.xlsx', 'invoice-2.txt']
            ```

        See also:
            `Document.save`, `Exported`, `OutputExists`, `CancelToken`.
        """
        source = self._require_source() if "pdf" in formats else self.source
        with self._lock:
            data = self._data
        paths = exporter.export(
            data,
            source or Path(data.source),
            destination,
            formats,
            name=name,
            language=language,
            cancel=cancel,
            on_unreadable_text=None
            if on_event is None
            else lambda pages: on_event(events.Warning("export.warning.unreadable_text_layer", {"pages": pages})),
        )
        if on_event is not None:
            on_event(events.Exported(self.doc_id, tuple(str(p) for p in paths)))
        return paths

    def _require_source(self) -> Path:
        if self.source is None or not self.source.is_file():
            raise SourceChanged(f"the source file {self._data.source} is not available", file=self._data.source)
        if not self._source_checked:
            if file_sha256(self.source) != self._data.sha256:
                raise SourceChanged(
                    f"{self.source.name} is not the file this document was read from", file=self.source.name
                )
            self._source_checked = True
        return self.source

    def _render(self, number: int, zoom: float) -> tuple[Image.Image, PageReading]:
        reading = self._data.pages[number - 1]
        source = self._require_source()
        with PDFIUM_LOCK:
            document = pdfium.PdfDocument(str(source))
            try:
                page = document[number - 1]
                try:
                    array = render_page(page, max(1, round(reading.dpi * zoom)))
                finally:
                    page.close()
            finally:
                document.close()
        return Image.fromarray(array), reading

    def page_image(self, number: int, zoom: float = 1.0) -> bytes:
        """Return a page as a PNG image, drawn from the source PDF.

        Args:
            number: The page number, counting from 1.
            zoom: A factor on the page's own resolution: the image is drawn at `round(dpi * zoom)` dpi, where `dpi` is
                `Page.dpi`. 1.0 gives the pixel grid that the word boxes use. Use 0.5 for a thumbnail.

        Returns:
            The PNG file as bytes.

        Raises:
            SourceChanged: The source PDF is missing (`Document.source` is `None`) or is not the file the document was
                read from.
            IndexError: `number` is larger than the number of pages.

        Note:
            Each call opens the PDF and draws the page again, under a lock shared by the whole process: keep the
            result if you need it twice. `0` is the last page, as for `tokens`.
        """
        image, _ = self._render(number, zoom)
        return _png(image)

    def word_crop(self, word_id: str, pad: int = 8) -> bytes:
        """Return the picture of one word as a PNG image, cut from the page.

        This is what a reviewer needs to compare a doubtful word with the print: the word's box (`Word.box`, in pixels
        of the page at its own resolution), made larger on every side.

        Args:
            word_id: The id of the word, such as `"p1w13"`.
            pad: Pixels added on every side of the word's box. 0 gives the box alone. The crop never goes outside the
                page.

        Returns:
            The PNG file as bytes.

        Raises:
            SourceChanged: The source PDF is missing (`Document.source` is `None`) or is not the file the document was
                read from.

        Note:
            Each call opens the PDF and draws the whole page before it cuts, under a lock shared by the whole process.
        """
        word = self.word(word_id)
        image, _ = self._render(word.page, 1.0)
        left, top, right, bottom = word.box
        box = (
            max(0, int(left) - pad),
            max(0, int(top) - pad),
            min(image.width, int(right) + pad + 1),
            min(image.height, int(bottom) + pad + 1),
        )
        return _png(image.crop(box))


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def _default_lexicon(languages: Sequence[str]):
    try:
        from .guard import lexicon
    except ImportError as error:
        raise EngineLoadError(f"the word lists are not installed: {error}") from error
    missing = [language for language in languages if language not in lexicon.available(languages)]
    if missing:
        raise EngineLoadError(
            f"the word lists for {', '.join(missing)} are missing; build them with: .venv/bin/python scripts/fetch_models.py --lists"
        )
    return lexicon


PACKAGED_GUARD_MODEL = Path(__file__).resolve().parent / "data" / "guard.model"


def _default_guard():
    """The model named by $EXTRACT_GUARD_MODEL; else the copy inside the package, which an installed wheel carries;
    else the one in the `models/` folder of a checkout."""
    try:
        from .guard import MODEL_ENV, Guard
    except ImportError as error:
        raise EngineLoadError(f"the guard is not installed: {error}") from error
    if os.environ.get(MODEL_ENV) or not PACKAGED_GUARD_MODEL.is_file():
        return Guard.load()
    return Guard.load(PACKAGED_GUARD_MODEL)


def _thorough_guard():
    """The guard of the thorough second reading: its own model file (`data/guard-thorough.model`, or the one named by
    $EXTRACT_GUARD_MODEL_THOROUGH), trained on the readings of the lines and the zones together. Without it the mode
    cannot judge what it reads, so it stops with a clear error rather than falling back to the other model."""
    try:
        from .guard import Guard, readings
    except ImportError as error:
        raise EngineLoadError(f"the guard is not installed: {error}") from error
    path = Path(os.environ.get(second_reading.THOROUGH_MODEL_ENV) or second_reading.THOROUGH_MODEL)
    if not path.is_file():
        raise EngineLoadError(
            f"the thorough second reading needs its own guard model, {path}, and this install does not have it; "
            f"reinstall Extract, set {second_reading.THOROUGH_MODEL_ENV} to the model file, or turn the second reading off "
            "(second_reading=False)",
            path=str(path),
            cause="thorough_model_missing",
        )
    try:
        guard = Guard.load(path, thorough=True)
    except ModelMismatch as error:
        raise EngineLoadError(f"the thorough guard model {path} cannot be used: {error}", path=str(path)) from error
    block = guard.model.fix
    trained = readings.ReadingSpec.from_meta(block.get("readings") if block else None)
    if trained is None or not trained.same_cell(readings.product_spec(True)):
        raise EngineLoadError(
            f"the guard model {path} was not trained on the thorough readings (lines and zones together; "
            f"meta.fix.readings is {'missing' if trained is None else trained.label})",
            path=str(path),
        )
    return guard


class Reader:
    """Worker processes and models kept loaded between reads, so that a batch pays the start-up cost once.

    `read` starts the workers for each call and stops them at the end; a `Reader` keeps them. Nothing is loaded when
    you create it: the first `read` loads the word lists, the guard and the OCR engine, and a missing piece raises
    `EngineLoadError` there. Use it as a context manager, so that the workers stop when the block ends.

    A `Reader` reads one document at a time. Do not call `Reader.read` from several threads at once; use one `Reader`
    per process or serialise the calls.

    Args:
        options: The `Options` of every read made with this `Reader`. The default is `Options()`.
        guard: Which guard judges the words. `None` loads the packaged guard model (the file named by
            `$EXTRACT_GUARD_MODEL` if that variable is set; `"thorough"` second reading uses its own model).
            `False` reads without a guard: every OCR word comes out `unscored` and each page sends
            `guard.warning.unscored` with the cause `no_guard`. Any other object with an `info` attribute and a
            `score_page(...)` method is used as a custom guard; that is for tests and research, not for normal use.
        lexicon: The word lists the guard compares against. `None` loads the packaged lists for
            `Options.languages`, `False` uses none, and any other object with the methods `has` and `share` is used
            as the lists.

    Attributes:
        options: The `Options` this reader was made with.

    Note:
        As with `read`, the workers import your main script again: create the `Reader` under
        `if __name__ == "__main__":`. With the default `Options.processes`, at most two worker processes run, and a
        `Reader` can use about 1.5 GB of memory for each.

    Example:
        ```python
        import extract

        if __name__ == "__main__":
            options = extract.Options(work_dir="work")
            with extract.Reader(options) as reader:
                for name in ("scan.pdf", "invoice.pdf"):
                    doc = reader.read(name)
                    print(name, doc.status, len(doc.doubtful()))
        ```

        ```text
        # output
        scan.pdf done 0
        invoice.pdf done 3
        ```

    See also:
        `read`, `Options`, `Reader.read`, `Reader.close`.
    """

    _engine_factory = DEFAULT_FACTORY

    def __init__(self, options: Options | None = None, *, guard: Any = None, lexicon: Any = None) -> None:
        self.options = options or Options()
        self._guard_choice = guard
        self._lexicon_choice = lexicon
        self._guard: Any = None
        self._lexicon: Any = None
        self._batch: BatchReader | None = None

    def _prepare(self) -> BatchReader:
        if self._batch is None:
            leave_if_started_by_an_unguarded_script()
            self._lexicon = (
                _default_lexicon(self.options.languages)
                if self._lexicon_choice is None
                else self._lexicon_choice or None
            )
            if self._guard_choice is not None:
                self._guard = self._guard_choice or None
            else:
                self._guard = _thorough_guard() if self.options.thorough else _default_guard()
            self._batch = BatchReader(self.options, self._lexicon, factory=self._engine_factory)
        return self._batch

    @property
    def job_log(self) -> list[dict[str, Any]]:
        """What the engine decided and measured while this `Reader` worked, as a list of dictionaries.

        Each entry has a `code` such as `ram.workers` (how many workers the free memory allows), `worker.started`,
        `worker.ready` (with the seconds the models took to load), `pool.page` (a page read by a worker),
        `worker.died` or `worker.recycled`. The list is a copy and is empty before the first `read`. The same entries
        are appended, one JSON line each, to `job.log` in the work folder.

        Every entry has a `time` and a `code`, and the other keys depend on the code: the `ram.workers` entry has
        `free_mb`, `budget_mb`, `processes`, `threads` and `backend`, among others. Pages that come from the page cache
        need no worker, so a read of cached pages leaves the list empty.
        """
        return list(self._batch.log.entries) if self._batch is not None else []

    def read(
        self,
        path: str | Path,
        on_event: Callable[[events.Event], None] | None = None,
        cancel: CancelToken | None = None,
        *,
        correct: str | None = None,
    ) -> Document:
        """Read one PDF with the workers and models this `Reader` already holds.

        This is what `extract.read` does, without starting and stopping the workers around it, and it raises what `read`
        raises. A page that was read before with the same file, resolution and engine comes from the page cache in the
        work folder and costs no OCR. A page that cannot be read is not an exception: the document comes back `partial`.
        Each call sends its own events, from `Started` to `Finished`; see `events`.

        Args:
            path: The PDF, as a string or a `Path`.
            on_event: A function called with each `Event`. It runs in the thread that called `read`, so it can call
                `cancel.cancel()` itself.
            cancel: A `CancelToken` to stop the read from another thread or from `on_event`.
            correct: `off`, `safe` or `more`, overriding `Options.correct` for this document only. `None` uses the
                option.

        Returns:
            The `Document`.

        Note:
            Every failure inside the read, and `Cancelled`, send `Finished` (`"failed"` or `"stopped"`) before the
            exception leaves. Errors found before the read starts (a bad `correct`, models or word lists that cannot
            be loaded) send no `Finished`. One read at a time per `Reader`.
        """
        emit = on_event or (lambda event: None)
        mode = self.options.correct if correct is None else correct
        if mode not in CORRECT_MODES:
            raise ValueError(f"correct must be one of {CORRECT_MODES}, got {mode!r}")
        batch = self._prepare()
        budget = pipeline.REVIEW_BUDGETS[self.options.review]
        languages = self.options.languages
        withheld: list[dict[str, Any]] = []
        notes: list[dict[str, Any]] = []

        def sent(event: events.Event) -> None:
            if isinstance(event, events.Warning) and event.code == "guard.warning.fixes_withheld":
                withheld.append(dict(event.params))
            if isinstance(event, events.Warning) and event.code in PAGE_NOTES:
                note = {"key": event.code, **event.params}
                if note not in notes:
                    notes.append(note)
            emit(event)

        def transform(reading: PageReading) -> PageReading:
            if pipeline.looks_sideways(reading):
                sent(events.Warning("page.warning.sideways", {"page": reading.number}))
            scored = pipeline.score_page(reading, self._guard, languages, budget, sent)
            return fixes.apply_fixes(scored, mode, sent)[0]

        try:
            reading = batch.read(path, sent, cancel, transform)
            pages, doc_type, invoice, locale, numbers = pipeline.analyse_fixed(reading.pages, languages, mode)
            options: dict[str, Any] = {
                "languages": list(languages),
                "review": self.options.review,
                "correct": mode,
                "second_reading": self.options.second_reading,
            }
            if withheld:
                options["fixes_withheld"] = withheld
            if notes:
                options["page_notes"] = sorted(notes, key=lambda note: (note["page"], note["key"]))
            if self.options.second_reading:
                failed = [page.number for page in pages if page.reread_report.get("reader_failed")]
                if failed:
                    options["second_reading_failed_pages"] = failed
            data = DocumentData(
                source=Path(path).name,
                sha256=reading.sha256,
                pages=pages,
                engine=ocr.fingerprint(),
                guard=pipeline.guard_id(self._guard),
                options=options,
                doc_type=doc_type,
                locale=locale,
                numbers=numbers,
                invoice=invoice,
            )
            document = Document(data, Path(path), languages=languages, work_dir=batch.work_dir)
            doubtful = sum(1 for w in document.words() if w.status in ("doubtful", "unscored"))
            checked = sum(1 for n in numbers if n.status == "checked")
            fixed = document.fixes["applied"]
            emit(
                events.DocumentRead(
                    document.doc_id, str(path), data.status, data.unreadable_pages, doubtful, checked, fixed
                )
            )
            emit(events.Finished("done"))
            return document
        except Cancelled:
            emit(events.Finished("stopped"))
            raise
        except Exception:
            emit(events.Finished("failed"))
            raise

    def close(self) -> None:
        """Stop the worker processes and the second-reading process.

        The OCR models live in the workers and go with them. Leaving a `with` block calls it. The `Reader` is still
        usable afterwards: the next `read` prepares everything again. Calling it twice, or before any `read`, does
        nothing.
        """
        if self._batch is not None:
            self._batch.close()
            self._batch = None

    def __enter__(self) -> Reader:
        """Start a `with` block.

        Returns:
            The `Reader` itself. Nothing is loaded yet; the first `read` does that.
        """
        return self

    def __exit__(self, *exc: object) -> None:
        """End a `with` block: call `close`.

        Args:
            *exc: The exception type, value and traceback if the block raised, else `None` three times. They are not
                used, and an exception from the block is never swallowed.
        """
        self.close()


def read(
    path: str | Path,
    options: Options | None = None,
    on_event: Callable[[events.Event], None] | None = None,
    cancel: CancelToken | None = None,
    *,
    correct: str | None = None,
) -> Document:
    """Read one PDF and return a `Document`.

    It starts the OCR worker processes and loads the models for this call, reads, and stops them again: several
    seconds before the first page, paid on every call. To read two files or more, make one `Reader` and call
    `Reader.read` for each.

    Digital PDFs whose text layer reads as text need no OCR, and scanned pages that were read before come from the
    page cache in the work folder, keyed by the file's hash, the page, the resolution and the engine.

    Args:
        path: The PDF, as a string or a `Path`.
        options: The `Options` of the read. `None` means `Options()`, whose work folder is `$XDG_CACHE_HOME/extract`, or
            `~/.cache/extract`; pass `Options(work_dir=...)` in tests, and wherever the page cache and the logs must go
            elsewhere.
        on_event: A function called with each `Event` of the read (`Started`, `PageRead`, `Progress`, `DocumentRead`,
            `Warning`, `Finished`). It runs in the thread that called `read`.
        cancel: A `CancelToken` that stops the read from another thread or from `on_event`.
        correct: `off`, `safe` or `more`, overriding `Options.correct` for this read. `None` uses the option. It is
            the only option `read` takes directly: `languages`, `review`, `second_reading` and the others go in
            `Options`.

    Returns:
        The `Document`. Check `Document.status`: a document with an unreadable page comes back `partial`, it is not
        an error.

    Raises:
        ValueError: `correct` is not `off`, `safe` or `more`.
        FileNotFoundError: `path` does not exist. This is the built-in exception, not an `ExtractError`.
        UnsupportedFile: The file is not a PDF that can be opened.
        PasswordProtected: The PDF needs a password to be opened.
        EngineLoadError: A piece the read needs is missing or cannot start: the word lists, the guard model, an OCR
            model, or a worker process. The message says which.
        ModelMismatch: The guard model was built for another version of the code.
        SourceChanged: The PDF changed on disk while it was being read.
        Cancelled: The `cancel` token was set. Pages already read stay in the cache.

    Note:
        The OCR workers are started with the `spawn` method, so they import your main script again. Put every call
        to `extract.read` (and to `Reader`) under `if __name__ == "__main__":`. Without the guard, a read that needs a
        worker (a scanned PDF) raises `EngineLoadError` with `cause="no_main_guard"` after a few seconds, and the
        worker prints one line on stderr. A digital PDF needs no worker and reads fine even without the guard, so
        the fault shows only on scans.

        With the default `Options.processes`, `read` starts up to two worker processes of about 1.5 GB each: do not
        run two reads at once on a small machine.

    Example:
        ```python
        import extract

        if __name__ == "__main__":
            doc = extract.read("invoice.pdf", extract.Options(work_dir="work"))
            print(doc.status, len(doc.doubtful()))
        ```

        ```text
        # output
        done 3
        ```

    See also:
        `Reader`, `Options`, `load`, `inspect`, `CancelToken`, `Document`.
    """
    with Reader(options) as reader:
        return reader.read(path, on_event, cancel, correct=correct)


def load(path: str | Path, source: str | Path | None = None, options: Options | None = None) -> Document:
    """Open a saved reading, a `.extract.json` file, as a `Document`, without reading anything again.

    `load` is the way back from `Document.save` or from the `json` export. It runs no OCR and loads no model, so it
    takes well under a second, and the result is equal (`==`) to the document that was saved. It reads files of schema
    version 2 and the older version 1. The PDF is needed only for `Document.page_image`, `Document.word_crop` and the
    `pdf` export.

    Args:
        path: The saved reading.
        source: The PDF the reading was made from. By default `load` looks for the file with the name recorded in the
            JSON in the same folder as the JSON, and uses it if it exists. Pass `source` when the PDF is somewhere
            else.
        options: Used only for `work_dir`, where later corrections are logged, and for `languages` when the file has
            none. The default `Options()` logs into the default work folder (`$XDG_CACHE_HOME/extract`, else
            `~/.cache/extract`), so pass `Options(work_dir=...)` when you will correct the loaded document.

    Returns:
        The `Document`. Its `source` is `None` if no PDF was found. A PDF whose hash differs from the one recorded
        is not detected here: `SourceChanged` is raised when you first use it (`page_image`, `word_crop`, the `pdf`
        export).

    Raises:
        FileNotFoundError: `path` does not exist. This is the built-in exception, not an `ExtractError`.
        UnsupportedFile: The file is not JSON, or it is JSON that is not an Extract document of schema version 1 or 2.
        KeyError: The file declares a supported schema version but lacks fields (a damaged or hand-edited file).

    Example:
        ```python
        import extract

        doc = extract.load("invoice.extract.json")
        print(doc.name, doc.source)

        elsewhere = extract.load("invoice.extract.json", source="scans/invoice.pdf")
        print(elsewhere.source)
        ```

        ```text
        # output
        invoice.pdf invoice.pdf
        scans/invoice.pdf
        ```

    See also:
        `Document.save`, `Document.export`, `read`, `Options`.
    """
    path = Path(path)
    data = jsondoc.load(path)
    if source is None:
        sibling = path.with_name(data.source)
        source = sibling if sibling.is_file() else None
    options = options or Options()
    languages = data.options.get("languages", options.languages)
    return Document(
        data, Path(source) if source is not None else None, languages=languages, work_dir=resolve_work_dir(options)
    )
