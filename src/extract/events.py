"""The events of a read: progress, notices and the end of the job.

Pass a function as `on_event` to `read`, `Reader.read` or `Document.export` and it is called with one event after
another, in the thread that called, while the job runs. All of them are frozen dataclasses and subclasses of `Event`;
all are importable from `extract` (`extract.PageRead`, `extract.Warning`, ...). A screen gets them as JSON through
`to_dict`, with i18n keys and parameters only.

The events of one `Reader.read` come in this order:

1. `Started`, once, with the page count.
2. For every page, a `PageRead` and then a `Progress`. Pages arrive in the order the workers finish them, which is not
   always 1, 2, 3: digital and cached pages come first.
3. `Warning`, at any moment between the first and the last event, as many as there are notices.
4. `DocumentRead`, once, with the summary of the document.
5. `Finished`, last: `"done"`, `"stopped"` after a cancel, or `"failed"` after an exception. The exception is raised
   after the event.

A read that fails before the PDF is open (a missing file, `UnsupportedFile`, `PasswordProtected`) sends
`Finished("failed")` and nothing before it.

`Document.export` sends `Warning("export.warning.unreadable_text_layer")` if it has to, and `Exported` when the files
are written.

Example:
    ```python
    import extract

    events = []

    if __name__ == "__main__":
        options = extract.Options(work_dir="work")
        extract.read("scan.pdf", options, on_event=events.append)
        print([type(event).__name__ for event in events])
    ```

    ```text
    # output
    ['Started', 'PageRead', 'Progress', 'DocumentRead', 'Finished']
    ```
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class Event:
    """The base of every event. It has no fields of its own.

    Use `isinstance(event, extract.PageRead)` to tell the events apart.

    Example:
        ```python
        import extract

        event = extract.Progress(doc_id="f79ddbc36632", done=1, total=3)
        print(event.to_dict())
        ```

        ```text
        # output
        {'event': 'Progress', 'doc_id': 'f79ddbc36632', 'done': 1, 'total': 3}
        ```

    See also:
        `Started`, `PageRead`, `Progress`, `DocumentRead`, `Exported`, `Warning`, `Finished`.
    """

    def to_dict(self) -> dict[str, Any]:
        """Return the event as a dictionary, ready for `json.dumps`.

        Returns:
            `{"event": <class name>, <every field>: <value>}`. Tuples stay tuples here, and `json.dumps` turns them
            into lists.
        """
        return {"event": type(self).__name__, **asdict(self)}


@dataclass(frozen=True)
class Started(Event):
    """A read has begun and its size is known.

    Sent once by `Reader.read` (and so by `read`), before the first page.

    Attributes:
        documents: How many documents this read covers. Always 1: each call reads one document.
        pages: The number of pages of the PDF.

    See also:
        `PageRead`, `Finished`.
    """

    documents: int
    pages: int


@dataclass(frozen=True)
class PageRead(Event):
    """One page is finished.

    Sent after each page, followed by its `Progress`. The pages do not come in order: workers finish when they
    finish.

    Attributes:
        doc_id: The id of the document, as `Document.doc_id` (the first 12 hex digits of its SHA-256).
        page: The page number, counting from 1.
        status: The status of the page: `native`, `ocr` or `unreadable`.
        words: How many words the page has.
        doubtful: How many words are `doubtful` or `unscored` on the page.
        fixed: How many words the guard changed on the page (a fix applied; 0 with the correction mode `off`).

    See also:
        `Progress`, `Page`.
    """

    doc_id: str
    page: int
    status: str
    words: int
    doubtful: int = 0
    fixed: int = 0


@dataclass(frozen=True)
class Progress(Event):
    """How far the read is, sent after every `PageRead`.

    Attributes:
        doc_id: The id of the document, as `Document.doc_id`.
        done: Pages finished so far.
        total: Pages in the document.

    See also:
        `PageRead`, `Started`.
    """

    doc_id: str
    done: int
    total: int


@dataclass(frozen=True)
class DocumentRead(Event):
    """The document is read, and this is its summary. Sent once, after the last page.

    Attributes:
        doc_id: The id of the document, as `Document.doc_id`.
        path: The path that was given to `read`, as a string, without change.
        status: The status of the document: `done`, or `partial` when some pages are unreadable.
        unreadable_pages: The numbers of the unreadable pages, 1-based. Empty when there is none.
        doubtful: How many words are `doubtful` or `unscored` in the whole document.
        numbers_checked: How many numbers ended `checked`.
        fixed: How many fixes the guard applied (0 with the correction mode `off`).

    See also:
        `Finished`, `Document`.
    """

    doc_id: str
    path: str
    status: str
    unreadable_pages: tuple[int, ...] = ()
    doubtful: int = 0
    numbers_checked: int = 0
    fixed: int = 0


@dataclass(frozen=True)
class Exported(Event):
    """The files of an export are written. Sent once by `Document.export` when you pass `on_event`.

    Attributes:
        doc_id: The id of the document, as `Document.doc_id`.
        paths: The paths of the files written, as strings, in the order of the formats.

    See also:
        `Document.export`.
    """

    doc_id: str
    paths: tuple[str, ...]


@dataclass(frozen=True)
class Warning(Event):  # noqa: A001 - the name the API documents
    """A notice about the read or the export. Nothing failed: the job goes on.

    This class is `extract.Warning`, and it hides the built-in `Warning`: after `from extract import *` the name
    `Warning` is this class, so import `extract` and write `extract.Warning`, or `from extract import events`.

    The codes, and the `params` each one carries:

    - `page.warning.low_resolution`: a scan is under 100 dpi. `page`, `dpi`. The page is read at 100 dpi.
    - `page.warning.sideways`: the lines run up or down. `page`. No word is changed.
    - `guard.warning.unscored`: some words could not be judged. `page`, `words`, `cause` (`no_guard`, `error`,
      `too_many_words` or `no_confidence`) and sometimes `detail`.
    - `guard.warning.fixes_withheld`: fixes were not applied on a page. `page` and `share` (the share of words the fixes
      would have changed, over the 8 % limit), or `page` and `cause="language"` when the page is not in a chosen
      language or no word list could judge it.
    - `guard.warning.readings_differ`: the second readings were ignored because the guard model was trained on others.
      `page`, `model`, `product`.
    - `guard.warning.no_lexicon`: there is no word list for a language. `language`, the chosen languages, such as
      `"es, en"`.
    - `export.warning.unreadable_text_layer`: unreadable pages still got their few words into the PDF text layer, in
      `Document.export`. `pages`.
    - Engine notices, with the values of the engine's own log entry: `ram.low` (`free_mb`, `threshold_mb`, `workers`: no
      worker starts until memory recovers), `engine.fallback` (`reason`, `pid`: the OCR engine fell back to another
      backend), `worker.died` (`pid`, `exit_code`, `chunk`), `worker.crash` (`pid`, `error`), `reader.skipped`
      (`free_mb`, `budget_mb`: not enough memory for the second reading) and `reader.failed` (`error`).

    Attributes:
        code: What the notice is about, an i18n key such as `"guard.warning.unscored"`. The codes are listed above.
        params: The values the text of the notice needs, such as `{"page": 2, "words": 11}`. The keys depend on the
            code.

    Example:
        ```python
        import extract

        def show(event):
            if isinstance(event, extract.Warning):
                print(event.code, event.params)

        if __name__ == "__main__":
            options = extract.Options(work_dir="work")
            with extract.Reader(options, guard=False) as reader:
                reader.read("scan.pdf", on_event=show)
        ```

        ```text
        # output
        guard.warning.unscored {'page': 1, 'words': 95, 'cause': 'no_guard'}
        ```

    See also:
        `read`, `Document.page_notes`, `Document.fixes`.
    """

    code: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Finished(Event):
    """The read is over. Sent last, once.

    Attributes:
        status: `"done"` when the document was read, `"stopped"` when the `CancelToken` was set, `"failed"` when an
            exception ended the read. For the last two, the exception is raised after this event.

    See also:
        `DocumentRead`, `Cancelled`.
    """

    status: str
