"""The errors Extract raises, and the exceptions it leaves to Python.

Every problem that Extract itself finds in a file, a model or a job is an `ExtractError`, so one `except` clause
catches them all. The subclasses say what went wrong, and `code` and `params` let a program react without parsing the
message.

Some errors are not `ExtractError` and are not caught by it:

- `FileNotFoundError`: the path given to `read`, `Reader.read`, `inspect` or `load` does not exist.
- `ValueError`: a setting or an argument is out of range (`Options`, the `correct` argument, a mode, a format), a
  word or a page that a method cannot act on, or an exported file that failed its check.
- `KeyError`: a word id that the document does not have (`Document.word`, `Document.correct`, ...), or a saved
  reading given to `load` that declares a supported schema version but lacks fields.
- `IndexError`: a page number past the end (`Document.tokens`, `Document.page_image`).
- `OSError`: a file cannot be written (`Document.save`, the work folder). The interface turns it into a message; the
  library lets it through.

The classes are importable from `extract` and defined in `extract.errors`, the module a traceback names.
"""

from __future__ import annotations


class ExtractError(Exception):
    """The base of every error Extract raises about a file, a model or a job. Catch this to catch them all.

    Args:
        message: The text of the error. Empty means that `str(error)` is the `code`.
        **params: Values that belong to the error, as keyword arguments, such as `file="a.pdf"`. They are kept in
            `params`.

    Attributes:
        code: An i18n key, such as `"error.unsupported_file"`. It is a class attribute: each subclass has its own, and
            a screen looks its text up by it. The base class has `"error.extract"`.
        params: A dictionary of the keyword arguments the error was made with. The keys a class uses are named in its
            own docstring: `file` (a file name), `path` (a file path) and `cause` (a short reason).

    Example:
        ```python
        import extract

        try:
            extract.load("scan.pdf")
        except extract.ExtractError as error:
            print(type(error).__name__, error.code)
        ```

        ```text
        # output
        UnsupportedFile error.unsupported_file
        ```

    See also:
        `UnsupportedFile`, `EngineLoadError`, `Cancelled`.
    """

    code = "error.extract"

    def __init__(self, message: str = "", **params: object) -> None:
        super().__init__(message or self.code)
        self.params = params


class UnsupportedFile(ExtractError):
    """The file is not a PDF that can be opened, or not a saved reading of an Extract document.

    Raised by `read`, `Reader.read` and `inspect` when pdfium cannot open the file as a PDF (a text file, a picture, a
    damaged PDF; the error carries `file=` with its name). Raised by `load` when the file is not JSON, or is JSON
    without the `schema_version` 1 or 2 of an Extract document (no `file`).

    Images and Office files are not read: put them in a PDF first. To read a PDF that needs a password, see
    `PasswordProtected`.

    Attributes:
        code: `"error.unsupported_file"`.
        params: `file`, the name of the PDF, when a PDF was refused; empty when `load` refused a saved reading.
    """

    code = "error.unsupported_file"


class PasswordProtected(ExtractError):
    """The PDF needs a password to be opened, and Extract has none.

    Raised by `read`, `Reader.read` and `inspect`. A PDF that has only an owner password (restrictions on printing or
    copying, but it opens without a password) is not refused: it is read like any other, and the searchable PDF is
    written without those restrictions.

    Remove the password with the tool that set it, then read the new file.

    Attributes:
        code: `"error.password_protected"`.
        params: `file`, the name of the PDF.
    """

    code = "error.password_protected"


class EngineLoadError(ExtractError):
    """A piece that reading needs is missing, or could not start: a model, a word list or a worker.

    Raised by `read` and `Reader.read` (never by `load`), before or during the first page. The message says which
    piece, and what to run when there is a fix:

    - the word lists are missing for a chosen language: run `.venv/bin/python scripts/fetch_models.py --lists`;
    - an OCR model file is missing (`file=` names it): run `python3 scripts/fetch_models.py`, with `--all` for the
      readers of the second reading;
    - the packaged guard is not installed;
    - the thorough second reading was asked for and its guard model is missing (`cause="thorough_model_missing"` and
      `path=`), or it was not trained on the thorough readings;
    - OpenVINO or RapidOCR could not be imported or started;
    - `Options(backend="openvino")` and the workers started another backend;
    - a worker process could not start;
    - the script that called `read` has no `if __name__ == "__main__":` guard (`cause="no_main_guard"`): each OCR
      worker imports that script again and would read at its top level. The worker prints one line on stderr and
      leaves, and `read` raises this error. A digital PDF needs no worker, so only a scanned PDF shows the fault. See
      `read`.

    Attributes:
        code: `"error.engine_load"`.
        params: Any of `file` (a model file name), `path` (the path of a guard model) and `cause` (a short reason,
            `"thorough_model_missing"` or `"no_main_guard"`). Most raises carry none.
    """

    code = "error.engine_load"


class ModelMismatch(ExtractError):
    """The guard model was built for another schema or library version than this code, and cannot be used.

    Raised while the guard is loaded, by the first `read` or `Reader.read` that needs it. It means that the model file
    and the installed code do not belong together, for example a guard model of another release in the file named by
    `$EXTRACT_GUARD_MODEL`. Install the model that ships with this version.

    Attributes:
        code: `"error.model_mismatch"`.
        params: `path`, the path of the model file, when the schema is the cause.
    """

    code = "error.model_mismatch"


class OutputExists(ExtractError):
    """An export cannot be written without overwriting a file.

    Raised by `Document.export` when `destination` is a file and not a folder, or when the name and its numbered
    copies (`name-2` to `name-999`) are all taken. An export never overwrites: a taken name gets the
    first free copy number instead, so this error is rare.

    Attributes:
        code: `"error.output_exists"`.
        params: `file`, the name that could not be used.
    """

    code = "error.output_exists"


class SourceChanged(ExtractError):
    """The PDF is not the file the document was read from, or it changed while it was being read.

    Raised by `read` and `Reader.read` when the size or the modification time of the PDF changes during the read.
    Raised by `Document.export` (with `"pdf"`), `Document.page_image` and `Document.word_crop` when the source PDF
    is missing (`Document.source` is `None`) or its SHA-256 differs from the one the document recorded. The check
    runs at the first use, never silently.

    Pass the right file as `source` to `load`, or restore the original. The formats that do not need the PDF
    (`xlsx`, `json`, `txt`) are not affected.

    Attributes:
        code: `"error.source_changed"`.
        params: `file`, the name of the PDF.
    """

    code = "error.source_changed"


class Cancelled(ExtractError):
    """The job was stopped through its `CancelToken`.

    Raised by `read` and `Reader.read` within about two seconds of `CancelToken.cancel`, after sending
    `Finished("stopped")`; the pages already read stay in the page cache. Raised by `Document.export` when the token
    is set before a format starts; the files written so far keep their names, and no partial file is left.

    It is the normal end of a job you stopped, not a failure, so it is usually caught on its own.

    Attributes:
        code: `"error.cancelled"`.
        params: Empty.
    """

    code = "error.cancelled"
