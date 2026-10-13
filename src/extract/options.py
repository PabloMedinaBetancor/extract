from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from . import second_reading as second_reading_decision
from .engine.pool import CHUNK_PAGES, RECYCLE_PAGES

LANGUAGES = ("es", "en")
REVIEW_LEVELS = ("fewer", "balanced", "more")
BACKENDS = ("auto", "openvino", "onnxruntime")
CORRECT_MODES = ("off", "safe", "more")


@dataclass(frozen=True)
class Options:
    """The settings of a read, as one immutable object.

    Make one with the fields you want to change; the others keep their defaults. Pass it to `read`, to `Reader`, and
    to `load` (which uses only `work_dir` and `languages`). The object is frozen: to change a setting, make a new
    `Options`, for instance with `dataclasses.replace`. Every field is checked when the object is made.

    Attributes:
        languages: The languages of the documents, a non-empty subset of `("es", "en")`. They choose the word lists
            the guard compares each word with, and the first one breaks a tie when the style of the amounts
            (`1.234,56` or `1,234.56`) cannot be decided from the numbers. A list is turned into a tuple. A plain
            string such as `"es"` is refused: write `("es",)`. The OCR models are the same for both languages. On a
            page in a language you did not choose, the guard applies no fix and sends `guard.warning.fixes_withheld`.
        review: How many words the guard flags as `doubtful`: `"fewer"`, `"balanced"` (the default) or `"more"`, each a
            fixed threshold set on the development pages so that about 3 %, 5 % and 8 % of their words are flagged. It is
            not a share of each page: a damaged page gets more flags and a clean page almost none. `"fewer"` asks you to
            check less and lets more wrong words through; `"more"` does the opposite.
        recycle_pages: How many pages an OCR worker reads before it is replaced by a fresh one, at least 4. The
            default is 120. A worker is also replaced earlier when its memory grows past about 1.4 GB (1.3 GB on a
            small machine). Lower it on a machine with little memory.
        work_dir: The work folder: the page cache, `job.log` and `corrections.jsonl`. A string is turned into a
            `Path`. `None` (the default) means `$XDG_CACHE_HOME/extract` if that variable is set, else
            `~/.cache/extract`. Deleting the folder removes every trace of the reads. The interface always uses
            `~/.cache/extract` and ignores `XDG_CACHE_HOME`.
        processes: How many OCR worker processes to run, at least 1. `None` (the default) chooses from the free memory
            and the physical cores, with a maximum of 2. Each process is planned at about 1.5 GB with the OpenVINO
            backend (the largest worker measured used 1,330 MB) and 700 MB with `onnxruntime`.
        threads: How many threads each worker uses, at least 1. `None` (the default) is the number of physical cores
            divided by `processes`, and at least 1.
        backend: The engine that runs the OCR: `"auto"` (the default), `"openvino"` or `"onnxruntime"`. `"auto"`
            picks OpenVINO when more than 2,000 MB are free and `onnxruntime` otherwise. Asking for `"openvino"` when
            the workers start another backend raises `EngineLoadError`.
        correct: How far the guard may change words: `"off"` (the default) changes nothing, `"safe"` applies the fixes
            of the strictest precision and `"more"` those with the best net balance; every fix can be undone. A page
            whose fixes would change more than 8 % of its words keeps none, and a digit changes only when a verifying
            rule confirms it (the arithmetic rules only in `"more"`). `read` can override this field for one document.
        second_reading: Whether to read the doubtful lines again with other recognisers, in a separate process:
            `False` (the default in 5.0.7), `True` or the exact string `"thorough"`. `0` and `1` are refused. With
            `True` the readings are computed, but the packaged guard model was trained without them, so it ignores
            them and each page sends `guard.warning.readings_differ`: that costs time and memory and changes nothing.
            `"thorough"` reads 16 lines with the two light recognisers and 40 zones with the deep one, judged by its
            own guard model; it adds 2.95 s a page on average on 60 pilot pages and a reader process of up to 1.6 GB.
            It needs the extra reader models (`python3 scripts/fetch_models.py --all`); without them the read raises
            `EngineLoadError`.

    Raises:
        ValueError: A field is out of range. The message names the field and the valid values: `languages` is empty
            or holds anything but `"es"` and `"en"`; `review`, `backend` or `correct` is not one of its values;
            `second_reading` is not `True`, `False` or `"thorough"`; `recycle_pages` is under 4; `processes` or
            `threads` is under 1.
        dataclasses.FrozenInstanceError: You assigned to a field of an existing `Options`.

    Example:
        ```python
        import extract

        options = extract.Options(languages=("es",), review="more", work_dir="work")
        print(options.languages, options.review, options.correct, options.thorough)

        try:
            extract.Options(review="most")
        except ValueError as error:
            print(error)
        ```

        ```text
        # output
        ('es',) more off False
        review must be one of ('fewer', 'balanced', 'more'), got 'most'
        ```

    See also:
        `read`, `Reader`, `load`, `Document.set_mode`.
    """

    languages: tuple[str, ...] = LANGUAGES
    review: str = "balanced"
    recycle_pages: int = RECYCLE_PAGES
    work_dir: Path | str | None = None
    processes: int | None = None
    threads: int | None = None
    backend: str = "auto"
    correct: str = "off"
    second_reading: bool | str = second_reading_decision.default_on()

    def __post_init__(self) -> None:
        if not self.languages or any(lang not in LANGUAGES for lang in self.languages):
            raise ValueError(f"languages must be a non-empty subset of {LANGUAGES}, got {self.languages!r}")
        if self.review not in REVIEW_LEVELS:
            raise ValueError(f"review must be one of {REVIEW_LEVELS}, got {self.review!r}")
        if self.backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {self.backend!r}")
        if self.correct not in CORRECT_MODES:
            raise ValueError(f"correct must be one of {CORRECT_MODES}, got {self.correct!r}")
        if not second_reading_decision.valid_mode(self.second_reading):
            raise ValueError(f"second_reading must be True, False or 'thorough', got {self.second_reading!r}")
        if self.recycle_pages < CHUNK_PAGES:
            raise ValueError(f"recycle_pages must be at least {CHUNK_PAGES}")
        if self.processes is not None and self.processes < 1:
            raise ValueError("processes must be at least 1")
        if self.threads is not None and self.threads < 1:
            raise ValueError("threads must be at least 1")
        object.__setattr__(self, "languages", tuple(self.languages))
        if self.work_dir is not None:
            object.__setattr__(self, "work_dir", Path(self.work_dir))

    @property
    def thorough(self) -> bool:
        """True when `second_reading` is the exact string `"thorough"`, and only then.

        `True` is not thorough: it selects the default second reading.
        """
        return second_reading_decision.is_thorough(self.second_reading)
