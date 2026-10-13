"""`Guard`: says how likely each OCR word is wrong, and never changes a word.

`Guard.score_page` runs in the main process after the OCR of a page. It needs no network and holds
nothing between pages except the loaded model, the dictionaries and their caches.
"""

import logging
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from extract.guard import fixer, fixing, readings, reasons, signals
from extract.guard.model import GuardModel, contributions
from extract.types import PageReading, Scored

LOG = logging.getLogger(__name__)
MAX_WORDS = 5000
MODEL_ENV = "EXTRACT_GUARD_MODEL"
BUDGETS = (0.03, 0.05, 0.08)


def default_model_path() -> Path:
    return Path(os.environ.get(MODEL_ENV) or Path(__file__).resolve().parents[3] / "models" / "guard.model")


class Guard:
    def __init__(self, model: GuardModel, thorough: bool = False) -> None:
        self.model = model
        self.thorough = thorough  # the second reading the product runs is the thorough mode (its model is this one)
        self.last_warnings: list[dict] = []

    @classmethod
    def load(cls, path: Path | str | None = None, thorough: bool = False) -> "Guard":
        """The packaged model, or the one at `path`. Raises ModelMismatch if its signal schema is not this one.
        `thorough` says the product reads the thorough mode, whose readings this model has to be trained on."""
        return cls(GuardModel.load(Path(path) if path else default_model_path()), thorough)

    @property
    def info(self) -> dict:
        meta = self.model.meta
        return {
            "model_id": self.model.model_id(),
            "schema": meta["schema"],
            "thresholds": meta["thresholds"],
            "number_divisor": meta["number_divisor"],
            "training": meta["training"],
            "lightgbm_version": meta["lightgbm_version"],
            "fix": self._fix_info(),
        }

    def _fix_info(self) -> dict | None:
        """What the model's fix block says about when it applies a fix (None for a model with no chooser)."""
        block = self.model.fix
        if block is None:
            return None
        keys = (
            "t_gen",
            "t_safe",
            "m",
            "precision_floor",
            "t5_floor",
            "safe_promise",
            "language_floor",
            "by_language",
            "language_sensor",
        )
        return {key: block[key] for key in keys if key in block}

    def score_page(
        self,
        page: PageReading,
        languages: Sequence[str] = ("es", "en"),
        locale: str = "auto",
        budget: float = 0.05,
    ) -> list[Scored]:
        """A verdict for every word of the page: `ok` or `doubtful` with its probability of being wrong, or
        `unscored` (with a warning in `last_warnings`) when the page cannot be looked at."""
        words = page.words
        self.last_warnings = []
        if not words:
            return []
        if len(words) > MAX_WORDS:
            return self._unscored(page, "too_many_words")
        if not any(_usable(w.conf) for w in words):
            return self._unscored(page, "no_confidence")
        thresholds = self.model.thresholds(budget)
        try:
            return self._score(page, tuple(languages), locale, thresholds)
        except Exception:
            LOG.exception("the guard failed on page %s; its %d words are left unscored", page.number, len(words))
            return self._unscored(page, "error")

    def _words_as_trained(self, page: PageReading) -> tuple:
        """The words of the page with the second readings the model was trained on: those of its cell (readers, cap),
        through the same function as the training loader. When the product reads another cell than the model's, or the model
        was trained with none, the readings are left out and the page says so in `last_warnings`."""
        words = page.words
        block = self.model.fix
        if block is None or not any(getattr(w, "readings", None) for w in words):
            return words
        trained = readings.ReadingSpec.from_meta(block.get("readings"))
        product = readings.product_spec(getattr(self, "thorough", False))
        if trained is not None and trained.same_cell(product):
            return readings.apply(words, trained)
        self.last_warnings.append(
            {
                "key": "guard.warning.readings_differ",
                "page": page.number,
                "model": "none" if trained is None else trained.label,
                "product": product.label,
            }
        )
        return readings.apply(words, None)

    def _score(self, page: PageReading, languages, locale, thresholds) -> list[Scored]:
        words = self._words_as_trained(page)
        locale = _locale(words, locale)
        matrix = signals.extract(words, languages, locale)
        raw = self.model.raw(matrix)
        probability = self.model.probability(raw)
        t_word, t_number = thresholds
        has_digit = matrix[:, signals.INDEX["has_digit"]] == 1.0
        doubtful = probability >= np.where(has_digit, t_number, t_word)
        explained = self._reasons(words, matrix, doubtful, languages)
        fixes: dict = {}
        if self.model.fix is not None:
            fixes, withheld = fixing.propose(self.model, words, matrix, raw, languages)
            if withheld:
                self.last_warnings.append(
                    {"key": "guard.warning.fixes_withheld", "page": page.number, "cause": withheld}
                )
            suggested = {}
        else:
            suggested = self._suggestions(words, matrix, doubtful, languages)
        out = []
        for i, (w, p, d) in enumerate(zip(words, probability, doubtful, strict=True)):
            fix = fixes.get(i)
            suggestion = suggested.get(i, (None, None))
            if fix is not None and fix.op == "text" and d:
                suggestion = (fix.text, fix.p_fix)  # the v4 button keeps working (A2.3)
            out.append(Scored(w.id, float(p), "doubtful" if d else "ok", explained.get(i, ()), *suggestion, fix))
        return out

    def _reasons(self, words, matrix, doubtful, languages) -> dict:
        rows = np.nonzero(doubtful)[0]
        if not len(rows):
            return {}
        parts = contributions(self.model.detector, matrix[rows])
        return {int(i): reasons.explain(parts[k], words[i], matrix[i], languages) for k, i in enumerate(rows)}

    def _suggestions(self, words, matrix, doubtful, languages) -> dict:
        """The spelling suggestion of each doubtful plain word whose best candidate is likely enough to be the truth."""
        fixer_meta = self.model.meta.get("fixer")
        if self.model.ranker is None or fixer_meta is None:
            return {}
        table = fixer_meta["calibration"]
        out = {}
        for i in np.nonzero(doubtful)[0]:
            word = words[i]
            candidates = fixer.generate(word.text, languages)
            if not candidates:
                continue
            rows = fixer.features(
                word.text,
                word.char_conf,
                word.conf,
                float(matrix[i, signals.INDEX["in_dictionary"]]),
                candidates,
                languages,
            )
            scores = self.model.rank(rows)
            best = int(np.argmax(scores))
            confidence = float(np.interp(scores[best], table["scores"], table["probs"]))
            if confidence >= fixer_meta["t_suggest"]:
                out[int(i)] = (candidates[best].text, confidence)
        return out

    def _unscored(self, page: PageReading, cause: str) -> list[Scored]:
        self.last_warnings = [
            {"key": "guard.warning.unscored", "page": page.number, "words": len(page.words), "cause": cause}
        ]
        return [Scored(w.id, None, "unscored") for w in page.words]


def _usable(conf) -> bool:
    """A confidence the guard can use: a finite number (None, NaN and inf say nothing)."""
    return conf is not None and bool(np.isfinite(conf))


def _locale(words, locale: str) -> str:
    if locale in ("es", "en"):
        return locale
    from extract import checks

    return checks.detect_locale(words)
