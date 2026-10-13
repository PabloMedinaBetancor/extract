"""Why a word is doubtful: the signals that push its score towards "wrong", grouped into a few readable reasons.

LightGBM's per-signal contributions add up to the raw score. For a doubtful word the three largest groups of
contributions that raise the score become reasons, each an i18n key with its parameters.
"""

import numpy as np

from extract.guard import dictionary, lexicon
from extract.guard.signals import INDEX
from extract.types import Reason

MAX_REASONS = 3
MIN_PUSH = 1e-6

GROUPS = {
    "guard.reason.low_char_conf": ("conf_mean", "conf_min", "low_conf_chars"),
    "guard.reason.not_in_dictionary": ("in_dictionary", "in_dictionary_folded", "log_frequency", "char_lm_logp"),
    "guard.reason.close_to": ("dict_distance",),
    "guard.reason.mixed_letters_digits": ("mixed_letters_digits", "has_digit"),
    "guard.reason.odd_chars": ("odd_chars",),
    "guard.reason.unusual_height": ("height_ratio",),
    "guard.reason.unusual_width": ("width_per_char_ratio",),
    "guard.reason.bad_number_format": ("number_format_ok",),
    "guard.reason.low_page_quality": ("page_conf_mean", "line_conf_mean"),
    "guard.reason.neighbours_unusual": ("prev_in_dictionary", "next_in_dictionary"),
}
KEYS = tuple(GROUPS)
_COLUMNS = {key: [INDEX[name] for name in names] for key, names in GROUPS.items()}
# LightGBM can credit the digit signals with a push on a word that has no digit at all (a misread accented word was
# shown as a badly written number), so these reasons are only given to words with a digit.
DIGIT_REASONS = ("guard.reason.bad_number_format", "guard.reason.mixed_letters_digits")


def explain(contributions: np.ndarray, word, signals_row: np.ndarray, languages) -> tuple[Reason, ...]:
    """The reasons of one doubtful word: `contributions` is its row of per-signal contributions (bias last)."""
    has_digit = any(c.isdigit() for c in word.text)
    pushes = {
        key: float(contributions[cols].sum()) for key, cols in _COLUMNS.items() if has_digit or key not in DIGIT_REASONS
    }
    chosen = [key for key, push in sorted(pushes.items(), key=lambda item: -item[1]) if push > MIN_PUSH][:MAX_REASONS]
    out = []
    for key in chosen:
        reason = _with_parameters(key, word, signals_row, languages)
        if reason is not None:
            out.append(reason)
    return tuple(out)


def _with_parameters(key: str, word, signals_row, languages) -> Reason | None:
    if key == "guard.reason.low_char_conf":
        chars = word.char_conf
        if not chars:
            return None
        position = min(range(len(chars)), key=lambda i: chars[i])
        if position >= len(word.text):
            return None
        return Reason(key, {"char": word.text[position], "conf": round(float(chars[position]), 2)})
    if key == "guard.reason.close_to":
        near = _nearest(word.text, languages)
        return Reason(key, {"word": near}) if near else None
    return Reason(key, {})


def _nearest(text: str, languages) -> str | None:
    """The listed word nearest to the letters of `text`, or None when there is none within two edits."""
    letters = "".join(c for c in text if c.isalpha())
    if not letters:
        return None
    best = None
    for language in lexicon.available(languages):
        book = dictionary.load(language)
        found = book.candidates(letters)
        if found and (best is None or (found[0][1], found[0][0]) < best[0]):
            best = ((found[0][1], found[0][0]), book.words[found[0][0]])
    return best[1] if best else None
