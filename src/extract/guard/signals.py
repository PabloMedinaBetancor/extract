"""The 21 signals the detector looks at for every word of a page.

Each signal is a float32 column; NaN means the signal does not apply (a word without digits has no
number format, a language without a word list has no dictionary signals). Continuous values are
rounded to 4 decimals so Linux and Windows give the same matrix. The schema (names, order, signs
and the extractor version) has a fingerprint that is stored in the model file.

A signal is a function of the OCR output only: nothing here reads a truth word.
"""

import hashlib
import json
import math
import re
from collections.abc import Sequence

import numpy as np

from extract.guard import dictionary, lexicon
from extract.types import Word

EXTRACTOR_VERSION = 1
LOW_CONFIDENCE = 0.90

# (name, monotone sign): -1 never more suspicious when higher, +1 never less, 0 free.
SCHEMA = (
    ("conf_mean", -1),
    ("conf_min", -1),
    ("low_conf_chars", 1),
    ("length", 0),
    ("in_dictionary", -1),
    ("in_dictionary_folded", -1),
    ("log_frequency", -1),
    ("dict_distance", 1),
    ("char_lm_logp", -1),
    ("has_digit", 0),
    ("mixed_letters_digits", 0),
    ("number_format_ok", -1),
    ("case_pattern", 0),
    ("odd_chars", 1),
    ("hyphen_fragment", 0),
    ("height_ratio", 0),
    ("width_per_char_ratio", 0),
    ("line_conf_mean", 0),
    ("page_conf_mean", 0),
    ("prev_in_dictionary", 0),
    ("next_in_dictionary", 0),
)
COLUMNS = tuple(name for name, _ in SCHEMA)
SIGNS = tuple(sign for _, sign in SCHEMA)
INDEX = {name: i for i, name in enumerate(COLUMNS)}

_LETTER_RUNS = re.compile(r"[^\W\d_]+")
_EDGES = re.compile(r"^\W+|\W+$")
_CURRENCY = "".join(map(chr, (0x20AC, 0xA3)))
_ORDINAL = "".join(map(chr, (0xBA, 0xAA, 0xB0)))
_PUNCTUATION = ".,;:!?()[]{}%-/&@#$'\"+*=<>"
_TYPOGRAPHIC = "".join(map(chr, (0xAB, 0xBB, 0x201C, 0x201D, 0x2018, 0x2019, 0xA7, 0xBF, 0xA1, 0x2013, 0x2014)))
_ALLOWED = set(_PUNCTUATION + _TYPOGRAPHIC + _CURRENCY + _ORDINAL)
_CLITICS = {"s", "t", "d", "m"}
_NUMBER_SHAPES = {
    "es": re.compile(rf"^[-+(]?[{_CURRENCY}$]?(\d{{1,3}}(\.\d{{3}})+(,\d+)?|\d+(,\d+)?)[{_CURRENCY}$%]?\)?$"),
    "en": re.compile(rf"^[-+(]?[{_CURRENCY}$]?(\d{{1,3}}(,\d{{3}})+(\.\d+)?|\d+(\.\d+)?)[{_CURRENCY}$%]?\)?$"),
}
_OTHER_SHAPES = (
    re.compile(r"^\d+([.]\d+)+[.]?$"),
    re.compile(rf"^\d+[{_ORDINAL}]$"),
    re.compile(r"^\d{1,2}:\d{2}(:\d{2})?$"),
    re.compile(r"^\+?\d{9,15}$"),
    re.compile(r"^\d{8}[A-Za-z]$"),
    re.compile(r"^[XYZxyz]\d{7}[A-Za-z]$"),
    re.compile(r"^[A-Za-z]\d{7}[0-9A-Ja-j]$"),
    re.compile(r"^[A-Za-z]{2}\d{2}[A-Za-z0-9]{10,30}$"),
)
_DATE = re.compile(r"^(\d{1,2})([/.\-])(\d{1,2})\2(\d{2}|\d{4})$")
_ISO_DATE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})$")


def schema_fingerprint() -> str:
    """SHA-256 of the schema: it changes whenever a signal, its order, its sign or the extractor changes."""
    payload = json.dumps({"version": EXTRACTOR_VERSION, "columns": SCHEMA}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def extract(words: Sequence[Word], languages=("es", "en"), locale: str = "es", cache: dict | None = None) -> np.ndarray:
    """The signal matrix (one row per word, in the order given) of one page."""
    matrix = np.full((len(words), len(COLUMNS)), np.nan, dtype=np.float32)
    if not words:
        return matrix
    cache = {} if cache is None else cache
    usable = lexicon.available(languages)
    _fill_cache(words, usable, locale, cache)
    tokens = [cache[(w.text, usable, locale)] for w in words]
    page_conf = _mean([w.conf for w in words])
    lines = _group_by_line(words)
    fragments = _hyphen_fragments(words, lines)
    for i, word in enumerate(words):
        row = matrix[i]
        for column, value in tokens[i].items():
            row[INDEX[column]] = value
        _confidence_signals(row, word)
        row[INDEX["hyphen_fragment"]] = 1.0 if i in fragments else 0.0
        row[INDEX["page_conf_mean"]] = page_conf
    for members in lines.values():
        _line_signals(matrix, words, members, tokens)
    return np.round(matrix, 4)


def _fill_cache(words: Sequence[Word], languages: tuple[str, ...], locale: str, cache: dict) -> None:
    """Compute the token signals of every text not seen yet; the character model is read once for all of them."""
    fresh = {}
    for word in words:
        key = (word.text, languages, locale)
        if key not in cache and key not in fresh:
            fresh[key] = _parse(word.text)
    lm = {}
    if languages:
        wanted = sorted({"".join(parts) for parts, has_digit in fresh.values() if parts and not has_digit})
        batch = dictionary.GramBatch(wanted)
        for language in languages:
            scored = dictionary.load(language).score(batch)
            lm.update({(language, word): value for word, value in scored.items()})
    for (text, _, _), (parts, has_digit) in fresh.items():
        cache[(text, languages, locale)] = _compute_token(text, parts, has_digit, languages, locale, lm)


def _parse(text: str) -> tuple[list[str], bool]:
    """The runs of letters of the word without the punctuation at its ends, and whether it has digits."""
    return _LETTER_RUNS.findall(_EDGES.sub("", text)), any(c.isdigit() for c in text)


def _compute_token(
    text: str, letters: list[str], has_digit: bool, languages: tuple[str, ...], locale: str, lm: dict
) -> dict[str, float]:
    core = _EDGES.sub("", text)
    out = {
        "length": float(len(text)),
        "has_digit": float(has_digit),
        "mixed_letters_digits": float(has_digit and any(c.isalpha() for c in core)),
        "number_format_ok": _number_format(core, locale) if has_digit else math.nan,
        "case_pattern": _case_pattern(letters),
        "odd_chars": float(sum(1 for c in text if not (c.isalnum() or c.isspace() or c in _ALLOWED))),
    }
    if languages and letters and not has_digit:
        out.update(_dictionary_signals(letters, languages, lm))
    return out


def _dictionary_signals(parts: list[str], languages: tuple[str, ...], lm: dict) -> dict[str, float]:
    """Dictionary signals over the page languages: a word listed in any of them is a word, and for one
    that is not, the nearest word of any language counts."""
    joined = "".join(parts)
    books = [dictionary.load(language) for language in languages]
    present = [[_part_in(book, p) for p in parts] for book in books]
    listed = [i for i, found in enumerate(present) if all(f[0] for f in found)]
    folded = any(all(f[1] for f in found) for found in present)
    if listed:
        best = max(
            listed, key=lambda i: (min(books[i].log_frequency(p) for p in parts), _logp(lm, languages[i], joined))
        )
        freq, distance, chosen = min(books[best].log_frequency(p) for p in parts), 0, best
    else:
        freq, distance, chosen = 0.0, dictionary.MAX_DISTANCE + 1, 0
        for i, book in enumerate(books):
            near = max(_distance(book, p) for p in parts)
            if near < distance or (
                near == distance and _logp(lm, languages[i], joined) > _logp(lm, languages[chosen], joined)
            ):
                distance, chosen = near, i
            if distance <= 1:
                break
    logp = lm[(languages[chosen], joined)]
    return {
        "in_dictionary": float(bool(listed)),
        "in_dictionary_folded": float(folded),
        "log_frequency": round(freq, 4),
        "dict_distance": float(distance),
        "char_lm_logp": math.nan if logp is None else logp,
    }


def _logp(lm: dict, language: str, joined: str) -> float:
    value = lm[(language, joined)]
    return -99.0 if value is None else value


def _part_in(book: dictionary.Dictionary, part: str) -> tuple[bool, bool]:
    if book.rank_of(part) >= 0 or (len(part) == 1 and part.casefold() in _CLITICS):
        return True, True
    return False, book.has_folded(part)


def _distance(book: dictionary.Dictionary, part: str) -> int:
    near = book.candidates(part)
    return near[0][1] if near else dictionary.MAX_DISTANCE + 1


def _case_pattern(parts: list[str]) -> float:
    """0 lower, 1 Title, 2 UPPER, 3 irregular; the parts of a hyphenated word must agree."""
    if not parts:
        return math.nan
    patterns = {_part_case(p) for p in parts}
    return patterns.pop() if len(patterns) == 1 else 3.0


def _part_case(part: str) -> float:
    if part.islower():
        return 0.0
    if part.isupper():
        return 2.0
    return 1.0 if part[0].isupper() and part[1:].islower() else 3.0


def _number_format(core: str, locale: str) -> float:
    if _NUMBER_SHAPES.get(locale, _NUMBER_SHAPES["es"]).match(core) or any(p.match(core) for p in _OTHER_SHAPES):
        return 1.0
    return float(_is_date(core))


def _is_date(core: str) -> bool:
    match = _DATE.match(core)
    if match:
        first, second = int(match.group(1)), int(match.group(3))
        return 1 <= min(first, second) and max(first, second) <= 31 and (first <= 12 or second <= 12)
    iso = _ISO_DATE.match(core)
    return bool(iso and 1 <= int(iso.group(2)) <= 12 and 1 <= int(iso.group(3)) <= 31)


def _confidence_signals(row: np.ndarray, word: Word) -> None:
    chars = [c for c in word.char_conf or () if math.isfinite(c)]
    if word.conf is not None and math.isfinite(word.conf):
        row[INDEX["conf_mean"]] = word.conf
    if chars:
        row[INDEX["conf_min"]] = min(chars)
        row[INDEX["low_conf_chars"]] = sum(1 for c in chars if c < LOW_CONFIDENCE)


def _mean(values) -> float:
    values = [v for v in values if v is not None and math.isfinite(v)]
    return float(np.mean(values)) if values else math.nan


def _group_by_line(words: Sequence[Word]) -> dict[int, list[int]]:
    lines: dict[int, list[int]] = {}
    for i, word in enumerate(words):
        lines.setdefault(word.line_id, []).append(i)
    for members in lines.values():
        members.sort(key=lambda i: words[i].idx_in_line)
    return lines


def _hyphen_fragments(words: Sequence[Word], lines: dict[int, list[int]]) -> set[int]:
    """Words that are one half of a word hyphenated at the end of a line."""
    ordered = [lines[k] for k in sorted(lines)]
    fragments: set[int] = set()
    for here, following in zip(ordered, ordered[1:], strict=False):
        last = words[here[-1]].text
        if len(last) > 1 and last.endswith("-") and any(c.isalpha() for c in last):
            fragments.update((here[-1], following[0]))
    return fragments


def _line_signals(matrix: np.ndarray, words: Sequence[Word], members: list[int], tokens: list[dict]) -> None:
    heights = np.array([words[i].box[3] - words[i].box[1] for i in members], dtype=np.float64)
    per_char = np.array(
        [(words[i].box[2] - words[i].box[0]) / max(1, len(words[i].text)) for i in members], dtype=np.float64
    )
    median_height, median_width = float(np.median(heights)), float(np.median(per_char))
    line_conf = _mean([words[i].conf for i in members])
    for position, i in enumerate(members):
        row = matrix[i]
        row[INDEX["height_ratio"]] = heights[position] / median_height if median_height > 0 else math.nan
        row[INDEX["width_per_char_ratio"]] = per_char[position] / median_width if median_width > 0 else math.nan
        row[INDEX["line_conf_mean"]] = line_conf
        if position > 0:
            row[INDEX["prev_in_dictionary"]] = tokens[members[position - 1]].get("in_dictionary", math.nan)
        if position + 1 < len(members):
            row[INDEX["next_in_dictionary"]] = tokens[members[position + 1]].get("in_dictionary", math.nan)
