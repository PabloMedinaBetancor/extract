"""Rows, anchors and numeric tokens: the geometry the invoice reader works on."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from rapidfuzz.distance import Levenshtein

from extract.types import Cell, Word

ANCHORS = json.loads((Path(__file__).with_name("anchors.json")).read_text(encoding="utf-8"))

NUMBER = re.compile(r"^[-+(.,']?[€$£]?\d[\d.,]*[)]?[€$£]?$|^\d[\d.,]*-$")
MIN_LINE_WORDS = 4
RATE = re.compile(r"^\d+(?:[.,]\d+)?%$")


def norm(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.replace("°", "º"))
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c for c in stripped.casefold() if c.isalnum())


def same_token(token: str, anchor: str) -> bool:
    """Equal, or one letter off in a word long enough that a misread is likelier than another word."""
    if token == anchor:
        return True
    reach = 2 if len(anchor) >= 9 else 1 if len(anchor) >= 5 else 0
    return reach > 0 and abs(len(token) - len(anchor)) <= reach and Levenshtein.distance(token, anchor) <= reach


def is_number(word: Word) -> bool:
    return NUMBER.match(word.text) is not None


def height(word: Word) -> float:
    return max(word.box[3] - word.box[1], 1.0)


@dataclass(frozen=True)
class Row:
    words: tuple[Word, ...]
    top: float
    bottom: float

    @property
    def centre(self) -> float:
        return (self.top + self.bottom) / 2

    def norms(self) -> list[tuple[str, int]]:
        return [(n, i) for i, w in enumerate(self.words) if (n := norm(w.text))]


def page_slope(words: Sequence[Word]) -> float:
    """The skew of the page as the median slope of its long text lines (y over x of the word centres), 0 if unsure."""
    slopes = []
    lines: dict[int, list[Word]] = {}
    for word in words:
        lines.setdefault(word.line_id, []).append(word)
    for line in lines.values():
        if len(line) < MIN_LINE_WORDS:
            continue
        xs = [(w.box[0] + w.box[2]) / 2 for w in line]
        ys = [(w.box[1] + w.box[3]) / 2 for w in line]
        mean_x, mean_y = sum(xs) / len(xs), sum(ys) / len(ys)
        spread = sum((x - mean_x) ** 2 for x in xs)
        if spread > 0:
            slopes.append(sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys, strict=True)) / spread)
    return median(slopes) if len(slopes) >= 3 else 0.0


def rows_of(words: Sequence[Word]) -> list[Row]:
    """Cluster words into text rows by the vertical centre of their boxes after undoing the page's skew, each row
    left to right."""
    slope = page_slope(words)

    def level(word: Word) -> float:
        return (word.box[1] + word.box[3]) / 2 - slope * (word.box[0] + word.box[2]) / 2

    ordered = sorted(words, key=level)
    groups: list[list[Word]] = []
    for word in ordered:
        if groups:
            last = groups[-1]
            reference = median(height(w) for w in last)
            if abs(level(word) - median(level(w) for w in last)) <= 0.5 * reference:
                last.append(word)
                continue
        groups.append([word])
    rows = []
    for group in groups:
        group.sort(key=lambda w: w.box[0])
        rows.append(Row(tuple(group), min(w.box[1] for w in group), max(w.box[3] for w in group)))
    return rows


def find_anchor(row: Row, anchors: list[list[str]]) -> tuple[int, int] | None:
    """The first anchor of the list found as consecutive words of the row, as (first word, one past the last)."""
    norms = row.norms()
    for anchor in anchors:
        for start in range(len(norms) - len(anchor) + 1):
            window = norms[start : start + len(anchor)]
            if all(same_token(n, a) for (n, _), a in zip(window, anchor, strict=True)):
                return norms[start][1], norms[start + len(anchor) - 1][1] + 1
    return None


def starts_with(row: Row, anchors: list[list[str]]) -> int | None:
    """One past the last word of an anchor that opens the row, or None."""
    found = find_anchor(row, anchors)
    if found is None:
        return None
    first = row.norms()[0][1] if row.norms() else None
    return found[1] if found[0] == first else None


def run_from(words: tuple[Word, ...], start: int) -> list[Word]:
    """Consecutive words from `start` until a gap wider than a column gutter."""
    taken: list[Word] = []
    for word in words[start:]:
        if taken and word.box[0] - taken[-1].box[2] > 2.5 * max(height(word), height(taken[-1])):
            break
        taken.append(word)
    return taken


def cell_of(words: list[Word] | tuple[Word, ...]) -> Cell | None:
    if not words:
        return None
    return Cell(" ".join(w.text for w in words), tuple(w.id for w in words), words[0].page)
