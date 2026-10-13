"""Second reading: which units of a page are read again, how a reader's words are aligned to the first reading's, and
the view of the readings that a cap leaves visible.

Everything here is a pure function of the first reading (or of the stored readings), so the same page always gives
the same units and the same rows, in training and at run time.
"""

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from extract import second_reading
from extract.types import Word

MAX_REREAD_LINES = 16
MAX_ZONES = 40
ZONE_WORDS = 5
TAU_CHAR = 0.9
TAU_WORD = 0.9

ADD_DROP_COST = 0.5
JOIN_CUT_COST = 1.0
UNALIGNED_SHARE = 0.5


STUDY_UNITS = 16
DEFAULT_UNIT, DEFAULT_READERS = second_reading.cell()  # the packaged decision (line + the cheap set)


@dataclass(frozen=True)
class RereadConfig:
    """What pass 1 keeps of a page for the second reading: the thresholds of the selection rule and how many units of
    each kind (the first `study_units` by rank) get their crop stored. They are part of the page's cache key; the unit
    kind and the readers are parameters of pass 2 and are not."""

    tau_char: float = TAU_CHAR
    tau_word: float = TAU_WORD
    study_units: int = STUDY_UNITS  # lines
    study_zones: int = MAX_ZONES

    def __post_init__(self) -> None:
        if not 1 <= self.study_units <= MAX_REREAD_LINES:
            raise ValueError(f"study_units must be between 1 and {MAX_REREAD_LINES}")
        if not 1 <= self.study_zones <= MAX_ZONES:
            raise ValueError(f"study_zones must be between 1 and {MAX_ZONES}")

    def key(self) -> str:
        return f"u:{self.tau_char}:{self.tau_word}:{MAX_REREAD_LINES}/{MAX_ZONES}:{self.study_units}/{self.study_zones}"

    def units(self, words: Sequence["Word"], unit: str) -> list["Unit"]:
        """The units of one kind that have a stored crop, most doubtful first."""
        limit = self.study_units if unit == "line" else self.study_zones
        if unit == "line":
            return select_lines(words, self.tau_char, self.tau_word, limit)
        if unit == "zone":
            return select_zones(words, self.tau_char, self.tau_word, limit)
        raise ValueError(f"unknown unit: {unit!r}")


@dataclass(frozen=True)
class Unit:
    """A run of words of one line that a reader reads again. `indices` are positions in the page's word tuple, in
    reading order; `suspects` are the ones that passed the selection rule; `rank` is 0 for the most doubtful unit."""

    kind: str
    line_id: int
    indices: tuple[int, ...]
    suspects: tuple[int, ...]
    rank: int = 0


def is_suspect(word: Word, tau_char: float = TAU_CHAR, tau_word: float = TAU_WORD) -> bool:
    """A word the recogniser doubted: one of its characters, or the word as a whole, fell below the thresholds."""
    if word.char_conf is None or word.conf is None:
        return False
    return min(word.char_conf, default=1.0) < tau_char or word.conf < tau_word


def lines_of(words: Sequence[Word]) -> list[list[int]]:
    lines: list[list[int]] = []
    last = None
    for index, word in enumerate(words):
        if word.line_id != last:
            lines.append([])
            last = word.line_id
        lines[-1].append(index)
    return lines


def _lowest(words: Sequence[Word], indices: Sequence[int]) -> float:
    return min((min(words[i].char_conf or (1.0,)) for i in indices), default=1.0)


def select_lines(
    words: Sequence[Word], tau_char: float = TAU_CHAR, tau_word: float = TAU_WORD, limit: int = MAX_REREAD_LINES
) -> list[Unit]:
    """The lines that hold suspect words, the ones with most suspects first and then the lowest character confidence."""
    found = []
    for position, line in enumerate(lines_of(words)):
        suspects = tuple(i for i in line if is_suspect(words[i], tau_char, tau_word))
        if suspects:
            found.append((-len(suspects), _lowest(words, line), position, line, suspects))
    found.sort(key=lambda item: item[:3])
    return [
        Unit("line", words[line[0]].line_id, tuple(line), suspects, rank)
        for rank, (_, _, _, line, suspects) in enumerate(found[:limit])
    ]


def _runs(positions: Sequence[int]) -> list[list[int]]:
    runs: list[list[int]] = []
    for position in positions:
        if runs and position == runs[-1][-1] + 1:
            runs[-1].append(position)
        else:
            runs.append([position])
    return runs


def _line_zones(suspects: Sequence[int], length: int) -> list[tuple[int, int, list[int]]]:
    """(first, last, suspects) windows of a line, in positions of the line: each run of suspects widened by one word on
    each side and runs that touch merged. A window of more than ZONE_WORDS words is cut by words: the suspects are
    grouped while they span at most ZONE_WORDS - 2 positions, and each group is widened by its neighbours without
    overlapping the group before or the one after, so no zone has more than ZONE_WORDS words and no word is in two."""
    windows: list[list] = []
    for run in _runs(suspects):
        first, last = max(0, run[0] - 1), min(length - 1, run[-1] + 1)
        if windows and first <= windows[-1][1] + 1:
            windows[-1][1] = last
            windows[-1][2].extend(run)
        else:
            windows.append([first, last, list(run)])
    zones: list[tuple[int, int, list[int]]] = []
    for first, last, held in windows:
        if last - first + 1 <= ZONE_WORDS:
            zones.append((first, last, held))
            continue
        groups: list[list[int]] = []
        for position in held:
            if groups and position - groups[-1][0] + 1 <= ZONE_WORDS - 2:
                groups[-1].append(position)
            else:
                groups.append([position])
        taken = first - 1
        for number, group in enumerate(groups):
            start = max(taken + 1, group[0] - 1, first)
            following = groups[number + 1][0] if number + 1 < len(groups) else None
            end = min(last, group[-1] + 1)
            if following is not None and end >= following:
                end = following - 1
            zones.append((start, end, group))
            taken = end
    return zones


def select_zones(
    words: Sequence[Word], tau_char: float = TAU_CHAR, tau_word: float = TAU_WORD, limit: int = MAX_ZONES
) -> list[Unit]:
    """Runs of suspect words with one neighbour on each side (at most ZONE_WORDS words), the most doubtful first."""
    found = []
    for line_position, line in enumerate(lines_of(words)):
        suspects = [p for p, i in enumerate(line) if is_suspect(words[i], tau_char, tau_word)]
        for first, last, held in _line_zones(suspects, len(line)):
            indices = tuple(line[first : last + 1])
            found.append((_lowest(words, indices), line_position, first, indices, tuple(line[p] for p in held)))
    found.sort(key=lambda item: item[:3])
    return [
        Unit("zone", words[indices[0]].line_id, indices, suspects, rank)
        for rank, (_, _, _, indices, suspects) in enumerate(found[:limit])
    ]


def select_units(
    words: Sequence[Word], unit: str, tau_char: float = TAU_CHAR, tau_word: float = TAU_WORD
) -> list[Unit]:
    if unit == "line":
        return select_lines(words, tau_char, tau_word)
    if unit == "zone":
        return select_zones(words, tau_char, tau_word)
    raise ValueError(f"unknown unit: {unit!r}")


def view(words: Sequence[Word], cap: int) -> tuple[Word, ...]:
    """The words as the chooser sees them at a runtime cap: the readings of units with rank `cap` or more are hidden (a reading with no rank belongs to no unit and stays),
    and a word left with none has `readings = None`. Words that lose nothing are returned as they are."""
    out = []
    for word in words:
        if word.readings is None:
            out.append(word)
            continue
        kept = tuple(r for r in word.readings if r.rank is None or r.rank < cap)
        if len(kept) == len(word.readings):
            out.append(word)
        else:
            out.append(dataclasses.replace(word, readings=kept or None))
    return tuple(out)


@dataclass(frozen=True)
class Aligned:
    """What one word of a unit got from a reader: its text (None when the reader gave nothing for it), the indices
    of the first reading's words the reader saw as one (a join, group of 2 or 3, else just the word) and the indices
    of the reader's words it came from."""

    text: str | None
    group: tuple[int, ...]
    read: tuple[int, ...]


def _squeeze(texts: Sequence[str]) -> str:
    return "".join(texts).replace(" ", "")


def align(first: Sequence[str], read: Sequence[str]) -> list[Aligned] | None:
    """Align the words of a reader to the first reading's words of one unit, by dynamic programming over groups of
    one to three words on either side (1:1, n:1 a join, 1:m a cut, and one word alone added or dropped). Returns one
    `Aligned` per first word, or None when the alignment costs more than half the unit's characters or the reader
    gave more than `len(first) + 1` words. A first word the reader gave nothing for has text None."""
    n, m = len(first), len(read)
    if n == 0 or m > n + 1:
        return None
    steps = [(1, 1), (2, 1), (3, 1), (1, 2), (1, 3), (1, 0), (0, 1)]
    best = [[float("inf")] * (m + 1) for _ in range(n + 1)]
    back: list[list[tuple[int, int] | None]] = [[None] * (m + 1) for _ in range(n + 1)]
    best[0][0] = 0.0
    for i in range(n + 1):
        for j in range(m + 1):
            if best[i][j] == float("inf"):
                continue
            for a, b in steps:
                ni, nj = i + a, j + b
                if ni > n or nj > m:
                    continue
                left, right = _squeeze(first[i:ni]), _squeeze(read[j:nj])
                if a == 0 or b == 0:
                    cost = len(left) + len(right) + ADD_DROP_COST
                else:
                    cost = Levenshtein.distance(left, right) + JOIN_CUT_COST * (a + b - 2)
                if best[i][j] + cost < best[ni][nj]:
                    best[ni][nj] = best[i][j] + cost
                    back[ni][nj] = (a, b)
    if best[n][m] > UNALIGNED_SHARE * max(1, len(_squeeze(first))):
        return None
    out: list[Aligned | None] = [None] * n
    i, j = n, m
    while (i, j) != (0, 0):
        step = back[i][j]
        a, b = step
        i0, j0 = i - a, j - b
        if b == 0:
            out[i0] = Aligned(None, (i0,), ())
        elif a:
            group = tuple(range(i0, i))
            joined = " ".join(read[j0:j])
            for k in range(i0, i):
                out[k] = Aligned(joined, group if a > 1 else (k,), tuple(range(j0, j)))
        i, j = i0, j0
    return [item for item in out if item is not None]
