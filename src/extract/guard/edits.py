"""Edit distance with the substitutions an OCR makes priced below the others."""

import json
from functools import lru_cache

from rapidfuzz.distance import Levenshtein

from extract.guard import lexicon
from extract.guard.dictionary import fold_accents


@lru_cache(maxsize=1)
def confusion_costs() -> dict[tuple[str, str], float]:
    """Cost of reading the first string for the second (and the reverse), from `data/confusions.json`."""
    costs = {}
    for entry in json.loads((lexicon.DATA_DIR / "confusions.json").read_text(encoding="utf-8")):
        first, second = fold_accents(entry["a"]), fold_accents(entry["b"])
        if first != second:
            costs[(first, second)] = costs[(second, first)] = float(entry["cost"])
    return costs


@lru_cache(maxsize=1)
def _two_character_swaps() -> tuple[tuple[str, str, float], ...]:
    """The confusions of two characters for one (`rn` for `m`), as (long, short, cost)."""
    return tuple((a, b, cost) for (a, b), cost in confusion_costs().items() if len(a) == 2 and len(b) == 1)


@lru_cache(maxsize=1 << 17)
def weighted_distance(word: str, candidate: str) -> float:
    """Edit distance in which the swaps an OCR makes (`c`/`e`, `0`/`o`, `rn`/`m`...) cost less than other edits;
    case and accents do not count."""
    costs = confusion_costs()
    word, candidate = fold_accents(word), fold_accents(candidate)
    for long, short, cost in _two_character_swaps():
        if (long in word and word.replace(long, short) == candidate) or (
            long in candidate and candidate.replace(long, short) == word
        ):
            return cost
    total = 0.0
    for tag, source, target in Levenshtein.editops(word, candidate):
        total += costs.get((word[source], candidate[target]), 1.0) if tag == "replace" else 1.0
    return total
