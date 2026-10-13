"""Folds of the development documents: every document in exactly one fold, the bands balanced by greedy fill.

A pure function of the document list and a seed, so the folds never depend on the order the pages are met in. The
nested protocol assigns again among the training documents of an outer fold to get its inner folds.
"""

import hashlib
from collections import Counter
from collections.abc import Mapping

FOLDS = 5


def _rank(seed: int, document: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{seed}\x1f{document}".encode()).digest()[:8], "big")


def assign_folds(documents: Mapping[str, tuple[str, int]], folds: int = FOLDS, seed: int = 1) -> dict[str, int]:
    """Document id -> fold. `documents` maps an id to (band, number of words). Within a band the documents go in the
    order of a hash of their id, each to the fold that holds the fewest words of that band so far (the lowest fold
    on a tie)."""
    if folds < 2:
        raise ValueError("at least two folds are needed")
    assigned: dict[str, int] = {}
    load: dict[str, Counter] = {}
    for document in sorted(documents, key=lambda d: (documents[d][0], _rank(seed, d), d)):
        band, words = documents[document]
        counts = load.setdefault(band, Counter())
        fold = min(range(folds), key=lambda f: (counts[f], f))
        counts[fold] += max(words, 1)
        assigned[document] = fold
    return assigned
