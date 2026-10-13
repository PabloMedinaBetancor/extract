"""The baselines of the chooser, on the same rows, folds, calibration and operating-point rule.

`keep` changes nothing. The nearest-word rule is what anyone reading the problem tries first: replace a word that
is not in the dictionary by the listed word nearest to it (the most frequent among the equally near), when it is close. The
logistic regression sees the same features with the imputation and missing-value indicators a linear model needs,
and one-hot kinds, so it is not a straw man.
"""

import numpy as np

from bench.guard.baselines import LogisticBaseline
from bench.guard.operating import BROKEN, FIXED, STILL_WRONG, Choice
from bench.guard.pairs import Table
from extract.guard.features import INDEX, KINDS, OPS

NEAREST_LIMIT = 1.0


def one_hot(x: np.ndarray) -> np.ndarray:
    """The matrix with the edit kind and the operation replaced by one indicator per value."""
    keep = [i for name, i in INDEX.items() if name not in ("kind", "op")]
    kinds = np.stack([x[:, INDEX["kind"]] == k for k in range(len(KINDS))], axis=1)
    ops = np.stack([x[:, INDEX["op"]] == k for k in range(len(OPS))], axis=1)
    return np.hstack([x[:, keep], kinds.astype(np.float32), ops.astype(np.float32)])


def logistic_learner(l2: float = 10.0):
    """Factory for the protocol: a logistic regression fitted on the training rows."""

    def fit(x, y):
        fitted = LogisticBaseline(l2).fit(one_hot(x), y)
        return lambda rows: fitted.score(one_hot(rows))

    return fit


def nearest_word_choice(table: Table) -> Choice:
    """The sites where the rule replaces the word: not in the dictionary and a listed word within the limit."""
    n = table.sites
    index = np.zeros(n, dtype=np.int64)
    p_fix = np.full(n, -np.inf)
    outcome = np.zeros(n, dtype=np.int8)
    for s in range(n):
        block = table.block(s)
        x = table.x[block]
        if x[0, INDEX["in_dictionary"]] != 0:
            continue
        near = [
            (x[i, INDEX["weighted_distance"]], -x[i, INDEX["cand_log_frequency"]], i)
            for i, candidate in enumerate(table.found[s])
            if "dictionary" in candidate.sources
        ]
        if not near or min(near)[0] > NEAREST_LIMIT:
            continue
        best = min(near)[2]
        index[s], p_fix[s] = best, 1.0
        outcome[s] = FIXED if table.y[block][best] else BROKEN if table.token_right[s] else STILL_WRONG
    return Choice(index, p_fix, np.zeros(n), outcome)
