"""Calibration of the detector's raw scores and the review thresholds, both fitted on validation only.

The isotonic fit is a build-time step (scipy); the model file keeps its table and the product applies it
with `numpy.interp`.
"""

import numpy as np
from scipy.optimize import isotonic_regression

MIN_WORDS = 1000


def fit_isotonic(raw: np.ndarray, y: np.ndarray, min_words: int = MIN_WORDS) -> dict[str, list[float]]:
    """Pool-adjacent-violators on the validation scores, as a table of (raw score, probability) points.

    Equal scores are pooled first. Each run of pooled scores becomes one point, at the weighted mean of its
    scores, and `numpy.interp` joins the points. A plain step function would give every score of a run the
    same probability, so a review budget could only flag whole runs; the line between run centres keeps the
    order of the scores and still reproduces each run's rate at its centre.
    """
    if len(raw) < min_words:
        raise ValueError(f"calibration needs at least {min_words} validation words, got {len(raw)}")
    values, inverse = np.unique(raw, return_inverse=True)
    weights = np.bincount(inverse).astype(np.float64)
    rate = np.bincount(inverse, weights=y.astype(np.float64)) / weights
    fitted = isotonic_regression(rate, weights=weights, increasing=True).x
    starts = np.concatenate([[0], np.nonzero(np.diff(fitted) != 0)[0] + 1])
    runs = np.split(np.arange(len(values)), starts[1:])
    centres = [float(np.average(values[r], weights=weights[r])) for r in runs]
    return {"scores": centres, "probs": [float(fitted[r[0]]) for r in runs]}


def apply_table(table: dict[str, list[float]], raw: np.ndarray) -> np.ndarray:
    return np.interp(raw, np.asarray(table["scores"]), np.asarray(table["probs"]))


def word_thresholds(probability: np.ndarray, budgets) -> dict[str, float]:
    """For each review budget, the probability a word needs to be flagged so that about that share is.

    Ties can flag a little more than the budget; `realised_shares` says how much.
    """
    ordered = np.sort(probability)[::-1]
    return {f"{b:.2f}": float(ordered[max(0, int(round(b * len(ordered))) - 1)]) for b in budgets}


def realised_shares(probability: np.ndarray, thresholds: dict[str, float]) -> dict[str, float]:
    return {budget: float((probability >= t).mean()) for budget, t in thresholds.items()}


def out_of_sample_probability(raw: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Calibrated probabilities that no word helped fit: the documents are split in two halves, each half is
    calibrated by the table fitted on the other one."""
    documents = np.unique(groups)
    first = np.isin(groups, documents[::2])
    out = np.empty(len(raw), dtype=np.float64)
    for fitted, held_out in ((first, ~first), (~first, first)):
        out[held_out] = apply_table(fit_isotonic(raw[fitted], y[fitted]), raw[held_out])
    return out
