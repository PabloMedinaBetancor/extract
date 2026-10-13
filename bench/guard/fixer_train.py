"""Trains and measures the spelling ranker.

Training uses wrong one-to-one words without digits whose truth is among the candidates; how often it is
(`candidate_recall`) caps what the fixer can do and is reported. The score of the best candidate is calibrated
on the words the detector flags, right ones included, because those are the words a suggestion is shown for;
`t_suggest` is the lowest threshold that keeps the shown suggestions at 85 % precision on validation.
"""

import re
from dataclasses import dataclass

import numpy as np

from bench.guard import calibrate
from extract.guard import fixer, model

PRECISION_FLOOR = 0.85
MIN_CALIBRATION = 200
_DIGIT = re.compile(r"\d")


@dataclass
class Queries:
    """One query per word: the candidates' feature rows, which of them is the truth, and what kind of error it was."""

    features: list
    candidates: list
    truth_at: list
    detail: list
    row: list

    def __len__(self) -> int:
        return len(self.row)


def build_queries(rows, indices, languages=("es", "en")) -> Queries:
    """Candidates and features for the given rows; `truth_at` is the position of the true text or -1."""
    queries = Queries([], [], [], [], [])
    for i in indices:
        text, truth = str(rows.text[i]), str(rows.truth[i])
        candidates = fixer.generate(text, languages)
        if not candidates:
            continue
        in_dictionary = float(rows.x[i, _column("in_dictionary")])
        conf = float(rows.x[i, _column("conf_mean")])
        queries.features.append(
            fixer.features(text, rows.chars[i], None if np.isnan(conf) else conf, in_dictionary, candidates, languages)
        )
        queries.candidates.append(candidates)
        queries.truth_at.append(next((k for k, c in enumerate(candidates) if c.text == truth), -1))
        queries.detail.append(str(rows.detail[i]))
        queries.row.append(int(i))
    return queries


def _column(name: str) -> int:
    from extract.guard.signals import INDEX

    return INDEX[name]


def wrong_pairs(rows) -> np.ndarray:
    """The rows the ranker learns from: one-to-one substitutions of plain words (no digits on either side)."""
    mask = (rows.kind == "substitution") & (rows.y == 1)
    plain = np.array(
        [not _DIGIT.search(str(t)) and not _DIGIT.search(str(u)) for t, u in zip(rows.text, rows.truth, strict=True)]
    )
    return np.nonzero(mask & plain)[0]


def candidate_recall(queries: Queries) -> dict:
    """The share of queries whose truth is among the candidates, overall and by kind of error."""
    present = np.array([k >= 0 for k in queries.truth_at])
    report = {"queries": len(queries), "candidate_recall": float(present.mean()) if len(present) else float("nan")}
    for detail in sorted(set(queries.detail)):
        mask = np.array([d == detail for d in queries.detail])
        report[f"by_{detail or 'unknown'}"] = {
            "queries": int(mask.sum()),
            "candidate_recall": float(present[mask].mean()),
        }
    return report


def fit_ranker(queries: Queries, seed: int = 1, threads: int = 1):
    """The lambdarank model on the queries whose truth is among their candidates."""
    kept = [i for i, k in enumerate(queries.truth_at) if k >= 0]
    if not kept:
        raise ValueError("no query has its truth among the candidates")
    x = np.vstack([queries.features[i] for i in kept])
    labels = np.concatenate([(np.arange(len(queries.features[i])) == queries.truth_at[i]).astype(int) for i in kept])
    groups = np.array([len(queries.features[i]) for i in kept])
    return model.train_ranker(x, labels, groups, seed=seed, threads=threads)


def top_choices(booster, queries: Queries, threads: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """For each query, the position of its best candidate and that candidate's raw score."""
    best, scores = [], []
    for features in queries.features:
        raw = model.raw_scores(booster, features, threads)
        best.append(int(np.argmax(raw)))
        scores.append(float(raw.max()))
    return np.array(best), np.array(scores)


def accuracy(booster, queries: Queries) -> dict:
    """Top-1 accuracy over the queries that have their truth among the candidates, overall and by kind of error."""
    best, _ = top_choices(booster, queries)
    present = np.array([k >= 0 for k in queries.truth_at])
    correct = best == np.array(queries.truth_at)
    report = {
        "top1_given_candidate": float(correct[present].mean()) if present.any() else float("nan"),
        "top1_all": float(correct.mean()),
    }
    for detail in sorted(set(queries.detail)):
        mask = np.array([d == detail for d in queries.detail]) & present
        if mask.any():
            report[f"top1_{detail or 'unknown'}"] = float(correct[mask].mean())
    return report


def choose_threshold(probability: np.ndarray, correct: np.ndarray, floor: float = PRECISION_FLOOR) -> dict:
    """The lowest probability that keeps the precision of the suggestions at or above the floor, and its coverage.

    Suggestions are taken in order of probability; the threshold is the lowest one whose cumulative precision is
    still at the floor. If none is, nothing is suggested (`threshold` is above 1) and the report says so.
    """
    order = np.argsort(-probability, kind="stable")
    hits = np.cumsum(correct[order])
    precision = hits / np.arange(1, len(order) + 1)
    ok = np.nonzero(precision >= floor)[0]
    if not len(ok):
        return {"threshold": 1.01, "precision": float("nan"), "coverage": 0.0, "suggested": 0}
    last = int(ok.max())
    return {
        "threshold": float(probability[order][last]),
        "precision": float(precision[last]),
        "coverage": float((last + 1) / len(order)),
        "suggested": last + 1,
    }


def calibrate_top_choice(scores: np.ndarray, correct: np.ndarray) -> dict:
    """Isotonic table from the best candidate's raw score to the probability that it is the truth."""
    return calibrate.fit_isotonic(scores, correct, min_words=MIN_CALIBRATION)


def shown_population(rows, doubtful: np.ndarray) -> np.ndarray:
    """The rows a suggestion would be shown for: words the detector flags that are plain (no digits on either side)."""
    plain = np.array([not _DIGIT.search(str(t)) for t in rows.text])
    return np.nonzero(doubtful & plain & (rows.x[:, _column("has_digit")] != 1.0))[0]


def study(train_rows, validation_rows, doubtful: np.ndarray, languages=("es", "en"), seed: int = 1, threads: int = 1):
    """Fit the ranker on training errors and measure it on validation.

    Returns (ranker, `meta["fixer"]` or None when validation has too few flagged words to calibrate, report). The calibration and `t_suggest` come from the words the detector
    flags on validation, right ones included, since those are the words a suggestion is shown for.
    """
    train_queries = build_queries(train_rows, wrong_pairs(train_rows), languages)
    ranker = fit_ranker(train_queries, seed, threads)
    errors = build_queries(validation_rows, wrong_pairs(validation_rows), languages)
    report = {
        "training_queries": len(train_queries),
        "candidate_recall": candidate_recall(errors),
        "accuracy": accuracy(ranker, errors),
    }
    shown = build_queries(validation_rows, shown_population(validation_rows, doubtful), languages)
    if len(shown) < MIN_CALIBRATION:
        report["shown"] = {
            "doubtful_plain_words": len(shown),
            "note": f"fewer than {MIN_CALIBRATION}: no calibration, no suggestions",
        }
        return ranker, None, report
    best, scores = top_choices(ranker, shown, threads)
    correct = np.array(
        [
            shown.candidates[k][b].text == str(validation_rows.truth[row])
            for k, (b, row) in enumerate(zip(best, shown.row, strict=True))
        ]
    )
    table = calibrate_top_choice(scores, correct)
    probability = np.interp(scores, table["scores"], table["probs"])
    threshold = choose_threshold(probability, correct)
    report["shown"] = {
        "doubtful_plain_words": len(shown),
        "right_words_among_them": int(
            np.sum([str(validation_rows.truth[r]) == str(validation_rows.text[r]) for r in shown.row])
        ),
        **threshold,
    }
    meta = {
        "feature_fingerprint": fixer.feature_fingerprint(),
        "calibration": table,
        "t_suggest": threshold["threshold"],
        "precision_floor": PRECISION_FLOOR,
    }
    return ranker, meta, report
