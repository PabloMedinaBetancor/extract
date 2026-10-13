"""The checks around the detector that say how far to trust its numbers.

All of them train on the training rows and look at the validation rows only.
"""

import hashlib
from types import SimpleNamespace

import numpy as np

from bench.guard import metrics, train
from extract.guard import model

REVIEW_SHARE = 0.05


def _fit_score(train_rows, validation_rows, cell, seed=train.SEED, threads=1, monotone=True, y=None):
    booster = model.train_detector(
        train_rows.x,
        train_rows.y if y is None else y,
        cell["leaves"],
        cell["trees"],
        seed=seed,
        threads=threads,
        monotone=monotone,
    )
    return model.raw_scores(booster, validation_rows.x, threads)


def _summary(validation_rows, score) -> dict:
    return {
        "ap": metrics.average_precision(validation_rows.y, score),
        "recall_at_review": metrics.recall_at_share(validation_rows.y, score, REVIEW_SHARE),
        "auc": metrics.auc(validation_rows.y, score),
    }


def document_subsets(train_rows, fractions) -> dict[float, np.ndarray]:
    """Row masks that keep the first `fraction` of the training documents in a fixed order (a hash of the id)."""
    documents = sorted(set(train_rows.group), key=lambda d: hashlib.sha256(str(d).encode()).hexdigest())
    masks = {}
    for fraction in fractions:
        kept = set(documents[: max(1, int(round(fraction * len(documents))))])
        masks[fraction] = np.array([g in kept for g in train_rows.group])
    return masks


def learning_curve(train_rows, validation_rows, cell, fractions=(0.25, 0.5, 1.0), seed=train.SEED) -> list[dict]:
    """Train on a growing share of the training documents: does more data still help?"""
    curve = []
    for fraction, mask in document_subsets(train_rows, fractions).items():
        subset = SimpleNamespace(x=train_rows.x[mask], y=train_rows.y[mask])
        curve.append(
            {
                "fraction": fraction,
                "documents": int(len(set(train_rows.group[mask]))),
                "words": int(mask.sum()),
                **_summary(validation_rows, _fit_score(subset, validation_rows, cell, seed)),
            }
        )
    return curve


def ablation(train_rows, validation_rows, cell, groups, seed=train.SEED) -> dict[str, dict]:
    """Drop one group of signals at a time (its columns become missing everywhere) and retrain."""
    results = {"none": _summary(validation_rows, _fit_score(train_rows, validation_rows, cell, seed))}
    for name, columns in groups.items():
        dropped_train, dropped_validation = train_rows.x.copy(), validation_rows.x.copy()
        dropped_train[:, columns] = np.nan
        dropped_validation[:, columns] = np.nan
        results[f"without_{name}"] = _summary(
            validation_rows,
            _fit_score(
                SimpleNamespace(x=dropped_train, y=train_rows.y), SimpleNamespace(x=dropped_validation), cell, seed
            ),
        )
    return results


def permutation_canary(train_rows, validation_rows, cell, seed=train.SEED, repeats=30, constrained_repeats=10) -> dict:
    """A model trained on labels shuffled over all words must find nothing in the true labels.

    One shuffled model is a random function of informative signals and can land far from 0.5 on its own, so
    every figure is the mean AUC over `repeats` shuffles with its standard error. Two checks must hold:

    * no leak: the shuffled model scored on the true validation labels is not above 0.5 by more than two
      standard errors (or 0.02). A leak would put every shuffle well above chance.
    * clean null: the same scores against validation labels shuffled too sit within two standard errors of 0.5
      (or 0.02) either way, which shows the metric code itself finds nothing in noise.

    The shuffled model scores *below* 0.5 on the true labels: the wrong words sit in sparse corners of the
    signal space (low confidence, odd shapes), where a model fitted to noise has few rows per leaf and
    predicts below the base rate. That is a property of the estimator, not a leak, and the doubly shuffled null
    shows it: it needs the true labels to appear. The constrained model, fitted without shuffling the sign of
    its own prior, is reported beside them.
    """
    true_labels, null_labels, constrained = [], [], []
    for k in range(repeats):
        shuffled = np.random.default_rng(seed + k).permutation(train_rows.y)
        score = _fit_score(train_rows, validation_rows, cell, seed, monotone=False, y=shuffled)
        true_labels.append(metrics.auc(validation_rows.y, score))
        null_y = np.random.default_rng(seed + 10_000 + k).permutation(validation_rows.y)
        null_labels.append(metrics.auc(null_y, score))
        if k < constrained_repeats:
            constrained.append(
                metrics.auc(validation_rows.y, _fit_score(train_rows, validation_rows, cell, seed, y=shuffled))
            )
    mean, error = _mean_and_error(true_labels)
    null_mean, null_error = _mean_and_error(null_labels)
    return {
        "auc": mean,
        "auc_se": error,
        "auc_sd": float(np.std(true_labels, ddof=1)),
        "auc_each": true_labels,
        "auc_both_shuffled": null_mean,
        "auc_both_shuffled_se": null_error,
        "auc_with_monotone_constraints": float(np.mean(constrained)),
        "no_leak": bool(mean - 0.5 <= max(0.02, 2 * error)),
        "clean_null": bool(abs(null_mean - 0.5) <= max(0.02, 2 * null_error)),
        "passes": bool(mean - 0.5 <= max(0.02, 2 * error) and abs(null_mean - 0.5) <= max(0.02, 2 * null_error)),
    }


def _mean_and_error(values: list[float]) -> tuple[float, float]:
    spread = float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")
    return float(np.mean(values)), spread / np.sqrt(len(values))


def monotone_cost(train_rows, validation_rows, cell, seed=train.SEED) -> dict:
    """The same cell with and without the monotone constraints: what do the constraints cost in recall?"""
    constrained = _summary(validation_rows, _fit_score(train_rows, validation_rows, cell, seed))
    free = _summary(validation_rows, _fit_score(train_rows, validation_rows, cell, seed, monotone=False))
    return {
        "constrained": constrained,
        "unconstrained": free,
        "recall_cost_points": 100 * (free["recall_at_review"] - constrained["recall_at_review"]),
    }


def pilot_sample_size(y, score, groups, share=REVIEW_SHARE) -> dict:
    """Per-document spread of the recall at the review budget, and the test documents it asks for."""
    flagged = metrics.flag_top(score, share)
    recalls = metrics.per_document_recall(y, flagged, groups)
    spread = float(recalls.std(ddof=1))
    return {
        "documents_with_errors": int(len(recalls)),
        "recall_sd": spread,
        "documents_needed_pooled": metrics.documents_needed(spread, 0.04),
        "documents_needed_per_band": metrics.documents_needed(spread, 0.07),
    }
