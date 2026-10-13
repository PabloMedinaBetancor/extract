"""Metrics of the guard with cluster-bootstrap intervals (documents are resampled, never words).

Everything works on plain arrays: `y` is 1 for a wrong word, `score` is higher for a more suspicious one,
`groups` holds the document of every row.
"""

import numpy as np


def ranking_order(score: np.ndarray) -> np.ndarray:
    """Row indices from most to least suspicious; ties keep their original order, so results are reproducible."""
    return np.argsort(-score, kind="stable")


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the precision-recall curve as the mean of the precision at every wrong word."""
    positives = int(y.sum())
    if positives == 0:
        return float("nan")
    ordered = y[ranking_order(score)]
    hits = np.cumsum(ordered)
    return float((hits[ordered == 1] / (np.nonzero(ordered == 1)[0] + 1)).sum() / positives)


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """Area under the ROC curve (the chance that a wrong word scores above a right one; ties count half)."""
    positives, negatives = int(y.sum()), int(len(y) - y.sum())
    if not positives or not negatives:
        return float("nan")
    values, inverse = np.unique(score, return_inverse=True)
    counts = np.bincount(inverse)
    ranks = np.cumsum(counts) - (counts - 1) / 2.0
    rank_sum = float((ranks[inverse] * (y == 1)).sum())
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def flag_top(score: np.ndarray, share: float) -> np.ndarray:
    """Boolean mask of the `share` of rows with the highest scores."""
    count = int(round(share * len(score)))
    mask = np.zeros(len(score), dtype=bool)
    mask[ranking_order(score)[:count]] = True
    return mask


def recall_precision(y: np.ndarray, flagged: np.ndarray) -> tuple[float, float]:
    """Recall of the wrong words and precision of the flags; NaN where there is nothing to divide by."""
    hits = int((y.astype(bool) & flagged).sum())
    wrong, flags = int(y.sum()), int(flagged.sum())
    return (hits / wrong if wrong else float("nan"), hits / flags if flags else float("nan"))


def recall_at_share(y: np.ndarray, score: np.ndarray, share: float) -> float:
    return recall_precision(y, flag_top(score, share))[0]


def precision_at_share(y: np.ndarray, score: np.ndarray, share: float) -> float:
    return recall_precision(y, flag_top(score, share))[1]


def reliability_table(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> list[dict]:
    """Equal-mass bins of the predictions with their mean prediction and observed rate."""
    order = np.argsort(probability, kind="stable")
    table = []
    for chunk in np.array_split(order, bins):
        if len(chunk):
            table.append(
                {
                    "words": int(len(chunk)),
                    "predicted": float(probability[chunk].mean()),
                    "observed": float(y[chunk].mean()),
                }
            )
    return table


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    """Weighted mean gap between predicted and observed rate over equal-mass bins."""
    table = reliability_table(y, probability, bins)
    total = sum(row["words"] for row in table)
    return float(sum(row["words"] * abs(row["predicted"] - row["observed"]) for row in table) / total)


def document_index(groups: np.ndarray) -> list[np.ndarray]:
    """For each document, the rows it owns."""
    order = np.argsort(groups, kind="stable")
    sorted_groups = groups[order]
    cuts = np.nonzero(sorted_groups[1:] != sorted_groups[:-1])[0] + 1
    return np.split(order, cuts)


def bootstrap_rows(documents: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """The rows of one resample: documents drawn with replacement, all their rows together."""
    picks = rng.integers(0, len(documents), len(documents))
    return np.concatenate([documents[i] for i in picks])


def cluster_bootstrap(
    groups: np.ndarray, statistics: dict, resamples: int = 2000, seed: int = 1
) -> dict[str, dict[str, float]]:
    """Point value, 95 % interval and standard error of each statistic over resampled documents.

    `statistics` maps a name to a function of the row indices; every statistic sees the same resamples, so
    differences between two of them can be bootstrapped too (`paired_difference`).
    """
    documents = document_index(groups)
    rng = np.random.default_rng(seed)
    everything = np.arange(len(groups))
    draws = {name: [] for name in statistics}
    for _ in range(resamples):
        rows = bootstrap_rows(documents, rng)
        for name, function in statistics.items():
            draws[name].append(function(rows))
    summary = {}
    for name, function in statistics.items():
        values = np.array(draws[name], dtype=np.float64)
        values = values[~np.isnan(values)]
        low, high = np.percentile(values, [2.5, 97.5]) if len(values) else (float("nan"),) * 2
        summary[name] = {
            "value": float(function(everything)),
            "low": float(low),
            "high": float(high),
            "se": float(values.std(ddof=1)) if len(values) > 1 else float("nan"),
        }
    return summary


def paired_difference(groups: np.ndarray, first, second, resamples: int = 2000, seed: int = 1) -> dict[str, float]:
    """The difference first - second with its 95 % interval, on the same resampled documents."""
    summary = cluster_bootstrap(groups, {"d": lambda rows: first(rows) - second(rows)}, resamples, seed)["d"]
    summary["excludes_zero"] = bool(summary["low"] > 0 or summary["high"] < 0)
    return summary


def per_document_recall(y: np.ndarray, flagged: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Recall of each document that has at least one wrong word (the spread that sizes the test)."""
    values = []
    for rows in document_index(groups):
        wrong = int(y[rows].sum())
        if wrong:
            values.append(int((y[rows].astype(bool) & flagged[rows]).sum()) / wrong)
    return np.array(values)


def documents_needed(spread: float, half_width: float = 0.04) -> int:
    """n = (1.96 * SD / half-width)^2, rounded up: documents for a pooled recall within the half-width."""
    return int(np.ceil((1.96 * spread / half_width) ** 2))
