"""The guard's numbers on a set of read pages, for a model, thresholds and baselines that are already frozen.

Nothing here fits or chooses: the detector, its calibration table, its thresholds, the ranker, `t_suggest` and the
logistic baseline come in finished, and the rows are only scored. Every figure carries the 95 % interval of a
bootstrap that resamples whole documents. `bench/read_test.py` calls this once on the sealed test; the tests call it
on made-up rows.
"""

import numpy as np

from bench.guard import analysis, fixer_train, metrics
from bench.guard import run_validation as study
from extract.guard import model, signals

RESAMPLES = 2000
BUDGETS = model.BUDGETS


def evaluate(rows, guard, baselines: dict[str, np.ndarray], resamples: int = RESAMPLES, seed: int = 1) -> dict:
    """All the detector figures of one slice of rows. `baselines` maps a name to its scores on the same rows."""
    y, groups, band = rows.y, rows.group, rows.band
    raw = guard.raw(rows.x)
    probability = guard.probability(raw)
    digits = rows.x[:, signals.INDEX["has_digit"]] == 1.0
    scores = {"detector": raw, **baselines}
    share = analysis.REVIEW_SHARE
    statistics = {}
    for name, score in scores.items():
        statistics[f"{name}.recall"] = lambda r, s=score: metrics.recall_at_share(y[r], s[r], share)
        statistics[f"{name}.precision"] = lambda r, s=score: metrics.precision_at_share(y[r], s[r], share)
        statistics[f"{name}.ap"] = lambda r, s=score: metrics.average_precision(y[r], s[r])
    flagged_by_budget = {}
    for budget in BUDGETS:
        t_word, t_number = guard.thresholds(budget)
        flagged = probability >= np.where(digits, t_number, t_word)
        flagged_by_budget[budget] = flagged
        key = f"shipped.{budget:.2f}"
        statistics[f"{key}.flagged_share"] = lambda r, f=flagged: float(f[r].mean())
        statistics[f"{key}.recall"] = lambda r, f=flagged: _recall(y[r], f[r])
        statistics[f"{key}.precision"] = lambda r, f=flagged: metrics.recall_precision(y[r], f[r])[1]
        statistics[f"numbers.{budget:.2f}.recall"] = lambda r, f=flagged: _recall(
            ((y[r] == 1) & digits[r]).astype(np.uint8), f[r]
        )
        statistics[f"numbers.{budget:.2f}.flagged_share"] = lambda r, f=flagged: (
            float(f[r][digits[r]].mean()) if digits[r].any() else float("nan")
        )
    statistics["ece.pooled"] = lambda r: metrics.expected_calibration_error(y[r], probability[r])
    bands = sorted(set(band))
    for name in bands:
        statistics[f"ece.{name}"] = lambda r, b=name: _band_ece(y, probability, band, r, b)
        statistics[f"band.{name}.recall"] = lambda r, b=name: _band_recall(y, raw, band, r, b, share)
    summary = metrics.cluster_bootstrap(groups, statistics, resamples, seed)
    best = max((n for n in baselines), key=lambda n: summary[f"{n}.recall"]["value"])
    report = {
        "rows": _row_summary(rows),
        "at_review_5pct": {name: {k: summary[f"{name}.{k}"] for k in ("recall", "precision", "ap")} for name in scores},
        "best_baseline": best,
        "detector_minus_best_baseline": {
            "recall": metrics.paired_difference(
                groups,
                lambda r: metrics.recall_at_share(y[r], raw[r], share),
                lambda r: metrics.recall_at_share(y[r], baselines[best][r], share),
                resamples,
                seed,
            ),
            "average_precision": metrics.paired_difference(
                groups,
                lambda r: metrics.average_precision(y[r], raw[r]),
                lambda r: metrics.average_precision(y[r], baselines[best][r]),
                resamples,
                seed,
            ),
        },
        "shipped": {
            f"{budget:.2f}": {k: summary[f"shipped.{budget:.2f}.{k}"] for k in ("flagged_share", "recall", "precision")}
            for budget in BUDGETS
        },
        "numbers": {
            "digit_tokens": int(digits.sum()),
            "wrong_digit_tokens": int(((y == 1) & digits).sum()),
            **{
                f"{budget:.2f}": {k: summary[f"numbers.{budget:.2f}.{k}"] for k in ("recall", "flagged_share")}
                for budget in BUDGETS
            },
        },
        "calibration": {
            "ece": {name: summary[f"ece.{name}"] for name in ("pooled", *bands)},
            "reliability": _reliability(rows, probability),
        },
        "by_band": {
            name: {
                "words": int((band == name).sum()),
                "wrong": int(y[band == name].sum()),
                "recall_at_review_5pct": summary[f"band.{name}.recall"],
            }
            for name in bands
        },
        "misses": study.misses_by_kind(rows, metrics.flag_top(raw, share)),
        "half_width_pooled_recall": _half_width(summary["detector.recall"]),
    }
    report["fixer"] = evaluate_fixer(rows, guard, flagged_by_budget[share], resamples, seed)
    return report


def evaluate_fixer(rows, guard, doubtful: np.ndarray, resamples: int = RESAMPLES, seed: int = 1) -> dict:
    """Candidate recall and top-1 accuracy on the wrong plain words, and precision and coverage of the suggestions
    the frozen ranker and `t_suggest` would show on the words the detector flags at the 5 % budget."""
    if guard.ranker is None or "fixer" not in guard.meta:
        return {"note": "the model has no ranker"}
    languages = ("es", "en")
    errors = fixer_train.build_queries(rows, fixer_train.wrong_pairs(rows), languages)
    report = {"queries": len(errors)}
    if len(errors):
        present = np.array([k >= 0 for k in errors.truth_at])
        best, _ = fixer_train.top_choices(guard.ranker, errors)
        correct = best == np.array(errors.truth_at)
        groups = rows.group[np.array(errors.row)]
        summary = metrics.cluster_bootstrap(
            groups,
            {
                "candidate_recall": lambda r: float(present[r].mean()),
                "top1_all": lambda r: float(correct[r].mean()),
                "top1_given_candidate": lambda r: float(correct[r][present[r]].mean()) if present[r].any() else np.nan,
            },
            resamples,
            seed,
        )
        report.update(summary)
        report["by_detail"] = {
            detail: {
                "queries": int(mask.sum()),
                "candidate_recall": float(present[mask].mean()),
                "top1_given_candidate": float(correct[mask & present].mean()) if (mask & present).any() else None,
            }
            for detail in sorted(set(errors.detail))
            for mask in [np.array([d == detail for d in errors.detail])]
        }
    report["shown"] = _shown(rows, guard, doubtful, languages, resamples, seed)
    return report


def _shown(rows, guard, doubtful, languages, resamples, seed) -> dict:
    table = guard.meta["fixer"]["calibration"]
    shown = fixer_train.build_queries(rows, fixer_train.shown_population(rows, doubtful), languages)
    if not len(shown):
        return {"doubtful_plain_words": 0}
    best, scores = fixer_train.top_choices(guard.ranker, shown)
    probability = np.interp(scores, table["scores"], table["probs"])
    suggested = probability >= guard.meta["fixer"]["t_suggest"]
    right = np.array(
        [
            shown.candidates[k][b].text == str(rows.truth[row])
            for k, (b, row) in enumerate(zip(best, shown.row, strict=True))
        ]
    )
    groups = rows.group[np.array(shown.row)]
    summary = metrics.cluster_bootstrap(
        groups,
        {
            "precision": lambda r: float(right[r][suggested[r]].mean()) if suggested[r].any() else np.nan,
            "coverage": lambda r: float(suggested[r].mean()),
        },
        resamples,
        seed,
    )
    return {
        "doubtful_plain_words": len(shown),
        "suggested": int(suggested.sum()),
        "t_suggest": guard.meta["fixer"]["t_suggest"],
        **summary,
    }


def _recall(y, flagged) -> float:
    wrong = int((y == 1).sum())
    return float(((y == 1) & flagged).sum() / wrong) if wrong else float("nan")


def _band_ece(y, probability, band, rows, name) -> float:
    mask = band[rows] == name
    return metrics.expected_calibration_error(y[rows][mask], probability[rows][mask]) if mask.sum() >= 100 else np.nan


def _band_recall(y, score, band, rows, name, share) -> float:
    mask = band[rows] == name
    return metrics.recall_at_share(y[rows][mask], score[rows][mask], share) if mask.any() else np.nan


def _half_width(summary: dict) -> float:
    return float((summary["high"] - summary["low"]) / 2)


def _reliability(rows, probability) -> dict:
    tables = {"pooled": metrics.reliability_table(rows.y, probability)}
    for name in sorted(set(rows.band)):
        mask = rows.band == name
        if mask.sum() >= 100:
            tables[name] = metrics.reliability_table(rows.y[mask], probability[mask])
    return tables


def _row_summary(rows) -> dict:
    return {
        "words": len(rows.y),
        "wrong": int(rows.y.sum()),
        "documents": len(set(rows.group)),
        "pages": len(set(rows.page)),
        **rows.counts,
    }
