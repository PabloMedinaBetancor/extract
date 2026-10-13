"""Fits the detector: a grid of six cells, validation picks one, then isotonic calibration and review thresholds.

Only the training rows teach the model and only the validation rows decide anything (the cell, the
calibration, the thresholds). The sealed test is never opened here.
"""

import numpy as np

from bench.guard import calibrate, metrics
from extract.guard import model, signals

GRID = tuple((leaves, trees) for leaves in (7, 15, 31) for trees in (100, 200))
BOOTSTRAP_FOR_SE = 200
SEED = 1


def run_grid(train, validation, seed: int = SEED, threads: int = 1) -> list[dict]:
    """Fit every cell on the training rows and score it by validation average precision with its cluster SE."""
    cells = []
    for leaves, trees in GRID:
        booster = model.train_detector(train.x, train.y, leaves, trees, seed=seed, threads=threads)
        score = model.raw_scores(booster, validation.x, threads)
        summary = metrics.cluster_bootstrap(
            validation.group,
            {"ap": lambda rows, s=score: metrics.average_precision(validation.y[rows], s[rows])},
            resamples=BOOTSTRAP_FOR_SE,
            seed=seed,
        )["ap"]
        cells.append({"leaves": leaves, "trees": trees, "ap": summary["value"], "se": summary["se"]})
    return cells


def choose_cell(cells: list[dict]) -> dict:
    """The cheapest cell (fewest trees, then fewest leaves) whose AP is within one cluster SE of the best."""
    best = max(cells, key=lambda c: c["ap"])
    eligible = [c for c in cells if c["ap"] >= best["ap"] - best["se"]]
    return min(eligible, key=lambda c: (c["trees"], c["leaves"]))


def fit_model(train, validation, cell: dict, seed: int = SEED, threads: int = 1, budgets=model.BUDGETS):
    """The detector of the chosen cell with its calibration table and thresholds, as a `GuardModel`."""
    booster = model.train_detector(train.x, train.y, cell["leaves"], cell["trees"], seed=seed, threads=threads)
    raw = model.raw_scores(booster, validation.x, threads)
    table = calibrate.fit_isotonic(raw, validation.y)
    probability = calibrate.apply_table(table, raw)
    thresholds = calibrate.word_thresholds(probability, budgets)
    settings = {
        **model.DETECTOR_FIXED,
        "num_leaves": cell["leaves"],
        "n_estimators": cell["trees"],
        "grid": [list(g) for g in GRID],
    }
    summary = {
        "train_words": int(len(train.y)),
        "train_wrong": int(train.y.sum()),
        "train_documents": int(len(set(train.group))),
        "validation_words": int(len(validation.y)),
        "validation_documents": int(len(set(validation.group))),
        "realised_shares": calibrate.realised_shares(probability, thresholds),
    }
    meta = model.new_meta(table, thresholds, settings, summary, seed)
    return model.GuardModel(booster, meta), raw, probability


def calibration_report(validation, probability: np.ndarray, raw: np.ndarray | None = None) -> dict:
    """Expected calibration error pooled and per damage band, with the reliability tables.

    `ece_in_sample` is measured on the rows the isotonic table was fitted on, so it is near zero by construction.
    With `raw`, `ece_out_of_sample` repeats it with each half of the documents calibrated by the other half.
    """
    tables = {"pooled": metrics.reliability_table(validation.y, probability)}
    report = {"ece_in_sample": _ece_by_band(validation, probability)}
    if raw is not None:
        held_out = calibrate.out_of_sample_probability(raw, validation.y, validation.group)
        report["ece_out_of_sample"] = _ece_by_band(validation, held_out)
    for band in sorted(set(validation.band)):
        mask = validation.band == band
        if mask.sum() >= 100 and 0 < validation.y[mask].sum():
            tables[band] = metrics.reliability_table(validation.y[mask], probability[mask])
    return {**report, "reliability": tables}


def _ece_by_band(validation, probability: np.ndarray) -> dict[str, float]:
    report = {"pooled": metrics.expected_calibration_error(validation.y, probability)}
    for band in sorted(set(validation.band)):
        mask = validation.band == band
        if mask.sum() >= 100 and 0 < validation.y[mask].sum():
            report[band] = metrics.expected_calibration_error(validation.y[mask], probability[mask])
    return report


def signal_groups() -> dict[str, list[int]]:
    """The five groups of signals the ablation drops one at a time."""
    names = {
        "engine": ("conf_mean", "conf_min", "low_conf_chars", "length"),
        "dictionary": ("in_dictionary", "in_dictionary_folded", "log_frequency", "dict_distance", "char_lm_logp"),
        "shape": (
            "has_digit",
            "mixed_letters_digits",
            "number_format_ok",
            "case_pattern",
            "odd_chars",
            "hyphen_fragment",
        ),
        "layout": ("height_ratio", "width_per_char_ratio", "line_conf_mean", "page_conf_mean"),
        "context": ("prev_in_dictionary", "next_in_dictionary"),
    }
    return {group: [signals.INDEX[n] for n in members] for group, members in names.items()}
