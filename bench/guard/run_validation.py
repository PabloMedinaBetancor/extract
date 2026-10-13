"""The whole detector study on the training and validation sets, for the model that ships.

Reads the pages that already carry an OCR reading (bench/run_corpus.py), builds the rows, checks the labels on
the undamaged pages, runs the baselines, the grid, the calibration and the thresholds, and writes the model and
`bench/results/guard-validation.json` (numbers only, no document text). The sealed test is never opened here.

    python -m bench.guard.run_validation [--engine FINGERPRINT] [--threads 2]
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np

from bench.guard import analysis, baselines, dataset, fixer_train, metrics, readback, train
from extract.guard import model, signals

RESAMPLES = 2000
AUDIT_SHEETS = 4
AUDIT_SEED = 6


def stat(function, groups, resamples):
    return metrics.cluster_bootstrap(groups, {"v": function}, resamples=resamples)["v"]


def baseline_scores(train_rows, validation_rows) -> dict[str, np.ndarray]:
    logistic = baselines.fit_logistic(train_rows.x, train_rows.y, validation_rows.x, validation_rows.y)
    x = validation_rows.x
    return {
        "lowest_mean_confidence": baselines.lowest_mean_confidence(x),
        "lowest_min_confidence": baselines.lowest_minimum_confidence(x),
        "outside_dictionary_first": baselines.outside_the_dictionary_first(x),
        "logistic_regression": logistic.score(x),
    }


def compare(validation_rows, scores: dict[str, np.ndarray], resamples: int) -> dict:
    """Recall and precision at the review budget with intervals, for the detector and every baseline."""
    y, groups = validation_rows.y, validation_rows.group
    report = {}
    for name, score in scores.items():
        report[name] = {
            "recall": stat(
                lambda rows, s=score: metrics.recall_at_share(y[rows], s[rows], analysis.REVIEW_SHARE),
                groups,
                resamples,
            ),
            "precision": stat(
                lambda rows, s=score: metrics.precision_at_share(y[rows], s[rows], analysis.REVIEW_SHARE),
                groups,
                resamples,
            ),
            "average_precision": stat(
                lambda rows, s=score: metrics.average_precision(y[rows], s[rows]), groups, resamples
            ),
        }
    best = max((n for n in scores if n != "detector"), key=lambda n: report[n]["recall"]["value"])
    detector, other = scores["detector"], scores[best]
    report["detector_minus_best_baseline_ap"] = {
        "baseline": best,
        **metrics.paired_difference(
            groups,
            lambda rows: metrics.average_precision(y[rows], detector[rows]),
            lambda rows: metrics.average_precision(y[rows], other[rows]),
            resamples,
        ),
    }
    report["detector_minus_best_baseline"] = {
        "baseline": best,
        "measure": "recall at the review budget",
        **metrics.paired_difference(
            groups,
            lambda rows: metrics.recall_at_share(y[rows], detector[rows], analysis.REVIEW_SHARE),
            lambda rows: metrics.recall_at_share(y[rows], other[rows], analysis.REVIEW_SHARE),
            resamples,
        ),
    }
    return report


def misses_by_kind(validation_rows, flagged: np.ndarray) -> dict:
    """What the detector misses: wrong words by kind of error, by confidence decile and for numbers."""
    wrong = validation_rows.y == 1
    by_kind = {}
    for kind in sorted(set(validation_rows.kind[wrong])):
        mask = wrong & (validation_rows.kind == kind)
        by_kind[kind] = {"wrong": int(mask.sum()), "caught": float(flagged[mask].mean())}
    conf = validation_rows.x[:, signals.INDEX["conf_mean"]]
    edges = np.nanpercentile(conf, np.arange(0, 101, 10))
    deciles = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        mask = wrong & (conf >= low) & (conf <= high)
        deciles.append(
            {
                "conf_from": float(low),
                "conf_to": float(high),
                "wrong": int(mask.sum()),
                "caught": float(flagged[mask].mean()) if mask.any() else None,
            }
        )
    return {"by_kind": by_kind, "by_confidence_decile": deciles}


def write_audit(validation_rows, out_dir: Path, data_dir: Path) -> int:
    picks = readback.audit_sample(validation_rows, seed=AUDIT_SEED)
    entries = [
        {
            "set": "val",
            "doc_id": str(validation_rows.group[i]),
            "page": int(str(validation_rows.page[i]).split("/")[1]),
            "box": validation_rows.box[i].tolist(),
            "ocr": str(validation_rows.text[i]),
            "truth": str(validation_rows.truth[i]),
        }
        for i in picks
    ]
    size = -(-len(entries) // AUDIT_SHEETS)
    for k in range(AUDIT_SHEETS):
        chunk = entries[k * size : (k + 1) * size]
        if chunk:
            readback.crop_sheet(chunk, out_dir / f"sheet-{k + 1}.png", data_dir)
    return len(entries)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--engine", help="refuse pages read by another engine fingerprint")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--out", type=Path, default=Path("bench/results/guard-validation.json"))
    parser.add_argument("--model", type=Path, default=Path("models/guard.model"))
    parser.add_argument("--audit-dir", type=Path, default=Path("work/audit"))
    args = parser.parse_args(argv)

    train_rows = dataset.load_rows(args.data_dir, "train", engine=args.engine)
    validation_rows = dataset.load_rows(args.data_dir, "val", engine=args.engine)
    report = {
        "rows": {
            "train": {"words": len(train_rows), "wrong": int(train_rows.y.sum()), **train_rows.counts},
            "validation": {
                "words": len(validation_rows),
                "wrong": int(validation_rows.y.sum()),
                **validation_rows.counts,
            },
        },
        "schema": signals.schema_fingerprint(),
    }
    report["g3_readback"] = readback.readback(readback.load_clean(args.data_dir))
    report["audit_crops"] = write_audit(validation_rows, args.audit_dir, args.data_dir)

    cells = train.run_grid(train_rows, validation_rows, threads=args.threads)
    cell = train.choose_cell(cells)
    guard, raw, probability = train.fit_model(train_rows, validation_rows, cell, threads=args.threads)
    report["grid"] = {"cells": cells, "chosen": cell}
    report["calibration"] = train.calibration_report(validation_rows, probability, raw)
    report["thresholds"] = guard.meta["thresholds"]
    report["realised_shares"] = guard.meta["training"]["realised_shares"]

    scores = {"detector": raw, **baseline_scores(train_rows, validation_rows)}
    report["comparison_at_5_percent"] = compare(validation_rows, scores, args.resamples)
    report["misses"] = misses_by_kind(validation_rows, metrics.flag_top(raw, analysis.REVIEW_SHARE))
    report["pilot_sample_size"] = analysis.pilot_sample_size(validation_rows.y, raw, validation_rows.group)
    report["learning_curve"] = analysis.learning_curve(train_rows, validation_rows, cell)
    report["ablation"] = analysis.ablation(train_rows, validation_rows, cell, train.signal_groups())
    report["permutation_canary"] = analysis.permutation_canary(train_rows, validation_rows, cell)
    report["monotone_constraints"] = analysis.monotone_cost(train_rows, validation_rows, cell)
    report["numbers"] = numbers_recall(validation_rows, guard)
    t_word, t_number = guard.thresholds(analysis.REVIEW_SHARE)
    digits = validation_rows.x[:, signals.INDEX["has_digit"]] == 1.0
    doubtful = probability >= np.where(digits, t_number, t_word)
    guard.ranker, fixer_meta, report["fixer"] = fixer_train.study(
        train_rows, validation_rows, doubtful, threads=args.threads
    )
    if fixer_meta is not None:
        guard.meta["fixer"] = fixer_meta

    guard.save(args.model)
    args.out.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(
        json.dumps(
            {k: report[k] for k in ("grid", "comparison_at_5_percent", "pilot_sample_size")}, indent=1, default=float
        )
    )


def numbers_recall(validation_rows, guard) -> dict:
    """Recall of wrong tokens that carry digits at the number threshold, and how much each budget flags.

    A budget flags words above the word threshold and tokens with digits above a third of it, so the share of
    all words actually flagged is a little over the budget; `flagged_share_all_words` says how much.
    """
    probability = guard.probability(guard.raw(validation_rows.x))
    digits = validation_rows.x[:, signals.INDEX["has_digit"]] == 1.0
    wrong = (validation_rows.y == 1) & digits
    out = {"digit_tokens": int(digits.sum()), "wrong_digit_tokens": int(wrong.sum())}
    for budget in model.BUDGETS:
        t_word, t_number = guard.thresholds(budget)
        out[f"recall_at_{budget:.2f}"] = float((probability[wrong] >= t_number).mean()) if wrong.any() else None
        out[f"flagged_share_digit_tokens_at_{budget:.2f}"] = float((probability[digits] >= t_number).mean())
        out[f"flagged_share_all_words_at_{budget:.2f}"] = float(
            (probability >= np.where(digits, t_number, t_word)).mean()
        )
    return out


if __name__ == "__main__":
    main()
