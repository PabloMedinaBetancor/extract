"""The model that ships: the detector and the chooser trained on all of D, and the `fix` block that the
out-of-fold run fitted.

    python3 -s -m bench.guard.final_fix --structure --out models/guard.model

The thresholds, the calibration tables and `t_gen` come from the pooled out-of-fold predictions of the protocol
(`train_chooser.run`) and nowhere else; the detector and the chooser that ship are fitted on every development document
with the chosen cells and seed. The chooser is trained on the sites the detector flags among the out-of-fold scores
(`>= t_gen`), the same words it will meet at run time. T5 decides nothing here.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

from bench.guard import calibrate, cells_paired, operating, pairs, train_chooser
from extract.guard import features, language, model, readings

LANGUAGE_FLOOR_MIN = 0.2
LANGUAGE_FLOOR_SHARE = 0.5
LANGUAGE_PERCENTILE = 1.0


def language_floor(pages: list[pairs.Page]) -> float:
    """Under this share of words in the es and en lists a page is not treated as Spanish or English: half of the 1st
    percentile of the development pages (never under 0.2), so no page of the corpus is withheld."""
    shares = np.array([p.context.dictionary_share for p in pages], dtype=np.float64)
    shares = shares[np.isfinite(shares)]
    if not len(shares):
        return LANGUAGE_FLOOR_MIN
    return float(max(LANGUAGE_FLOOR_MIN, LANGUAGE_FLOOR_SHARE * np.percentile(shares, LANGUAGE_PERCENTILE)))


def documents_report(table, raw, thresholds: dict, pages: list, min_group: int, min_positive: int) -> dict:
    """What each rule applies, fixes and breaks in each document, in total and in each language of `thresholds` (the table's
    language of each site): the counts that the paired comparison of two models resamples. `documents` lists the
    documents in order, `language` the label of each in the split."""
    documents = sorted({page.doc for page in pages})
    choice, _, languages, _, _, masks = operating.points_by_language(
        table, raw, True, thresholds, min_group=min_group, min_positive=min_positive
    )
    out: dict = {
        "documents": documents,
        "language": {page.doc: page.language or pairs.DEFAULT_LANGUAGE for page in pages},
        "total": {},
        "by_language": {name: {} for name in thresholds},
    }
    for rule, mask in masks.items():
        out["total"][rule] = cells_paired.per_document(table, choice, mask, documents)
        for name in thresholds:
            out["by_language"][name][rule] = cells_paired.per_document(
                table, choice, mask & (languages == name), documents
            )
    return out


def build(
    pages: list[pairs.Page],
    chooser_cell=train_chooser.CHOOSER_CELL,
    seed: int = 1,
    threads: int = 2,
    structure: bool = True,
    minima: dict | None = None,
    min_group: int = operating.MIN_GROUP,
    min_positive: int = operating.MIN_POSITIVE,
    budgets=model.BUDGETS,
    floor: float = operating.SAFE_FLOOR,
    sensor: "float | dict | None" = None,
    promise: str = "exact",
    spec: readings.ReadingSpec | None = None,
) -> tuple[model.GuardModel, dict]:
    """The model with its `fix` block and the out-of-fold report behind it. `spec` is the cell of the second readings the
    pages were loaded with: it is written in the model, and the runtime compares it with what the product reads."""
    cache = pairs.SiteCache(pages)
    learner = train_chooser.lightgbm_learner(chooser_cell, seed=seed, threads=threads)
    run = train_chooser.run(pages, {"chooser": learner}, seed=seed, threads=threads, cache=cache, structure=structure)
    words, outer = run.words, run.detector_raw
    detector_leaves, detector_trees = train_chooser.DETECTOR_CELL
    detector = train_chooser.fit_detector(words, np.arange(len(words)), seed, threads)
    detector_table = calibrate.fit_isotonic(outer, words.y, min_words=1)
    thresholds = calibrate.word_thresholds(calibrate.apply_table(detector_table, outer), budgets)
    # the chooser's own training sites: the words the detector flags out of fold, text and structure
    flagged = outer >= run.t_gen
    site_row = np.array([pairs.is_site(pages[p], w) for p, w in zip(words.page, words.word, strict=True)])
    sites = train_chooser.sites_of(words, np.nonzero(site_row & flagged)[0])
    if structure:
        sites = train_chooser.with_structure(pages, words, np.nonzero(flagged)[0], sites)
    training = cache.table(sites)
    chooser = model.train_chooser(training.x, training.y, chooser_cell[0], chooser_cell[1], seed=seed, threads=threads)
    raw = run.raw["chooser"]
    tables = operating.calibration_tables(run.table, raw, min_group, min_positive)
    # the thresholds are chosen on the cross-fitted probabilities (no row helps fit its own calibration), and applied
    # at run time to the probabilities of the whole tables: the in-sample counts at these points are reported apart
    probability = operating.calibrate_rows_by_op(run.table, raw, True, min_group, min_positive)
    choice = operating.protect_proper(run.table, operating.choose(run.table, probability))
    ops, languages = operating.site_ops(run.table), operating.site_languages(run.table)
    kinds = operating.site_kinds(run.table, choice)
    eligible = operating.eligible_groups(run.table, min_group, min_positive)
    hits = operating.safe_outcome(
        run.table, choice, promise
    )  # the promise that `safe` keeps (a choice of the maintainer)
    points = operating.thresholds_by_op(
        choice, ops, floor=floor, minima=minima, kinds=kinds, eligible=eligible, safe_outcome=hits
    )
    # the product picks the block of a page by its SENSED language (see `language.py`); a page that is `unsure` applies nothing. So the
    # thresholds that ship are fitted on the sensed partition. The partition by the split's label (the optimistic one: it
    # does not make the sensor's mistakes) is kept apart, in the report, for reference.
    by_label = by_language = None
    sensed_table = run.table
    if sensor is not None:
        labelled = languages
        sensed = np.array(
            [pages[int(p)].sensed or pages[int(p)].language or pairs.DEFAULT_LANGUAGE for p in run.table.page],
            dtype=object,
        )
        sensed_table = dataclasses.replace(run.table, language=sensed)
        by_label = operating.thresholds_by_language(
            choice, ops, labelled, floor=floor, minima=minima, kinds=kinds, eligible=eligible, safe_outcome=hits
        )
        by_language = operating.thresholds_by_language(
            choice, ops, sensed, floor=floor, minima=minima, kinds=kinds, eligible=eligible, safe_outcome=hits
        )
        by_language.pop(language.UNSURE, None)
    fix = {
        "feature_fingerprint": features.feature_fingerprint(),
        "t_gen": float(run.t_gen),
        "processed_share": 0.20,
        "calibration": tables,
        "t_safe": {op: p["t_safe"] for op, p in points.items()},
        "m": {op: p["m"] for op, p in points.items() if "|" not in op},
        "precision_floor": floor,
        "t5_floor": operating.T5_FLOOR,
        "safe_promise": promise,
        "more_broken_share": operating.MORE_BROKEN_SHARE,
        "minima": minima or operating.MIN_APPLIED,
        "language_floor": language_floor(pages),
        "chooser_cell": list(chooser_cell),
        "readings": None if spec is None else spec.as_meta(),
        "structure_sites": structure,
        "training": {
            "documents": len({p.doc for p in pages}),
            "pages": len(pages),
            "chooser_sites": int(training.sites),
            "chooser_rows": int(len(training.y)),
            "oof_sites": int(run.table.sites),
        },
    }
    if by_language is not None:
        # B12: the page's sensed language picks its block (and its word list); a language with no block applies nothing
        fix["language_sensor"] = language.sensor_settings(sensor)
        fix["by_language"] = {
            lang: {
                "t_safe": {op: p["t_safe"] for op, p in block.items()},
                "m": {op: p["m"] for op, p in block.items() if "|" not in op},
            }
            for lang, block in by_language.items()
        }
    settings = {
        **model.DETECTOR_FIXED,
        "num_leaves": detector_leaves,
        "n_estimators": detector_trees,
        "source": "out-of-fold on the development pool",
    }
    summary = {
        "train_words": int(len(words.y)),
        "train_wrong": int(words.y.sum()),
        "train_documents": len({p.doc for p in pages}),
        "realised_shares": calibrate.realised_shares(calibrate.apply_table(detector_table, outer), thresholds),
    }
    meta = model.new_meta(detector_table, thresholds, settings, summary, seed)
    meta["fix"] = fix
    delivered = {op: dict(p) for op, p in points.items()}
    report = {
        "delivered": delivered,
        "cross_fitted": operating.report_by_op(
            run.table,
            raw,
            cross_fit=True,
            thresholds=delivered,
            min_group=min_group,
            min_positive=min_positive,
            with_quality=True,
        ),
        "in_sample": operating.report_by_op(
            run.table, raw, cross_fit=False, thresholds=delivered, min_group=min_group, min_positive=min_positive
        ),
    }
    if by_language is not None:
        report["delivered_by_language"] = by_language
        report["by_language_cross_fitted"] = operating.report_by_language(
            sensed_table, raw, cross_fit=True, thresholds=by_language, min_group=min_group, min_positive=min_positive
        )
        report["by_language_in_sample"] = operating.report_by_language(
            sensed_table, raw, cross_fit=False, thresholds=by_language, min_group=min_group, min_positive=min_positive
        )
        # the same with the languages of the split's label and the thresholds fitted on them (reference, optimistic)
        report["by_label_thresholds"] = by_label
        report["by_label_cross_fitted"] = operating.report_by_language(
            run.table, raw, cross_fit=True, thresholds=by_label, min_group=min_group, min_positive=min_positive
        )
        report["by_document"] = {
            "sensed": documents_report(sensed_table, raw, by_language, pages, min_group, min_positive),
            "label": documents_report(run.table, raw, by_label, pages, min_group, min_positive),
        }
    return model.GuardModel(detector, meta, None, chooser), report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    pairs.add_data_arguments(parser)
    parser.add_argument("--out", type=Path, default=Path("models/guard.model"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--structure", action="store_true")
    parser.add_argument(
        "--promise",
        choices=operating.PROMISES,
        default="exact",
        help="what counts as a hit when `safe` is swept: the word ends exact (signed) or never ends worse; "
        "a choice of the maintainer, `exact` until changed",
    )
    parser.add_argument(
        "--no-readings",
        action="store_true",
        help="train without the second readings (a test run only: the product has them)",
    )
    args = parser.parse_args(argv)
    if args.cells is not None and not args.no_readings:
        parser.error(
            "--cells: the model is trained on the cell of the product (`data/second_reading.json`), not on one of "
            "its own; leave --cells out (or --no-readings for a test model)"
        )
    if args.no_readings:
        args.cells = []
    started = time.time()
    pages = pairs.pages_from_args(args)
    guard_model, report = build(
        pages,
        threads=args.threads,
        structure=args.structure,
        sensor=getattr(args, "sensor", None),
        promise=args.promise,
        spec=args.spec,
    )
    guard_model.save(args.out)
    print({"model": str(args.out), "id": guard_model.model_id(), "seconds": round(time.time() - started, 1)})
    print(report["delivered"])
    Path(f"bench/results/fix-final-report-{args.out.stem}.json").write_text(
        json.dumps(report, indent=1, default=float), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
