"""The first out-of-fold result of the chooser on the cached v4 pages: no engine extras, one-word edits only.

python3 -s -m bench.guard.first_result --sets train val --out bench/results/fix-first.json
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from bench.guard import baselines5, operating, pairs, train_chooser
from extract.guard import proper

FLOORS = {"gate": 100, "design": operating.MIN_APPLIED_TEXT}
TOP_COUNTS = (50, 100, 200, 300, 500, 800, 1200)


def top_errors(table, choice, top: int = 1200, listed: int = 80) -> dict:
    """The sites that are not fixed among the `top` best by `p_fix`: how many by outcome, edit kind and op, and a list of
    them (token as read, truth, the candidate applied) so that the first errors of the safe range can be read."""
    order = np.argsort(-choice.p_fix, kind="stable")[:top]
    wrong = [int(s) for s in order if choice.outcome[s] != operating.FIXED and np.isfinite(choice.p_fix[s])]
    kinds: Counter = Counter()
    for s in wrong:
        candidate = table.found[s][int(choice.index[s])]
        kinds[(int(choice.outcome[s]), str(table.family[s]), str(candidate.kind))] += 1
    rows = []
    for s in wrong[:listed]:
        candidate = table.found[s][int(choice.index[s])]
        rows.append(
            {
                "rank": int(np.nonzero(order == s)[0][0]),
                "outcome": {operating.BROKEN: "broken", operating.STILL_WRONG: "still_wrong"}[int(choice.outcome[s])],
                "op": str(table.family[s]),
                "kind": str(candidate.kind),
                "token": str(table.token[s]),
                "truth": str(table.truth[s]),
                "applied": candidate.text,
                "p_fix": round(float(choice.p_fix[s]), 4),
                "doc": str(table.doc[s]),
            }
        )
    return {
        "top": top,
        "errors": len(wrong),
        "by_outcome_op_kind": {"|".join(map(str, k)): n for k, n in kinds.most_common()},
        "first": rows,
    }


GROUP_TOPS = (50, 100, 300, 800)
QUALITY_TOPS = (100, 200, 300, 500, 800, 1200, 2000)


def precision_by_group(table, choice, floors=(0.995, 0.99)) -> dict:
    """For each `op|kind` group of the best candidates: its sites, the precision of its `k` best by `p_fix` for a few `k`
    and the most it could apply at each floor (the largest `k` whose precision is at or over it): where the per-group door
    could open."""
    ops, kinds = operating.site_ops(table), operating.site_kinds(table, choice)
    out = {}
    for op in sorted(set(ops.tolist())):
        for kind in sorted(set(kinds[ops == op].tolist())):
            members = np.nonzero((ops == op) & (kinds == kind) & np.isfinite(choice.p_fix))[0]
            if not len(members):
                continue
            order = members[np.argsort(-choice.p_fix[members], kind="stable")]
            fixed = np.cumsum(choice.outcome[order] == operating.FIXED)
            precision = fixed / np.arange(1, len(order) + 1)
            row = {"sites": int(len(order)), "fixed_in_group": int(fixed[-1])}
            row["precision_at"] = {str(k): float(precision[k - 1]) for k in GROUP_TOPS if k <= len(order)}
            for floor in floors:
                over = np.nonzero(precision >= floor)[0]
                row[f"largest_k_at_{floor}"] = int(over[-1] + 1) if len(over) else 0
            out[f"{op}|{kind}"] = row
    return out


def still_wrong_classes(table, choice, mask) -> dict:
    """Among the applied edits that end with the word still wrong, what separates the edit from the truth: only the marks
    around the word (`core_right`: the letters are the truth's, the punctuation or the case of the ends is not), the
    letters themselves (`core_wrong`), or the span is not one word of the truth (a merge, a split: `other`). `core_right`
    is what a candidate that combined the fix of the letters with the other readers' marks could close."""
    from extract.guard.forms import split_word

    counts: Counter = Counter()
    examples: dict[str, list] = {"core_right": [], "core_wrong": []}
    for s in np.nonzero(mask & (choice.outcome == operating.STILL_WRONG))[0]:
        applied = table.found[s][int(choice.index[s])].text
        truth = str(table.truth[s])
        a, t = split_word(applied), split_word(truth)
        if a is None or t is None or " " in truth or " " in applied:
            counts["other"] += 1
            continue
        key = "core_right" if a[1] == t[1] else "core_wrong"
        counts[key] += 1
        if len(examples[key]) < 12:
            examples[key].append([str(table.token[s]), applied, truth])
    return {"counts": dict(counts), "examples": examples}


PROPER_CLASSES = proper.CLASSES
proper_class = proper.proper_class


def by_proper_class(table, choice, masks: dict, resamples: int = 300) -> dict:
    """For each class of `proper_class` (of the token as read) and each mask (`more`, the best N...): how many sites are in
    it, how many of the applied edits are exact, broken (the word was right) or leave it wrong, with intervals over documents."""
    classes = np.array([proper_class(str(t)) for t in table.token], dtype=object)
    out = {"sites": {c: int((classes == c).sum()) for c in PROPER_CLASSES}}
    for name, mask in masks.items():
        out[name] = {}
        for c in PROPER_CLASSES:
            members = mask & (classes == c)
            out[name][c] = (
                operating.with_intervals(choice, members, table.doc, resamples) if members.any() else {"applied": 0}
            )
    return out


def report_learner(table, raw, cross_fit: bool) -> dict:
    """Thresholds and counts of one learner's out-of-fold scores at both minimum counts."""
    probability = operating.calibrate_rows(table, raw, cross_fit=cross_fit)
    choice = operating.protect_proper(table, operating.choose(table, probability))
    docs = table.doc
    out = {"sites": int(table.sites), "rows": int(len(table.y)), "reachable_sites": int(table.reachable.sum())}
    for name, minimum in FLOORS.items():
        t_safe = operating.safe_threshold(choice.safe_score, choice.outcome, operating.SAFE_FLOOR, minimum)
        mask = operating.applied(choice, "safe", t_safe)
        out[f"safe_min_{minimum}"] = {
            "rule": name,
            "t_safe": t_safe,
            **operating.with_intervals(choice, mask, docs, resamples=500),
        }
    order = np.argsort(-choice.p_fix)
    margin = operating.more_margin(choice.margin, choice.outcome)
    out["top_errors"] = top_errors(table, choice)
    if cross_fit:  # what the edits do to the words (signed precision and damage): the out-of-fold run only, it costs a bootstrap per point
        distances = operating.edit_distances(table, choice)
        out["quality"] = {
            "top": {
                str(n): operating.quality(table, choice, mask, 300, distances)
                for n, mask in operating.top_masks(choice, QUALITY_TOPS).items()
            },
            "more": operating.quality(table, choice, operating.applied(choice, "more", margin), 300, distances),
        }
        more_mask = operating.applied(choice, "more", margin)
        tops = operating.top_masks(choice, (300, 500, 800))
        out["proper_names"] = by_proper_class(
            table, choice, {"more": more_mask, **{f"top{n}": m for n, m in tops.items()}}
        )
        out["quality"]["more"]["still_wrong_classes"] = still_wrong_classes(
            table, choice, operating.applied(choice, "more", margin)
        )
    out["by_group"] = precision_by_group(table, choice)
    out["precision_at_applied"] = {
        str(k): float((choice.outcome[order][:k] == operating.FIXED).mean()) for k in TOP_COUNTS if k <= table.sites
    }
    out["more"] = {
        "m": margin,
        **operating.with_intervals(choice, operating.applied(choice, "more", margin), docs, 500),
    }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    pairs.add_data_arguments(parser)
    parser.add_argument("--out", type=Path, default=Path("bench/results/fix-first.json"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--structure", action="store_true", help="joins, segmentations and drops are sites too")
    args = parser.parse_args(argv)
    started = time.time()
    pages = pairs.pages_from_args(args)
    learners = {
        "lightgbm": train_chooser.lightgbm_learner(threads=args.threads),
        "logistic": baselines5.logistic_learner(),
    }
    run = train_chooser.run(pages, learners, threads=args.threads, structure=args.structure)
    result = {
        "pages": len(pages),
        "documents": len(set(page.doc for page in pages)),
        "structure_sites": args.structure,
        "readings": args.spec.as_meta() if args.spec else {"cells": args.cells, "cap": args.cap},
        "t_gen_raw": run.t_gen,
        "flagged_sites": int(run.table.sites),
        "seconds": None,
    }
    for name, raw in run.raw.items():
        result[name] = {
            "in_sample_calibration": report_learner(run.table, raw, cross_fit=False),
            "cross_fitted_calibration": report_learner(run.table, raw, cross_fit=True),
        }
        if args.structure:
            result[name]["by_op"] = operating.report_by_op(run.table, raw, cross_fit=True)
    nearest = baselines5.nearest_word_choice(run.table)
    mask = np.isfinite(nearest.p_fix)
    result["nearest_word"] = operating.with_intervals(nearest, mask, run.table.doc, resamples=500)
    result["keep"] = {"applied": 0}
    best = result["lightgbm"]["cross_fitted_calibration"]
    result["gate_f"] = {
        "safe_applied_at_floor": best["safe_min_100"]["applied"],
        "more_net": best["more"]["net"],
        "more_net_interval": [best["more"]["intervals"]["net"]["low"], best["more"]["intervals"]["net"]["high"]],
        "passes": bool(best["safe_min_100"]["applied"] >= 100 or best["more"]["intervals"]["net"]["low"] > 0),
    }
    result["seconds"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k in ("pages", "documents", "flagged_sites", "seconds")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
