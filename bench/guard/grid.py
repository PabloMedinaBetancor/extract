"""The grid of the chooser: nine LightGBM cells, one run of the protocol with nine learners, and the choice of
the cheapest cell within one cluster standard error of the best number of correct fixes applied at the `safe` point.

    python3 -s -m bench.guard.grid --structure --out bench/results/chooser-grid.json

Every learner sees the same sites and the same detector scores (one run), so the cells differ only in the model. A
cell over the cost cap is not eligible. The cost of a cell is `num_leaves * n_estimators`, the number of leaf
lookups of one scored row; a tie in cost goes to the lower log-loss.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from bench.guard import operating, pairs, train_chooser

LEAVES = (15, 31, 63)
TREES = (100, 200, 400)
CELLS = tuple((leaves, trees) for leaves in LEAVES for trees in TREES)


def name_of(cell: tuple[int, int]) -> str:
    return f"{cell[0]}x{cell[1]}"


def cost_of(cell: tuple[int, int]) -> int:
    return cell[0] * cell[1]


def choose_cell(results: dict, eligible: set | None = None) -> dict:
    """`results` maps a cell to `{"fixed": the correct fixes applied at safe, "se": the cluster standard error of that
    number, "logloss": ...}`. The best eligible cell sets the bar (its `fixed` minus its `se`); the cheapest cell that
    reaches it wins, a tie in cost going to the lower log-loss."""
    pool = {c: r for c, r in results.items() if eligible is None or c in eligible}
    if not pool:
        return {"cell": None, "reason": "no eligible cell"}
    best = max(pool, key=lambda c: (pool[c]["fixed"], -cost_of(c)))
    bar = pool[best]["fixed"] - pool[best]["se"]
    within = [c for c, r in pool.items() if r["fixed"] >= bar]
    pick = min(within, key=lambda c: (cost_of(c), pool[c]["logloss"], c))
    return {"cell": pick, "best": best, "bar": bar, "within_one_se": sorted(within), "cost": cost_of(pick)}


SAFE_CELLS_SHARE = 2 / 3  # the share of the cells that must apply something at safe for safe to be the metric


def by_metric(results: dict) -> tuple[dict, str]:
    """The numbers `choose_cell` compares: the correct fixes applied at `safe` when safe is on in at least two thirds of the
    cells, else the net balance of `more`, which then carries the success rule. A safe point that one cell of nine
    reaches, by a handful of fixes over the minimum count, is the luck of that cell and not a property of the model: the
    first real run had 301 applied in 15x200 and nothing in the other eight. Returns the results renamed to
    `fixed`/`se` and the name of the metric."""
    on = sum(1 for r in results.values() if r["fixed"] > 0)
    if on >= SAFE_CELLS_SHARE * len(results):
        return results, "safe_fixed"
    return {c: {**r, "fixed": r["more_net"], "se": r["more_net_se"]} for c, r in results.items()}, "more_net"


def logloss(table, probability: np.ndarray) -> float:
    p = np.clip(probability, 1e-9, 1 - 1e-9)
    return float(-np.mean(table.y * np.log(p) + (1 - table.y) * np.log(1 - p)))


def evaluate(run, resamples: int = 300) -> dict:
    """Per cell: the correct fixes applied at safe (and their standard error), the net at `more`, the log-loss."""
    out = {}
    for cell in CELLS:
        raw = run.raw[name_of(cell)]
        report = operating.report_by_op(run.table, raw, cross_fit=True, resamples=resamples)
        fixed = report["safe"]["total"]["intervals"]["fixed"]
        probability = operating.calibrate_rows_by_op(run.table, raw, cross_fit=True)
        out[cell] = {
            "fixed": fixed["value"],
            "se": fixed["se"],
            "logloss": logloss(run.table, probability),
            "safe_applied": report["safe"]["total"]["applied"],
            "more_net": report["more"]["total"]["net"],
            "more_net_se": report["more"]["total"]["intervals"]["net"]["se"],
            "thresholds": report["thresholds"],
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    pairs.add_data_arguments(parser)
    parser.add_argument("--out", type=Path, default=Path("bench/results/chooser-grid.json"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--structure", action="store_true")
    parser.add_argument("--ineligible", nargs="*", default=[], help="cells over the cost cap, as 31x200")
    args = parser.parse_args(argv)
    started = time.time()
    pages = pairs.pages_from_args(args)
    learners = {name_of(c): train_chooser.lightgbm_learner(c, threads=args.threads) for c in CELLS}
    run = train_chooser.run(pages, learners, threads=args.threads, structure=args.structure)
    results = evaluate(run)
    eligible = {c for c in CELLS if name_of(c) not in set(args.ineligible)}
    compared, metric = by_metric(results)
    choice = choose_cell(compared, eligible)
    payload = {
        "pages": len(pages),
        "structure_sites": args.structure,
        "cells": {name_of(c): r for c, r in results.items()},
        "metric": metric,
        "choice": {**choice, "cell": name_of(choice["cell"]) if choice["cell"] else None},
        "seconds": round(time.time() - started, 1),
    }
    payload["choice"].pop("best", None)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=1, default=float), encoding="utf-8")
    print(json.dumps(payload["choice"], default=list))
    return 0


if __name__ == "__main__":
    sys.exit(main())
