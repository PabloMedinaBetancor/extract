"""Ablations of the chooser: drop one group of features and see what the safe point loses.

Every learner sees the same sites and detector scores; a group is dropped by holding its columns at zero in training and
in scoring, so the model has to do without what the group says.

    python3 -s -m bench.guard.ablation --structure --out bench/results/fix-ablation.json
"""

import argparse
import json
import sys
import time
from pathlib import Path

from bench.guard import operating, pairs, train_chooser
from extract.guard import features, signals

GROUPS = {
    "signals": signals.COLUMNS,
    "edit": features.EDIT,
    "candidate": features.CANDIDATE,
    "page": features.PAGE,
    "engine": features.ENGINE,
    "readings": features.READINGS,
    "structure": features.STRUCTURE,
    "context": features.CONTEXT,
}


def columns_of(group: str) -> list[int]:
    return [features.INDEX[name] for name in GROUPS[group]]


def without_columns(factory, columns: list[int]):
    """A learner factory that holds `columns` at zero in the rows it is fitted on and in those it scores."""

    def fit(x, y, *rest):
        masked = x.copy()
        masked[:, columns] = 0.0
        scorer = factory(masked, y, *rest)

        def score(rows):
            held = rows.copy()
            held[:, columns] = 0.0
            return scorer(held)

        return score

    if getattr(factory, "needs_groups", False):
        fit.needs_groups = True
    return fit


def ablation_learners(factory) -> dict:
    """The full learner and one without each group of features."""
    return {"all": factory, **{f"without_{g}": without_columns(factory, columns_of(g)) for g in GROUPS}}


def evaluate(run, resamples: int = 300) -> dict:
    """Per learner: the correct fixes applied at safe and the net at `more`, each with its standard error."""
    out = {}
    for name, raw in run.raw.items():
        report = operating.report_by_op(run.table, raw, cross_fit=True, resamples=resamples)
        safe, more = report["safe"]["total"], report["more"]["total"]
        out[name] = {
            "safe_fixed": safe["intervals"]["fixed"]["value"],
            "safe_fixed_se": safe["intervals"]["fixed"]["se"],
            "safe_applied": safe["applied"],
            "more_net": more["net"],
            "more_net_se": more["intervals"]["net"]["se"],
        }
    base = out["all"]
    for name, row in out.items():
        row["safe_fixed_lost"] = None if name == "all" else round(base["safe_fixed"] - row["safe_fixed"], 1)
        row["more_net_lost"] = None if name == "all" else round(base["more_net"] - row["more_net"], 1)
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    pairs.add_data_arguments(parser)
    parser.add_argument("--out", type=Path, default=Path("bench/results/fix-ablation.json"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--structure", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    pages = pairs.pages_from_args(args)
    run = train_chooser.run(
        pages,
        ablation_learners(train_chooser.lightgbm_learner(threads=args.threads)),
        threads=args.threads,
        structure=args.structure,
    )
    result = {"pages": len(pages), "structure_sites": args.structure, "learners": evaluate(run)}
    result["seconds"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(
        json.dumps(
            {
                k: {"safe_fixed_lost": v["safe_fixed_lost"], "more_net_lost": v["more_net_lost"]}
                for k, v in result["learners"].items()
            }
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
