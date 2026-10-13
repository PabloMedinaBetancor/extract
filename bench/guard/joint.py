"""Joint against single-language training: the chooser trained on Spanish and English together against one
trained on each alone, out of fold on the same folds, with the language of every number reported.

    python3 -s -m bench.guard.joint --structure --by-language --out bench/results/chooser-joint.json

The three runs use the pages of the manifest in their own folds (the folds of the documents do not depend on which pages are
loaded, so the English documents are in the same fold in the joint run and in the English-only one). The joint model ships
unless on Spanish its number is lower than the Spanish-only one by more than one cluster standard error (the rule of the
grid); then there are two chooser files, selected by the sensor. The Spanish-only model on the English pages is the transfer
figure and is reported only. English is report-only: what it measures decides nothing for Spanish.
"""

import argparse
import json
import sys
import time
from pathlib import Path

from bench.guard import operating, pairs, train_chooser

METRICS = ("safe_fixed", "more_net")


def metric_of(report: dict, language: str) -> dict:
    """The number a language's block of a `report_by_language` is judged by: the correct fixes applied at `safe` when it applies
    any, else the net balance of `more`, with its cluster standard error."""
    part = report["by_language"].get(language)
    if part is None:
        return {"metric": None, "value": None, "se": None}
    safe = part["safe"]["total"]
    if safe["applied"] > 0:
        interval = safe["intervals"]["fixed"]
        return {"metric": "safe_fixed", "value": interval["value"], "se": interval["se"], "applied": safe["applied"]}
    interval = part["more"]["total"]["intervals"]["net"]
    return {
        "metric": "more_net",
        "value": interval["value"],
        "se": interval["se"],
        "applied": part["more"]["total"]["applied"],
    }


def choose_chooser(spanish_alone: dict, joint: dict) -> dict:
    """`joint` ships unless the Spanish-only chooser beats it on Spanish by more than one standard error of the Spanish-only
    number; the two must be judged by the same metric, else the comparison is not made and the joint model ships."""
    if spanish_alone["value"] is None or joint["value"] is None:
        return {"ships": "joint", "reason": "no Spanish number to compare"}
    if spanish_alone["metric"] != joint["metric"]:
        return {"ships": "joint", "reason": f"different metrics ({spanish_alone['metric']}, {joint['metric']})"}
    cost = spanish_alone["value"] - joint["value"]
    ships = "split" if cost > spanish_alone["se"] else "joint"
    return {"ships": ships, "cost_to_spanish": cost, "one_se": spanish_alone["se"], "metric": joint["metric"]}


def only(pages: list, language: str) -> list:
    return [p for p in pages if (p.language or pairs.DEFAULT_LANGUAGE) == language]


def run_variant(pages: list, learner, threads: int, structure: bool, resamples: int) -> dict:
    run = train_chooser.run(pages, {"chooser": learner}, threads=threads, structure=structure)
    return operating.report_by_language(run.table, run.raw["chooser"], cross_fit=True, resamples=resamples)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    pairs.add_data_arguments(parser)
    parser.add_argument("--out", type=Path, default=Path("bench/results/chooser-joint.json"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--structure", action="store_true")
    args = parser.parse_args(argv)
    started = time.time()
    pages = pairs.pages_from_args(args)
    learner = train_chooser.lightgbm_learner(threads=args.threads)
    variants = {"joint": pages, "es_only": only(pages, "es"), "en_only": only(pages, "en")}
    reports = {
        name: run_variant(subset, learner, args.threads, args.structure, 300)
        for name, subset in variants.items()
        if subset
    }
    out = {
        "pages": {name: len(subset) for name, subset in variants.items()},
        "numbers": {name: {lang: metric_of(report, lang) for lang in ("es", "en")} for name, report in reports.items()},
        "exploratory": "31x200",
        "seconds": None,
    }
    if "es_only" in reports and "joint" in reports:
        out["decision"] = choose_chooser(out["numbers"]["es_only"]["es"], out["numbers"]["joint"]["es"])
    out["seconds"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("pages", "decision") if k in out}, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
