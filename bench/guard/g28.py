"""The source-dropping and canary experiments, out of fold on the development pages: each candidate source and each reader dropped from
the pool, LambdaRank against the binary chooser, the detector with and without its monotone constraints, the learning
curve at 25, 50 and 100 % of the documents, and the canary (labels permuted inside each site).

    python3 -s -m bench.guard.g28 sources  --structure --out bench/results/g28-sources.json
    python3 -s -m bench.guard.g28 learners --structure --out bench/results/g28-learners.json
    python3 -s -m bench.guard.g28 curve    --structure --out bench/results/g28-curve.json
    python3 -s -m bench.guard.g28 canary   --structure --out bench/results/g28-canary.json

Everything is exploratory: one chooser cell (31x200), the pages and folds of the manifest. A source stays in the runtime only
if dropping it costs at least 2 correct fixes per 100 wrong words at `safe` or at least 2 net fixed words per 100 wrong words
in `more`: the cost is reported per 100 wrong words of the development pages.
"""

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from bench.guard import challenger, operating, pairs, train_chooser
from extract.guard import candidates, model

SOURCES = ("dictionary", "fold", "punct", "page", "ctc_alt", "reading", "reading:latin", "reading:wide", "reading:deep")
KEEP_PER_100 = 2.0


@contextmanager
def without_sources(dropped):
    """The candidate pool with the sources in `dropped` removed: a name drops that source, `reading` drops every reader."""
    real = candidates.by_source

    def filtered(ctx, index, context_rule=True):
        found = real(ctx, index, context_rule)
        return {
            name: texts
            for name, texts in found.items()
            if name not in dropped and not (name.startswith("reading:") and "reading" in dropped)
        }

    with mock.patch.object(candidates, "by_source", filtered):
        yield


def summary(run, resamples: int = 300) -> dict:
    """`safe` and `more` of every learner of a run, with the number of wrong words that the per-100 figures divide by."""
    wrong = int(run.words.y.sum())
    out = {"wrong_words": wrong}
    for name, raw in run.raw.items():
        report = operating.report_by_op(run.table, raw, cross_fit=True, resamples=resamples)
        safe, more = report["safe"]["total"], report["more"]["total"]
        out[name] = {
            "safe_applied": safe["applied"],
            "safe_fixed": safe["fixed"],
            "more_applied": more["applied"],
            "more_net": more["net"],
            "more_net_se": more["intervals"]["net"]["se"],
            "more_broken": more["broken"],
            "sites": int(run.table.sites),
            "reachable_sites": int(run.table.reachable.sum()),
        }
    return out


def per_100(value: float, wrong: int) -> float:
    return 100.0 * value / wrong if wrong else float("nan")


def source_verdicts(base: dict, dropped: dict, wrong: int) -> dict:
    """For each dropped source: what dropping it costs at `safe` and in `more`, per 100 wrong words, and whether the rule
    keeps it (a cost of at least 2 per 100 in either)."""
    out = {}
    for name, row in dropped.items():
        safe = per_100(base["safe_fixed"] - row["safe_fixed"], wrong)
        more = per_100(base["more_net"] - row["more_net"], wrong)
        out[name] = {
            "safe_fixed_lost_per_100_wrong": safe,
            "more_net_lost_per_100_wrong": more,
            "keeps": bool(safe >= KEEP_PER_100 or more >= KEEP_PER_100),
            "reachable_sites": row["reachable_sites"],
        }
    return out


def curve_points(results: dict) -> dict:
    """The learning curve: the net of `more` (and its SE) at each share of the documents."""
    return {
        share: {"more_net": r["more_net"], "se": r["more_net_se"], "safe_fixed": r["safe_fixed"]}
        for share, r in results.items()
    }


def canary_verdict(row: dict) -> dict:
    """Canary: the chooser trained on labels permuted inside each site applies nothing at `safe` and nets at most zero in `more`
    (up to one standard error: a net of 0 +- noise is a pass, a net clearly over zero is a leak)."""
    return {
        "safe_applied": row["safe_applied"],
        "more_net": row["more_net"],
        "more_net_se": row["more_net_se"],
        "passes": bool(row["safe_applied"] == 0 and row["more_net"] <= row["more_net_se"]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=("sources", "learners", "curve", "canary"))
    pairs.add_data_arguments(parser)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--structure", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="sources: drop only these")
    args = parser.parse_args(argv)
    started = time.time()
    pages = pairs.pages_from_args(args)
    learner = train_chooser.lightgbm_learner(threads=args.threads)
    result: dict = {"pages": len(pages), "structure_sites": args.structure, "exploratory": "31x200"}

    def run(learners, **kwargs):
        return train_chooser.run(pages, learners, threads=args.threads, structure=args.structure, **kwargs)

    if args.what == "sources":
        base = summary(run({"chooser": learner}))
        result["base"] = base
        result["dropped"] = {}
        for name in args.only or SOURCES:
            with without_sources({name}):
                result["dropped"][name] = summary(run({"chooser": learner}))["chooser"]
        result["verdicts"] = source_verdicts(base["chooser"], result["dropped"], base["wrong_words"])
    elif args.what == "learners":
        result["learners"] = summary(
            run({"binary": learner, "lambdarank": challenger.lambdarank_learner(threads=args.threads)})
        )
        real = model.train_detector
        with mock.patch.object(
            train_chooser.model, "train_detector", lambda *a, **k: real(*a, **{**k, "monotone": False})
        ):
            result["detector_without_monotone"] = summary(run({"chooser": learner}))["chooser"]
    elif args.what == "curve":
        cache = pairs.SiteCache(pages)
        result["points"] = {}
        for share in (0.25, 0.5, 1.0):
            result["points"][str(share)] = summary(run({"chooser": learner}, cache=cache, train_share=share))["chooser"]
        result["curve"] = curve_points(result["points"])
    else:
        permuted = summary(run({"chooser": learner}, permute_labels=True))
        result["canary"] = canary_verdict(permuted["chooser"])
    result["seconds"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(
        json.dumps({k: v for k, v in result.items() if k in ("verdicts", "curve", "canary", "seconds")}, default=float)
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
