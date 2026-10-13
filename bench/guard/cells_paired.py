"""The value of the second reading cell by cell, out of fold, with the paired bootstrap of every difference.

Six runs of the same protocol on the same pages and folds, each seeing different readings: none, L(latin), L(wide),
L(latin+wide), Z(deep) and L+Z. Each run keeps, per document, what `safe` and `more` apply, fix and break; a comparison of
two runs resamples the same documents in both and takes the difference of the sums (2,000 resamples), so the interval says
what a single run's standard error cannot.

    python3 -s -m bench.guard.cells_paired run  --structure --config none latin --out-dir bench/results/cells
    python3 -s -m bench.guard.cells_paired compare --out-dir bench/results/cells --out bench/results/cells-paired.json
"""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np

from bench.guard import baselines5, operating, pairs, train_chooser
from extract.guard import readings

CONFIGS: dict[str, tuple[list[str], tuple[str, ...] | None]] = {
    "none": ([], None),
    "latin": (["L"], ("latin",)),
    "wide": (["L"], ("wide",)),
    "latin+wide": (["L"], ("latin", "wide")),
    "deep": (["Z"], ("deep",)),
    "latin+wide+deep": (["L", "Z"], ("latin", "wide", "deep")),
}
LEARNERS = ("lightgbm", "logistic")
READER_OF_CELL = {"L": ("latin", "wide"), "Z": ("deep",)}


def parse_config(name: str):
    """`latin+wide+deep` (all the units of the cells) or `latin+wide+deep@4/5`: the lines of L read up to rank 4 and the zones
    of Z up to rank 5 (the runtime caps of the two cells, hidden by reader); returns the cells, the readers and the caps."""
    base, _, cap = name.partition("@")
    if base not in CONFIGS:
        raise ValueError(f"unknown config {name!r}")
    cells, readers = CONFIGS[base]
    caps = None
    if cap:
        lines, zones = (int(x) for x in cap.split("/"))
        caps = {**{r: lines for r in READER_OF_CELL["L"]}, "deep": zones}
    return cells, readers, caps


SEED = 1
ES_SETS = ("train", "val", "test")
SAFE_EDGE = 1.1


def per_document(table, choice, mask, documents: list[str]) -> dict:
    """What a rule applies in each document: applied, fixed and broken, in the order of `documents`."""
    index = {d: i for i, d in enumerate(documents)}
    out = {name: np.zeros(len(documents), dtype=int) for name in ("applied", "fixed", "broken")}
    docs = np.asarray(table.doc)
    for site in np.nonzero(mask)[0]:
        i = index[str(docs[site])]
        out["applied"][i] += 1
        code = int(choice.outcome[site])
        if code == operating.FIXED:
            out["fixed"][i] += 1
        elif code == operating.BROKEN:
            out["broken"][i] += 1
    return {k: v.tolist() for k, v in out.items()}


def trace_memory(every: float = 10.0) -> None:
    """A thread that prints the resident memory and where the main thread is (to find what makes a run big)."""
    import sys
    import threading

    main = threading.main_thread().ident

    def loop():
        while True:
            rss = int(open("/proc/self/status").read().split("VmRSS:")[1].split()[0]) // 1024
            frame = sys._current_frames().get(main)
            chain = []
            while frame is not None and len(chain) < 4:
                chain.append(f"{frame.f_code.co_name}:{frame.f_lineno}")
                frame = frame.f_back
            print(f"[mem] {rss} MB  {' < '.join(chain)}", file=sys.stderr, flush=True)
            time.sleep(every)

    threading.Thread(target=loop, daemon=True).start()


def safe_edge(applied: int) -> bool:
    """True when `safe` applies something but under 1.1 times its minimum count: a figure that the cliff of the minimum can
    turn into zero (the cap changes it from 355 to 0), not a stable one."""
    return bool(0 < applied < SAFE_EDGE * operating.MIN_APPLIED_TEXT)


def run_config(name: str, args) -> dict:
    cells, readers, caps = parse_config(name)
    spec = readings.ReadingSpec("line", readers) if readers else None
    started = time.time()
    pages = pairs.load_pages(args.data, args.sets, manifest=args.manifest, cells=cells, spec=spec, caps=caps)
    documents = sorted({page.doc for page in pages})
    makers = {
        "lightgbm": lambda: train_chooser.lightgbm_learner(threads=args.threads),
        "logistic": baselines5.logistic_learner,  # a dense float64 copy of the rows: +1 GB at its peak
    }
    learners = {name: makers[name]() for name in args.learners}
    run = train_chooser.run(pages, learners, seed=SEED, threads=args.threads, structure=args.structure)
    wrong = int(run.words.y.sum())
    out = {
        "config": name,
        "cells": cells,
        "readers": list(readers or ()),
        "caps": caps,
        "pages": len(pages),
        "sets": list(args.sets),
        "languages": dict(Counter(page.language or "es" for page in pages)),
        "documents": documents,
        "wrong_words": wrong,
        "sites": int(run.table.sites),
        "reachable_sites": int(run.table.reachable.sum()),
        "learners": {},
    }
    for learner, raw in run.raw.items():
        choice, ops, kinds, thresholds, masks = operating.points_by_op(run.table, raw, cross_fit=True)
        row = {"thresholds": thresholds}
        for rule, mask in masks.items():
            counts = operating.with_intervals(choice, mask, run.table.doc, 500)
            row[rule] = {
                "applied": counts["applied"],
                "fixed": counts["fixed"],
                "broken": counts["broken"],
                "net": counts["net"],
                "precision": counts["precision"],
                "net_se": counts["intervals"]["net"]["se"],
                "net_interval": [counts["intervals"]["net"]["low"], counts["intervals"]["net"]["high"]],
                "by_document": per_document(run.table, choice, mask, documents),
            }
        # `safe` of words hangs on a minimum count: a figure just over it is not a stable one (the cliff of 300)
        row["safe"]["safe_edge"] = safe_edge(row["safe"]["applied"])
        out["learners"][learner] = row
    out["seconds"] = round(time.time() - started, 1)
    return out


def paired(a: dict, b: dict, key: str, resamples: int, seed: int = 1) -> dict:
    """`a - b` of the sum of `key` over the documents (the same documents resampled in both), with its 95 % interval."""
    x, y = np.asarray(a[key], dtype=float), np.asarray(b[key], dtype=float)
    diff = x - y
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(diff), size=(resamples, len(diff)))
    sums = diff[picks].sum(axis=1)
    low, high = np.percentile(sums, [2.5, 97.5])
    return {
        "difference": float(diff.sum()),
        "se": float(sums.std(ddof=1)),
        "low": float(low),
        "high": float(high),
        "excludes_zero": bool(low > 0 or high < 0),
    }


def net_of(row: dict) -> dict:
    d = row["by_document"]
    return {"net": [f - b for f, b in zip(d["fixed"], d["broken"], strict=True)], **d}


COMPARISONS = (
    ("latin", "none"),
    ("wide", "none"),
    ("latin+wide", "none"),
    ("deep", "none"),
    ("latin+wide+deep", "none"),
    ("latin+wide", "latin"),
    ("latin+wide", "wide"),
    ("latin+wide+deep", "latin+wide"),
    ("latin+wide+deep", "deep"),
    ("deep", "latin+wide"),
)


SPEED = Path("bench/results/reader-speed.json")


ARMS = Path("bench/results/reader-arms.json")


def arms_cost(lines: int, zones: int, arms: dict) -> dict:
    """The seconds a page costs L read up to `lines` and Z up to `zones`, from the per-page, per-unit seconds of
    `reader-arms.json` (60 pages; the readers' time adds up over the units), each cell scaled so that all its units give
    the batched figures of the product (as `pilot_cells.cost_of`). Mean and p95 are taken over the pages of the sum, so
    the p95 is the joint one, not a bound; any cap is allowed, not only the budgets of reader-speed."""
    total = None
    for cell, budget in (("L", lines), ("Z", zones)):
        entry = arms["cells"][cell]
        full = np.array([p.get("fixed_s", 0.0) + sum(p["unit_s"]) for p in entry["per_page"]])
        scale = float(entry["seconds_per_page_mean"]) / float(full.mean())
        part = np.array([p.get("fixed_s", 0.0) + sum(p["unit_s"][:budget]) for p in entry["per_page"]]) * scale
        total = part if total is None else total + part
    return {
        "mean_s": round(float(total.mean()), 3),
        "p95_s": round(float(np.percentile(total, 95)), 3),
        "pages": len(total),
    }


def reading_cost(name: str, speed: dict | None) -> dict | None:
    """The seconds a page costs the readings of a config, from the official `reader-speed.json`: the cells at their budgets
    (`L` 4/8/16 lines, `Z` 5/10/20/40 zones) add up in the mean; the p95 is the sum of the two p95 (an upper bound: the
    pages that are slow in L are not all slow in Z). None when a budget was not measured."""
    if speed is None:
        return None
    base, _, cap = name.partition("@")
    cells, _ = CONFIGS[base]
    if base in ("latin", "wide", "latin+wide", "deep", "latin+wide+deep") and not cap:
        full = {"L": 16, "Z": 40}
        budgets = {c: full[c] for c in cells}
    elif cap:
        lines, zones = (int(x) for x in cap.split("/"))
        budgets = {"L": lines, "Z": zones}
    else:
        return {"mean_s": 0.0, "p95_s": 0.0}
    mean = p95 = 0.0
    for cell, budget in budgets.items():
        if budget == 0:  # the cell is not read at all
            continue
        entry = speed["cells"][cell]["budgets"].get(str(budget))
        if entry is None:  # a cap that reader-speed did not measure: only the per-page model of the arms has it
            mean = p95 = None
            break
        if base in ("latin", "wide") and cell == "L":  # one reader of the two: its own mean, at the full budget only
            reader = speed["cells"]["L"]["readers"][base]
            entry = {"mean": reader["mean"], "p95": reader["p95"]}
        mean += entry["mean"]
        p95 += entry["p95"]
    if mean is None:
        return (
            {"per_page_arms": arms_cost(lines, zones, json.loads(ARMS.read_text(encoding="utf-8")))}
            if ARMS.is_file()
            else None
        )
    out = {"mean_s": round(mean, 3), "p95_s_upper_bound": round(p95, 3)}
    if cap and ARMS.is_file():
        out["per_page_arms"] = arms_cost(lines, zones, json.loads(ARMS.read_text(encoding="utf-8")))
    return out


def same_pages(runs: dict[str, dict]) -> None:
    """The runs are paired by position in `by_document`: refuse runs that do not hold the same documents in the same order
    (or were not read from the same sets)."""
    first_name, first = next(iter(runs.items()))
    for name, run in runs.items():
        sets = run.get("sets"), first.get("sets")  # the first runs did not record them
        if run["documents"] != first["documents"] or (None not in sets and sets[0] != sets[1]):
            raise ValueError(f"run {name!r} does not hold the same documents/sets as {first_name!r}: cannot be paired")


def compare(runs: dict[str, dict], resamples: int) -> dict:
    out: dict = {"resamples": resamples, "runs": {}, "paired": {}, "learner_pairs": {}}
    speed = json.loads(SPEED.read_text(encoding="utf-8")) if SPEED.is_file() else None
    same_pages(runs)
    capped = sorted((n for n in runs if "@" in n), key=lambda n: tuple(int(x) for x in n.partition("@")[2].split("/")))
    comparisons = list(COMPARISONS) + [(name, other) for name in capped for other in ("latin+wide+deep", "none")]
    comparisons += list(zip(capped[1:], capped[:-1], strict=True))  # each cap against the next cheaper one
    chain = [f"latin+wide+deep@{cap}" for cap in ("4/4", "4/8", "8/8", "8/16")]  # the caps of the pilot
    comparisons += [pair for pair in zip(chain[1:], chain[:-1], strict=True) if pair not in comparisons]
    for name, run in runs.items():
        out["runs"][name] = {
            "reading_cost": reading_cost(name, speed),
            "wrong_words": run["wrong_words"],
            "sites": run["sites"],
            "reachable_sites": run["reachable_sites"],
            "seconds": run["seconds"],
            "learners": {
                learner: {
                    "safe": {k: v for k, v in row["safe"].items() if k != "by_document"},
                    "more": {k: v for k, v in row["more"].items() if k != "by_document"},
                    "thresholds": row["thresholds"],
                }
                for learner, row in run["learners"].items()
            },
        }
    for learner in LEARNERS:
        if not all(learner in run["learners"] for run in runs.values()):
            continue
        out["paired"][learner] = {}
        for a, b in comparisons:
            if a not in runs or b not in runs:
                continue
            ra, rb = runs[a]["learners"][learner], runs[b]["learners"][learner]
            wrong = runs[b]["wrong_words"]
            more = paired(net_of(ra["more"]), net_of(rb["more"]), "net", resamples)
            safe_fixed = paired(net_of(ra["safe"]), net_of(rb["safe"]), "fixed", resamples)
            safe_applied = paired(net_of(ra["safe"]), net_of(rb["safe"]), "applied", resamples)
            out["paired"][learner][f"{a} - {b}"] = {
                "more_net": more,
                "more_net_per_100_wrong": 100 * more["difference"] / wrong,
                "safe_fixed": safe_fixed,
                "safe_applied": safe_applied,
            }
    for name, run in runs.items():  # the logistic chooser against the binary LightGBM, same pages
        if "logistic" not in run["learners"]:
            continue
        a, b = run["learners"]["lightgbm"], run["learners"]["logistic"]
        out["learner_pairs"][name] = {
            "more_net": paired(net_of(a["more"]), net_of(b["more"]), "net", resamples),
            "safe_fixed": paired(net_of(a["safe"]), net_of(b["safe"]), "fixed", resamples),
        }
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("what", choices=("run", "compare"))
    pairs.add_data_arguments(parser)
    parser.set_defaults(
        sets=list(ES_SETS)
    )  # the default of the other runners includes the English set (+60 % of RAM, other figures)
    parser.add_argument("--structure", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--config", nargs="+", default=list(CONFIGS))
    parser.add_argument("--out-dir", type=Path, nargs="+", default=[Path("bench/results/cells")])
    parser.add_argument("--out", type=Path, default=Path("bench/results/cells-paired.json"))
    parser.add_argument("--learners", nargs="+", default=["lightgbm"], choices=list(LEARNERS))
    parser.add_argument("--trace-memory", action="store_true")
    parser.add_argument("--resamples", type=int, default=2000)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.what == "run":
        if args.trace_memory:
            trace_memory()
        out_dir = args.out_dir[0]
        out_dir.mkdir(parents=True, exist_ok=True)
        for name in args.config:
            result = run_config(name, args)
            (out_dir / f"{name.replace('/', '-')}.json").write_text(json.dumps(result, default=float), encoding="utf-8")
            more = result["learners"]["lightgbm"]["more"]
            safe = result["learners"]["lightgbm"]["safe"]
            print(
                json.dumps(
                    {
                        "config": name,
                        "seconds": result["seconds"],
                        "more_net": more["net"],
                        "more_se": more["net_se"],
                        "safe_applied": safe["applied"],
                        "safe_fixed": safe["fixed"],
                        "safe_broken": safe["broken"],
                    }
                ),
                flush=True,
            )
        return 0
    runs = {}
    for folder in args.out_dir:
        for path in sorted(folder.glob("*.json")):
            run = json.loads(path.read_text(encoding="utf-8"))
            runs[run["config"]] = run
    result = compare(runs, args.resamples)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    print(json.dumps(result["paired"]["lightgbm"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
