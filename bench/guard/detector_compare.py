"""The retrained detector against the v4 detector on the same rows, at the 5 % review budget.

The v4 detector is the frozen file `models/guard-v4.model` (byte for byte the one the v4 test read registered; its
digest is `V4_SHA256` and a test compares them). Nothing is fitted here: both detectors score the same rows (the signals
of the first reading of each page, as `bench.guard.dataset` builds them), and the report says, per group of rows
(a language, a slice), for the v4 file and for each v5 model:

* recall, precision and average precision at the 5 % review budget (the share of the highest scores flagged, the
  measure of v4's headline `guard.*.recall_at_review_5pct`), each with the 95 % interval of a bootstrap that
  resamples whole documents;
* what the shipped thresholds of that budget flag (share of words, recall);
* the paired difference v5 - v4 on the same resampled documents, and the non-inferiority verdict: the v5 detector «loses at most
  1 point of recall» when `v4 recall - v5 recall <= 0.01` at the point value (the pre-registered rule is a point); the
  interval is reported next to it (`interval_clears`: its lower bound is >= -0.01).

The raw OCR's WER is not here: it is `wer_as_read` of the same pages in `bench/read_test_v5.py`.

    from bench.guard import detector_compare
    groups = detector_compare.rows_by_group(records, key)          # key(record) -> "es" | "en" | ...
    report = detector_compare.compare(groups, {"default": default_model, "thorough": thorough_model})
"""

import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

from bench.guard import analysis, dataset, metrics
from extract.guard import signals
from extract.guard.model import GuardModel

REFERENCE = Path(__file__).resolve().parents[2] / "models" / "guard-v4.model"
V4_SHA256 = "344d973db341c83768f9750d088f4b99e4d0b98162f0b61892dff3044c874297"
SHARE = analysis.REVIEW_SHARE
TOLERANCE = 0.01  # non-inferiority: at most 1 point of recall lost
RESAMPLES = 2000


def load_reference(path: Path = REFERENCE) -> GuardModel:
    """The v4 detector; refuses a file that is not the registered one."""
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != V4_SHA256:
        raise ValueError(f"{path} is not the frozen v4 detector: {digest} != {V4_SHA256}")
    return GuardModel.load(Path(path))


def _join(parts: list) -> "dataset.Rows":
    names = ("x", "y", "group", "band", "kind", "detail", "page", "text", "truth", "box", "chars")
    counts: dict = {}
    for part in parts:
        for name, value in part.counts.items():
            counts[name] = counts.get(name, 0) + value
    if not parts:
        return dataset.rows_from_records([])
    return dataset.Rows(*[np.concatenate([getattr(part, name) for part in parts]) for name in names], counts=counts)


def rows_by_group(records, key, languages=("es", "en"), pooled: bool = False) -> dict[str, "dataset.Rows"]:
    """The rows of (path, record) pairs split by `key(record)`; with `pooled` also the group `all` with every page (the published tables are per language: it is off by default). The pages are
    taken one at a time (a record is several MB; the records are never all in memory). Pages the engine called
    unreadable have no rows, as everywhere in `bench.guard.dataset`."""
    parts = defaultdict(list)
    for pair in records:
        parts[key(pair[1])].append(dataset.rows_from_records([pair], languages))
    groups = {name: _join(items) for name, items in sorted(parts.items())}
    if pooled:
        groups["all"] = _join([part for items in parts.values() for part in items])
    return groups


def _shipped(model: GuardModel, raw: np.ndarray, digits: np.ndarray, budget: float) -> np.ndarray:
    t_word, t_number = model.thresholds(budget)
    return model.probability(raw) >= np.where(digits, t_number, t_word)


def page_table(rows, model: GuardModel, share: float = SHARE) -> list[dict]:
    """One row per page for one detector: its words, its wrong words and how many of them are among the `share` of
    the group's words with the highest scores (the flags of the pooled ranking, as the recall above). Sums to the
    recall of `compare_rows`; for a bootstrap by document that keeps the flags of the whole group."""
    flagged = metrics.flag_top(model.raw(rows.x), share)
    wrong = rows.y == 1
    table = []
    for index in metrics.document_index(rows.page):
        page = str(rows.page[index[0]])
        table.append(
            {
                "doc": str(rows.group[index[0]]),
                "page": page,
                "words": int(len(index)),
                "wrong_words": int(wrong[index].sum()),
                "flagged": int(flagged[index].sum()),
                "wrong_flagged": int((wrong & flagged)[index].sum()),
            }
        )
    return table


def compare_rows(
    rows, reference: GuardModel, candidates: dict[str, GuardModel], resamples: int = RESAMPLES, seed: int = 1
) -> dict:
    """The figures of one group of rows for the v4 detector and each candidate."""
    y, groups = rows.y, rows.group
    digits = rows.x[:, signals.INDEX["has_digit"]] == 1.0
    models = {"v4": reference, **candidates}
    raw = {name: model.raw(rows.x) for name, model in models.items()}
    flagged = {name: _shipped(model, raw[name], digits, SHARE) for name, model in models.items()}
    statistics = {}
    for name in models:
        statistics[f"{name}.recall"] = lambda r, s=raw[name]: metrics.recall_at_share(y[r], s[r], SHARE)
        statistics[f"{name}.precision"] = lambda r, s=raw[name]: metrics.precision_at_share(y[r], s[r], SHARE)
        statistics[f"{name}.ap"] = lambda r, s=raw[name]: metrics.average_precision(y[r], s[r])
        statistics[f"{name}.shipped_recall"] = lambda r, f=flagged[name]: metrics.recall_precision(y[r], f[r])[0]
        statistics[f"{name}.shipped_flagged_share"] = lambda r, f=flagged[name]: float(f[r].mean())
    for name in candidates:
        statistics[f"{name}.minus_v4.recall"] = lambda r, n=name: (
            metrics.recall_at_share(y[r], raw[n][r], SHARE) - metrics.recall_at_share(y[r], raw["v4"][r], SHARE)
        )
    summary = metrics.cluster_bootstrap(groups, statistics, resamples, seed)
    out = {
        "rows": int(len(y)),
        "wrong": int(y.sum()),
        "documents": int(len(set(groups.tolist()))),
        "v4": {k: summary[f"v4.{k}"] for k in ("recall", "precision", "ap", "shipped_recall", "shipped_flagged_share")},
    }
    for name in candidates:
        difference = summary[f"{name}.minus_v4.recall"]
        loss = summary["v4.recall"]["value"] - summary[f"{name}.recall"]["value"]
        out[name] = {
            k: summary[f"{name}.{k}"] for k in ("recall", "precision", "ap", "shipped_recall", "shipped_flagged_share")
        }
        out[name]["minus_v4_recall"] = {
            **difference,
            "excludes_zero": bool(difference["low"] > 0 or difference["high"] < 0),
        }
        out[name]["g31"] = {
            "loss": float(loss),
            "tolerance": TOLERANCE,
            "met": bool(loss <= TOLERANCE + 1e-12),
            "interval_clears": bool(difference["low"] >= -TOLERANCE),
        }
    return out


def compare(
    groups: dict,
    candidates: dict[str, GuardModel],
    reference: GuardModel | None = None,
    resamples: int = RESAMPLES,
    seed: int = 1,
) -> dict:
    """`groups` maps a name to its rows; `candidates` maps a name (`default`, `thorough`) to a v5 model. A group
    without wrong words, or without rows, is reported as such and not scored."""
    reference = reference if reference is not None else load_reference()
    report = {
        "share": SHARE,
        "tolerance": TOLERANCE,
        "reference": {"file": "models/guard-v4.model", "sha256": V4_SHA256, "model_id": reference.model_id()},
        "candidates": {name: model.model_id() for name, model in candidates.items()},
        "groups": {},
    }
    for name, rows in groups.items():
        if not len(rows.y) or not int(rows.y.sum()):
            report["groups"][name] = {"rows": int(len(rows.y)), "wrong": 0, "note": "no wrong words: nothing to score"}
            continue
        report["groups"][name] = compare_rows(rows, reference, candidates, resamples, seed)
    return report


def main(argv: list[str] | None = None) -> None:
    """The comparison on development sets (never a sealed one: the loader refuses them), per language of the set."""
    import argparse
    import json
    import os

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sets", nargs="+", default=["val", "dev5en"])
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--pooled", action="store_true", help="also the pooled ES+EN group (not for publication)")
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    records = (pair for name in args.sets for pair in dataset.iter_page_records(args.data, name))
    groups = rows_by_group(records, lambda record: "en" if record["set"].endswith("en") else "es", pooled=args.pooled)
    root = REFERENCE.parent
    candidates = {
        "default": GuardModel.load(root / "guard.model"),
        "thorough": GuardModel.load(root / "guard-thorough.model"),
    }
    report = compare(groups, candidates, resamples=args.resamples)
    report["sets"] = args.sets
    args.out.write_text(json.dumps(report, indent=1, default=float) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
