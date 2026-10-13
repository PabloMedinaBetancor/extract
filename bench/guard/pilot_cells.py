"""Gate P, the pilot's cells : what each cell of the second reading adds to the guard's
pool, what it costs, and which cell the pre-registered rule picks.

The pilot reads 60 pages with the engine (pass 1, with the alternatives and the crops) and then, for every cell, pass 2
(`bench/run_pass2.py --cell L|Z|LD|ZC`): L = (line, cheap readers), Z = (zone, deep), and the crossed LD and ZC. A cell
stored at its largest budget holds the readings of every unit; a smaller budget is its prefix (units of rank below the
budget: `extract.engine.reread.view`), so one pass 2 serves every budget, and a reader is a subset of its sources.

The **pool recall gain** of a variant is the written-form share of the wrong words (components weighted by their truth
words, structure included: a pair needs the right text among the sources of its site, a split a join, a merge a
segmentation, an invented word a drop site) that the pool reaches with the second readings, minus the share it reaches
without them (the dictionary, the vocabulary, the alternatives and `masked` are in both). Its interval resamples the
documents. The **harm** counts the right words a reader would change, and `net100` is the wrong words with the right
text among a reader's texts minus the right words it contradicts, per 100 wrong.

`choose_reader` is the pre-registered rule as a pure function. Gate P: the written-form recall of the plain wrong
one-to-one words with the second readings must reach 85 %; under 80 % the build stops and the maintainer is told.

    python3 -s -m bench.guard.pilot_cells --out bench/results/pilot-cells.json
"""

import argparse
import dataclasses
import json
import os
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

from bench.guard import accounting, metrics, pairs, pool_recall
from extract.engine.reread import view
from extract.guard import candidates, structure
from extract.guard.forms import normalise

CELLS = {"L": ("line", "cheap"), "Z": ("zone", "deep"), "LD": ("line", "deep"), "ZC": ("zone", "cheap")}
BUDGETS = {"line": (4, 8, 16), "zone": (5, 10, 20, 40)}
FAMILIES = ("replace", "join", "segment", "drop", "other")
PLAIN_KINDS = ("letters", "accent", "punctuation", "case")
MEAN_CAP_S = 1.6
P95_CAP_S = 3.5
RSS_CAP_MB = 1500.0  # the reader process's cap (`extract.engine.readerproc.RSS_CAP_MB`, pinned by a test)
PERMISSIVE = ("Apache-2.0", "MIT", "BSD-3-Clause", "BSD-2-Clause")
CHEAP_MODELS = ("ppocrv6-small", "latin-ppocrv5-mobile")  # what the cheap readers run (bake-off names)
DEEP_MODELS = ("ppocrv6-medium",)
BETWEEN_ACTION = "raise the candidate cap from 16 to 20 and repeat the pool recall; stop only if it stays under 80 %"
TIE_SECONDS = 0.05  # two cells whose mean seconds a page differ by less than this are tied in cost
GATE_PASS = 0.85
GATE_STOP = 0.80
MIN_GAIN_DOCS = 5


# ---------------------------------------------------------------------------------------------------------------------
# The pool reach of one variant of a page
# ---------------------------------------------------------------------------------------------------------------------


def family_of(component: accounting.Component) -> str:
    return {"pair": "replace", "split": "join", "merge": "segment", "invented": "drop"}.get(component.kind, "other")


def with_readings(page: pairs.Page, cap: int | None, sources: frozenset[str] | None) -> pairs.Page:
    """The page as the chooser would see it with the readings of the units of rank below `cap` and, when given, of the
    readers in `sources` only; the first reading's own extras (alternatives, masked) stay."""
    words = page.words if cap is None else view(page.words, cap)
    if sources is not None:
        kept = []
        for word in words:
            if word.readings is None:
                kept.append(word)
                continue
            readings = tuple(r for r in word.readings if r.source in sources)
            kept.append(dataclasses.replace(word, readings=readings or None))
        words = tuple(kept)
    return dataclasses.replace(page, words=tuple(words), _context=None)


def without_readings(page: pairs.Page) -> pairs.Page:
    return dataclasses.replace(
        page, words=tuple(dataclasses.replace(w, readings=None) for w in page.words), _context=None
    )


def reached(page: pairs.Page, component: accounting.Component, spans: set) -> bool:
    """Whether the pool of the page holds the truth of a wrong component, in written form."""
    ctx = page.context
    target = normalise(component.truth)
    if component.kind == "pair":
        index = component.words[0]
        if ctx.parts[index] is None:
            return False
        return any(normalise(t) == target for texts in candidates.by_source(ctx, index).values() for t in texts)
    if component.kind == "split":
        span = tuple(sorted(component.words))
        if span not in spans:
            return False
        return any(normalise(c.text) == target for c in structure.generate_structure(ctx, "join", span)[1:])
    if component.kind == "merge" and len(component.words) == 1:
        index = component.words[0]
        if not structure.segment_ok(ctx, index):
            return False
        return any(normalise(c.text) == target for c in structure.generate_structure(ctx, "segment", (index,))[1:])
    if component.kind == "invented":
        return structure.drop_ok(ctx, component.words[0])
    return False


def is_plain_wrong_pair(page: pairs.Page, component: accounting.Component) -> bool:
    """The population of the gate: a wrong one-to-one word of letters, an accent, a mark or a case, with a plain token."""
    if component.kind != "pair":
        return False
    index = component.words[0]
    kind = pool_recall.error_kind(page.labels[index], page.words[index])
    return kind in PLAIN_KINDS and page.context.parts[index] is not None


@dataclass
class Rows:
    """The wrong components of the pilot pages, one row each, in a fixed order, and the right words for the harm."""

    doc: list[str] = field(default_factory=list)
    weight: list[int] = field(default_factory=list)
    family: list[str] = field(default_factory=list)
    plain: list[bool] = field(default_factory=list)
    keys: list[tuple[str, int]] = field(default_factory=list)  # (page key, component index)

    def array(self) -> dict[str, np.ndarray]:
        return {
            "doc": np.array(self.doc, dtype=object),
            "weight": np.array(self.weight, dtype=np.float64),
            "family": np.array(self.family, dtype=object),
            "plain": np.array(self.plain, dtype=bool),
        }


def wrong_rows(pages: list[pairs.Page]) -> Rows:
    rows = Rows()
    for page in pages:
        texts = [w.text for w in page.words]
        for number, component in enumerate(page.components):
            if accounting.is_right(component, texts):
                continue
            rows.doc.append(page.doc)
            rows.weight.append(component.weight)
            rows.family.append(family_of(component))
            rows.plain.append(is_plain_wrong_pair(page, component))
            rows.keys.append((page.key, number))
    return rows


def reach_vector(pages: list[pairs.Page]) -> np.ndarray:
    """For every row of `wrong_rows(pages)`, whether the pool of that variant of the page reaches the truth."""
    out = []
    for page in pages:
        spans = set(structure.join_spans(page.context))
        texts = [w.text for w in page.words]
        for component in page.components:
            if not accounting.is_right(component, texts):
                out.append(reached(page, component, spans))
    return np.array(out, dtype=bool)


def unreached_causes(pages: list[pairs.Page], reach: np.ndarray, rows: Rows) -> dict:
    """Why the plain wrong words the pool does not reach are out of it: the truth is no dictionary word, it is a
    dictionary word within `CEILING_EDITS` letters that no source found, or it is farther than that."""
    by_key = {page.key: page for page in pages}
    causes: Counter = Counter()
    for hit, plain, (key, number) in zip(reach, rows.plain, rows.keys, strict=True):
        if hit or not plain:
            continue
        page = by_key[key]
        component = page.components[number]
        word = page.words[component.words[0]]
        if not pool_recall._in_dictionary(component.truth):
            causes["truth_outside_the_dictionary"] += 1
        elif (
            Levenshtein.distance(pool_recall.letters(word.text), pool_recall.letters(component.truth))
            <= pool_recall.CEILING_EDITS
        ):
            causes["within_3_edits_but_no_source_found_it"] += 1
        else:
            causes["farther_than_3_edits"] += 1
    return dict(causes)


def harm_counts(pages: list[pairs.Page]) -> Counter:
    """What the readings of a variant do to the words: the right words they contradict (harm), the wrong words whose
    right text is among their texts (fixed) and the words whose readings say something else than the word (changed)."""
    out: Counter = Counter()
    for page in pages:
        texts = [w.text for w in page.words]
        for component in page.components:
            if component.kind != "pair":
                continue
            word = page.words[component.words[0]]
            right = accounting.is_right(component, texts)
            readings = [
                r
                for r in (word.readings or ())
                if r.text is not None and not candidates.joins_words(r) and " " not in r.text.strip()
            ]
            if not readings:
                out["right_unread" if right else "wrong_unread"] += 1
                continue
            differing = [r.text for r in readings if normalise(r.text) != normalise(word.text)]
            if right:
                out["right_read"] += 1
                out["harmed"] += bool(differing)
                out["changed"] += bool(differing)
            else:
                out["wrong_read"] += 1
                hit = any(normalise(t) == normalise(component.truth) for t in differing)
                out["fixed"] += hit
                out["changed"] += bool(differing)
                out["changed_right"] += hit
    return out


def share(num: float, den: float) -> float | None:
    return round(num / den, 4) if den else None


def gain_statistic(weights: np.ndarray, gain: np.ndarray):
    def statistic(rows: np.ndarray) -> float:
        total = weights[rows].sum()
        return float(gain[rows].sum() / total) if total else float("nan")

    return statistic


def variant_report(
    base_reach: np.ndarray, reach: np.ndarray, harm: Counter, arrays: dict[str, np.ndarray], resamples: int = 1000
) -> dict:
    """The gain of one variant over the engine without readings, with its document-bootstrap interval, per family and on
    the plain wrong words, and what it does to right words."""
    weights, doc = arrays["weight"], arrays["doc"]
    gain = (reach & ~base_reach) * weights
    total = float(weights.sum())
    groups = np.unique(doc, return_inverse=True)[1]
    point = gain_statistic(weights, gain)
    summary = metrics.cluster_bootstrap(groups, {"gain": point}, resamples=resamples)["gain"] if len(weights) else {}
    by_family = {}
    for family in FAMILIES:
        mask = arrays["family"] == family
        if not mask.any():
            continue
        by_family[family] = {
            "wrong_weight": float(weights[mask].sum()),
            "base": share(float((base_reach * weights)[mask].sum()), float(weights[mask].sum())),
            "with_readings": share(float((reach * weights)[mask].sum()), float(weights[mask].sum())),
        }
    plain = arrays["plain"]
    plain_rows = np.nonzero(plain)[0]
    plain_ci = None
    if len(plain_rows):
        plain_groups = np.unique(doc[plain_rows], return_inverse=True)[1]
        interval = metrics.cluster_bootstrap(
            plain_groups, {"r": lambda rows: float(reach[plain_rows][rows].mean())}, resamples=resamples
        )["r"]
        plain_ci = [round(interval["low"], 4), round(interval["high"], 4)]
    wrong_read = harm["wrong_read"] + harm["wrong_unread"]
    return {
        "wrong_weight": total,
        "base": share(float((base_reach * weights).sum()), total),
        "with_readings": share(float((reach * weights).sum()), total),
        "gain": round(float(summary.get("value", float("nan"))), 4),
        "gain_ci95": [round(summary["low"], 4), round(summary["high"], 4)] if summary else None,
        "by_family": by_family,
        "plain_wrong_pairs": {
            "n": int(plain.sum()),
            "base": share(int((base_reach & plain).sum()), int(plain.sum())),
            "with_readings": share(int((reach & plain).sum()), int(plain.sum())),
            "with_readings_ci95": plain_ci,
        },
        "harm": {
            "right_words_read": harm["right_read"],
            "right_words_contradicted": harm["harmed"],
            "harm_rate": share(harm["harmed"], harm["right_read"]),
            "wrong_words_read": harm["wrong_read"],
            "wrong_words_fixed": harm["fixed"],
            "change_precision": share(harm["changed_right"], harm["changed"]),
            "net100": round(100 * (harm["fixed"] - harm["harmed"]) / wrong_read, 1) if wrong_read else None,
        },
    }


# ---------------------------------------------------------------------------------------------------------------------
# Costs and the choice
# ---------------------------------------------------------------------------------------------------------------------


def cost_of(arms: dict, cell: str, budget: int, largest: int) -> dict | None:
    """The measured cost of a cell at a budget, from `reader-arms.json` (run_pass2) or `reader-speed.json`, by the first
    form that fits:
    - `cells[cell]["per_page"]`, a list of `{"fixed_s": seconds a page costs whatever the budget, "unit_s": [seconds of the
      unit of rank 0, of rank 1, ...]}`: the reader's time is additive over the crops, so the cost of a budget is the
      fixed part plus the units of rank below it, and the mean and the p95 are taken over the pages (one pass 2 serves
      every budget; scaled to the batched figures when the cell has them);
    - the key `cell@budget`, or `cells[cell]["budgets"][budget]`, a summary measured at that budget (`mean`/`p95` as
      reader-speed.json writes them, or the longer names; the memory of the cell if the budget has none);
    - `cells[cell]`, one summary, taken for the cell's largest budget unless it names its `budget`.
    None when nothing was measured for it."""
    cells = arms.get("cells", arms)
    entry = parent = cells.get(f"{cell}@{budget}")
    if entry is None:
        entry = parent = cells.get(cell)
        if isinstance(entry, dict) and isinstance(entry.get("per_page"), list):
            return _cost_from_units(entry, budget)
        if isinstance(entry, dict) and "budgets" in entry:
            entry = entry["budgets"].get(str(budget))
        elif isinstance(entry, dict) and int(entry.get("budget", largest)) != budget:
            entry = None
    if not isinstance(entry, dict):
        return None
    mean = entry.get("seconds_per_page_mean", entry.get("pass2_seconds_mean", entry.get("mean")))
    p95 = entry.get("seconds_per_page_p95", entry.get("pass2_seconds_p95", entry.get("p95")))
    rss = entry.get("peak_rss_mb", entry.get("peak_reader_rss_mb"))
    if rss is None and isinstance(parent, dict):  # the memory of the reader process is one figure for the cell
        rss = parent.get("peak_rss_mb", parent.get("peak_reader_rss_mb"))
    if mean is None or p95 is None:
        return None
    return {"mean_s": float(mean), "p95_s": float(p95), "rss_mb": None if rss is None else float(rss)}


def _cost_from_units(entry: dict, budget: int) -> dict:
    """Mean and p95 of the pages' seconds at a budget. When the cell also holds the batched pass 2 as the product runs it
    (`seconds_per_page_mean` and `_p95` over all the units), the per-unit times are scaled so that the full budget gives
    exactly those figures: a batch pads its crops to the widest, so the units read one by one add up to less."""
    fixed = [p.get("fixed_s", 0.0) for p in entry["per_page"]]
    seconds = np.array([f + sum(p["unit_s"][:budget]) for f, p in zip(fixed, entry["per_page"], strict=True)])
    full = np.array([f + sum(p["unit_s"]) for f, p in zip(fixed, entry["per_page"], strict=True)])
    mean_scale = p95_scale = 1.0
    batch_mean, batch_p95 = entry.get("seconds_per_page_mean"), entry.get("seconds_per_page_p95")
    if batch_mean is not None and batch_p95 is not None and full.mean() > 0 and np.percentile(full, 95) > 0:
        mean_scale = float(batch_mean) / float(full.mean())
        p95_scale = float(batch_p95) / float(np.percentile(full, 95))
    rss = entry.get("peak_rss_mb", entry.get("peak_reader_rss_mb"))
    return {
        "mean_s": float(seconds.mean()) * mean_scale,
        "p95_s": float(np.percentile(seconds, 95)) * p95_scale,
        "rss_mb": None if rss is None else float(rss),
        "batch_scale": [round(mean_scale, 3), round(p95_scale, 3)],
    }


def licence_in_order(bakeoff: dict | None, cell: str) -> bool:
    """Whether the models a cell runs have a permissive licence, a download address and a sha256 in the bake-off file."""
    if not bakeoff:
        return False
    names = DEEP_MODELS if CELLS[cell][1] == "deep" else CHEAP_MODELS
    models = bakeoff.get("models", {})
    return all(names_ok(models.get(name)) for name in names)


def names_ok(model: dict | None) -> bool:
    return bool(model and model.get("licence") in PERMISSIVE and model.get("url") and model.get("sha256"))


@dataclass(frozen=True)
class Option:
    """One cell at one budget: its cost (None when not measured), whether its model's licence and download are in order
    and its gain per wrong component (the weighted reach it adds, row by row)."""

    cell: str
    budget: int
    mean_s: float | None
    p95_s: float | None
    rss_mb: float | None
    gain: np.ndarray
    licence_ok: bool = True


def infeasible_reasons(
    o: Option, mean_cap: float = MEAN_CAP_S, p95_cap: float = P95_CAP_S, rss_cap: float = RSS_CAP_MB
) -> list[str]:
    why = []
    if o.mean_s is None or o.p95_s is None:
        why.append("seconds_not_measured")
    else:
        if o.mean_s > mean_cap:
            why.append("mean_seconds_over_R6")
        if o.p95_s > p95_cap:
            why.append("p95_seconds_over_R6")
    if o.rss_mb is None:
        why.append("memory_not_measured")
    elif o.rss_mb > rss_cap:
        why.append("memory_over_R7")
    if not o.licence_ok:
        why.append("licence_or_download")
    return why


def choose_reader(
    options: list[Option],
    weights: np.ndarray,
    groups: np.ndarray,
    mean_cap: float = MEAN_CAP_S,
    p95_cap: float = P95_CAP_S,
    rss_cap: float = RSS_CAP_MB,
    tie_seconds: float = TIE_SECONDS,
    resamples: int = 2000,
    seed: int = 1,
) -> dict:
    """The pre-registered rule. Feasible: mean and p95 seconds inside the time budget, memory inside the memory budget, licence in order. The cap of a cell
    is its largest feasible budget. Of the two cells with the larger gain at their caps, the first wins when the paired
    document-bootstrap interval of the difference excludes zero; otherwise the cheaper in mean seconds; and when the
    cost is also tied (within `tie_seconds`), Z, the maintainer's. When no cell is feasible the options are handed to the maintainer."""
    table = [
        {
            "cell": o.cell,
            "budget": o.budget,
            "gain": round(float(o.gain.sum() / weights.sum()), 4) if weights.sum() else None,
            "mean_s": o.mean_s,
            "p95_s": o.p95_s,
            "rss_mb": o.rss_mb,
            "infeasible": infeasible_reasons(o, mean_cap, p95_cap, rss_cap),
        }
        for o in options
    ]
    feasible = [o for o, row in zip(options, table, strict=True) if not row["infeasible"]]
    if not feasible:
        return {
            "decision": "options_for_maintainer",
            "reason": "no cell fits the time and memory budgets: the options with their measured cost go to the maintainer",
            "options": sorted(table, key=lambda r: -(r["gain"] or 0)),
        }
    caps: dict[str, Option] = {}
    for o in feasible:
        if o.cell not in caps or o.budget > caps[o.cell].budget:
            caps[o.cell] = o

    def total(o: Option) -> float:
        return float(o.gain.sum() / weights.sum()) if weights.sum() else 0.0

    ranked = sorted(caps.values(), key=lambda o: (-total(o), o.mean_s, o.cell))
    out: dict = {"options": table, "caps": {c: o.budget for c, o in caps.items()}}
    if len(ranked) == 1:
        first = ranked[0]
        return {
            **out,
            "decision": "choice",
            "cell": first.cell,
            "budget": first.budget,
            "reason": "the only feasible cell",
        }
    first, second = ranked[:2]
    difference = metrics.paired_difference(
        groups, gain_statistic(weights, first.gain), gain_statistic(weights, second.gain), resamples, seed
    )
    out["difference_first_minus_second"] = {
        "first": first.cell,
        "second": second.cell,
        **{k: round(v, 4) if isinstance(v, float) else v for k, v in difference.items()},
    }
    if difference["excludes_zero"]:
        return {
            **out,
            "decision": "choice",
            "cell": first.cell,
            "budget": first.budget,
            "reason": "the larger gain at the cap; the interval of the difference excludes zero",
        }
    if abs(first.mean_s - second.mean_s) > tie_seconds:
        cheaper = first if first.mean_s < second.mean_s else second
        return {
            **out,
            "decision": "choice",
            "cell": cheaper.cell,
            "budget": cheaper.budget,
            "reason": "gains not told apart (the interval includes zero); the cheaper in seconds a page",
        }
    pick = next((o for o in (first, second) if o.cell == "Z"), min((first, second), key=lambda o: o.p95_s))
    return {
        **out,
        "decision": "choice",
        "cell": pick.cell,
        "budget": pick.budget,
        "reason": "gains and cost tied: Z, the maintainer's"
        if pick.cell == "Z"
        else "gains and cost tied, Z not among them: the lower p95",
    }


def knee(curve: list[tuple[float, float]]) -> int:
    """The index of the knee of a curve of (seconds, gain) points ordered by budget: the point farthest from the chord
    between the first and the last; the last when there are fewer than three."""
    if len(curve) < 3:
        return len(curve) - 1
    x = np.array([c[0] for c in curve], dtype=np.float64)
    y = np.array([c[1] for c in curve], dtype=np.float64)
    x = (x - x[0]) / ((x[-1] - x[0]) or 1.0)
    y = (y - y[0]) / ((y[-1] - y[0]) or 1.0)
    return int(np.argmax(np.abs(y - x)))


def gate_p(recall: float | None) -> str:
    """Gate P on the written-form recall of the plain wrong words with the second readings."""
    if recall is None:
        return "no_data"
    if recall >= GATE_PASS:
        return "pass"
    return "stop" if recall < GATE_STOP else "between_80_and_85"


# ---------------------------------------------------------------------------------------------------------------------
# The pilot
# ---------------------------------------------------------------------------------------------------------------------


def pilot_paths(manifest: Path, data: Path) -> list[Path]:
    entries = json.loads(manifest.read_text(encoding="utf-8"))["pilot"]
    return [data / "pages" / e["set"] / e["doc_id"] / f"{e['page']}.json" for e in entries]


def load_variants(
    paths: list[Path], cells
) -> tuple[list[pairs.Page], dict[str, list[pairs.Page]], Counter, str | None]:
    """The pilot pages with the first reading alone (`base`) and, per cell, with its readings; the pages a cell has not
    read are counted, and a cell that does not have all of them is skipped (every variant is over the same pages). The
    fourth value is the engine key of the first page (it says how many units of each kind the engine stores)."""
    from bench.run_pass2 import load_cell
    from extract.engine.cache import reading_from_dict

    missing: Counter = Counter()
    base: list[pairs.Page] = []
    by_cell: dict[str, list[pairs.Page]] = {c: [] for c in cells}
    records = []
    for path in paths:
        record = json.loads(path.read_text(encoding="utf-8"))
        first = reading_from_dict(record["ocr"]["reading"]) if record.get("ocr") else None
        if first is None or first.status != "ocr" or not first.words:
            missing["no_first_reading"] += 1
            continue
        records.append((path, record, first))
    for path, record, first in records:
        pages = {}
        for cell in cells:
            second = load_cell(path, cell)
            if second is None:
                missing[f"no_pass2_{cell}"] += 1
            else:
                pages[cell] = pairs.page_from_reading(record, second)
        base.append(pairs.page_from_reading(record, first))
        for cell, page in pages.items():
            by_cell[cell].append(page)
    return base, by_cell, missing, (records[0][1]["ocr"].get("engine") if records else None)


def engine_unit_limit(engine: str | None, unit: str) -> int | None:
    """How many units of a kind (`line`, `zone`) the engine stores a page, from the key it read the page with; None when
    the key does not carry its configuration."""
    from bench.run_pass2 import study_of
    from extract.engine import reread

    if not engine:
        return None
    try:
        config = study_of(engine)
    except (StopIteration, ValueError):
        return None
    if unit == "line":
        return min(config.study_units, reread.MAX_REREAD_LINES)
    return min(getattr(config, "study_zones", config.study_units), reread.MAX_ZONES)


def readers_of(pages: list[pairs.Page]) -> list[str]:
    return sorted({r.source for p in pages for w in p.words for r in (w.readings or ())})


def load_arms(*sources: Path) -> dict:
    """The cells of the cost files, a later file replacing the fields it has of a cell and keeping the others of the
    earlier one (and the budgets of a cell merged by budget); a missing file is skipped."""
    arms: dict = {}
    for source in sources:
        if not source.is_file():
            continue
        for cell, entry in json.loads(source.read_text(encoding="utf-8")).get("cells", {}).items():
            old = arms.get(cell)
            if not (isinstance(old, dict) and isinstance(entry, dict)):
                arms[cell] = entry
                continue
            arms[cell] = {
                **old,
                **entry,
                **({"budgets": {**old.get("budgets", {}), **entry["budgets"]}} if "budgets" in entry else {}),
            }
    return arms


def gate_for(result: dict, key: str = "choice") -> dict:
    """Gate P for the option the rule chose (or, when none fits and the options go to the maintainer, the one with the largest
    gain): the written-form recall of the plain wrong words with the second readings, its interval over documents and
    how many words it rests on, the verdict and what to do when it is between the two bars."""
    choice = result[key]
    if choice["decision"] == "choice":
        cell, budget, which = choice["cell"], choice["budget"], "chosen"
    else:
        best = max(choice["options"], key=lambda o: o["gain"] or 0)
        cell, budget, which = best["cell"], best["budget"], "best_gain_not_chosen"
    plain = result["cells"][cell]["budgets"][str(budget)]["plain_wrong_pairs"]
    verdict = gate_p(plain["with_readings"])
    action = {"between_80_and_85": BETWEEN_ACTION, "stop": "stop the build and tell the maintainer with the table"}.get(
        verdict
    )
    return {
        "option": which,
        "cell": cell,
        "budget": budget,
        "n_plain_wrong_words": plain["n"],
        "recall_with_readings": plain["with_readings"],
        "recall_ci95": plain["with_readings_ci95"],
        "base": plain["base"],
        "decision": verdict,
        "action": action,
    }


CHEAP_READERS = frozenset({"latin", "bin", "clahe", "wide"})
SUBSETS = {
    "latin": frozenset({"latin"}),
    "latin+wide": frozenset({"latin", "wide"}),
    "latin+wide+clahe": frozenset({"latin", "wide", "clahe"}),
    "all_cheap": CHEAP_READERS,
    "deep": frozenset({"deep"}),
    "latin+wide+deep": frozenset({"latin", "wide", "deep"}),
    "all_cheap+deep": CHEAP_READERS | {"deep"},
}
ABLATION_BUDGETS = {"line": (4, 8, 16), "zone": (10, 20, 40)}
COST_CELLS = {"line": {"cheap": "L", "deep": "LD"}, "zone": {"cheap": "ZC", "deep": "Z"}}


def merged_readings(cheap: list[pairs.Page], deep: list[pairs.Page]) -> list[pairs.Page]:
    """The pages of the cheap and the deep cell of one unit kind as one: every word with the readings of both (the
    units and their ranks are the same, they depend on the first reading alone)."""
    out = []
    for first, second in zip(cheap, deep, strict=True):
        words = tuple(
            dataclasses.replace(a, readings=(*(a.readings or ()), *(b.readings or ())) or None)
            for a, b in zip(first.words, second.words, strict=True)
        )
        out.append(dataclasses.replace(first, words=words, _context=None))
    return out


def estimated_cost(arms: dict, unit: str, subset: frozenset, budget: int, largest: int) -> dict | None:
    """Seconds of a set of readers at a budget, estimated from the measured cost of the cell that runs them: the cell's
    cost at that budget times the share the subset's readers have of the cell's reader seconds (reader-speed.json,
    `readers.<name>.mean`). Cheap readers come from the cheap cell and `deep` from the deep one; the estimate is the
    sum, so it is an estimate and says so."""
    total = {"mean_s": 0.0, "p95_s": 0.0}
    for kind, readers in (("cheap", subset & CHEAP_READERS), ("deep", subset & {"deep"})):
        if not readers:
            continue
        cell = COST_CELLS[unit][kind]
        cost = cost_of(arms, cell, budget, largest)
        per_reader = arms.get("cells", arms).get(cell, {}).get("readers", {}) if isinstance(arms, dict) else {}
        weights = {r: v["mean"] for r, v in per_reader.items() if isinstance(v, dict) and "mean" in v}
        if cost is None or not weights or sum(weights.values()) <= 0:
            return None
        share_of_cell = sum(weights.get(r, 0.0) for r in readers) / sum(weights.values())
        total["mean_s"] += cost["mean_s"] * share_of_cell
        total["p95_s"] += cost["p95_s"] * share_of_cell
    return {**total, "estimate": True}


def fits_r6(seconds: dict | None) -> bool | None:
    """Whether a cost is inside the time budget (mean and p95 seconds a page); None when there is no cost."""
    if seconds is None:
        return None
    return bool(seconds["mean_s"] <= MEAN_CAP_S and seconds["p95_s"] <= P95_CAP_S)


def measured_cost(per_reader: dict, arms: dict, unit: str, subset: frozenset, budget: int) -> dict | None:
    """Seconds of a set of readers at a budget from what the speed measurement gives for each reader alone (`per_reader.<name>.
    page_seconds.<unit>`: the seconds of all the units a page stores, mean and p95 over the pilot pages): their sum
    scaled by the share of those units that a page has at the budget (`units_per_page_mean` of the cheap cell of the
    speed file at the budget over the same at its largest). A sum of p95s is an upper bound of the p95."""
    cell = COST_CELLS[unit]["cheap"]
    units = {
        int(b): v["units_per_page_mean"]
        for b, v in arms.get("cells", arms).get(cell, {}).get("budgets", {}).items()
        if isinstance(v, dict) and "units_per_page_mean" in v
    }
    if budget not in units or not units or not subset <= set(per_reader):
        return None
    ratio = units[budget] / units[max(units)]
    pages = [per_reader[r].get("page_seconds", {}).get(unit) for r in subset]
    if any(not isinstance(x, dict) or "mean" not in x or "p95" not in x for x in pages):
        return None
    return {
        "mean_s": ratio * sum(x["mean"] for x in pages),
        "p95_s": ratio * sum(x["p95"] for x in pages),
        "estimate": True,
        "source": "per_reader",
        "note": "approximation: sum of the readers alone (an upper bound of a batch) and the p95 scaled by the units of the budget; the batched measure of the chosen set decides R6",
    }


def readers_ablation(
    base_reach: np.ndarray,
    arrays: dict,
    by_cell: dict,
    arms: dict,
    stored: dict,
    resamples: int,
    per_reader: dict | None = None,
) -> dict:
    """What each set of readers adds, per unit kind and budget: the pool gain with its interval, the harm and net100, and
    the estimated seconds against the time budget. Needs the cheap and the deep cell of the unit read by pass 2."""
    out: dict = {}
    for unit, names in COST_CELLS.items():
        cheap, deep = by_cell.get(names["cheap"]), by_cell.get(names["deep"])
        if not cheap or not deep or len(cheap) != len(deep):
            continue
        limit = stored.get(unit)
        budgets = [b for b in ABLATION_BUDGETS[unit] if limit is None or b <= limit]
        largest = max(budgets)
        pages = merged_readings(cheap, deep)
        table: dict = {}
        for name, subset in SUBSETS.items():
            table[name] = {}
            for budget in budgets:
                variant = [with_readings(p, budget, subset) for p in pages]
                report = variant_report(base_reach, reach_vector(variant), harm_counts(variant), arrays, resamples)
                seconds = (per_reader and measured_cost(per_reader, arms, unit, subset, budget)) or estimated_cost(
                    arms, unit, subset, budget, largest
                )
                fits = fits_r6(seconds)
                table[name][str(budget)] = {
                    "gain": report["gain"],
                    "gain_ci95": report["gain_ci95"],
                    "plain_recall": report["plain_wrong_pairs"]["with_readings"],
                    "harm_rate": report["harm"]["harm_rate"],
                    "net100": report["harm"]["net100"],
                    "estimated_seconds": seconds,
                    "fits_R6_estimated": fits,
                }
        out[unit] = table
    return out


def per_reader_of(path: Path) -> dict | None:
    """The seconds of every reader alone from the pilot's cost file, or None when it has none."""
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8")).get("per_reader") or None


def run(args: argparse.Namespace) -> dict:
    started = time.time()
    paths = pilot_paths(args.manifest, args.data)
    cells = [c for c in args.cells if c in CELLS]
    base, by_cell, missing, engine = load_variants(paths, cells)
    # the time budget is the criterion on the speed set (reader-speed.json, the official cost); the pilot's own pages (reader-arms.json)
    # give a second cost beside it, and the rule is run under each
    sources = {"speed": load_arms(args.speed), "pilot": load_arms(args.arms)}
    bakeoff = json.loads(args.bakeoff.read_text(encoding="utf-8")) if args.bakeoff.is_file() else None
    result: dict = {"pages": len(base), "missing": dict(missing), "cells": {}}
    if not base:
        result["note"] = "no pilot page is read yet: run_corpus --second-reading and run_pass2 first"
        return result
    base_pages = [without_readings(p) for p in base]
    rows = wrong_rows(base_pages)
    arrays = rows.array()
    groups = np.unique(arrays["doc"], return_inverse=True)[1]
    base_reach = reach_vector(base_pages)
    result["wrong_components"] = len(rows.weight)
    result["wrong_weight"] = float(arrays["weight"].sum())
    result["plain_wrong_pairs"] = int(arrays["plain"].sum())
    options: dict[str, list[Option]] = {name: [] for name in sources}
    stored_by_unit: dict = {}
    for cell in cells:
        pages = by_cell[cell]
        if len(pages) != len(base):
            result["cells"][cell] = {"skipped": "not every pilot page has its pass 2", "pages": len(pages)}
            continue
        unit = CELLS[cell][0]
        budgets = tuple(args.line_budgets if unit == "line" else args.zone_budgets)
        ranks = [r.rank for p in pages for w in p.words for r in (w.readings or ()) if r.rank is not None]
        observed = 1 + max(ranks) if ranks else None
        # what the engine stores comes from its own configuration; the highest rank seen only stands in when the key of
        # the pages does not say (an older engine), and it is reported beside as a check
        stored = engine_unit_limit(engine, unit) or observed  # None: nothing to collapse
        stored_by_unit[unit] = stored
        largest = min(max(budgets), stored) if stored is not None else max(budgets)
        entry = {
            "unit": unit,
            "readers": readers_of(pages),
            "units_stored": stored,
            "units_observed": observed,
            "budgets": {},
            "collapsed": {},
        }
        licence = licence_in_order(bakeoff, cell)
        curve = []
        for budget in budgets:
            if stored is not None and budget > stored:  # the engine keeps no more units than `stored` a page
                entry["collapsed"][str(budget)] = min(budget, stored)
                if min(budget, stored) in {int(b) for b in entry["budgets"]}:
                    continue
                budget = min(budget, stored)
            variant = [with_readings(p, budget, None) for p in pages]
            reach = reach_vector(variant)
            report = variant_report(base_reach, reach, harm_counts(variant), arrays, args.resamples)
            gain_vector = (reach & ~base_reach) * arrays["weight"]
            for name, arms in sources.items():
                cost = cost_of(arms, cell, budget, largest)
                report[f"cost_{name}"] = cost
                mean_s, p95_s, rss_mb = (cost["mean_s"], cost["p95_s"], cost["rss_mb"]) if cost else (None, None, None)
                options[name].append(Option(cell, budget, mean_s, p95_s, rss_mb, gain_vector, licence))
            report["gate_p_plain_written"] = gate_p(report["plain_wrong_pairs"]["with_readings"])
            entry["budgets"][str(budget)] = report
            official = report["cost_speed"] or report["cost_pilot"]
            if official:
                curve.append((official["mean_s"], report["gain"]))
        if len(curve) >= 3:
            k = knee(curve)
            budgets_seen = list(entry["budgets"])
            entry["knee"] = {"budget": int(budgets_seen[k]), "gain_per_second": share(curve[k][1], curve[k][0])}
        by_reader = {}
        for reader in entry["readers"]:
            sub = [with_readings(p, largest, frozenset({reader})) for p in pages]
            by_reader[reader] = variant_report(base_reach, reach_vector(sub), harm_counts(sub), arrays, args.resamples)
        entry["by_reader_at_the_cap"] = {
            r: {"gain": v["gain"], "gain_ci95": v["gain_ci95"], "harm": v["harm"]} for r, v in by_reader.items()
        }
        result["cells"][cell] = entry
    for name, key, gate_key in (("speed", "choice", "gate_p"), ("pilot", "choice_pilot_costs", "gate_p_pilot_costs")):
        if options[name]:
            result[key] = choose_reader(options[name], arrays["weight"], groups, resamples=args.resamples)
            result[gate_key] = gate_for(result, key)
    if "gate_p" in result:
        gate = result["gate_p"]
        chosen = [with_readings(p, gate["budget"], None) for p in by_cell[gate["cell"]]]
        gate["unreached_plain_wrong_words_by_cause"] = unreached_causes(chosen, reach_vector(chosen), rows)
        gate["base_unreached_by_cause"] = unreached_causes(base_pages, base_reach, rows)
    result["readers_ablation"] = readers_ablation(
        base_reach, arrays, by_cell, sources["speed"], stored_by_unit, args.resamples, per_reader_of(args.arms)
    )
    result["seconds"] = round(time.time() - started, 1)
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    data = Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data"))
    parser.add_argument("--data", type=Path, default=data)
    parser.add_argument("--manifest", type=Path, default=Path("bench/corpus/split-v5.json"))
    parser.add_argument("--cells", nargs="+", default=list(CELLS))
    parser.add_argument("--line-budgets", nargs="+", type=int, default=list(BUDGETS["line"]))
    parser.add_argument("--zone-budgets", nargs="+", type=int, default=list(BUDGETS["zone"]))
    parser.add_argument("--arms", type=Path, default=Path("bench/results/reader-arms.json"))
    parser.add_argument("--speed", type=Path, default=Path("bench/results/reader-speed.json"))
    parser.add_argument("--bakeoff", type=Path, default=Path("bench/results/reader-bakeoff.json"))
    parser.add_argument("--resamples", type=int, default=1000)
    parser.add_argument("--out", type=Path, default=Path("bench/results/pilot-cells.json"))
    args = parser.parse_args(argv)
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(
        json.dumps(
            {k: result[k] for k in ("pages", "missing", "choice", "choice_pilot_costs") if k in result}, indent=1
        )[:2500]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
