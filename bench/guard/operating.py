"""From raw chooser scores to operating points: calibration per kind, the best fix of each site, and the thresholds.

Everything here receives out-of-fold predictions and nothing else. A site is processed when the detector flags it;
its best non-KEEP candidate `c*` is the one with the highest calibrated `p_fix`. `safe` applies `c*` when `p_fix`
reaches `t_safe`; `more` applies it when `p_fix - p_keep` reaches `m`. A fix is *fixed* when `c*` is the truth,
*broken* when the token was right and *changed, still wrong* when the token was wrong and `c*` is not the truth.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from rapidfuzz.distance import Levenshtein

from bench.guard import calibrate, metrics
from bench.guard.pairs import Table
from extract.guard import select as selection
from extract.guard.features import INDEX, KINDS, OPS

MIN_GROUP = 500
MIN_POSITIVE = 100
# the floor of the sweep stays 99.5 % (the 99.0 % that was tried earlier was undone); T5 asks `T5_FLOOR`
SAFE_FLOOR = 0.995
T5_FLOOR = 0.99
PROMISES = ("exact", "no_worse")  # what counts as a hit at `safe`: the word ends exact, or never ends worse
MORE_BROKEN_SHARE = 0.05
MIN_APPLIED_TEXT = 300
MIN_APPLIED_STRUCTURE = 100
MIN_APPLIED = {
    "text": MIN_APPLIED_TEXT,
    "join": MIN_APPLIED_STRUCTURE,
    "segment": MIN_APPLIED_STRUCTURE,
    "drop": MIN_APPLIED_STRUCTURE,
}
FIXED, BROKEN, STILL_WRONG = 1, 2, 3


def calibrate_rows(
    table: Table, raw: np.ndarray, cross_fit: bool = False, min_group: int = MIN_GROUP, min_positive: int = MIN_POSITIVE
) -> np.ndarray:
    """Calibrated probability of every row from its raw score: one isotonic table per edit kind that has at least
    `min_group` rows and `min_positive` true candidates, the pooled table for the others. With `cross_fit` the documents are split in two halves and
    each half is calibrated by the tables fitted on the other, so no row helps fit its own probability."""
    kind = table.x[:, INDEX["kind"]].astype(int)
    docs = table.doc[table.site]
    if not cross_fit:
        return _apply(raw, table.y, kind, raw, kind, min_group, min_positive)
    out = np.empty(len(raw), dtype=np.float64)
    names = np.unique(docs)
    first = np.isin(docs, names[::2])
    for fitted, held_out in ((first, ~first), (~first, first)):
        out[held_out] = _apply(
            raw[fitted], table.y[fitted], kind[fitted], raw[held_out], kind[held_out], min_group, min_positive
        )
    return out


def _apply(fit_raw, fit_y, fit_kind, raw, kind, min_group, min_positive) -> np.ndarray:
    pooled = calibrate.fit_isotonic(fit_raw, fit_y, min_words=1)
    out = calibrate.apply_table(pooled, raw)
    for code in np.unique(fit_kind):
        members = fit_kind == code
        if members.sum() >= min_group and fit_y[members].sum() >= min_positive:
            table = calibrate.fit_isotonic(fit_raw[members], fit_y[members], min_words=1)
            target = kind == code
            out[target] = calibrate.apply_table(table, raw[target])
    return out


@dataclass
class Choice:
    """The best non-KEEP candidate of every site and what applying it would do."""

    index: np.ndarray
    p_fix: np.ndarray
    p_keep: np.ndarray
    outcome: np.ndarray
    safe_p_fix: np.ndarray | None = (
        None  # `p_fix` as `safe` sees it: -inf where `safe` leaves the site alone (proper names)
    )

    @property
    def safe_score(self) -> np.ndarray:
        return self.p_fix if self.safe_p_fix is None else self.safe_p_fix

    @property
    def margin(self) -> np.ndarray:
        return self.p_fix - self.p_keep


def choose(table: Table, probability: np.ndarray) -> Choice:
    """The candidate with the highest `p_fix` of each site that has one besides the token as read. A site with only
    the token has `p_fix` of -inf and is never applied."""
    index = np.zeros(table.sites, dtype=np.int64)
    p_fix = np.full(table.sites, -np.inf)
    p_keep = np.zeros(table.sites)
    outcome = np.zeros(table.sites, dtype=np.int8)
    for s in range(table.sites):
        block = table.block(s)
        p = probability[block]
        p_keep[s] = p[0]
        if len(p) > 1:
            best = 1 + int(np.argmax(p[1:]))
            index[s], p_fix[s] = best, p[best]
            labels = table.y[block]
            outcome[s] = FIXED if labels[best] else BROKEN if table.token_right[s] else STILL_WRONG
    return Choice(index, p_fix, p_keep, outcome)


def safe_threshold(p_fix: np.ndarray, outcome: np.ndarray, floor: float, min_applied: int) -> float | None:
    """The lowest threshold whose precision is at least `floor` with at least `min_applied` applied, found scanning
    from the highest down: the cut is just above the first threshold, past the minimum count, at which the
    precision falls under the floor, so a lucky dip lower down is never picked. None when there is none."""
    return _scan(p_fix, outcome, min_applied, lambda fixed, broken, wrong, n: fixed >= floor * n)


def more_margin(
    margin: np.ndarray, outcome: np.ndarray, cap: float = MORE_BROKEN_SHARE, min_applied: int = 1
) -> float | None:
    """The smallest margin at which the broken fixes stay within `cap` of the fixed ones and the net balance is
    positive, scanning from the highest margin down and cutting before the first failure."""
    return _scan(
        margin, outcome, min_applied, lambda fixed, broken, wrong, n: (broken <= cap * fixed) & (fixed > broken)
    )


def _scan(score: np.ndarray, outcome: np.ndarray, min_applied: int, holds) -> float | None:
    usable = np.isfinite(score)
    if (
        not usable.any()
    ):  # a group with no site that safe may apply (every candidate dropped, every name protected): off
        return None
    score, outcome = score[usable], outcome[usable]
    order = np.argsort(-score, kind="stable")
    score, outcome = score[order], outcome[order]
    ends = np.nonzero(np.append(score[1:] != score[:-1], True))[0]
    fixed = np.cumsum(outcome == FIXED)[ends]
    broken = np.cumsum(outcome == BROKEN)[ends]
    wrong = np.cumsum(outcome == STILL_WRONG)[ends]
    count = ends + 1
    failing = np.nonzero((count >= min_applied) & ~holds(fixed, broken, wrong, count))[0]
    last = int(failing[0]) - 1 if len(failing) else len(ends) - 1
    if last < 0 or count[last] < min_applied:
        return None
    return float(score[ends[last]])


def applied(choice: Choice, rule: str, threshold: float | None) -> np.ndarray:
    """Which sites apply their best candidate: `safe` on `p_fix`, `more` on the margin."""
    if threshold is None:
        return np.zeros(len(choice.index), dtype=bool)
    score = choice.safe_score if rule == "safe" else choice.margin
    return np.isfinite(score if rule == "safe" else choice.p_fix) & (score >= threshold)


def counts(choice: Choice, mask: np.ndarray) -> dict:
    taken = choice.outcome[mask]
    fixed, broken, wrong = (int((taken == code).sum()) for code in (FIXED, BROKEN, STILL_WRONG))
    return {
        "applied": int(mask.sum()),
        "fixed": fixed,
        "broken": broken,
        "changed_still_wrong": wrong,
        "precision": fixed / mask.sum() if mask.sum() else None,
        "net": fixed - broken,
        # what the 73 % hides (B7): the damage per 100 applied, and the share that leaves another wrong word
        "broken_per_100_applied": 100 * broken / mask.sum() if mask.sum() else None,
        "still_wrong_share": wrong / mask.sum() if mask.sum() else None,
    }


def with_intervals(choice: Choice, mask: np.ndarray, docs: np.ndarray, resamples: int = 2000, seed: int = 1) -> dict:
    """The counts of an operating point with 95 % cluster-bootstrap intervals over documents."""
    out = counts(choice, mask)

    def fixed(rows):
        return float((choice.outcome[rows][mask[rows]] == FIXED).sum())

    def net(rows):
        taken = choice.outcome[rows][mask[rows]]
        return float((taken == FIXED).sum() - (taken == BROKEN).sum())

    def precision(rows):
        taken = choice.outcome[rows][mask[rows]]
        return float((taken == FIXED).mean()) if len(taken) else float("nan")

    out["intervals"] = metrics.cluster_bootstrap(
        docs, {"fixed": fixed, "net": net, "precision": precision}, resamples, seed
    )
    return out


def kind_name(code: int) -> str:
    return KINDS[int(code)]


# ---------------------------------------------------------------------------------------------------------------------
# What an edit does to the word, not only whether it ends exact: the promises "ends right" and "never worse"
# ---------------------------------------------------------------------------------------------------------------------


@dataclass
class Distances:
    """Per site: the edit distance of the token as read to the truth, of the best candidate to the truth, and the length of
    the truth (NaN for a site with no candidate)."""

    before: np.ndarray
    after: np.ndarray
    reference: np.ndarray


def edit_distances(table: Table, choice: Choice) -> Distances:
    n = len(choice.index)
    before, after, reference = (np.full(n, np.nan) for _ in range(3))
    for s in range(n):
        if not np.isfinite(choice.p_fix[s]):
            continue
        truth = str(table.truth[s])
        before[s] = Levenshtein.distance(str(table.token[s]), truth)
        after[s] = Levenshtein.distance(table.found[s][int(choice.index[s])].text, truth)
        reference[s] = len(truth)
    return Distances(before, after, reference)


def quality(
    table: Table, choice: Choice, mask: np.ndarray, resamples: int = 300, distances: Distances | None = None
) -> dict:
    """What the applied edits (`mask`) do, with 95 % intervals over documents. Signed precision is `fixed / applied` (the
    promise "it ends right", B7); `precision_no_worse` adds the edits that leave the word wrong but closer to the truth
    ("it never makes a word worse"); the damage is the broken words and the edits that end farther from the truth
    (`damage_strict`), and with the ones that end as far (`damage_with_equal`). The changes are per 100 applied: the edit
    distance to the truth (negative is better), the word errors (`after - before`), and the character error rate over the
    characters of the truth of the applied sites."""
    d = distances or edit_distances(table, choice)
    outcome = choice.outcome
    still_wrong = outcome == STILL_WRONG
    classes = {
        "fixed": outcome == FIXED,
        "broken": outcome == BROKEN,
        "improved": still_wrong & (d.after < d.before),
        "equal": still_wrong & (d.after == d.before),
        "worse": still_wrong & (d.after > d.before),
    }
    change = np.where(mask, np.nan_to_num(d.after - d.before), 0.0)
    word_before = (~table.token_right).astype(float)
    word_after = (outcome != FIXED).astype(float)
    reference = np.nan_to_num(d.reference)

    def share(*names):
        flags = np.zeros(len(mask), dtype=bool)
        for name in names:
            flags |= classes[name]

        def stat(rows):
            applied = mask[rows].sum()
            return float((flags[rows] & mask[rows]).sum() / applied) if applied else float("nan")

        return stat

    def per_100(values):
        def stat(rows):
            applied = mask[rows].sum()
            return float(100 * values[rows].sum() / applied) if applied else float("nan")

        return stat

    def error_rate(rows):
        total = reference[rows][mask[rows]].sum()
        return float(change[rows].sum() / total) if total else float("nan")

    statistics = {
        "precision_signed": share("fixed"),
        "precision_no_worse": share("fixed", "improved"),
        "damage_strict": share("broken", "worse"),
        "damage_with_equal": share("broken", "worse", "equal"),
        "improved_share": share("improved"),
        "equal_share": share("equal"),
        "worse_share": share("worse"),
        "broken_share": share("broken"),
        "edit_distance_change_per_100": per_100(change),
        "word_errors_change_per_100": per_100(np.where(mask, word_after - word_before, 0.0)),
        "char_error_rate_change": error_rate,
    }
    out = {"applied": int(mask.sum()), **{k: int((v & mask).sum()) for k, v in classes.items()}}
    out["intervals"] = metrics.cluster_bootstrap(table.doc, statistics, resamples, 1)
    return out


def protect_proper(table: Table, choice: Choice) -> Choice:
    """The choice as `safe` sees it: the sites whose token is a proper name that no word list knows (any word of the
    span, for a join or a segmentation) have `-inf` for `safe`, so that they are neither swept for its threshold nor applied
    by it. `more` keeps the real probabilities."""
    from extract.guard import proper

    protected = np.array([proper.protected_in_safe(str(t).split()) for t in table.token], dtype=bool)
    shown = np.where(protected, -np.inf, choice.p_fix)
    return Choice(choice.index, choice.p_fix, choice.p_keep, choice.outcome, shown)


def safe_outcome(
    table: Table, choice: Choice, promise: str = "exact", distances: Distances | None = None
) -> np.ndarray:
    """The outcomes that `safe` is swept on: with `exact` those of `choose` (a hit is a word that ends exact); with `no_worse`
    an edit that leaves the word wrong but closer to the truth is a hit too (the promise "never makes a word worse").
    The choice of the promise is the maintainer's, and only moves the threshold: both precisions are always reported."""
    if promise not in PROMISES:
        raise ValueError(f"promise must be one of {PROMISES}, not {promise!r}")
    outcome = choice.outcome.copy()
    if promise == "no_worse":
        d = distances or edit_distances(table, choice)
        outcome[(outcome == STILL_WRONG) & (d.after < d.before)] = FIXED
    return outcome


def top_masks(choice: Choice, tops: Sequence[int]) -> dict[int, np.ndarray]:
    """The `n` best sites by `p_fix` (those that have a candidate), for each `n` that the sites can fill."""
    order = np.argsort(-choice.p_fix, kind="stable")
    order = order[np.isfinite(choice.p_fix[order])]
    out = {}
    for n in tops:
        if n <= len(order):
            mask = np.zeros(len(choice.p_fix), dtype=bool)
            mask[order[:n]] = True
            out[n] = mask
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Per op (text, join, segment, drop): calibration by (op, kind), thresholds, overlaps
# ---------------------------------------------------------------------------------------------------------------------


def site_ops(table: Table) -> np.ndarray:
    """The op of every site: the family of its candidates (`text` for the sites of a word)."""
    family = table.family if table.family is not None else np.array(["text"] * table.sites, dtype=object)
    return np.asarray(family, dtype=object)


def calibrate_rows_by_op(
    table: Table, raw: np.ndarray, cross_fit: bool = False, min_group: int = MIN_GROUP, min_positive: int = MIN_POSITIVE
) -> np.ndarray:
    """Calibrated probability of every row: one isotonic table per (op, kind) group with at least `min_group` rows and
    `min_positive` true candidates, else the pooled table of its op (when the op has `min_positive` true candidates),
    else the table of all rows. With `cross_fit` the documents are split in two halves and each half is calibrated by
    the tables fitted on the other."""
    kind = table.x[:, INDEX["kind"]].astype(int)
    op = site_ops(table)[table.site]
    docs = table.doc[table.site]
    if not cross_fit:
        return _apply_by_op(raw, table.y, op, kind, raw, op, kind, min_group, min_positive)
    out = np.empty(len(raw), dtype=np.float64)
    first = np.isin(docs, np.unique(docs)[::2])
    for fitted, held_out in ((first, ~first), (~first, first)):
        out[held_out] = _apply_by_op(
            raw[fitted],
            table.y[fitted],
            op[fitted],
            kind[fitted],
            raw[held_out],
            op[held_out],
            kind[held_out],
            min_group,
            min_positive,
        )
    return out


def _apply_by_op(fit_raw, fit_y, fit_op, fit_kind, raw, op, kind, min_group, min_positive) -> np.ndarray:
    pooled = calibrate.fit_isotonic(fit_raw, fit_y, min_words=1)
    out = calibrate.apply_table(pooled, raw)
    for name in np.unique(fit_op):
        members = fit_op == name
        if fit_y[members].sum() < min_positive:
            continue
        op_table = calibrate.fit_isotonic(fit_raw[members], fit_y[members], min_words=1)
        target = op == name
        out[target] = calibrate.apply_table(op_table, raw[target])
        for code in np.unique(fit_kind[members]):
            group = members & (fit_kind == code)
            if group.sum() >= min_group and fit_y[group].sum() >= min_positive:
                table = calibrate.fit_isotonic(fit_raw[group], fit_y[group], min_words=1)
                chosen = target & (kind == code)
                out[chosen] = calibrate.apply_table(table, raw[chosen])
    return out


def site_kinds(table: Table, choice: Choice) -> np.ndarray:
    """The edit kind of the best candidate of every site (`accent`, `letters`, `structure`...; `keep` for a site that has
    none): the group the calibration of its row belongs to."""
    rows = table.starts + choice.index
    codes = table.x[rows, INDEX["kind"]].astype(int)
    return np.array([KINDS[c] for c in codes], dtype=object)


def eligible_groups(table: Table, min_group: int = MIN_GROUP, min_positive: int = MIN_POSITIVE) -> set[str]:
    """The `op|kind` groups that have a calibration table of their own (`calibration_tables`): at least `min_group` rows
    and `min_positive` true candidates, in an op that has `min_positive` itself. Only these can have a threshold of
    their own."""
    kind = table.x[:, INDEX["kind"]].astype(int)
    op = site_ops(table)[table.site]
    found = set()
    for name in OPS:
        members = op == name
        if not members.any() or table.y[members].sum() < min_positive:
            continue
        for code in np.unique(kind[members]):
            group = members & (kind == code)
            if group.sum() >= min_group and table.y[group].sum() >= min_positive:
                found.add(f"{name}|{KINDS[int(code)]}")
    return found


def thresholds_by_op(
    choice: Choice,
    ops: np.ndarray,
    floor: float = SAFE_FLOOR,
    minima: dict | None = None,
    cap: float = MORE_BROKEN_SHARE,
    kinds: np.ndarray | None = None,
    eligible: set | None = None,
    safe_outcome: np.ndarray | None = None,
) -> dict:
    """`t_safe` and `m` of every op that has sites (None where the op cannot meet its condition: off in that mode).
    The per-group door: for an op whose `t_safe` is None, with `kinds` and the `eligible` groups, the same scan runs on
    each `op|kind` group that has a calibration table of its own, with the op's minimum count, and writes
    `{"t_safe": t or None, "m": None}` under the key `op|kind` (None: that group is off). Chosen on these rows only."""
    minima = minima or MIN_APPLIED
    hits = choice.outcome if safe_outcome is None else safe_outcome  # what counts as a hit at `safe` (the promise)
    out = {}
    for name in OPS:
        members = ops == name
        if not members.any():
            continue
        out[name] = {
            "t_safe": safe_threshold(choice.safe_score[members], hits[members], floor, minima[name]),
            "m": more_margin(choice.margin[members], choice.outcome[members], cap),
        }
    if kinds is not None and eligible:
        for name in [n for n in out if out[n]["t_safe"] is None]:
            for kind in sorted(set(kinds[ops == name].tolist())):
                key = f"{name}|{kind}"
                if key not in eligible:
                    continue
                members = (ops == name) & (kinds == kind)
                out[key] = {
                    "t_safe": safe_threshold(choice.safe_score[members], hits[members], floor, minima[name]),
                    "m": None,
                }
    return out


def applied_by_op(
    choice: Choice, ops: np.ndarray, rule: str, thresholds: dict, kinds: np.ndarray | None = None
) -> np.ndarray:
    """Which sites apply their best candidate, each op against its own threshold (an op that is off applies nothing). The
    `op|kind` entries override their op in `safe` and only there; they need `kinds`."""
    mask = np.zeros(len(choice.index), dtype=bool)
    key = "t_safe" if rule == "safe" else "m"
    for name, values in thresholds.items():
        if "|" in name:
            if rule != "safe" or kinds is None:
                continue
            op, kind = name.split("|")
            members = (ops == op) & (kinds == kind)
        else:
            members = ops == name
        mask[members] = applied(_members(choice, members), rule, values[key])
    return mask


def without_overlaps(table: Table, choice: Choice, mask: np.ndarray) -> np.ndarray:
    """`mask` after the selection of a page (`select.select`): of the applied sites that share a word, the one with the
    larger gain `p_fix - p_keep` stays and the others are suggestions only."""
    keep = mask.copy()
    ops = site_ops(table)
    by_page: dict[int, list[int]] = {}
    for s in np.nonzero(mask)[0]:
        by_page.setdefault(int(table.page[s]), []).append(int(s))
    for sites in by_page.values():
        proposals = {
            selection.Proposal(tuple(table.span[s]), str(ops[s]), float(choice.margin[s]), str(s)): s for s in sites
        }
        taken, _ = selection.select(proposals)
        keep[sites] = False
        for proposal in taken:
            keep[proposals[proposal]] = True
    return keep


def points_by_op(
    table: Table,
    raw: np.ndarray,
    cross_fit: bool = True,
    thresholds: dict | None = None,
    floor: float = SAFE_FLOOR,
    minima: dict | None = None,
    min_group: int = MIN_GROUP,
    min_positive: int = MIN_POSITIVE,
) -> tuple:
    """What `report_by_op` measures before it counts: the choice, the op and kind of each site, the thresholds (chosen unless
    given) and the sites each rule applies (`safe`, `more`) once the overlaps are resolved. For the runs that need the
    sites themselves (the paired comparison of two runs by document)."""
    probability = calibrate_rows_by_op(table, raw, cross_fit, min_group, min_positive)
    choice = protect_proper(table, choose(table, probability))
    ops, kinds = site_ops(table), site_kinds(table, choice)
    if thresholds is None:
        thresholds = thresholds_by_op(
            choice, ops, floor, minima, kinds=kinds, eligible=eligible_groups(table, min_group, min_positive)
        )
    masks = {
        rule: without_overlaps(table, choice, applied_by_op(choice, ops, rule, thresholds, kinds))
        for rule in ("safe", "more")
    }
    return choice, ops, kinds, thresholds, masks


def report_by_op(
    table: Table,
    raw: np.ndarray,
    cross_fit: bool = True,
    resamples: int = 500,
    thresholds: dict | None = None,
    floor: float = SAFE_FLOOR,
    minima: dict | None = None,
    min_group: int = MIN_GROUP,
    min_positive: int = MIN_POSITIVE,
    with_quality: bool = False,
) -> dict:
    """The operating points of a pooled out-of-fold run, per op and in total: calibration by (op, kind), `t_safe[op]`
    at the precision floor with its minimum count (and `t_safe[op|kind]` for the ops that miss it), `m[op]` at
    5 % broken, overlaps resolved by the page selection, and the counts with their document-bootstrap intervals. With
    `thresholds` (the points that ship) nothing is chosen: their counts are measured on these probabilities,
    cross-fitted or in sample."""
    choice, ops, kinds, thresholds, masks = points_by_op(
        table, raw, cross_fit, thresholds, floor, minima, min_group, min_positive
    )
    out: dict = {"sites": int(table.sites), "cross_fit": cross_fit, "thresholds": thresholds}
    for rule in ("safe", "more"):
        mask = masks[rule]
        out[rule] = {"total": with_intervals(choice, mask, table.doc, resamples)}
        if with_quality:  # what the edits do to the words ; needs the candidates' texts and the truth
            out[rule]["quality"] = quality(table, choice, mask, resamples)
        for name in thresholds:
            if "|" in name:
                if rule == "safe":
                    op, kind = name.split("|")
                    out[rule][name] = counts(choice, mask & (ops == op) & (kinds == kind))
            else:
                out[rule][name] = counts(choice, mask & (ops == name))
    return out


# ---------------------------------------------------------------------------------------------------------------------
# Per language: the same scan and the same floors on the rows of one language only
# ---------------------------------------------------------------------------------------------------------------------


def site_languages(table: Table) -> np.ndarray:
    """The language of every site (the page's; `es` for a table that carries none)."""
    if table.language is None:
        return np.array(["es"] * table.sites, dtype=object)
    return np.asarray(table.language, dtype=object)


def _members(choice: Choice, members: np.ndarray) -> Choice:
    return Choice(
        choice.index[members],
        choice.p_fix[members],
        choice.p_keep[members],
        choice.outcome[members],
        None if choice.safe_p_fix is None else choice.safe_p_fix[members],
    )


def thresholds_by_language(
    choice: Choice,
    ops: np.ndarray,
    languages: np.ndarray,
    floor: float = SAFE_FLOOR,
    minima: dict | None = None,
    cap: float = MORE_BROKEN_SHARE,
    kinds: np.ndarray | None = None,
    eligible: set | None = None,
    safe_outcome: np.ndarray | None = None,
) -> dict:
    """`{language: {op: {t_safe, m}}}`: `thresholds_by_op` over the sites of each language alone, so that the thresholds
    of a language never depend on another language's rows. A (language, op) that cannot meet its condition is None
    there and nowhere else."""
    return {
        str(language): thresholds_by_op(
            _members(choice, languages == language),
            ops[languages == language],
            floor,
            minima,
            cap,
            None if kinds is None else kinds[languages == language],
            eligible,
            None if safe_outcome is None else safe_outcome[languages == language],
        )
        for language in sorted(set(languages.tolist()))
    }


def applied_by_language(
    choice: Choice,
    ops: np.ndarray,
    languages: np.ndarray,
    rule: str,
    thresholds: dict,
    kinds: np.ndarray | None = None,
) -> np.ndarray:
    """Which sites apply their best candidate, each against the thresholds of its own language and op (a language with no
    block applies nothing)."""
    mask = np.zeros(len(choice.index), dtype=bool)
    for language, block in thresholds.items():
        members = languages == language
        mask[members] = applied_by_op(
            _members(choice, members), ops[members], rule, block, None if kinds is None else kinds[members]
        )
    return mask


def points_by_language(
    table: Table,
    raw: np.ndarray,
    cross_fit: bool = True,
    thresholds: dict | None = None,
    floor: float = SAFE_FLOOR,
    minima: dict | None = None,
    min_group: int = MIN_GROUP,
    min_positive: int = MIN_POSITIVE,
) -> tuple:
    """What `report_by_language` measures before it counts: the choice, the op, language and kind of every site, the
    thresholds (chosen on the rows of each language unless given) and the sites each rule applies once the overlaps of the
    page selection are resolved. For the runs that need the sites themselves (the counts by document)."""
    probability = calibrate_rows_by_op(table, raw, cross_fit, min_group, min_positive)
    choice = protect_proper(table, choose(table, probability))
    ops, languages, kinds = site_ops(table), site_languages(table), site_kinds(table, choice)
    if thresholds is None:
        thresholds = thresholds_by_language(
            choice,
            ops,
            languages,
            floor=floor,
            minima=minima,
            kinds=kinds,
            eligible=eligible_groups(table, min_group, min_positive),
        )
    masks = {
        rule: without_overlaps(table, choice, applied_by_language(choice, ops, languages, rule, thresholds, kinds))
        for rule in ("safe", "more")
    }
    return choice, ops, languages, kinds, thresholds, masks


def report_by_language(
    table: Table,
    raw: np.ndarray,
    cross_fit: bool = True,
    resamples: int = 500,
    thresholds: dict | None = None,
    floor: float = SAFE_FLOOR,
    minima: dict | None = None,
    min_group: int = MIN_GROUP,
    min_positive: int = MIN_POSITIVE,
) -> dict:
    """`report_by_op` for each language: one calibration (pooled by op and kind), the thresholds of each language
    from its own rows (and, for an op that misses its floor, of each `op|kind` group of its own rows), the overlaps of
    the page selection, and the counts with document-bootstrap intervals computed on the rows of that language
    (`by_language[language]`)."""
    choice, ops, languages, kinds, thresholds, masks = points_by_language(
        table, raw, cross_fit, thresholds, floor, minima, min_group, min_positive
    )
    out: dict = {"sites": int(table.sites), "cross_fit": cross_fit, "thresholds": thresholds, "by_language": {}}
    for language in thresholds:
        members = languages == language
        part: dict = {"sites": int(members.sum()), "documents": len(set(table.doc[members].tolist()))}
        for rule, mask in masks.items():
            part[rule] = {
                "total": with_intervals(_members(choice, members), mask[members], table.doc[members], resamples)
            }
            for name in thresholds[language]:
                if "|" in name:
                    if rule == "safe":
                        op, kind = name.split("|")
                        part[rule][name] = counts(choice, mask & members & (ops == op) & (kinds == kind))
                else:
                    part[rule][name] = counts(choice, mask & members & (ops == name))
        out["by_language"][language] = part
    return out


def calibration_tables(
    table: Table, raw: np.ndarray, min_group: int = MIN_GROUP, min_positive: int = MIN_POSITIVE
) -> dict[str, dict]:
    """The isotonic tables of a pooled out-of-fold run as the model file keeps them: `all` (every row), one per op with
    `min_positive` true candidates (`text`) and one per (op, kind) group with `min_group` rows and `min_positive` true
    candidates (`text|accent`). `extract.guard.fixing.calibrate` applies them with the same precedence as
    `calibrate_rows_by_op` has in sample: the group, else its op, else all."""
    kind = table.x[:, INDEX["kind"]].astype(int)
    op = site_ops(table)[table.site]
    tables = {"all": calibrate.fit_isotonic(raw, table.y, min_words=1)}
    for name in OPS:
        members = op == name
        if not members.any() or table.y[members].sum() < min_positive:
            continue
        tables[name] = calibrate.fit_isotonic(raw[members], table.y[members], min_words=1)
        for code in np.unique(kind[members]):
            group = members & (kind == code)
            if group.sum() >= min_group and table.y[group].sum() >= min_positive:
                tables[f"{name}|{KINDS[int(code)]}"] = calibrate.fit_isotonic(raw[group], table.y[group], min_words=1)
    return tables
