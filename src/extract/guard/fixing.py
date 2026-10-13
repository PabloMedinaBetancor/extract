"""Fixes at run time: for the words the detector flags, the best edit of each site, with its calibrated probabilities.

A word is *processed* when its raw detector score reaches `t_gen` (at most a fifth of the page, the highest first). Each
processed word is a text site (the candidates of `candidates.generate`) and may be part of structure sites (a join of
two or three words, a segmentation, a drop). The chooser scores every (site, candidate) row, the scores are
calibrated by (op, kind) with the tables of the model file and the best candidate that is not the token as read, `c*`,
becomes a `Fix`: `p_fix` is its probability of being the truth, `p_keep` that of the word as read. `modes` says where
the model's thresholds apply it: `safe` when `p_fix` reaches `t_safe[op]`, `more` when `p_fix - p_keep` reaches
`m[op]`; an op that has no threshold is off in that mode. Applying is the app's: it holds no statistics.

Overlaps are resolved by `select.select` (the larger gain `p_fix - p_keep` wins), once for each mode, so a proposal that
loses in a mode keeps the other modes it wins in. A page whose share of words in the es and en lists is under the
model's `language_floor` is not looked at as a Spanish or English page: its proposals are returned with no mode and the
caller says so.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from extract.guard import candidates, features, language, lexicon, proper, select, signals, structure
from extract.guard.features import INDEX, KINDS, OPS
from extract.guard.forms import normalise, split_word
from extract.types import Fix, Reason, Word

PROCESSED_SHARE = 0.20
MODES = ("safe", "more")
REASON_PREFIX = "guard.fix."
MAX_REASONS = 3
READERS_AGREE_MIN = 2
DEFAULT_LANGUAGE_FLOOR = 0.4


def processed_words(raw: np.ndarray, t_gen: float, share: float = PROCESSED_SHARE) -> list[int]:
    """The words whose raw detector score reaches `t_gen`, at most `share` of the page (the highest scores first)."""
    flagged = np.nonzero(raw >= t_gen)[0]
    cap = math.ceil(share * len(raw))
    if len(flagged) > cap:
        flagged = np.array(sorted(sorted(flagged.tolist(), key=lambda i: (-raw[i], i))[:cap]), dtype=int)
    return [int(i) for i in flagged]


def calibrate(tables: dict, raw: np.ndarray, ops: np.ndarray, kinds: np.ndarray) -> np.ndarray:
    """Calibrated probabilities of rows: the table of the (op, kind) group when the model has one, else the table of its
    op, else the table of all rows (`operating.calibration_tables` writes them with the same precedence)."""
    out = np.interp(raw, tables["all"]["scores"], tables["all"]["probs"])
    for op in np.unique(ops):
        members = ops == op
        if op in tables:
            out[members] = np.interp(raw[members], tables[op]["scores"], tables[op]["probs"])
        for kind in np.unique(kinds[members]):
            key = f"{op}|{kind}"
            if key in tables:
                chosen = members & (kinds == kind)
                out[chosen] = np.interp(raw[chosen], tables[key]["scores"], tables[key]["probs"])
    return out


@dataclass
class Site:
    """A site of the page: its op, the words it spans, its candidates (the token as read first) and their feature rows."""

    op: str
    span: tuple[int, ...]
    found: list[candidates.Candidate]
    x: np.ndarray
    first_row: int = 0
    proper: bool = False  # a proper name that no list knows is in the span: `safe` leaves it alone


def sites_of(ctx: candidates.PageContext, matrix: np.ndarray, flagged: Sequence[int]) -> list[Site]:
    """The sites that hold a flagged word and have a candidate besides the token as read, in a fixed order (text sites
    in reading order, then joins, segmentations and drops)."""
    marked = set(flagged)
    sites: list[Site] = []
    for index in sorted(marked):
        if ctx.is_site(index):
            found = candidates.generate(ctx, index)
            if len(found) > 1:
                x = features.feature_rows(ctx, index, found, matrix[index])
                sites.append(Site("text", (index,), found, x, proper=proper.protected_in_safe([ctx.words[index].text])))
    wanted: list[tuple[str, tuple[int, ...]]] = [("join", s) for s in structure.join_spans(ctx) if marked & set(s)]
    wanted += [("segment", (i,)) for i in sorted(marked) if structure.segment_ok(ctx, i)]
    wanted += [("drop", (i,)) for i in sorted(marked) if structure.drop_ok(ctx, i)]
    for family, span in wanted:
        found = structure.generate_structure(ctx, family, span)
        if len(found) > 1:
            x = features.structure_rows(ctx, family, span, found, matrix[span[0]])
            sites.append(
                Site(family, span, found, x, proper=proper.protected_in_safe([ctx.words[i].text for i in span]))
            )
    return sites


def propose(
    model, words: Sequence[Word], matrix: np.ndarray, raw: np.ndarray, languages: Sequence[str]
) -> tuple[dict[int, Fix], str | None]:
    """The fixes of a page, keyed by the index of the first word of their span, and the cause that withheld them
    (`language`) or None. `model` is a `GuardModel` with a `fix` block; `raw` the detector's raw scores."""
    block = model.fix
    if not lexicon.available(languages):  # no usable word list: nothing can be said, flagged words or not
        return {}, "language"
    flagged = processed_words(raw, block["t_gen"])
    if not flagged:
        return {}, None
    fragments = frozenset(np.nonzero(matrix[:, signals.INDEX["hyphen_fragment"]] == 1)[0].tolist())
    sensor = block.get("language_sensor")
    sensed = language.sensed_language(words, sensor) if sensor else None
    ctx = candidates.PageContext(words, language.dictionaries(sensed, languages), fragments)
    sites = sites_of(ctx, matrix, flagged)
    share = ctx.dictionary_share
    floor = block.get("language_floor", DEFAULT_LANGUAGE_FLOOR)
    # a page with no usable word list (no share) is withheld too: `not >=` is true for NaN, `<` is not
    withheld = "language" if not share >= floor else None
    if sensor and sensed not in languages:  # `unsure`, or a language the caller did not declare
        withheld = "language"
    # the thresholds of the language of the page; a language the model has no block for applies nothing
    thresholds = block["by_language"].get(sensed, {"t_safe": {}, "m": {}}) if sensor else block
    if not sites:
        return {}, withheld
    x = np.vstack([site.x for site in sites])
    ops = np.repeat([s.op for s in sites], [len(s.found) for s in sites]).astype(object)
    kinds = np.array([KINDS[int(code)] for code in x[:, INDEX["kind"]]], dtype=object)
    probability = calibrate(block["calibration"], model.choose(x), ops, kinds)
    proposals = []
    start = 0
    for site in sites:
        p = probability[start : start + len(site.found)]
        start += len(site.found)
        best = 1 + int(np.argmax(p[1:]))
        proposals.append((site, best, float(p[best]), float(p[0])))
    modes = _modes(proposals, thresholds, withheld)
    # one Fix per first word, chosen after the modes: the one that applies (at most one can, they share the word),
    # else the largest gain; so what is delivered is what the operating points measured
    order = sorted(
        range(len(proposals)),
        key=lambda k: (not modes[k], -(proposals[k][2] - proposals[k][3]), proposals[k][0].span, proposals[k][0].op),
    )
    chosen: dict[int, int] = {}
    for k in order:
        chosen.setdefault(proposals[k][0].span[0], k)
    kept = [proposals[k] for k in chosen.values()]
    modes = [modes[k] for k in chosen.values()]
    fixes = {}
    for k, (site, best, p_fix, p_keep) in enumerate(kept):
        candidate = site.found[best]
        fixes[site.span[0]] = Fix(
            op=site.op,
            kind=candidate.kind,
            text=candidate.text,
            ids=tuple(words[i].id for i in site.span),
            source=candidate.sources[0],
            sources=tuple(candidate.sources),
            p_fix=p_fix,
            p_keep=p_keep,
            rule=None,
            modes=modes[k],
            reasons=reasons_of(site, candidate, ctx, words),
        )
    return fixes, withheld


def _modes(kept: list[tuple], block: dict, withheld: str | None) -> list[tuple[str, ...]]:
    """Where each proposal applies: the thresholds of its op, then the overlaps resolved by the page selection. `more` is
    a relaxation of `safe`: what wins in `safe` applies in both, and a proposal that passes only `more` applies there
    when it shares no word with a winner of `safe` and wins its own overlaps (`select.select`: the larger gain)."""
    if withheld:
        return [() for _ in kept]

    def contenders(indexes) -> dict:
        return {select.Proposal(kept[k][0].span, kept[k][0].op, kept[k][2] - kept[k][3], str(k)): k for k in indexes}

    safe_pass = [
        k
        for k, (site, best, p_fix, p_keep) in enumerate(kept)
        if _passes("safe", site.op, p_fix, p_keep, block, site.found[best].kind) and not site.proper
    ]
    more_pass = [
        k
        for k, (site, _, p_fix, p_keep) in enumerate(kept)
        if k not in safe_pass and _passes("more", site.op, p_fix, p_keep, block)
    ]
    safe_proposals = contenders(safe_pass)
    safe_taken, _ = select.select(safe_proposals)
    safe_wins = {safe_proposals[p] for p in safe_taken}
    used = {i for k in safe_wins for i in kept[k][0].span}
    more_proposals = contenders(k for k in more_pass if not used & set(kept[k][0].span))
    more_taken, _ = select.select(more_proposals)
    more_wins = {more_proposals[p] for p in more_taken}
    return [("safe", "more") if k in safe_wins else ("more",) if k in more_wins else () for k in range(len(kept))]


def _passes(mode: str, op: str, p_fix: float, p_keep: float, block: dict, kind: str | None = None) -> bool:
    if mode == "safe":
        # the `op|kind` threshold when the model has one for the group (None: that group is off), else the op's
        table = block["t_safe"]
        key = f"{op}|{kind}"
        threshold = table[key] if key in table else table.get(op)
        return threshold is not None and p_fix >= threshold
    threshold = block["m"].get(op)
    return threshold is not None and p_fix - p_keep >= threshold


def reasons_of(
    site: Site, candidate: candidates.Candidate, ctx: candidates.PageContext, words: Sequence[Word]
) -> tuple:
    """At most three `guard.fix.*` keys, by rule, in the order of the design: what the edit is, then what supports it."""
    keys: list[tuple[str, dict]] = []
    if site.op == "text":
        if candidate.kind in ("accent", "accent_case"):
            keys.append(("accent_restored", {}))
        if candidate.kind in ("case", "accent_case"):
            keys.append(("case_restored", {}))
        if candidate.kind == "punct":
            keys.append(("punctuation_restored", {}))
    else:
        keys.append(({"join": "words_joined", "segment": "word_split", "drop": "speck_dropped"}[site.op], {}))
    word = words[site.span[0]]
    wanted = normalise(candidate.text)
    heard = [r for r in getattr(word, "readings", None) or () if r.text is not None]
    agreeing = [r for r in heard if normalise(r.text) == wanted]
    if len(agreeing) >= READERS_AGREE_MIN and site.op == "text":
        keys.append(("readers_agree", {"n": len(agreeing), "of": len(heard)}))
    if "ctc_alt" in candidate.sources:
        detail = _alternative(word, candidate.text)
        if detail:
            keys.append(("engine_alternative", detail))
    if "dictionary" in candidate.sources and site.op == "text":
        keys.append(("dictionary_nearest", {"word": candidate.text}))
    if "page" in candidate.sources and site.op == "text":
        keys.append(("page_vocabulary", {"count": int(ctx.written.get(_core(candidate.text), 0))}))
    return tuple(Reason(REASON_PREFIX + key, params) for key, params in keys[:MAX_REASONS])


def _core(text: str) -> str:
    parts = split_word(text)
    return parts[1] if parts else text


def _alternative(word: Word, text: str) -> dict | None:
    """The character the engine's alternatives gave and how likely they said it was: the first position where the
    candidate differs from the word as read (same length), looked up among the alternatives of that position."""
    alts = getattr(word, "char_alts", None)
    if not alts or len(text) != len(word.text):
        return None
    for position, (old, new) in enumerate(zip(word.text, text, strict=True)):
        if old != new and position < len(alts):
            for char, probability in alts[position]:
                if char == new:
                    return {"char": new, "conf": round(float(probability), 4)}
            return None
    return None


__all__ = ["OPS", "MODES", "propose", "processed_words", "calibrate", "reasons_of"]
