"""The features of a (site, candidate) row that the chooser reads.

One row per candidate, the token as read included, so that the model also sees what a right word looks like. A
feature is a function of the OCR output (the token, its neighbours, the page, the engine's alternatives and the
second readings) and of the word lists: nothing here reads a truth word. NaN means the feature does not apply, or
that its source did not run; a source that ran and did not propose the candidate gives 0.
"""

import hashlib
import json
import math
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
from rapidfuzz.distance import Levenshtein

from extract.guard import signals, structure
from extract.guard.candidates import KEEP, Candidate, PageContext
from extract.guard.dictionary import GramBatch
from extract.guard.edits import confusion_costs, weighted_distance
from extract.guard.forms import case_pattern, normalise, split_word

FEATURE_VERSION = 2
# The reader the reader bake-off named (PP-OCRv6 medium); the model is trained and served with the same one.
STRONG_READER = "deep"
INDEPENDENT_READERS = ("latin", "medium")
CONFUSION_PAIRS = 20
MARKS = "()[]{}\u00ab\u00bb\"'\u00bf\u00a1.,;:!?-\u00b7*\u2022/\u20ac%&"
KINDS = ("keep", "accent", "case", "accent_case", "letters", "punct", "structure")
OPS = ("text", "join", "segment", "drop")
CASE_CODES = {"lower": 0.0, "title": 1.0, "upper": 2.0, "irregular": 3.0}

EDIT = (
    "kind",
    "n_edits",
    "weighted_distance",
    "distance",
    "length_difference",
    "changed_chars_min_conf",
    "changed_chars_mean_conf",
    "confusion_id",
    "edit_punct_id",
)
CANDIDATE = (
    "cand_in_dictionary",
    "cand_log_frequency",
    "cand_rank_by_frequency",
    "cand_frequency_ratio",
    "cand_char_lm_gain",
    "folded_unique",
    "same_fold_count",
    "cand_case_pattern",
)
PAGE = (
    "page_count_token",
    "page_count_cand",
    "page_count_token_written",
    "page_count_cand_written",
    "page_count_cand_confident",
    "page_vocab_proposed",
    "page_dict_share",
    "n_candidates",
    "n_sources",
)
ENGINE = ("masked_differs", "alt_prob_min", "alt_prob_product", "alt_positions", "ctc_proposed")
READINGS = (
    "readers_run",
    "readers_aligned",
    "readers_equal_candidate",
    "readers_equal_token",
    "best_reader_conf_candidate",
    "independent_agree",
    "strong_run",
    "strong_equal_candidate",
    "strong_equal_token",
    "strong_conf",
    "zone_rank",
    "in_zone_as_context",
)
STRUCTURE = (
    "op",
    "n_tokens",
    "gap_ratio_mean",
    "gap_ratio_max",
    "parts_in_dictionary_share",
    "joined_in_dictionary",
    "joined_log_frequency",
    "parts_log_frequency_min",
    "split_position_ratio",
    "reader_group_support",
    "box_height_ratio",
    "box_area_ratio",
    "span_conf_min",
    "span_conf_mean",
)
CONTEXT = ("next_is_title", "ends_line", "starts_line", "prev_ends_sentence", "punct_context")
FEATURES = (*signals.COLUMNS, *EDIT, *CANDIDATE, *PAGE, *ENGINE, *READINGS, *STRUCTURE, *CONTEXT)
INDEX = {name: i for i, name in enumerate(FEATURES)}
CATEGORICAL = ("kind", "op")


def feature_fingerprint() -> str:
    """SHA-256 of the feature schema: it changes with a feature, its order, the version or the strong reader."""
    payload = json.dumps({"version": FEATURE_VERSION, "features": FEATURES, "strong_reader": STRONG_READER})
    return hashlib.sha256(payload.encode()).hexdigest()


@lru_cache(maxsize=1)
def confusion_ids() -> dict[frozenset[str], int]:
    """The `CONFUSION_PAIRS` most listed single-character confusions, as unordered pairs of lower-case characters."""
    pairs: dict[frozenset[str], int] = {}
    for (first, second), _ in confusion_costs().items():
        if len(first) == len(second) == 1 and len(pairs) < CONFUSION_PAIRS:
            pairs.setdefault(frozenset((first, second)), len(pairs))
    return pairs


def feature_rows(
    ctx: PageContext, index: int, candidates: Sequence[Candidate], signal_row: np.ndarray, reader: str | None = None
) -> np.ndarray:
    """The feature matrix of one site: a row per candidate, in the order given, `FEATURES` as columns."""
    word = ctx.words[index]
    _, core, _ = ctx.parts[index]
    reader = reader or STRONG_READER
    rows = np.full((len(candidates), len(FEATURES)), np.nan, dtype=np.float32)
    rows[:, : len(signals.COLUMNS)] = signal_row
    frequencies = [_log_frequency(ctx, _core_of(c)) for c in candidates]
    best_frequency = max(frequencies)
    order = sorted(range(len(candidates)), key=lambda i: -frequencies[i])
    rank = {i: place for place, i in enumerate(order)}
    cores = [_core_of(c) for c in candidates]
    logp = _char_logp_table(ctx, [core, *cores])
    token_lm = logp[core]
    shared = _shared(ctx, index)
    for i, candidate in enumerate(candidates):
        row = rows[i]
        ccore = _core_of(candidate)
        _set(row, _edit_features(word, core, candidate, ccore))
        _set(row, _candidate_features(ctx, ccore, frequencies[i], best_frequency, rank[i], logp[ccore] - token_lm))
        _set(row, _page_features(ctx, word, core, ccore, candidate, len(candidates)))
        _set(row, _engine_features(word, candidate))
        _set(row, _reading_features(word, candidate, reader))
        _set(row, shared)
        row[INDEX["op"]] = OPS.index("text")
    return np.round(rows, 4)


def _set(row: np.ndarray, values: dict[str, float]) -> None:
    for name, value in values.items():
        row[INDEX[name]] = value


def _core_of(candidate: Candidate) -> str:
    parts = split_word(candidate.text)
    return parts[1] if parts else candidate.text


def _log_frequency(ctx: PageContext, letters: str) -> float:
    return max((book.log_frequency(letters) for book in ctx.books), default=0.0)


def _char_logp_table(ctx: PageContext, words: Sequence[str]) -> dict[str, float]:
    """The best mean log-probability per character of each word over the page languages (NaN when none has one)."""
    batch = GramBatch(words)
    scored = [book.score(batch) for book in ctx.books]
    table = {}
    for word in dict.fromkeys(words):
        found = [value for value in (s.get(word) for s in scored) if value is not None]
        table[word] = max(found) if found else math.nan
    return table


def _shared(ctx: PageContext, index: int) -> dict[str, float]:
    """What the context says about the site, the same for every candidate."""
    word = ctx.words[index]
    following = ctx.parts[index + 1] if index + 1 < len(ctx.words) else None
    previous = ctx.parts[index - 1] if index else None
    starts = index == 0 or ctx.words[index - 1].line_id != word.line_id
    ends = ctx.ends_line(index)
    return {
        "next_is_title": math.nan if following is None else float(case_pattern(following[1]) == "title"),
        "ends_line": float(ends),
        "starts_line": float(starts),
        "prev_ends_sentence": math.nan if previous is None or starts else float(previous[2][:1] in (".", "?", "!")),
        "punct_context": float(ends or ctx.next_starts_sentence(index)),
    }


def _edit_features(word, core: str, candidate: Candidate, ccore: str) -> dict[str, float]:
    token = word.text
    text = candidate.text
    out = {"kind": float(KINDS.index(candidate.kind)), "n_edits": float(Levenshtein.distance(token, text))}
    out["weighted_distance"] = weighted_distance(core, ccore)
    out["distance"] = float(Levenshtein.distance(core.casefold(), ccore.casefold()))
    out["length_difference"] = float(len(ccore) - len(core))
    chars = word.char_conf
    ops = Levenshtein.editops(token, text)
    changed = [op.src_pos for op in ops if op.tag in ("replace", "delete")]
    if chars and len(chars) == len(token) and changed:
        confidences = [chars[i] for i in changed]
        out["changed_chars_min_conf"] = min(confidences)
        out["changed_chars_mean_conf"] = sum(confidences) / len(confidences)
    marks = [
        (token[op.src_pos] if op.tag != "insert" else text[op.dest_pos])
        for op in ops
        if op.tag in ("delete", "replace", "insert")
    ]
    marks = [m for m in marks if not m.isalnum()]
    if marks:
        out["edit_punct_id"] = float(MARKS.index(marks[0]) if marks[0] in MARKS else len(MARKS))
    if len(ops) == 1 and ops[0].tag == "replace":
        pair = frozenset((token[ops[0].src_pos].casefold(), text[ops[0].dest_pos].casefold()))
        out["confusion_id"] = float(confusion_ids().get(pair, CONFUSION_PAIRS))
    return out


def _candidate_features(
    ctx: PageContext, ccore: str, frequency: float, best: float, rank: int, lm_gain: float
) -> dict[str, float]:
    listed = ctx.fold_group(ccore)
    out = {
        "cand_in_dictionary": float(any(book.rank_of(ccore) >= 0 for book in ctx.books)),
        "cand_log_frequency": frequency,
        "cand_rank_by_frequency": float(rank),
        "cand_frequency_ratio": 10 ** (frequency - best) if best > 0 else math.nan,
        "cand_char_lm_gain": lm_gain,
        "folded_unique": float(len(listed) == 1),
        "same_fold_count": float(len(listed)),
        "cand_case_pattern": CASE_CODES[case_pattern(ccore)] if ccore else math.nan,
    }
    return out


def _page_features(
    ctx: PageContext, word, core: str, ccore: str, candidate: Candidate, n_candidates: int
) -> dict[str, float]:
    return {
        "page_count_token": float(ctx.counts[core.casefold()]),
        "page_count_cand": float(ctx.counts[ccore.casefold()]),
        "page_count_token_written": float(ctx.written[core]),
        "page_count_cand_written": float(ctx.written[ccore]),
        "page_count_cand_confident": float(ctx.confident_counts[ccore.casefold()]),
        "page_vocab_proposed": float("page" in candidate.sources),
        "page_dict_share": ctx.dictionary_share,
        "n_candidates": float(n_candidates),
        "n_sources": float(len(candidate.sources)),
    }


def _engine_features(word, candidate: Candidate) -> dict[str, float]:
    alternatives = getattr(word, "char_alts", None)
    if alternatives is None or len(alternatives) != len(word.text):
        return {}
    out = {
        "masked_differs": float(getattr(word, "masked", None) is not None),
        "ctc_proposed": float(any(source == "ctc_alt" for source in candidate.sources)),
    }
    token, text = word.text, candidate.text
    swaps = [op for op in Levenshtein.editops(token, text) if op.tag == "replace"]
    if swaps and len(swaps) == len(Levenshtein.editops(token, text)):
        probabilities = []
        for op in swaps:
            options = dict(alternatives[op.src_pos])
            probabilities.append(options.get(text[op.dest_pos], 0.0))
        out["alt_prob_min"] = min(probabilities)
        out["alt_prob_product"] = float(np.prod(probabilities))
        out["alt_positions"] = float(sum(len(alternatives[op.src_pos]) > 1 for op in swaps))
    return out


def _reading_features(word, candidate: Candidate, reader: str) -> dict[str, float]:
    readings = getattr(word, "readings", None)
    if readings is None:
        return {}
    wanted, token = normalise(candidate.text), normalise(word.text)
    texts = [None if r.text is None else normalise(r.text) for r in readings]
    agreeing = [r for r, text in zip(readings, texts, strict=True) if text == wanted]
    independent = {*INDEPENDENT_READERS, reader}
    out = {
        "readers_run": float(len(readings)),
        "readers_aligned": float(sum(text is not None for text in texts)),
        "readers_equal_candidate": float(len(agreeing)),
        "readers_equal_token": float(sum(text == token for text in texts)),
        "independent_agree": float(any(r.source in independent for r in agreeing)),
    }
    confidences = [r.conf for r in agreeing if r.conf is not None]
    if confidences:
        out["best_reader_conf_candidate"] = max(confidences)
    strong = [(r, text) for r, text in zip(readings, texts, strict=True) if r.source == reader]
    out["strong_run"] = float(bool(strong))
    if strong and strong[0][1] is not None:
        out["strong_equal_candidate"] = float(strong[0][1] == wanted)
        out["strong_equal_token"] = float(strong[0][1] == token)
    if strong and strong[0][0].conf is not None:
        out["strong_conf"] = strong[0][0].conf
    ranks = [r.rank for r in readings if getattr(r, "rank", None) is not None]
    if ranks:
        out["zone_rank"] = float(min(ranks))
    out["in_zone_as_context"] = float(any(getattr(r, "context", False) for r in readings))
    return out


def structure_rows(
    ctx: PageContext,
    family: str,
    span: Sequence[int],
    candidates: Sequence[Candidate],
    signal_row: np.ndarray,
    reader: str | None = None,
) -> np.ndarray:
    """The feature matrix of a structure site (a join of a run of words, a segmentation or a drop of one): a row per
    candidate in the order given. The signals are those of the first word; what only a text edit has stays NaN."""
    reader = reader or STRONG_READER
    span = tuple(span)
    words = [ctx.words[i] for i in span]
    cores = [(ctx.parts[i] or ("", "", ""))[1] for i in span]  # a speck of marks only (a drop) has no parts
    token_letters = "".join(cores)
    rows = np.full((len(candidates), len(FEATURES)), np.nan, dtype=np.float32)
    rows[:, : len(signals.COLUMNS)] = signal_row
    token = " ".join(w.text for w in words)
    confidences = [w.conf for w in words if w.conf is not None]
    gaps = [structure.gap_ratio(ctx, a, b) for a, b in zip(span, span[1:], strict=False)]
    parts_frequency = [structure.frequency(ctx, core) for core in cores]
    listed_parts = [structure.listed(ctx, core) for core in cores]
    supports = len(structure.reader_groups(ctx, span)) if family == "join" else len(structure.reader_cuts(ctx, span[0]))
    run = getattr(words[0], "readings", None) is not None
    height = max(ctx.line_height(span[0]), 1e-6)
    line_areas = sorted(
        (w.box[2] - w.box[0]) * (w.box[3] - w.box[1]) for w in ctx.words if w.line_id == words[0].line_id
    )
    shared = _shared(ctx, span[0])
    shared["ends_line"] = float(ctx.ends_line(span[-1]))
    shared["punct_context"] = float(ctx.ends_line(span[-1]) or ctx.next_starts_sentence(span[-1]))
    frequencies = [
        structure.frequency(ctx, "".join(ch for ch in c.text if ch.isalpha())) if c.op == "join" else None
        for c in candidates
    ]
    known = [f for f in frequencies if f is not None]
    best = max(known, default=0.0)
    order = sorted(known, reverse=True)
    for i, candidate in enumerate(candidates):
        row = rows[i]
        keep = candidate.op == "keep"
        letters = "".join(ch for ch in candidate.text if ch.isalpha())
        row[INDEX["kind"]] = KINDS.index("keep" if keep else "structure")
        row[INDEX["n_edits"]] = structure.distance(token, candidate.text)
        row[INDEX["weighted_distance"]] = weighted_distance(token_letters, letters)
        row[INDEX["distance"]] = Levenshtein.distance(token_letters.casefold(), letters.casefold())
        row[INDEX["length_difference"]] = len(letters) - len(token_letters)
        row[INDEX["n_candidates"]] = len(candidates)
        row[INDEX["n_sources"]] = len(candidate.sources)
        row[INDEX["page_dict_share"]] = ctx.dictionary_share
        _set(row, _reading_features(words[0], candidate, reader))
        _set(row, shared)
        row[INDEX["op"]] = OPS.index(family)
        row[INDEX["n_tokens"]] = len(span)
        if confidences:
            row[INDEX["span_conf_min"]] = min(confidences)
            row[INDEX["span_conf_mean"]] = sum(confidences) / len(confidences)
        if run and family != "drop":
            row[INDEX["reader_group_support"]] = supports
        if family == "join":
            row[INDEX["gap_ratio_mean"]] = sum(gaps) / len(gaps)
            row[INDEX["gap_ratio_max"]] = max(gaps)
            row[INDEX["parts_log_frequency_min"]] = min(parts_frequency)
            row[INDEX["parts_in_dictionary_share"]] = sum(listed_parts) / len(listed_parts)
            frequency = frequencies[i]
            if frequency is not None:
                _set(row, _joined_features(ctx, letters, candidate, frequency, best, order.index(frequency)))
        elif family == "segment":
            pieces = [p for p in candidate.text.split(" ") if p] if not keep else [token_letters]
            pieces = ["".join(ch for ch in p if ch.isalpha()) for p in pieces]
            row[INDEX["parts_in_dictionary_share"]] = sum(structure.listed(ctx, p) for p in pieces) / len(pieces)
            row[INDEX["parts_log_frequency_min"]] = min(structure.frequency(ctx, p) for p in pieces)
            if not keep and token_letters:
                row[INDEX["split_position_ratio"]] = len(pieces[0]) / len(token_letters)
        else:
            box = words[0].box
            row[INDEX["box_height_ratio"]] = (box[3] - box[1]) / height
            area = (box[2] - box[0]) * (box[3] - box[1])
            row[INDEX["box_area_ratio"]] = area / max(line_areas[len(line_areas) // 2], 1e-6)
    return np.round(rows, 4)


def _joined_features(
    ctx: PageContext, letters: str, candidate: Candidate, frequency: float, best: float, rank: int
) -> dict[str, float]:
    listed = ctx.fold_group(letters)
    return {
        "cand_in_dictionary": float(structure.listed(ctx, letters)),
        "cand_log_frequency": frequency,
        "cand_rank_by_frequency": float(rank),
        "cand_frequency_ratio": 10 ** (frequency - best) if best > 0 else math.nan,
        "joined_in_dictionary": float(structure.listed(ctx, letters)),
        "joined_log_frequency": frequency,
        "folded_unique": float(len(listed) == 1),
        "same_fold_count": float(len(listed)),
        "cand_case_pattern": CASE_CODES[case_pattern(letters)] if letters else math.nan,
        "page_count_cand": float(ctx.counts[letters.casefold()]),
        "page_count_cand_written": float(ctx.written[letters]),
        "page_count_cand_confident": float(ctx.confident_counts[letters.casefold()]),
        "page_vocab_proposed": float(letters.casefold() in ctx.vocabulary),
    }


def keep_candidate(token: str) -> Candidate:
    return Candidate(token, "keep", KEEP, (KEEP,))
