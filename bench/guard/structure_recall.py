"""Recall of the structure candidates, measured on the cached v4 pages: how many of the split, merged and invented
words a join, a segmentation or a drop could repair, and how many structure sites there are per 1,000 words.

    python3 -s -m bench.guard.structure_recall --sets train val --out bench/results/structure-recall.json
"""

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from unittest import mock

from bench.guard import accounting, pairs
from extract.guard import candidates, structure
from extract.guard.forms import normalise

FLOORS = {"join": 0.55, "segment": 0.70, "drop": 0.70}
MIN_CONCLUSIVE = 20  # a floor over fewer eligible components than this is "not conclusive", never pass or fail
NOT_PLAIN_REASONS = (
    "digit",
    "internal_marks",
    "fragment",
    "blocked_or_long",
    "hyphen_end",
    "mixed_capitals",
    "acronym",
    "other",
)


def not_plain_reason(ctx, index: int) -> str | None:
    """Why a word is not a flat token , or None when it is one: the reason the protection rules give."""
    if ctx.is_site(index):
        return None
    parts, text = ctx.parts[index], ctx.words[index].text
    if parts is None:
        return "digit" if any(c.isdigit() for c in text) else "internal_marks"
    if index in ctx.fragments:
        return "fragment"
    if len(text) > candidates.MAX_LENGTH or candidates.BLOCKED_CHARACTERS & set(text):
        return "blocked_or_long"
    prefix, core, suffix = parts
    if any(c in normalise(prefix + suffix) for c in "-'") or candidates.FINAL_MARK in prefix + suffix[:-1]:
        return "hyphen_end"
    if sum(1 for c in core if c.isupper()) >= 2 and not core.isupper():
        return "mixed_capitals"
    return "acronym" if core.isupper() and len(core) <= candidates.ACRONYM_LENGTH else "other"


def share(count: int, total: int) -> float | None:
    return round(count / total, 4) if total else None


def join_recall(page: pairs.Page, counts: Counter, spans: set) -> None:
    for component in page.components:
        if component.kind != "split":
            continue
        counts["components"] += 1
        words = component.words
        if not 2 <= len(words) <= structure.MAX_JOIN:
            counts["lost_more_than_three_tokens" if len(words) > 3 else "lost_one_token"] += 1
            continue
        span = tuple(sorted(words))
        if span not in spans:
            why = _why_not_a_span(page, span)
            counts["lost_" + why] += 1
            if why == "protected_or_not_plain":
                counts["by_reason_" + _first_reason(page, span)] += 1
            continue
        counts["span_found"] += 1
        found = structure.generate_structure(page.context, "join", span)
        truth = normalise(component.truth)
        if any(normalise(c.text) == truth for c in found[1:]):
            counts["reached"] += 1
            for c in found[1:]:
                if normalise(c.text) == truth:
                    counts["reached_by_" + "+".join(c.sources)] += 1
                    break
        else:
            counts["lost_text_not_proposed"] += 1


def _first_reason(page: pairs.Page, span: tuple[int, ...]) -> str:
    return next(r for r in (not_plain_reason(page.context, i) for i in span) if r is not None)


def _why_not_a_span(page: pairs.Page, span: tuple[int, ...]) -> str:
    ctx = page.context
    if span != tuple(range(span[0], span[0] + len(span))):
        return "not_adjacent_in_reading_order"
    if len({ctx.words[i].line_id for i in span}) > 1:
        return "across_lines"
    if not all(ctx.is_site(i) for i in span):
        return "protected_or_not_plain"
    for a, b in zip(span, span[1:], strict=False):
        if ctx.parts[a][2] or ctx.parts[b][0]:
            return "punctuation_between"
        if structure.gap_ratio(ctx, a, b) > structure.JOIN_GAP:
            return "gap_too_wide"
    return "other"


def segment_recall(page: pairs.Page, counts: Counter) -> None:
    for component in page.components:
        if component.kind != "merge" or len(component.words) != 1:
            continue
        counts["components"] += 1
        index = component.words[0]
        if not page.context.is_site(index):
            counts["lost_protected_or_not_plain"] += 1
            counts["by_reason_" + not_plain_reason(page.context, index)] += 1
        elif not structure.segment_ok(page.context, index):
            counts["lost_shorter_than_five_letters"] += 1
        else:
            counts["eligible"] += 1
            found = structure.generate_structure(page.context, "segment", (index,))
            truth = normalise(component.truth)
            if any(normalise(c.text) == truth for c in found[1:]):
                counts["reached"] += 1
            else:
                counts["lost_cut_not_in_dictionary_or_over_cap"] += 1


def drop_recall(page: pairs.Page, counts: Counter) -> None:
    for component in page.components:
        if component.kind != "invented":
            continue
        counts["components"] += 1
        index = component.words[0]
        text = page.words[index].text
        if len(text) > structure.DROP_MAX_CHARACTERS:
            counts["lost_longer_than_two_characters"] += 1
            continue
        counts["short"] += 1
        ctx = page.context
        if structure.drop_ok(ctx, index):
            counts["reached"] += 1
        elif any(c.isdigit() for c in text):
            counts["lost_digit"] += 1
        elif ctx.parts[index] is not None and structure.listed(ctx, ctx.parts[index][1]):
            counts["lost_listed_word"] += 1
        elif ctx.parts[index] is None and any(c.isalnum() for c in text):
            counts["lost_not_letters_or_marks"] += 1
        elif not ctx.is_site(index) and ctx.parts[index] is not None:
            counts["lost_protected"] += 1
        else:
            counts["lost_not_small_or_doubtful_or_lone_dash"] += 1


def cost(page: pairs.Page, counts: Counter) -> None:
    ctx = page.context
    counts["words"] += len(page.words)
    spans = structure.join_spans(ctx)
    counts["join_sites"] += len(spans)
    counts["join_candidates"] += sum(len(structure.generate_structure(ctx, "join", s)) - 1 for s in spans)
    for i in range(len(page.words)):
        if structure.segment_ok(ctx, i):
            counts["segment_sites"] += 1
            counts["segment_candidates"] += len(structure.generate_structure(ctx, "segment", (i,))) - 1
        if structure.drop_ok(ctx, i):
            counts["drop_sites"] += 1


def join_variants(pages: list[pairs.Page]) -> dict:
    """The join recall under each rule for the joined text as written (the cost of the restriction)."""
    out = {}
    for mode in ("listed", "listed_or_page", "any"):
        counts: Counter = Counter()
        with mock.patch.object(structure, "JOIN_AS_WRITTEN", mode):
            for page in pages:
                join_recall(page, counts, set(structure.join_spans(page.context)))
        out[mode] = {"reached": counts["reached"], "recall": share(counts["reached"], counts["components"])}
    return out


def tally(pages: list[pairs.Page]) -> dict:
    """The raw counts of a list of pages, additive across lists (a set, or train and val together)."""
    out = {k: Counter() for k in ("join", "segment", "drop", "cost", "wrong", "excluded")}
    out["pages"], out["documents"] = 0, set()
    for page in pages:
        out["pages"] += 1
        out["documents"].add(page.doc)
        out["excluded"].update(page.artifacts)
        spans = set(structure.join_spans(page.context))
        join_recall(page, out["join"], spans)
        segment_recall(page, out["segment"])
        drop_recall(page, out["drop"])
        cost(page, out["cost"])
        for component in page.components:
            if not accounting.is_right(component, [w.text for w in page.words]):
                out["wrong"][component.kind] += component.weight
    return out


def add(a: dict, b: dict) -> dict:
    out = {k: a[k] + b[k] for k in ("join", "segment", "drop", "cost", "wrong", "excluded")}
    out["pages"], out["documents"] = a["pages"] + b["pages"], a["documents"] | b["documents"]
    return out


def verdict(recall: float | None, eligible: int, floor: float) -> bool | None:
    """Whether a floor is met: None (not conclusive) when the eligible components are too few to say."""
    if recall is None or eligible < MIN_CONCLUSIVE:
        return None
    return recall >= floor


def finish(t: dict, pages: list[pairs.Page] | None = None) -> dict:
    """The report of a tally. Two denominators, both shown: `recall` over every grounded component of the kind (the
    label of the criterion: not a text-layer artifact) and `recall_of_eligible` over the flat tokens the protection rules allow
    an edit on (no digit, no internal mark, no hyphen end...); the floors are judged on the eligible ones."""
    out = {
        "pages": t["pages"],
        "documents": len(t["documents"]),
        "wrong_weight_by_component": dict(t["wrong"]),
        "artifact_components_left_out": dict(t["excluded"]),
    }
    for name, total in (("join", "components"), ("segment", "components"), ("drop", "short")):
        counts = t[name]
        eligible = {
            "join": counts["span_found"],
            "segment": counts["eligible"],
            "drop": counts["short"] - counts["lost_digit"],
        }[name]
        recall_of_eligible = share(counts["reached"], eligible)
        reasons = {k[len("by_reason_") :]: v for k, v in sorted(counts.items()) if k.startswith("by_reason_")}
        out[name] = {
            **{k: v for k, v in counts.items() if not k.startswith("by_reason_")},
            "eligible": eligible,
            "recall_of_eligible": recall_of_eligible,
            "recall": share(counts["reached"], counts[total]),
            "floor": FLOORS[name],
            "passes": verdict(recall_of_eligible, eligible, FLOORS[name]),
            "conclusive": eligible >= MIN_CONCLUSIVE,
            "lost_protected_or_not_plain_by_reason": reasons,
        }
    if pages is not None:
        out["join_as_written_variants"] = join_variants(pages)
    spent = t["cost"]
    words = max(spent["words"], 1)
    out["cost"] = {
        **spent,
        "join_sites_per_1000_words": round(1000 * spent["join_sites"] / words, 1),
        "segment_sites_per_1000_words": round(1000 * spent["segment_sites"] / words, 1),
        "drop_sites_per_1000_words": round(1000 * spent["drop_sites"] / words, 1),
    }
    return out


def report(pages: list[pairs.Page]) -> dict:
    return finish(tally(pages), pages)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--sets", nargs="+", default=["train", "val"])
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--out", type=Path, default=Path("bench/results/structure-recall.json"))
    args = parser.parse_args(argv)
    started = time.time()
    result = {}
    for marked, suffix in ((True, ""), (False, "_all_components")):
        tallies, loaded = {}, []
        for set_name in args.sets:
            pages = pairs.load_pages(args.data, [set_name], mark_artifacts=marked)
            tallies[set_name] = tally(pages)
            loaded += pages
            result[set_name + suffix] = finish(tallies[set_name], pages if marked else None)
        if len(args.sets) > 1:
            joint = tallies[args.sets[0]]
            for name in args.sets[1:]:
                joint = add(joint, tallies[name])
            result["+".join(args.sets) + suffix] = finish(joint, loaded if marked else None)
    result["seconds"] = round(time.time() - started, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1, default=sorted), encoding="utf-8")
    names = [n for n in result if n != "seconds"]
    print(
        json.dumps(
            {n: {k: [result[n][k]["recall_of_eligible"], result[n][k]["recall"]] for k in FLOORS} for n in names}
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
