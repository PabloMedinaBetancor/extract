"""The accounting of fixes: one function for the out-of-fold report and for the test.

The unit is the component of the v4 labelling: a one-to-one pair of an OCR word and a truth word, a merge (one OCR
word over several truth words), a split (several OCR words over one truth word), a mixed group, or an invented word
(an OCR word over nothing). Only grounded components count. A component weighs the number of truth words it holds
(an invented word weighs one), so joining, segmenting and dropping words keep the same denominator before and
after. A component is right when the final text of its OCR words, joined with single spaces, equals its truth
words joined with single spaces after `normalise`; an invented word is right when it was dropped.
"""

from collections import Counter
from dataclasses import dataclass

from rapidfuzz.distance import Levenshtein

from bench.corpus.labels import Label
from extract.guard.forms import normalise

OUTCOMES = ("fixed", "broken", "changed_still_wrong", "left_right", "left_wrong")


@dataclass(frozen=True)
class Component:
    kind: str
    words: tuple[int, ...]
    truth: str
    weight: int
    right_before: bool


@dataclass(frozen=True)
class Edit:
    """An applied fix: `op` over the word indexes in `words`, with the text the first word takes (`text`, `join`,
    `segment`); `drop` leaves no text."""

    op: str
    words: tuple[int, ...]
    text: str = ""


def components_of(texts: list[str], labels: list[Label]) -> list[Component]:
    """The grounded components of a page: `texts` are the OCR words as read and `labels` their labels."""
    groups: dict[tuple, list[int]] = {}
    out: list[Component] = []
    for i, label in enumerate(labels):
        if label.wrong is None:
            continue
        if label.kind in ("merge", "split", "mixed"):
            groups.setdefault((label.kind, label.truth_ids), []).append(i)
        elif label.kind == "invented":
            out.append(Component("invented", (i,), "", 1, False))
        else:
            out.append(Component("pair", (i,), label.truth_text, 1, not label.wrong))
    for (kind, _), members in groups.items():
        truth = labels[members[0]].truth_text
        out.append(Component(kind, tuple(members), truth, len(truth.split(" ")) if kind != "split" else 1, False))
    return out


def final_texts(texts: list[str], edits: list[Edit]) -> list[str | None]:
    """The text of every word after the edits; `None` marks a word that no longer exists (absorbed or dropped). An
    edit that touches a word an earlier one used is skipped, as the application does."""
    final: list[str | None] = list(texts)
    used: set[int] = set()
    for edit in edits:
        if used & set(edit.words):
            continue
        used.update(edit.words)
        if edit.op in ("text", "segment"):
            final[edit.words[0]] = edit.text
        elif edit.op == "join":
            final[edit.words[0]] = edit.text
            for index in edit.words[1:]:
                final[index] = None
        elif edit.op == "drop":
            for index in edit.words:
                final[index] = None
        else:
            raise ValueError(f"unknown edit {edit.op!r}")
    return final


def component_text(component: Component, final: list[str | None]) -> str:
    return " ".join(normalise(final[i]) for i in component.words if final[i])


def is_right(component: Component, final: list[str | None]) -> bool:
    if component.kind == "invented":
        return final[component.words[0]] is None
    return component_text(component, final) == normalise(component.truth)


def tally(components: list[Component], texts: list[str], edits: list[Edit]) -> dict:
    """Weighted and unweighted outcomes of the edits, the characters after and before, and the edited components."""
    final = final_texts(texts, edits)
    words, parts = Counter(), Counter()
    distance_before = distance_after = 0
    for component in components:
        touched = any(final[i] != texts[i] for i in component.words)
        right_after = is_right(component, final)
        if component.right_before and right_after:
            outcome = "left_right"
        elif component.right_before:
            outcome = "broken"
        elif right_after:
            outcome = "fixed"
        else:
            outcome = "changed_still_wrong" if touched else "left_wrong"
        words[outcome] += component.weight
        parts[outcome] += 1
        if touched:
            truth = normalise(component.truth)
            distance_before += Levenshtein.distance(" ".join(normalise(texts[i]) for i in component.words), truth)
            distance_after += Levenshtein.distance(component_text(component, final), truth)
    return {
        "weights": dict(words),
        "components": dict(parts),
        "truth_words": sum(c.weight for c in components),
        "truth_characters": sum(len(normalise(c.truth)) for c in components if c.kind != "invented"),
        "distance_before": distance_before,
        "distance_after": distance_after,
    }


def combine(pages: list[dict]) -> dict:
    """The sum of the tallies of many pages."""
    total = {"weights": Counter(), "components": Counter(), "truth_words": 0, "truth_characters": 0}
    total.update(distance_before=0, distance_after=0)
    for page in pages:
        total["weights"].update(page["weights"])
        total["components"].update(page["components"])
        for name in ("truth_words", "truth_characters", "distance_before", "distance_after"):
            total[name] += page[name]
    return total


def precision(tallied: dict, by: str = "weights") -> float | None:
    """Fixed over (fixed + broken + changed still wrong): a changed word that stays wrong is a miss."""
    counts = tallied[by]
    applied = counts.get("fixed", 0) + counts.get("broken", 0) + counts.get("changed_still_wrong", 0)
    return counts.get("fixed", 0) / applied if applied else None


def net_change(tallied: dict) -> float:
    """`d_WER`: broken weight minus fixed weight, over the truth words; below zero the text has fewer errors."""
    weights = tallied["weights"]
    return (weights.get("broken", 0) - weights.get("fixed", 0)) / tallied["truth_words"]
