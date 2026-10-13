"""Components whose truth is an oddity of the PDF text layer, not an error of the reading.

The text layer of a digital PDF is the truth of the benchmark, and it has habits of its own: a word written two or
three times (an overprint), two words with no space between them, a heading with its letters spaced out, a drop cap
set apart from its word. In each the OCR read what is on the page and is right; counting its words as errors would
reward a join, a cut or a duplicate that damages a right reading. Such a component is marked as *not grounded*
(`Label.wrong` is None, `kind` is `artifact`, `detail` is the reason), so it stays out of the detector's rows, the
labels of the chooser, the accounting and every metric, exactly as ink the layer does not describe does.

The rule is a function of the layer's words and of the OCR tokens as read, never of what the guard proposes, and it
is fixed before the test is read: the same rule marks the development documents and the test.

Audit (2026-10-02, `bench/guard/artifact_audit.py`, crops read by the corrector): of 30 `layer_glue` crops, 0 are an error of
the OCR (the ink shows the two words apart); of 30 `layer_duplicate` crops, 0 (the word is on the page once), and of the
5 outside the one scanned form that holds 143 of the 148, 1 is a real merge (two stars read as `**`). Taken together 1 in
65 sampled, far under the 10 % that would narrow the rule.
"""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import replace
from functools import lru_cache

from bench.corpus.labels import Alignment, Label, TruthWord, label_words
from bench.guard import accounting
from extract.guard import dictionary, lexicon
from extract.guard.forms import normalise, split_word

LANGUAGES = ("es", "en")
REASONS = ("layer_duplicate", "layer_spaced", "layer_dropcap", "layer_glue")
MIN_NAME_LETTERS = 3


@lru_cache(maxsize=1)
def _books() -> tuple:
    return tuple(dictionary.load(language) for language in lexicon.available(LANGUAGES))


def listed(letters: str) -> bool:
    """Whether a word of the Spanish or English lists, ignoring case."""
    return any(book.rank_of(letters) >= 0 for book in _books())


def reason(
    component: accounting.Component, texts: Sequence[str], is_listed: Callable[[str], bool] = listed
) -> str | None:
    """Why the truth of a merge or a split is an oddity of the layer, or None when it is an error of the reading.

    A merge whose truth words are all the same word is a word the layer wrote twice or three times; one whose truth
    words are single letters is a heading the layer spaced out; one whose truth is a single letter and an unlisted
    rest is a drop cap the layer set apart. A split whose truth is exactly the OCR words written together, an
    unlisted word, when each of them is a listed word or a capitalised name, is a pair the layer glued without a
    space."""
    if component.kind == "merge":
        pieces = [normalise(piece).casefold() for piece in component.truth.split(" ")]
        if len(pieces) > 1 and len(set(pieces)) == 1:
            return "layer_duplicate"
        cores = [(split_word(piece) or ("", piece, ""))[1] for piece in pieces]
        if len(pieces) >= 3 and all(len(core) == 1 for core in cores):
            return "layer_spaced"
        if len(pieces) == 2 and len(pieces[0]) == 1 and not is_listed(pieces[1]):
            return "layer_dropcap"
        return None
    if component.kind == "split":
        parts = [split_word(texts[i]) for i in component.words]
        if any(part is None for part in parts):
            return None
        if normalise(component.truth) != normalise("".join(texts[i] for i in component.words)):
            return None
        cores = [part[1] for part in parts]
        if not all(is_listed(core) or (core[:1].isupper() and len(core) >= MIN_NAME_LETTERS) for core in cores):
            return None
        return None if is_listed("".join(cores)) else "layer_glue"
    return None


def mark(
    texts: Sequence[str], labels: list[Label], is_listed: Callable[[str], bool] = listed
) -> tuple[list[Label], Counter]:
    """The labels with the words of every artifact component made ungrounded, and the components counted by reason."""
    marked = list(labels)
    counts: Counter = Counter()
    for component in accounting.components_of(list(texts), labels):
        why = reason(component, texts, is_listed)
        if why is None:
            continue
        counts[why] += 1
        for index in component.words:
            marked[index] = replace(
                labels[index], wrong=None, kind="artifact", detail=why, truth_text=None, truth_ids=()
            )
    return marked, counts


def align(
    read: Sequence[tuple], truth: Sequence[TruthWord], ungrounded: Sequence = (), mark_artifacts: bool = True
) -> tuple[Alignment, Counter]:
    """The one place the labels of a page are made: the alignment of the OCR words with the truth, with the artifact
    components of the text layer made ungrounded (unless `mark_artifacts` is off) and the kinds counted again. Every
    loader of labelled pages goes through here, so no figure mixes a denominator with the artifacts and one without."""
    alignment = label_words(list(read), list(truth), [tuple(box) for box in ungrounded])
    if not mark_artifacts:
        return alignment, Counter()
    labels, found = mark([word[1] for word in read], alignment.labels)
    return replace(alignment, labels=labels, kinds=Counter(label.kind for label in labels)), found
