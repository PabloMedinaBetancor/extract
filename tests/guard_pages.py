"""Synthetic pages with a text layer and an OCR reading that has known errors, for the chooser's tests."""

import random
from collections import Counter

from bench.corpus.labels import TruthWord, label_words
from bench.guard import artifacts, pairs
from extract.guard import signals
from extract.types import Word
from tests.guard_words import page
from tests.test_guard_dataset import page_record
from tests.words import WORDS

VOCABULARY = WORDS["vocabulary"]
WIDTH = 10.0


def damaged(word: str, rng: random.Random) -> str:
    """A misreading of a word: a lost accent when it has one, else a letter swapped for a look-alike."""
    folded = word.replace("ó", "o").replace("í", "i").replace("ú", "u").replace("é", "e").replace("á", "a")
    if folded != word:
        return folded
    position = rng.randrange(len(word))
    return word[:position] + rng.choice("xqz") + word[position + 1 :]


def record(doc: str, number: int, rng: random.Random, errors: float = 0.25, words: int = 40, band: str = "medium"):
    """A page record of `words` vocabulary words in lines of ten, a share of them misread."""
    truth, read = [], []
    for i in range(words):
        right = rng.choice(VOCABULARY)
        line, column = divmod(i, 10)
        box = (10.0 + 90 * column, 20.0 * line, 10.0 + 90 * column + WIDTH * len(right), 20.0 * line + 12.0)
        text = damaged(right, rng) if rng.random() < errors else right
        truth.append((right, box))
        read.append(Word(f"p{number}w{i}", number, text, box, 0.9, (0.9,) * len(text), line, column))
    return page_record(doc, number, band, read, truth)


def corpus(documents: int = 30, seed: int = 1, **kwargs) -> list[pairs.Page]:
    rng = random.Random(seed)
    pages = []
    for d in range(documents):
        page = pairs.page_of(record(f"d{d:03d}", 1, rng, **kwargs))
        pages.append(page)
    return pages


def labelled_page(tokens, truth, gap=4.0, conf=0.95, mark_artifacts=False, **extras):
    """A labelled page: `tokens` are the OCR words on one line, `truth` is a list of (text, ocr indexes, part) where
    `part` = (k, m) says the truth word is the k-th of m cut from the box of a single OCR word (a merge)."""
    words = page(" ".join(tokens), gap=gap, conf=conf, **extras)
    read = [(w.id, w.text, w.box) for w in words]
    boxes = []
    for text, members, part in truth:
        left = min(words[i].box[0] for i in members)
        right = max(words[i].box[2] for i in members)
        if part:
            k, m = part
            width = (right - left) / m
            left, right = left + k * width, left + (k + 1) * width
        boxes.append(TruthWord(f"t{len(boxes)}", text, (left, 0.0, right, 12.0)))
    labels = label_words(read, boxes).labels
    found = Counter()
    if mark_artifacts:
        labels, found = artifacts.mark([w.text for w in words], labels)
    return pairs.Page("k", "d1", "light", tuple(words), signals.extract(words), labels, frozenset(), artifacts=found)


def structure_record(
    doc: str, number: int, rng: random.Random, errors: float = 0.1, splits: float = 0.1, words: int = 40
):
    """Like `record`, but a share of the words is cut in two read words (a split) and a share of the damaged ones are two
    truth words read as one (a merge): the pages the structure candidates are for. The words are far enough apart that
    only the split halves are close enough to be joined."""
    truth, read = [], []
    column = 0
    for _ in range(words):
        right = rng.choice([w for w in VOCABULARY if len(w) >= 6])
        x = 10.0 + 14.0 * column
        box = (x, 0.0, x + WIDTH * len(right), 12.0)
        draw = rng.random()
        index = len(read)
        if draw < splits:
            cut = rng.randrange(2, len(right) - 2)
            left_box = (box[0], 0.0, box[0] + WIDTH * cut, 12.0)
            right_box = (box[0] + WIDTH * cut + 2.0, 0.0, box[2] + 2.0, 12.0)
            read.append(Word(f"p{number}w{index}", number, right[:cut], left_box, 0.9, (0.9,) * cut, 0, index))
            read.append(
                Word(
                    f"p{number}w{index + 1}",
                    number,
                    right[cut:],
                    right_box,
                    0.9,
                    (0.9,) * (len(right) - cut),
                    0,
                    index + 1,
                )
            )
            truth.append((right, box))
        else:
            text = damaged(right, rng) if draw < splits + errors else right
            read.append(Word(f"p{number}w{index}", number, text, box, 0.9, (0.9,) * len(text), 0, index))
            truth.append((right, box))
        column += len(right) + 2
    return page_record(doc, number, "medium", read, truth)
