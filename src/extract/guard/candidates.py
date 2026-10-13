"""Candidate texts for the words of a page, from several sources.

A site is a plain token (letters with punctuation attached at its ends) that no protection rule excludes. For a
site the candidates are the token as read (always first) and what each source proposes: the dictionary, the
accent and case restorations, the punctuation variants and the vocabulary of the page. The result depends only on
the OCR output and the word lists: no truth word is ever read here.
"""

import math
from collections import Counter
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from extract.guard import dictionary, lexicon
from extract.guard.dictionary import fold_accents
from extract.guard.edits import weighted_distance
from extract.guard.forms import apply_case, case_variants, normalise, split_word, strip_accents
from extract.types import Word

MAX_CANDIDATES = 16
DICTIONARY_POOL = 20
NEAR_POOL = 60
CASE_VARIANT_WORDS = 5
COMBINED_BASES = 3
COMBINED_MARKS = (",", ".")
COMBINED_DISTANCE = 1.0
WRITTEN_EDIT_WEIGHT = 0.1
MAX_LENGTH = 40
PAGE_EDITS = 2
PAGE_MIN_COUNT = 2
PUNCTUATION = (",", ".", ";", ":", ")", "?")
BLOCKED_CHARACTERS = frozenset("@/_#\\")
FINAL_MARK = ":"
URL_SCHEMES = frozenset({"http", "https", "ftp", "ftps", "mailto", "www", "file"})
ACRONYM_LENGTH = 5
ACRONYM_REPEATS = 3
CONFIDENT = 0.97

CTC_POSITIONS = 3
CTC_SUBSTITUTIONS = 2
CTC_CANDIDATES = 12

KEEP = "keep"
SOURCE_ORDER = (
    "dictionary",
    "fold",
    "punct",
    "page",
    "ctc_alt",
    "masked",
    "reading",
    "joined",
    "join_edit",
    "segment",
    "drop",
)


@dataclass(frozen=True)
class Candidate:
    """One proposal for a site. `sources` lists every source that proposed this text, in `SOURCE_ORDER`; `span` holds
    the indexes of the words the proposal replaces (the site's word for a text edit, a run of words for a join)."""

    text: str
    op: str
    kind: str
    sources: tuple[str, ...]
    span: tuple[int, ...] = ()


class PageContext:
    """What the sources know about one page: its tokens split into letters and punctuation, and its vocabulary."""

    def __init__(self, words: Sequence[Word], languages=("es", "en"), fragments: Collection[int] = ()) -> None:
        self.words = words
        self.books = [dictionary.load(language) for language in lexicon.available(languages)]
        self.fragments = frozenset(fragments)
        self.parts = [split_word(w.text) for w in words]
        written = Counter(parts[1] for parts in self.parts if parts)
        self.written = written
        folded = Counter()
        self.written_forms: dict[str, list[str]] = {}
        for core, count in sorted(written.items()):
            folded[core.casefold()] += count
            self.written_forms.setdefault(core.casefold(), []).append(core)
        self.counts = folded
        self.confident_counts = Counter(
            parts[1].casefold()
            for word, parts in zip(words, self.parts, strict=True)
            if parts and word.char_conf and min(word.char_conf) > CONFIDENT
        )
        cores = [parts[1] for parts in self.parts if parts]
        self.dictionary_share = lexicon.share(cores, languages) if cores and self.books else math.nan
        self.vocabulary = [core for core, count in sorted(folded.items()) if count >= PAGE_MIN_COUNT]
        self._near_vocabulary = lru_cache(maxsize=4096)(self._find_near_vocabulary)
        self._fold_group = lru_cache(maxsize=4096)(self._find_fold_group)
        self.nearest: dict[int, list[str]] = {}
        self._line_heights: dict[int, float] = {}
        self.piece_words: dict[str, tuple[tuple[str, int], ...]] = {}

    def is_site(self, index: int) -> bool:
        """Whether the word may be edited at all: a plain token that none of the protection rules excludes."""
        parts = self.parts[index]
        text = self.words[index].text
        if parts is None or index in self.fragments or len(text) > MAX_LENGTH or BLOCKED_CHARACTERS & set(text):
            return False
        prefix, core, suffix = parts
        # A hyphen or an apostrophe at either end belongs to a joined or elided word (`obra-`, `l'`, `d'`), and a
        # colon only closes a word when it is the very last character (`Urbanisticos:`, never `http:`).
        marks = normalise(prefix + suffix)
        if "-" in marks or "'" in marks or FINAL_MARK in prefix or FINAL_MARK in suffix[:-1]:
            return False
        if FINAL_MARK in suffix and core.casefold() in URL_SCHEMES:
            return False
        capitals = sum(1 for c in core if c.isupper())
        if capitals >= 2 and not core.isupper():
            return False
        return not (core.isupper() and len(core) <= ACRONYM_LENGTH and self.written[core] >= ACRONYM_REPEATS)

    def fold_group(self, letters: str) -> frozenset[str]:
        """The listed words (any language of the page) that differ from `letters` only in case and accents."""
        return self._fold_group(letters)

    def _find_fold_group(self, letters: str) -> frozenset[str]:
        return frozenset(word for book in self.books for word in book.same_fold(letters))

    def line_height(self, index: int) -> float:
        """The median height of the boxes of the line the word is on."""
        line = self.words[index].line_id
        if line not in self._line_heights:
            heights = sorted(w.box[3] - w.box[1] for w in self.words if w.line_id == line)
            self._line_heights[line] = heights[len(heights) // 2]
        return self._line_heights[line]

    def ends_line(self, index: int) -> bool:
        line = self.words[index].line_id
        return index + 1 >= len(self.words) or self.words[index + 1].line_id != line

    def next_starts_sentence(self, index: int) -> bool:
        if self.ends_line(index):
            return False
        following = self.parts[index + 1]
        return bool(following and following[1][:1].isupper())

    def near_vocabulary(self, core: str) -> tuple[tuple[str, int], ...]:
        """(word, edits) of the page's repeated words within `PAGE_EDITS` of the letters, nearest first."""
        return self._near_vocabulary(core.casefold())

    def _find_near_vocabulary(self, folded: str) -> tuple[tuple[str, int], ...]:
        if not self.vocabulary:
            return ()
        distances = process.cdist(
            [folded], self.vocabulary, scorer=Levenshtein.distance, score_cutoff=PAGE_EDITS, dtype=np.uint8
        )[0]
        near = [(self.vocabulary[i], int(d)) for i, d in enumerate(distances) if 0 < d <= PAGE_EDITS]
        return tuple(sorted(near, key=lambda item: (item[1], -self.counts[item[0]], item[0])))


def edit_kind(token: str, candidate: str) -> str:
    """The family of a text edit: punct, case, accent, accent_case or letters."""
    before, after = split_word(token), split_word(candidate)
    if before is None or after is None:
        return "letters"
    old, new = before[1], after[1]
    if old == new:
        return "punct"
    if old.casefold() == new.casefold():
        return "case"
    if strip_accents(old) == strip_accents(new):
        return "accent"
    return "accent_case" if fold_accents(old) == fold_accents(new) else "letters"


def dictionary_texts(ctx: PageContext, index: int) -> list[str]:
    """S1: of the listed words within two edits, the nearest by weighted distance (the most frequent among equals),
    written in the token's form."""
    if index in ctx.nearest:
        return ctx.nearest[index]
    prefix, core, suffix = ctx.parts[index]
    found: dict[str, float] = {}
    for book in ctx.books:
        for rank, _ in book.candidates(core)[:NEAR_POOL]:
            word = book.words[rank]
            found[word] = max(found.get(word, 0.0), float(book.log_freq[rank]))
    ordered = sorted(found, key=lambda word: (weighted_distance(core, word), -found[word], word))[:DICTIONARY_POOL]
    texts = [prefix + apply_case(word, core) + suffix for word in ordered]
    for word in ordered[:CASE_VARIANT_WORDS]:
        texts.extend(prefix + variant + suffix for variant in case_variants(word))
    ctx.nearest[index] = list(dict.fromkeys(texts))
    return ctx.nearest[index]


def fold_texts(ctx: PageContext, index: int) -> list[str]:
    """S2: the token's letters, and every listed word with the same letters and accents folded, in each regular case."""
    prefix, core, suffix = ctx.parts[index]
    words = [core, *sorted(ctx.fold_group(core))]
    seen: dict[str, None] = {}
    for word in words:
        for variant in case_variants(word):
            seen[prefix + variant + suffix] = None
    return list(seen)


def punctuation_texts(ctx: PageContext, index: int, context_rule: bool = True) -> list[str]:
    """S3: the token with its trailing punctuation replaced by each usual mark or removed (a mark is added only
    where the sentence or the line ends, with `context_rule`), and with a stray mark in front of it dropped. The
    same marks also go on the nearest dictionary words, because a misread letter often comes with a misread comma."""
    prefix, core, suffix = ctx.parts[index]
    texts = []
    if suffix or not context_rule or ctx.ends_line(index) or ctx.next_starts_sentence(index):
        texts = [prefix + core + mark for mark in (*PUNCTUATION, "") if mark != suffix]
    if prefix:
        texts.append(core + suffix)
    for base in _combination_bases(ctx, index):
        texts.extend(prefix + base + mark for mark in (*COMBINED_MARKS, "") if mark != suffix)
        if prefix:
            texts.append(base + suffix)
    return texts


def _combination_bases(ctx: PageContext, index: int) -> list[str]:
    """The nearest listed words, written in the token's case, that the marks are combined with."""
    _, core, _ = ctx.parts[index]
    bases = []
    for text in dictionary_texts(ctx, index)[:COMBINED_BASES]:
        letters = split_word(text)[1]
        if letters != core and weighted_distance(core, letters) <= COMBINED_DISTANCE:
            bases.append(letters)
    return bases


def page_texts(ctx: PageContext, index: int) -> list[str]:
    """S4: the repeated words of the same page within two edits, in the token's form and in the page's own."""
    prefix, core, suffix = ctx.parts[index]
    texts: dict[str, None] = {}
    for word, _ in ctx.near_vocabulary(core):
        texts[prefix + apply_case(word, core) + suffix] = None
        for written in ctx.written_forms[word]:
            texts[prefix + written + suffix] = None
    return list(texts)


def engine_alternative_texts(word: Word) -> list[str]:
    """S5: the token with up to `CTC_SUBSTITUTIONS` of its `CTC_POSITIONS` least certain characters replaced by the
    classes the recogniser put next, the likeliest first; none when the word carries no alternatives."""
    alternatives = getattr(word, "char_alts", None)
    text = word.text
    if not alternatives or len(alternatives) != len(text):
        return []
    doubtful = sorted(
        (i for i, options in enumerate(alternatives) if len(options) > 1), key=lambda i: (alternatives[i][0][1], i)
    )[:CTC_POSITIONS]
    swaps = [(i, char, prob) for i in doubtful for char, prob in alternatives[i][1:]]
    proposals: dict[str, float] = {}
    for first in range(len(swaps)):
        _place(proposals, text, [swaps[first]])
        for second in range(first + 1, len(swaps)):
            if swaps[first][0] != swaps[second][0]:
                _place(proposals, text, [swaps[first], swaps[second]])
    ordered = sorted(proposals, key=lambda found: (-proposals[found], found))
    return [found for found in ordered if split_word(found) is not None][:CTC_CANDIDATES]


def _place(proposals: dict[str, float], text: str, swaps) -> None:
    chars = list(text)
    probability = 1.0
    for index, char, prob in swaps:
        chars[index] = char
        probability *= prob
    proposals.setdefault("".join(chars), probability)


def joins_words(reading) -> bool:
    """Whether a reader saw the word together with its neighbours: its `group` holds two or three first-reading words.
    A reading of one word alone has the word itself as its group (the aligner always says which words it saw as one)."""
    return len(reading.group) > 1


def reading_texts(word: Word) -> dict[str, list[str]]:
    """S6: what the other readers and the Latin-set decode read for the word, by source, when it differs from the
    token. A reader that cut the word or read it together with its neighbours is evidence for a structure edit, not
    a text. (A reading aligned one to one has the word's own id as its group; only a group of two or more is a join.)"""
    found: dict[str, list[str]] = {}
    masked = getattr(word, "masked", None)
    if masked is not None and masked != word.text and split_word(masked) is not None:
        found["masked"] = [masked]
    for reading in getattr(word, "readings", None) or ():
        text = reading.text
        if text is None or joins_words(reading) or text == word.text or split_word(text) is None:
            continue
        found.setdefault(f"reading:{reading.source}", []).append(text)
    return found


def by_source(ctx: PageContext, index: int, context_rule: bool = True) -> dict[str, list[str]]:
    """The texts each source proposes for a site, before deduplication and before the cap. The engine's sources
    appear only when the word carries what they read."""
    word = ctx.words[index]
    out = {
        "dictionary": dictionary_texts(ctx, index),
        "fold": fold_texts(ctx, index),
        "punct": punctuation_texts(ctx, index, context_rule),
        "page": page_texts(ctx, index),
    }
    if alternatives := engine_alternative_texts(word):
        out["ctc_alt"] = alternatives
    out.update(reading_texts(word))
    return out


def generate(ctx: PageContext, index: int) -> list[Candidate]:
    """The candidates of a site: the token as read first, then at most `MAX_CANDIDATES` - 1 others, one per
    normalised written form, each with every source that proposed it. The order is fixed: most sources, then the
    nearest (weighted distance of the letters plus a tenth for every character the written form changes), then
    the most frequent word, then the text."""
    token = ctx.words[index].text
    token_key = normalise(token)
    proposed: dict[str, tuple[str, list[str]]] = {}
    for source, texts in by_source(ctx, index).items():
        for text in texts:
            key = normalise(text)
            if key == token_key:
                continue
            sources = proposed.setdefault(key, (text, []))[1]
            if source not in sources:
                sources.append(source)
    core = ctx.parts[index][1]

    def order(item: tuple[str, tuple[str, list[str]]]) -> tuple:
        text, sources = item[1]
        letters = split_word(text)[1]
        written = weighted_distance(core, letters) + WRITTEN_EDIT_WEIGHT * Levenshtein.distance(token, text)
        return (-len(sources), written, -_frequency(ctx, letters), text)

    chosen = sorted(proposed.items(), key=order)[: MAX_CANDIDATES - 1]
    span = (index,)
    out = [Candidate(token, "keep", KEEP, (KEEP,), span)]
    for _, (text, sources) in chosen:
        out.append(Candidate(text, "text", edit_kind(token, text), tuple(sorted(sources, key=source_rank)), span))
    return out


def source_rank(source: str) -> tuple[int, str]:
    return SOURCE_ORDER.index(source.split(":")[0]), source


def _frequency(ctx: PageContext, letters: str) -> float:
    return max((book.log_frequency(letters) for book in ctx.books), default=0.0)
