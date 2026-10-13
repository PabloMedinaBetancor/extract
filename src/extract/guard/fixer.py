"""Spelling candidates for a doubtful word and the features the ranker orders them by.

The fixer only ever proposes: nothing here changes a word. A token with digits, hyphens or apostrophes gets
no candidates. Candidates are the listed words within two edits of the word (the 20 most frequent), the
word's own letters under the other casings, each written with the word's own case pattern and attached
punctuation.
"""

import hashlib
import json
from dataclasses import dataclass

import numpy as np
from rapidfuzz.distance import Levenshtein

from extract.guard import dictionary, lexicon
from extract.guard.dictionary import fold_accents
from extract.guard.edits import weighted_distance
from extract.guard.forms import apply_case, split_word

MAX_CANDIDATES = 20
FEATURES = (
    "weighted_distance",
    "distance",
    "log_frequency",
    "frequency_rank",
    "folded_equal",
    "case_equal",
    "length_difference",
    "lowest_changed_confidence",
    "lm_gain",
    "word_in_dictionary",
    "word_conf_mean",
    "candidates",
)
FEATURE_VERSION = 1


@dataclass(frozen=True)
class Candidate:
    text: str
    core: str
    distance: int
    log_frequency: float


def feature_fingerprint() -> str:
    payload = json.dumps({"version": FEATURE_VERSION, "features": FEATURES})
    return hashlib.sha256(payload.encode()).hexdigest()


def generate(text: str, languages) -> list[Candidate]:
    """The candidates of one word; empty when the token cannot be fixed by spelling."""
    parts = split_word(text)
    if parts is None:
        return []
    prefix, core, suffix = parts
    books = [dictionary.load(language) for language in lexicon.available(languages)]
    found: dict[str, tuple[int, float]] = {}
    for book in books:
        for rank, distance in book.candidates(core)[: MAX_CANDIDATES * 3]:
            word = book.words[rank]
            frequency = float(book.log_freq[rank])
            if word not in found or found[word][1] < frequency:
                found[word] = (distance, frequency)
    ordered = sorted(found.items(), key=lambda item: -item[1][1])[:MAX_CANDIDATES]
    candidates: dict[str, Candidate] = {}
    for word, (distance, frequency) in ordered:
        surface = prefix + apply_case(word, core) + suffix
        if surface != text:
            candidates.setdefault(surface, Candidate(surface, word, distance, frequency))
    for variant in {core.lower(), core.capitalize(), core.upper()} - {core}:
        surface = prefix + variant + suffix
        if surface != text:
            candidates.setdefault(
                surface, Candidate(surface, variant.lower(), 0 if variant.lower() == core.lower() else 1, 0.0)
            )
    return list(candidates.values())


def features(
    text: str, char_conf, conf_mean, in_dictionary: float, candidates: list[Candidate], languages
) -> np.ndarray:
    """One row of ranker features per candidate (the columns of `FEATURES`)."""
    parts = split_word(text)
    core = parts[1] if parts else text
    books = [dictionary.load(language) for language in lexicon.available(languages)]
    word_lm = max((b.char_logp(core) for b in books if b.char_logp(core) is not None), default=None)
    by_frequency = sorted(range(len(candidates)), key=lambda i: -candidates[i].log_frequency)
    rank_of = {index: place for place, index in enumerate(by_frequency)}
    rows = np.full((len(candidates), len(FEATURES)), np.nan, dtype=np.float32)
    for i, candidate in enumerate(candidates):
        lm = max((b.char_logp(candidate.core) for b in books if b.char_logp(candidate.core) is not None), default=None)
        rows[i] = (
            weighted_distance(core, candidate.core),
            Levenshtein.distance(core.casefold(), candidate.core.casefold()),
            candidate.log_frequency,
            rank_of[i],
            float(fold_accents(candidate.core) == fold_accents(core)),
            float(candidate.core.casefold() == core.casefold()),
            len(candidate.core) - len(core),
            _lowest_changed_confidence(core, candidate.core, char_conf, len(parts[0]) if parts else 0),
            np.nan if lm is None or word_lm is None else lm - word_lm,
            in_dictionary,
            np.nan if conf_mean is None else conf_mean,
            len(candidates),
        )
    return rows


def _lowest_changed_confidence(core: str, candidate: str, char_conf, offset: int) -> float:
    """The lowest confidence among the characters of the word that the candidate changes or removes."""
    if not char_conf:
        return float("nan")
    changed = [
        offset + source
        for tag, source, _ in Levenshtein.editops(fold_accents(core), fold_accents(candidate))
        if tag in ("replace", "delete") and offset + source < len(char_conf)
    ]
    return min((char_conf[i] for i in changed), default=float("nan"))
