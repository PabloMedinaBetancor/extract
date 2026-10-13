"""Word lists with frequencies, nearest-word search and a character language model.

The files in `data/` are built by `scripts/build_dictionaries.py`. Words are ranked by frequency
(rank 0 is the most frequent), and every structure is addressed by rank.

Spelling search is a symmetric-delete index: a word is filed under the hash of every deletion (up
to two characters) of its first PREFIX characters, so the candidates of a query are the words that
share one of its own deletions. Candidates are checked with the real distance, so a hash collision
only costs a little time, and no word within two edits is ever missed.
"""

import gzip
import unicodedata
import zlib
from functools import lru_cache

import numpy as np
from rapidfuzz import process
from rapidfuzz.distance import Levenshtein

from extract.guard.lexicon import DATA_DIR

PREFIX = 7
INDEX_BUCKETS = 1 << 21
LM_BUCKETS = 1 << 20
LM_ORDER = 4
LM_SMOOTHING = 0.05
LM_ALPHABET = 32
MAX_DISTANCE = 2
CACHE_SIZE = 65536


@lru_cache(maxsize=1 << 17)
def fold_accents(word: str) -> str:
    """Lower case without any diacritic: accents, the tilde of the Spanish n and the diaeresis of u go too, because
    an OCR drops them all the same way."""
    if word.isascii():
        return word.casefold()
    return "".join(c for c in unicodedata.normalize("NFD", word.casefold()) if not unicodedata.combining(c))


def deletes(word: str, depth: int = MAX_DISTANCE) -> set[str]:
    seen, layer = {word}, {word}
    for _ in range(depth):
        layer = {s[:i] + s[i + 1 :] for s in layer for i in range(len(s))}
        seen |= layer
    return seen


def bucket(text: str, buckets: int) -> int:
    return zlib.crc32(text.encode("utf-8")) & (buckets - 1)


class Dictionary:
    """The word list of one language."""

    def __init__(self, language: str) -> None:
        self.language = language
        words, freqs = [], []
        with gzip.open(DATA_DIR / f"dict_{language}.tsv.gz", "rt", encoding="utf-8") as handle:
            for line in handle:
                word, freq = line.rstrip("\n").split("\t")
                words.append(word)
                freqs.append(int(freq))
        self.words = words
        self.log_freq = np.log10(np.array(freqs, dtype=np.float64) + 1.0)
        self.rank = {word: i for i, word in enumerate(words)}
        self.folded = {fold_accents(word) for word in words}
        index = np.load(DATA_DIR / f"symdel_{language}.npy", mmap_mode="r").view(np.ndarray)
        self._offsets = index[: INDEX_BUCKETS + 1]
        self._ids = index[INDEX_BUCKETS + 1 :]
        self._model = np.load(DATA_DIR / f"charlm_{language}.npy")
        self._lengths = np.array([len(w) for w in words], dtype=np.uint8)
        self._cached_candidates = lru_cache(maxsize=CACHE_SIZE)(self._find_candidates)
        self._logp_memo: dict[str, float | None] = {}
        self._by_fold: dict[str, list[str]] | None = None

    def clear_caches(self) -> None:
        """Forget the per-word results (the reader calls this when it is done with a document)."""
        self._cached_candidates.cache_clear()
        self._logp_memo.clear()

    def rank_of(self, word: str) -> int:
        """Rank of the word ignoring case, or -1."""
        return self.rank.get(word.casefold(), -1)

    def has_folded(self, word: str) -> bool:
        return fold_accents(word) in self.folded

    def same_fold(self, word: str) -> list[str]:
        """The listed words that differ from `word` only in case and accents, most frequent first."""
        if self._by_fold is None:
            index: dict[str, list[str]] = {}
            for listed in self.words:
                index.setdefault(fold_accents(listed), []).append(listed)
            self._by_fold = index
        return self._by_fold.get(fold_accents(word), [])

    def log_frequency(self, word: str) -> float:
        rank = self.rank_of(word)
        return float(self.log_freq[rank]) if rank >= 0 else 0.0

    def candidates(self, word: str, max_distance: int = MAX_DISTANCE) -> list[tuple[int, int]]:
        """(rank, distance) of every listed word within `max_distance` edits, nearest first, then most frequent."""
        return list(self._cached_candidates(word.casefold(), max_distance))

    def _find_candidates(self, word: str, max_distance: int) -> tuple[tuple[int, int], ...]:
        mask = INDEX_BUCKETS - 1
        offsets, ids = self._offsets, self._ids
        pieces = []
        for text in deletes(word[:PREFIX], max_distance):
            bucket = zlib.crc32(text.encode("utf-8")) & mask
            pieces.append(ids[offsets[bucket] : offsets[bucket + 1]])
        found = np.unique(np.concatenate(pieces))
        found = found[np.abs(self._lengths[found].astype(np.int16) - len(word)) <= max_distance]
        if not found.size:
            return ()
        distances = process.cdist(
            [word],
            [self.words[r] for r in found.tolist()],
            scorer=Levenshtein.distance,
            score_cutoff=max_distance,
            dtype=np.uint8,
        )[0]
        near = np.nonzero(distances <= max_distance)[0]
        return tuple(
            sorted(zip(found[near].tolist(), distances[near].tolist(), strict=True), key=lambda p: (p[1], p[0]))
        )

    def char_logp(self, word: str) -> float | None:
        """Mean log-probability per character of the letters of the word under the 4-gram model."""
        return self.char_logp_many([word])[0]

    def char_logp_many(self, words: list[str]) -> list[float | None]:
        """`char_logp` of many words at once."""
        scored = self.score(GramBatch(words))
        return [scored.get(word) for word in words]

    def score(self, batch: "GramBatch") -> dict[str, float | None]:
        """The mean log-probability of every word of a batch; the model is read in one pass over all their 4-grams."""
        out: dict[str, float | None] = {}
        missing = []
        for word in batch.words:
            if word in self._logp_memo:
                out[word] = self._logp_memo[word]
            else:
                missing.append(word)
        if missing:
            wanted = set(missing)
            keep = [i for i, word in enumerate(batch.words) if word in wanted]
            rows = np.concatenate([np.arange(batch.starts[i], batch.starts[i] + batch.lengths[i]) for i in keep])
            seen = self._model[0, batch.grams[rows]].astype(np.float64)
            context = self._model[1, batch.contexts[rows]].astype(np.float64)
            logs = np.log((seen + LM_SMOOTHING) / (context + LM_SMOOTHING * LM_ALPHABET))
            offsets = np.concatenate([[0], np.cumsum([batch.lengths[i] for i in keep])[:-1]])
            means = np.add.reduceat(logs, offsets) / np.array([batch.lengths[i] for i in keep])
            for i, mean in zip(keep, means.tolist(), strict=True):
                out[batch.words[i]] = self._logp_memo[batch.words[i]] = mean
            if len(self._logp_memo) > CACHE_SIZE:
                self._logp_memo.clear()
        for word in batch.skipped:
            out[word] = None
        return out


class GramBatch:
    """The hashed 4-grams and 3-character contexts of the letters of many words, computed once for every language."""

    def __init__(self, words) -> None:
        mask = LM_BUCKETS - 1
        self.words: list[str] = []
        self.skipped: list[str] = []
        grams: list[int] = []
        contexts: list[int] = []
        self.starts: list[int] = []
        self.lengths: list[int] = []
        for word in dict.fromkeys(words):
            letters = "".join(c for c in word.casefold() if c.isalpha())
            if not letters:
                self.skipped.append(word)
                continue
            padded = "^" * (LM_ORDER - 1) + letters + "$"
            count = len(padded) - LM_ORDER + 1
            self.words.append(word)
            self.starts.append(len(grams))
            self.lengths.append(count)
            for i in range(count):
                grams.append(zlib.crc32(padded[i : i + LM_ORDER].encode("utf-8")) & mask)
                contexts.append(zlib.crc32(padded[i : i + LM_ORDER - 1].encode("utf-8")) & mask)
        self.grams = np.array(grams, dtype=np.int64)
        self.contexts = np.array(contexts, dtype=np.int64)


@lru_cache(maxsize=None)
def load(language: str) -> Dictionary:
    return Dictionary(language)
