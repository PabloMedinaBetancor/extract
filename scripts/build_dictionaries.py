"""Builds the guard's word lists, spelling index and character language models.

Run once by a developer; the files it writes are shipped in `src/extract/guard/data`.
wordfreq (CC BY-SA 4.0 data, Apache code) is a build tool, not a dependency of the product, and
no text of the document catalogue is used anywhere here, so the test documents cannot leak into
the dictionary.

    python scripts/build_dictionaries.py [--out src/extract/guard/data]
"""

import argparse
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import wordfreq

from extract.guard.dictionary import INDEX_BUCKETS, LM_BUCKETS, LM_ORDER, PREFIX, bucket, deletes, fold_accents

WORDS_PER_LANGUAGE = 100_000
MAX_WORD_LENGTH = 25
DOMAIN_FREQUENCY = 20
TYPO_RATIO = 8
SINGLE_LETTERS = {"es": set("aeouy"), "en": set("ai")}
DOMAIN_FILE = Path(__file__).with_name("domain_words.json")

CONFUSIONS = [
    ("rn", "m", 0.25),
    ("cl", "d", 0.3),
    ("ii", "u", 0.4),
    ("vv", "w", 0.25),
    ("0", "o", 0.3),
    ("0", "O", 0.3),
    ("1", "l", 0.3),
    ("1", "I", 0.3),
    ("l", "I", 0.2),
    ("5", "S", 0.3),
    ("8", "B", 0.3),
    ("c", "e", 0.5),
    ("a", "o", 0.6),
    ("u", "v", 0.5),
    ("n", "h", 0.6),
    ("t", "f", 0.6),
    ("i", "l", 0.4),
    ("i", "j", 0.4),
    ("a", "\u00e1", 0.2),
    ("e", "\u00e9", 0.2),
    ("i", "\u00ed", 0.2),
    ("o", "\u00f3", 0.2),
    ("u", "\u00fa", 0.2),
    ("u", "\u00fc", 0.2),
    ("n", "\u00f1", 0.3),
    (".", ",", 0.3),
]


def ranked_words(language: str, domain: list[str], typo_ratio: int = TYPO_RATIO) -> list[tuple[str, int]]:
    """The most frequent plain words of a language, most frequent first, with domain words added."""
    frequency = {}
    for word in wordfreq.top_n_list(language, WORDS_PER_LANGUAGE * 2, wordlist="large"):
        if _plain(word, language) and len(frequency) < WORDS_PER_LANGUAGE:
            frequency[word] = round(wordfreq.word_frequency(word, language, wordlist="large") * 1e9)
    for word in domain:
        if _plain(word, language):
            frequency.setdefault(word, DOMAIN_FREQUENCY)
    _drop_unaccented_typos(frequency, typo_ratio)
    return sorted(frequency.items(), key=lambda item: (-item[1], item[0]))


def _drop_unaccented_typos(frequency: dict[str, int], typo_ratio: int) -> None:
    """Social-media text is full of words with the accent left out ("informacion"). A word without
    accents is dropped when an accented spelling of it is at least `typo_ratio` times as frequent, so
    that the guard can still tell a lost accent from a word."""
    best: dict[str, int] = {}
    for word, freq in frequency.items():
        folded = fold_accents(word)
        if folded != word:
            best[folded] = max(best.get(folded, 0), freq)
    for word in [w for w in frequency if w in best]:
        if frequency[word] * typo_ratio <= best[word]:
            del frequency[word]


def _plain(word: str, language: str) -> bool:
    if not word.isalpha() or len(word) > MAX_WORD_LENGTH:
        return False
    return len(word) > 1 or word in SINGLE_LETTERS[language]


def build_symmetric_delete_index(words: list[str]) -> np.ndarray:
    """One uint32 array: bucket offsets (INDEX_BUCKETS + 1 of them), then word ranks grouped by bucket.

    Every word is filed under the buckets of all deletions (up to two) of its first PREFIX
    characters. A query finds its candidates from its own deletions and checks them with the real
    distance, so hash collisions only cost time.
    """
    pairs = []
    for rank, word in enumerate(words):
        for text in deletes(word[:PREFIX]):
            pairs.append((bucket(text, INDEX_BUCKETS), rank))
    pairs = np.unique(np.array(pairs, dtype=np.uint32), axis=0)
    counts = np.bincount(pairs[:, 0], minlength=INDEX_BUCKETS)
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.uint32)
    return np.concatenate([offsets, pairs[np.argsort(pairs[:, 0], kind="stable")][:, 1]]).astype(np.uint32)


def build_char_model(entries: list[tuple[str, int]]) -> np.ndarray:
    """Hashed counts of 4-grams and of their 3-character contexts, weighted by log-frequency."""
    grams = defaultdict(float)
    contexts = defaultdict(float)
    for word, freq in entries:
        weight = math.log10(freq + 1) + 1.0
        padded = "^" * (LM_ORDER - 1) + word + "$"
        for i in range(len(padded) - LM_ORDER + 1):
            grams[padded[i : i + LM_ORDER]] += weight
            contexts[padded[i : i + LM_ORDER - 1]] += weight
    table = np.zeros((2, LM_BUCKETS), dtype=np.float32)
    for gram, count in grams.items():
        table[0, bucket(gram, LM_BUCKETS)] += count
    for context, count in contexts.items():
        table[1, bucket(context, LM_BUCKETS)] += count
    return table


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path("src/extract/guard/data"))
    parser.add_argument(
        "--typo-ratio", type=int, default=TYPO_RATIO, help="drop unaccented words this many times rarer"
    )
    args = parser.parse_args()
    domain = json.loads(DOMAIN_FILE.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    for language in ("es", "en"):
        entries = ranked_words(language, domain[language], args.typo_ratio)
        with gzip.GzipFile(args.out / f"dict_{language}.tsv.gz", "wb", mtime=0) as handle:
            handle.write("".join(f"{word}\t{freq}\n" for word, freq in entries).encode("utf-8"))
        np.save(args.out / f"symdel_{language}.npy", build_symmetric_delete_index([w for w, _ in entries]))
        np.save(args.out / f"charlm_{language}.npy", build_char_model(entries))
        print(language, len(entries), "words")
    (args.out / "confusions.json").write_text(
        json.dumps([{"a": a, "b": b, "cost": cost} for a, b, cost in CONFUSIONS], indent=1, ensure_ascii=True),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
