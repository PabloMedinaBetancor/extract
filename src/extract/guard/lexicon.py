"""The word sets of the supported languages, loaded on first use.

This module imports nothing heavy (no numpy, LightGBM or rapidfuzz): the engine's main process
uses it to decide whether a page's native text layer is readable, before any page is handed to a
worker. Words are compared case-insensitively and with their accents.
"""

import gzip
from pathlib import Path

DATA_DIR = Path(__file__).with_name("data")

_SETS: dict[str, frozenset[str]] = {}


class NoLexicon(LookupError):
    """None of the requested languages has a word list."""


def available(languages) -> tuple[str, ...]:
    """The requested languages that have a word list."""
    return tuple(language for language in languages if (DATA_DIR / f"dict_{language}.tsv.gz").is_file())


def has(word: str, languages=("es", "en")) -> bool:
    """Whether the word is in the list of any of the languages."""
    folded = word.casefold()
    return any(folded in _words(language) for language in _usable(languages))


def share(words, languages=("es", "en")) -> float:
    """The fraction of the words found in the list of any of the languages; 0.0 for no words."""
    sets = [_words(language) for language in _usable(languages)]
    words = [w.casefold() for w in words]
    if not words:
        return 0.0
    return sum(any(w in s for s in sets) for w in words) / len(words)


def _usable(languages) -> tuple[str, ...]:
    usable = available(languages)
    if not usable:
        raise NoLexicon(f"no word list for {tuple(languages)}")
    return usable


def _words(language: str) -> frozenset[str]:
    if language not in _SETS:
        with gzip.open(DATA_DIR / f"dict_{language}.tsv.gz", "rt", encoding="utf-8") as handle:
            _SETS[language] = frozenset(line.split("\t", 1)[0] for line in handle)
    return _SETS[language]
