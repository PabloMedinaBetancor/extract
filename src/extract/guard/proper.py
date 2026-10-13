"""Proper names, by code: a token that looks like a surname, a name, a street or a company and that no word list
knows. `safe` never changes, joins or splits such a token (a name has no spelling to restore from a dictionary: the nearest
listed word is a different name); `more` is for the reviewer.
"""

from collections.abc import Sequence

from extract.guard import lexicon
from extract.guard.forms import split_word

TITLE = "title_not_listed"
UPPER = "upper_not_listed"
TITLE_LISTED = "title_listed"
OTHER = "other"
CLASSES = (TITLE, UPPER, TITLE_LISTED, OTHER)
PROTECTED_IN_SAFE = frozenset({TITLE, UPPER})
UPPER_MIN_LETTERS = 3


def proper_class(token: str) -> str:
    """`title_not_listed`: an initial capital and the rest not all capitals, in neither word list; `upper_not_listed`: all
    capitals, three letters or more, in neither list (a surname written in capitals); `title_listed`: an initial capital
    that the lists know (a word at the start of a sentence); `other`: everything else, a token that is not a plain word
    included."""
    parts = split_word(token)
    if parts is None:
        return OTHER
    core = parts[1]
    listed = lexicon.has(core, ("es", "en"))
    if core.isupper() and len(core) >= UPPER_MIN_LETTERS:
        return OTHER if listed else UPPER
    if core[:1].isupper() and not core.isupper():
        return TITLE_LISTED if listed else TITLE
    return OTHER


def protected_in_safe(tokens: Sequence[str]) -> bool:
    """Whether any token of the span of an edit is a proper name that `safe` leaves alone."""
    return any(proper_class(t) in PROTECTED_IN_SAFE for t in tokens)
