"""How a word is written: its letters, the punctuation attached to its ends, and its case pattern.

Everything the candidate sources and the labelling share about the written form of a word lives here, so that
the guard and the benchmark compare words the same way.
"""

import re
import unicodedata

_PLAIN = re.compile(r"^([^\w]*)([^\W\d_]+)([^\w]*)$")
_INVISIBLE = dict.fromkeys(map(ord, "­​‌‍⁠﻿"))
_TYPOGRAPHIC = {
    **dict.fromkeys(map(ord, "‘’‚‛′"), "'"),
    **dict.fromkeys(map(ord, "“”„‟″"), '"'),
    **dict.fromkeys(map(ord, "‐‑‒–—―−"), "-"),
    **dict.fromkeys(map(ord, "·∙●▪◦•"), "•"),
    **dict.fromkeys(map(ord, "°˚"), "o"),
}


def normalise(text: str) -> str:
    """NFKC, minus the characters a reader cannot see (soft hyphen, zero-width marks), with the typographic variants
    an OCR cannot tell apart folded: curly and straight quotes and apostrophes, every dash and minus to the hyphen,
    every bullet to one bullet and the degree sign to the ordinal indicator's letter. Both sides of a comparison go
    through this."""
    return unicodedata.normalize("NFKC", text).translate(_INVISIBLE).translate(_TYPOGRAPHIC)


def strip_accents(text: str) -> str:
    """The text without any diacritic (the tilde of the Spanish n and the diaeresis of u included), case kept."""
    return "".join(c for c in unicodedata.normalize("NFD", text) if not unicodedata.combining(c))


def split_word(text: str) -> tuple[str, str, str] | None:
    """(attached punctuation before, letters, attached punctuation after), or None when the token is not a plain word."""
    match = _PLAIN.match(text)
    return match.groups() if match else None


def case_pattern(core: str) -> str:
    """`lower`, `title`, `upper` or `irregular`; a single capital counts as upper."""
    if core.islower():
        return "lower"
    if core.isupper():
        return "upper"
    return "title" if core[:1].isupper() and core[1:].islower() else "irregular"


def apply_case(core: str, pattern_of: str) -> str:
    """The word written in the case pattern of another: lower, Title or UPPER (an irregular pattern becomes lower)."""
    pattern = case_pattern(pattern_of)
    if pattern == "upper":
        return core.upper()
    if pattern == "title":
        return core[:1].upper() + core[1:]
    return core


def case_variants(core: str) -> tuple[str, ...]:
    """The three regular writings of a word: lower, Title and UPPER."""
    return (core.lower(), core[:1].upper() + core[1:].lower(), core.upper())
