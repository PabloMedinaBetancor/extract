"""The language of a page, sensed from its function words.

The word lists cannot tell Spanish from English (88 % of the words of an English sentence are in the Spanish list), so
the guard counts the share of the page's tokens that belong to a short list of function words of each language. The two
lists are disjoint: a word that is both (`a`, `no`, `me`, `he`, `son`, `has`...) is in neither. `es` when the Spanish share
reaches the floor and is at least twice the English one, `en` the other way round, else `unsure`: a page of numbers, of
non-words or of too few words says nothing. Pure python, a function of the tokens only: no model, no numpy.
"""

import re
from collections.abc import Sequence

ES = frozenset(
    """el la los las un una unos unas de del al y o u que en con por para es está están como se su sus lo le les más
    pero sin sobre entre este esta estos estas ese esa esos esas muy ya también cuando donde porque según hasta desde
    durante mediante cada todo todos todas otro otra otros otras ser fue han ha sido hay si ni tiene tienen puede pueden
    debe deben nos mi tu ante bajo contra tras cual cuales quien quienes cuyo qué cómo aquí así aún""".split()
)
EN = frozenset(
    """the of and to in is that for it as was with be by on not this are or his from at which but have an had they you
    were their one all we her there been if more when will would who so what its also than other into only
    these those such any may shall should must about after between each under over through during both where while
    because whether our your them then out up some could do does did being those""".split()
)
# the words that are in both languages (`a`, `no`, `me`, `he`, `son`, `has`, `solo`, `can`...) are in neither list

UNSURE = "unsure"
MIN_TOKENS = 20  # under this many words with letters the function words say nothing
DEFAULT_FLOOR = 0.12  # replaced by the floor fitted on the development pages and written in the model file
RATIO = 2.0
# the second stage: the character models of the two word lists. `LM_EN` and `LM_ES` are on the mean per word of
# `char_logp(en) - char_logp(es)`: at or over `LM_EN` the words are English, at or under `LM_ES` Spanish, between them
# nothing is said. Measured by B on D (1st-99th percentile of the OCR words): catalogue es [-0.60, -0.23], invoices es
# [-0.57, -0.24], invoices en [+0.11, +0.27]; French +0.07 and Catalan -0.07 fall between. To be fitted on the language of
# the manifest once there is one for each language: not on the models' own labels.
LM_EN = 0.05
LM_ES = -0.10
LM_MIN_LETTERS = 2
LM_MIN_TOKENS = 10  # under this many words with letters the character models say nothing either
LM_ALONE_BELOW = 100  # a page of this many words or more is prose: if the function words say nothing, nothing is said

_LETTERS = re.compile(r"[^\W\d_]+")


def tokens(words: Sequence) -> list[str]:
    """The casefolded letters of every word that has any (punctuation and digits dropped; a word with none is no token)."""
    out = []
    for word in words:
        parts = _LETTERS.findall(word.text if hasattr(word, "text") else word)
        if parts:
            out.append("".join(parts).casefold())
    return out


def shares(words: Sequence) -> tuple[float, float, int]:
    """(Spanish share, English share, number of tokens): the share of tokens that are function words of each language."""
    found = tokens(words)
    if not found:
        return 0.0, 0.0, 0
    return (
        sum(t in ES for t in found) / len(found),
        sum(t in EN for t in found) / len(found),
        len(found),
    )


def char_difference(words: Sequence) -> float | None:
    """The mean over the words of `char_logp(en) - char_logp(es)`: positive for English, negative for Spanish. Only the
    words of at least `LM_MIN_LETTERS` letters and no digit count; None when none does."""
    from extract.guard import dictionary

    letters = []
    for word in words:
        text = word.text if hasattr(word, "text") else word
        if any(c.isdigit() for c in text):
            continue
        found = "".join(_LETTERS.findall(text)).casefold()
        if len(found) >= LM_MIN_LETTERS:
            letters.append(found)
    if not letters:
        return None
    english = dictionary.load("en").char_logp_many(letters)
    spanish = dictionary.load("es").char_logp_many(letters)
    pairs = [(e, s) for e, s in zip(english, spanish, strict=True) if e is not None and s is not None]
    if not pairs:
        return None
    return float(sum(e - s for e, s in pairs) / len(pairs))


def _floor(floor: float | dict, language: str) -> float:
    return floor.get(language, DEFAULT_FLOOR) if isinstance(floor, dict) else floor


def function_language(words: Sequence, floor: float | dict = DEFAULT_FLOOR, min_tokens: int = MIN_TOKENS) -> str:
    """First stage: `es`, `en` or `unsure` from the function words alone: the language whose share reaches its floor and
    is at least twice the other's. `floor` is one number or one per language."""
    es, en, count = shares(words)
    if count < min_tokens:
        return UNSURE
    if es >= _floor(floor, "es") and es >= RATIO * en:
        return "es"
    if en >= _floor(floor, "en") and en >= RATIO * es:
        return "en"
    return UNSURE


def character_language(difference: float | None, lm_en: float = LM_EN, lm_es: float = LM_ES) -> str:
    """Second stage: `en` at or over `lm_en`, `es` at or under `lm_es`, else `unsure` (and for no measure at all)."""
    if difference is None:
        return UNSURE
    if difference >= lm_en:
        return "en"
    if difference <= lm_es:
        return "es"
    return UNSURE


def page_language(
    words: Sequence,
    floor: float | dict = DEFAULT_FLOOR,
    min_tokens: int = MIN_TOKENS,
    lm_en: float = LM_EN,
    lm_es: float = LM_ES,
    second_stage: bool = True,
    alone_below: int = LM_ALONE_BELOW,
) -> str:
    """`es`, `en` or `unsure`. The function words speak first; the character models speak alone when those say `unsure`
    on a page of fewer than `alone_below` words (an invoice of labels; a long page that the function words cannot place is
    another language or no prose, and stays `unsure`), and a language the function words give is kept only
    if the character models say the same, or have nothing to say for lack of words: a page they contradict or cannot place
    is `unsure`. French and Catalan, full of Spanish function words, fall there. A page of another language
    that both stages place in Spanish (Portuguese) cannot be told apart; the README says so."""
    first = function_language(words, floor, min_tokens)
    if not second_stage or len(tokens(words)) < LM_MIN_TOKENS:
        return first
    difference = char_difference(words)
    if difference is None:
        return first
    second = character_language(difference, lm_en, lm_es)
    if first == UNSURE:
        return second if len(tokens(words)) < alone_below else UNSURE
    return first if second == first else UNSURE


def dictionaries(sensed: str | None, declared: Sequence[str]) -> tuple[str, ...]:
    """The word lists a page's candidates come from: the list of its sensed language when that is one of the declared
    ones (a Spanish page never gets English candidates), else all the declared lists."""
    if sensed in declared:
        return (sensed,)
    return tuple(declared)


def sensor_settings(sensor: "float | dict | None" = None) -> dict:
    """The settings of the sensor as the model file keeps them: a floor (one number or one per language), the minimum
    number of words of the function-word stage, the share ratio, the two thresholds of the character stage and the size
    from which a page of unplaced prose stays `unsure`. A bare number is the floor."""
    given = {"floor": sensor} if isinstance(sensor, (int, float)) else dict(sensor or {})
    return {
        "floor": given.get("floor", DEFAULT_FLOOR),
        "min_tokens": given.get("min_tokens", MIN_TOKENS),
        "ratio": given.get("ratio", RATIO),
        "lm_en": given.get("lm_en", LM_EN),
        "lm_es": given.get("lm_es", LM_ES),
        "alone_below": given.get("alone_below", LM_ALONE_BELOW),
        "second_stage": given.get("second_stage", True),
    }


def sensed_language(words: Sequence, sensor: "float | dict | None" = None) -> str:
    """`page_language` with the settings of a sensor (see `sensor_settings`)."""
    s = sensor_settings(sensor)
    return page_language(
        words,
        s["floor"],
        s["min_tokens"],
        lm_en=s["lm_en"],
        lm_es=s["lm_es"],
        second_stage=s["second_stage"],
        alone_below=s["alone_below"],
    )
