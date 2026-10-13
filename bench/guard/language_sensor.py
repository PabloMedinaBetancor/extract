"""The page-language sensor against the language of the documents: the confusion matrix over the development
pages, the floor fitted on them, and the two floors of the criterion.

    python3 -s -m bench.guard.language_sensor --out bench/results/language-sensor.json

The language of a page is the one its document has in the manifest (`language`); the catalogue pages that do not say have
Spanish. The invoices count under their own language when the manifest gives one and are left out until it does. The
sealed sets are not read here (they are read once, at the final gate).
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from bench.guard import pairs
from extract.guard import language

CATALOGUE_MIN_WORDS = 100
FLOORS = {
    "catalog": {"right": 0.99, "wrong": 0.005},
    "invoice": {"right_or_unsure": 0.95, "wrong": 0.01},
}
FLOOR_PERCENTILE = (
    0.5  # the 1st percentile left 98.8 % of the Spanish pages sensed (the rest `unsure`): under the required 99 %
)


def truth_language(page: pairs.Page) -> str | None:
    """The language of the document of a page, or None when it is not known (an invoice the manifest gives none)."""
    if page.language:
        return page.language
    return "es" if page.source in (None, "catalog") else None


def fit_floors(pages: list[pairs.Page], percentile: float = FLOOR_PERCENTILE) -> dict[str, float]:
    """The floor of the share of each language: a percentile of that language's share over the catalogue pages of that
    language that are long enough to say it, so that no page of the corpus falls under its floor by construction (the
    contrast that is not circular is the sealed read). A language with no page keeps the default."""
    found: dict[str, list[float]] = {"es": [], "en": []}
    for page in pages:
        wanted = truth_language(page)
        es, en, count = language.shares(page.words)
        if wanted in found and page.source != "invoice" and count >= CATALOGUE_MIN_WORDS:
            found[wanted].append(es if wanted == "es" else en)
    return {
        lang: float(np.percentile(values, percentile)) if values else language.DEFAULT_FLOOR
        for lang, values in found.items()
    }


def fit_floor(pages: list[pairs.Page], percentile: float = FLOOR_PERCENTILE) -> float:
    """The floor of the Spanish share (the one of a corpus of Spanish pages)."""
    return fit_floors(pages, percentile)["es"]


def confusion(pages: list[pairs.Page], sensor: "float | dict") -> dict:
    """{source: {language of the document: {sensed: pages}}} and the two rates of the criterion per source."""
    matrix: dict = defaultdict(lambda: defaultdict(Counter))
    for page in pages:
        wanted = truth_language(page)
        if wanted is None:
            continue
        long_enough = len(language.tokens(page.words)) >= CATALOGUE_MIN_WORDS
        source = "invoice" if page.source == "invoice" else "catalog"
        if source == "catalog" and not long_enough:
            continue
        matrix[source][wanted][language.sensed_language(page.words, sensor)] += 1
    out = {
        "sensor": language.sensor_settings(sensor),
        "matrix": {s: {w: dict(c) for w, c in m.items()} for s, m in matrix.items()},
        "rates": {},
    }
    for source, by_language in matrix.items():
        total = sum(sum(c.values()) for c in by_language.values())
        right = sum(c[w] for w, c in by_language.items())
        wrong = sum(n for w, c in by_language.items() for sensed, n in c.items() if sensed not in (w, language.UNSURE))
        out["rates"][source] = {
            "pages": total,
            "right": right / total if total else None,
            "wrong": wrong / total if total else None,
            "right_or_unsure": (total - wrong) / total if total else None,
        }
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    pairs.add_data_arguments(parser)
    parser.add_argument("--out", type=Path, default=Path("bench/results/language-sensor.json"))
    args = parser.parse_args(argv)
    pages = pairs.pages_from_args(args)
    floors = fit_floors(pages)
    result = confusion(pages, {"floor": floors})
    result["pages"] = len(pages)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({"floors": floors, **result["rates"]}, default=float))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
