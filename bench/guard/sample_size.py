"""The n of the sealed tests from the final arrays: how many catalogue documents of each language T5 and T5-EN need so that
the net balance of `more` (fixed - broken words, per 100 truth words) is known within a quarter of its own size.

    python3 -s -m bench.guard.sample_size --by-language --default-report bench/results/fix-final-report-guard.json \\
        --thorough-report bench/results/fix-final-report-guard-thorough.json --out bench/results/sample-size.json

For each language (the sensed one, what the product uses) and each model: over the catalogue documents with truth words in that
language, the rate of a document is `100 * net / truth words`, the pooled effect is `100 * sum(net) / sum(truth words)`, the
spread is the SD of the rates between documents, the half width is `h = effect / 4`, and `n = (1.96 * SD / h)^2` (metrics.documents_needed).
The thorough model is also taken paired against the default one (the rate of the difference of their nets).
"""

import argparse
import json
from pathlib import Path

import numpy as np

from bench.guard import metrics, pairs

QUARTER = 4.0


def rates(net: np.ndarray, truth: np.ndarray) -> dict:
    """The effect, the spread, the half width and n of a set of documents: `net` and `truth` are per document (words)."""
    keep = truth > 0
    net, truth = net[keep], truth[keep]
    rate = 100.0 * net / truth
    effect = 100.0 * net.sum() / truth.sum()
    spread = float(rate.std(ddof=1)) if len(rate) > 1 else float("nan")
    half = effect / QUARTER
    needed = metrics.documents_needed(spread, half) if effect > 0 and np.isfinite(spread) else None
    return {
        "documents": int(keep.sum()),
        "truth_words": int(truth.sum()),
        "net": int(net.sum()),
        "effect_per_100": float(effect),
        "sd_between_documents": spread,
        "half_width": float(half),
        "n": needed,
    }


def truth_by_document(pages: list, documents: list[str]) -> dict:
    """{language: truth words per document} over the pages whose sensed language is it, and the catalogue mask of the documents."""
    index = {d: i for i, d in enumerate(documents)}
    out: dict = {}
    catalogue = np.zeros(len(documents), dtype=bool)
    for page in pages:
        i = index[page.doc]
        catalogue[i] = catalogue[i] or page.source == "catalog"
        language = page.sensed or page.language or pairs.DEFAULT_LANGUAGE
        out.setdefault(language, np.zeros(len(documents)))[i] += page.truth_words
    return out, catalogue


def net_of(report: dict, language: str) -> np.ndarray:
    block = report["by_document"]["sensed"]["by_language"][language]["more"]
    return np.asarray(block["fixed"], dtype=float) - np.asarray(block["broken"], dtype=float)


def compute(default: dict, thorough: dict | None, truth: dict, catalogue: np.ndarray) -> dict:
    out: dict = {
        "rule": "n = (1.96 * SD / h)^2 with h = effect / 4, rate = 100 * net / truth words per catalogue document"
    }
    for language in ("es", "en"):
        if language not in truth:
            continue
        mask = catalogue
        g = np.where(mask, truth[language], 0.0)
        block = {"default": rates(net_of(default, language), g)}
        if thorough is not None:
            block["thorough"] = rates(net_of(thorough, language), g)
            block["thorough_minus_default"] = rates(net_of(thorough, language) - net_of(default, language), g)
        out[language] = block
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    pairs.add_data_arguments(parser)
    parser.add_argument("--default-report", type=Path, required=True)
    parser.add_argument("--thorough-report", type=Path)
    parser.add_argument("--out", type=Path, default=Path("bench/results/sample-size.json"))
    args = parser.parse_args(argv)
    args.cells = []  # no second readings: only the sensed language and the truth words of each page are needed
    pages = pairs.pages_from_args(args)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
    default = read(args.default_report)
    documents = default["by_document"]["sensed"]["documents"]
    truth, catalogue = truth_by_document(pages, documents)
    result = compute(default, read(args.thorough_report) if args.thorough_report else None, truth, catalogue)
    args.out.write_text(json.dumps(result, indent=1, default=float), encoding="utf-8")
    for language in ("es", "en"):
        for name, row in result.get(language, {}).items():
            print(f"{language} {name:24s} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
