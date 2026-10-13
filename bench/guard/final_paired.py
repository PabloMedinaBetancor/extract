"""The thorough model against the default one, out of fold on the same development pages, with the paired bootstrap by document
: what `more` adds in each language, with the language of the page sensed (what the product does) and with the
split's label of the page (the optimistic reference), and on the documents of each label.

    python3 -s -m bench.guard.final_paired --default bench/results/fix-final-report-guard.json \\
        --thorough bench/results/fix-final-report-guard-thorough.json --out bench/results/final-paired.json
"""

import argparse
import json
from pathlib import Path

from bench.guard.cells_paired import paired

PARTITIONS = ("sensed", "label")


def net_of(counts: dict, rule: str, keep=None) -> dict:
    """`net`, `fixed`, `broken` and `applied` per document of a rule, over the documents `keep` (indexes) when given."""
    block = counts[rule]
    names = {"fixed": block["fixed"], "broken": block["broken"], "applied": block["applied"]}
    names["net"] = [f - b for f, b in zip(block["fixed"], block["broken"], strict=True)]
    if keep is not None:
        names = {k: [v[i] for i in keep] for k, v in names.items()}
    return names


def compare_parts(a: dict, b: dict, resamples: int, keep=None) -> dict:
    out = {}
    for rule, key in (("more", "net"), ("more", "applied"), ("safe", "fixed")):
        out[f"{rule}_{key}"] = paired(net_of(a, rule, keep), net_of(b, rule, keep), key, resamples)
    out["more_net_each"] = {
        "thorough": sum(net_of(a, "more", keep)["net"]),
        "default": sum(net_of(b, "more", keep)["net"]),
    }
    return out


def compare(thorough: dict, default: dict, resamples: int = 2000) -> dict:
    """The comparison of two final reports (their `by_document` blocks must hold the same documents in the same order)."""
    out: dict = {"resamples": resamples}
    for partition in PARTITIONS:
        a, b = thorough["by_document"][partition], default["by_document"][partition]
        if a["documents"] != b["documents"]:
            raise ValueError("the two reports do not hold the same documents: cannot be paired")
        block = {"total": compare_parts(a["total"], b["total"], resamples)}
        for language in sorted(set(a["by_language"]) & set(b["by_language"])):
            block[f"language_{language}"] = compare_parts(
                a["by_language"][language], b["by_language"][language], resamples
            )
        for label in sorted(set(a["language"].values())):  # the documents of each label of the split, all their pages
            keep = [i for i, doc in enumerate(a["documents"]) if a["language"][doc] == label]
            block[f"documents_labelled_{label}"] = {
                "documents": len(keep),
                **compare_parts(a["total"], b["total"], resamples, keep),
            }
        out[partition] = block
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--thorough", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("bench/results/final-paired.json"))
    parser.add_argument("--resamples", type=int, default=2000)
    args = parser.parse_args(argv)
    read = lambda path: json.loads(path.read_text(encoding="utf-8"))  # noqa: E731
    result = compare(read(args.thorough), read(args.default), args.resamples)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    for partition in PARTITIONS:
        for name, part in result[partition].items():
            if isinstance(part, dict) and "more_net" in part:
                m = part["more_net"]
                print(
                    f"{partition:7s} {name:28s} thorough - default more net {m['difference']:+.0f} [{m['low']:+.0f},{m['high']:+.0f}]"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
