"""Font families of the documents in each split, and how many the test documents have that training never saw.

The guard's signals carry no font identity, so this is only a description of how far the test is
from the training pages; it costs one `pdffonts` call per selected page.

    python -m bench.corpus.fonts [--out bench/results/font-families.json]
"""

import argparse
import json
import re
import subprocess
from pathlib import Path

from bench.corpus import catalog

_SUBSET = re.compile(r"^[A-Z]{6}\+")
_STYLE = re.compile(
    r"[-,_ ](bold|italic|oblique|regular|light|medium|black|semibold|condensed|narrow|mt|ps|roman|book|it|bd).*$", re.I
)


def family(font_name: str) -> str:
    """The family of a PDF font name: subset prefix, style words and non-letters removed, lower case."""
    name = _STYLE.sub("", _SUBSET.sub("", font_name))
    return re.sub(r"[^a-z]", "", name.lower())


def page_families(pdf: Path, page: int) -> set[str]:
    run = subprocess.run(
        ["pdffonts", "-f", str(page), "-l", str(page), str(pdf)],
        capture_output=True,
        text=True,
        errors="replace",
        timeout=60,
    )
    return {family(line.split()[0]) for line in run.stdout.splitlines()[2:] if line.split()} - {""}


def report(plan: dict, documents: dict[str, catalog.Document]) -> dict:
    families: dict[str, set[str]] = {"train": set(), "val": set(), "test": set()}
    for entry in plan["pages"]:
        if entry["set"] in families:
            families[entry["set"]].update(page_families(documents[entry["doc_id"]].path, entry["page"]))
    seen_in_training = families["train"] | families["val"]
    return {
        "families": {name: len(found) for name, found in families.items()},
        "test_families_never_in_training": len(families["test"] - seen_in_training),
        "test_families_never_in_training_share": round(
            len(families["test"] - seen_in_training) / max(1, len(families["test"])), 3
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog-root", type=Path, default=catalog.default_catalog_root())
    parser.add_argument("--split", type=Path, default=Path("bench/corpus/split.json"))
    parser.add_argument("--out", type=Path, default=Path("bench/results/font-families.json"))
    args = parser.parse_args()
    documents = {d.doc_id: d for d in catalog.list_documents(args.catalog_root)[0]}
    plan = json.loads(args.split.read_text(encoding="utf-8"))
    result = report(plan, documents)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
