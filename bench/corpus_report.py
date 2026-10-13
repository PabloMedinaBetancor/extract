"""The development pool D by language, for the published numbers: pages, documents, bands and the share of invoices.

    python -m bench.corpus_report [--out bench/results/corpus-by-language.json]

Read from the manifest (`bench/corpus/split-v5.json`) and nothing else: no page is opened, the sealed ones included.
D is every page whose role is `dev` (the pages a model is trained and cross-fitted on, with a fold) or `shift` (the v4
unseen-damage slice, which never trains); `aside` (invoices of held-out layouts and fonts), `aux`, `readback`, `test5`,
`unseen5` and `test5en` are reported on their own lines so that nothing is counted twice.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

SPLIT = Path("bench/corpus/split-v5.json")
POOL_ROLES = ("dev", "shift")


def share(part: int, whole: int) -> float | None:
    return round(part / whole, 4) if whole else None


def of_role(pages: list[dict], roles: tuple[str, ...]) -> dict:
    """Counts of the pages of the given roles, by language: pages, documents, bands, folds and the invoice share."""
    out: dict[str, dict] = {}
    for language in sorted({p["language"] for p in pages}):
        chosen = [p for p in pages if p["role"] in roles and p["language"] == language]
        if not chosen:
            continue
        invoices = [p for p in chosen if p["source"] == "invoice"]
        out[language] = {
            "pages": len(chosen),
            "documents": len({p["doc_id"] for p in chosen}),
            "catalogue_pages": len(chosen) - len(invoices),
            "catalogue_documents": len({p["doc_id"] for p in chosen if p["source"] == "catalog"}),
            "invoice_pages": len(invoices),
            "invoice_documents": len({p["doc_id"] for p in invoices}),
            "invoice_share_of_pages": share(len(invoices), len(chosen)),
            "pages_by_band": dict(sorted(Counter(p["band"] for p in chosen).items())),
            "pages_by_fold": dict(sorted(Counter(p.get("fold") for p in chosen if p.get("fold")).items())),
        }
    return out


def report(body: dict) -> dict:
    pages = body["pages"]
    pool = of_role(pages, POOL_ROLES)
    total = sum(v["pages"] for v in pool.values())
    return {
        "development_pool_D": {
            "roles": list(POOL_ROLES),
            "pages": total,
            "by_language": pool,
            "language_share_of_pages": {language: share(v["pages"], total) for language, v in pool.items()},
        },
        "dev_only": of_role(pages, ("dev",)),
        "shift": of_role(pages, ("shift",)),
        "aside_invoices": of_role(pages, ("aside",)),
        "readback": of_role(pages, ("readback",)),
        "sealed_not_opened": {
            "t5_pages": dict(Counter(p["language"] for p in pages if p["role"] in ("test5", "unseen5"))),
            "t5_english_invoices": sum(
                1 for p in pages if p["role"] == "test5" and p["source"] == "invoice" and p["language"] == "en"
            ),
            "t5en_pages": dict(Counter(p["language"] for p in pages if p["role"] == "test5en")),
            "t5en_candidate_documents": len(body.get("t5en", {}).get("documents", [])),
        },
        "folds_en": body.get("folds_en", {}).get("pages_per_fold_and_band"),
        "folds": body["folds"]["pages_per_fold_and_band"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--split", type=Path, default=SPLIT)
    parser.add_argument("--out", type=Path, default=Path("bench/results/corpus-by-language.json"))
    args = parser.parse_args()
    found = report(json.loads(args.split.read_text(encoding="utf-8")))
    args.out.write_text(json.dumps(found, indent=1), encoding="utf-8")
    pool = found["development_pool_D"]
    print(
        json.dumps(
            {
                "D_pages": pool["pages"],
                "by_language": {k: v["pages"] for k, v in pool["by_language"].items()},
                "share": pool["language_share_of_pages"],
            }
        )
    )


if __name__ == "__main__":
    main()
