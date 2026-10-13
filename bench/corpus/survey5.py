"""The v5 survey: which catalogue documents nobody has used can serve as the new sealed test T5.

    python -m bench.corpus.survey5 [--cap 150]

No OCR and no damaged render: every filter is the v4 eligibility of `catalog.check_page`, plus the rule against
republished text of v5: a page is dropped when the first 500 normalised characters repeat a page of the
development pool D or of an earlier T5 page, or when its 5-word-shingle Jaccard against any such page is 0.5 or more.
Documents are walked in a seeded random order and the walk stops at `--cap` eligible documents, so the 12 GB of the
catalogue are never read whole. Progress goes to `<data>/survey5/survey.jsonl`, one line per document, and the truth of
each eligible page to `<data>/selection/survey5/`, so a stopped survey resumes where it stopped.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import pypdfium2 as pdfium

from bench.corpus import catalog
from bench.corpus.split import stable_rng

SEED = 20261003
SHINGLE = 5
MAX_JACCARD = 0.5
MIN_PAGES = 2
SPARE_PAGES = 4
ORDER_SALT = "document-order-v5"
LONG_PAGE_WORDS = 300


def tokens(words) -> list[str]:
    """The words of a page lower-cased and stripped to letters and digits, empty ones dropped."""
    out = (re.sub(r"\W+", "", getattr(w, "text", w).lower()) for w in words)
    return [t for t in out if t]


def shingles(words) -> frozenset[int]:
    toks = tokens(words)
    if len(toks) < SHINGLE:
        return frozenset({hash_of(" ".join(toks))}) if toks else frozenset()
    return frozenset(hash_of(" ".join(toks[i : i + SHINGLE])) for i in range(len(toks) - SHINGLE + 1))


def hash_of(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


class DuplicateIndex:
    """Pages by their shingles, so the largest Jaccard of a new page against all of them costs one pass."""

    def __init__(self) -> None:
        self.sizes: list[int] = []
        self.owners: dict[int, list[int]] = {}

    def add(self, page_shingles: frozenset[int]) -> None:
        number = len(self.sizes)
        self.sizes.append(len(page_shingles))
        for shingle in page_shingles:
            self.owners.setdefault(shingle, []).append(number)

    def max_jaccard(self, page_shingles: frozenset[int]) -> float:
        shared: Counter[int] = Counter()
        for shingle in page_shingles:
            shared.update(self.owners.get(shingle, ()))
        best = 0.0
        for number, count in shared.items():
            best = max(best, count / (len(page_shingles) + self.sizes[number] - count))
        return best


def kept_pages(set_dir: Path):
    """(path, record) of every kept truth under `<set_dir>/<doc>/<n>.json`."""
    for path in sorted(set_dir.glob("*/*.json")):
        yield path, json.loads(path.read_text(encoding="utf-8"))


def development_index(data_dir: Path, everything: bool = False) -> tuple[DuplicateIndex, set[str]]:
    """The shingles and the 500-character keys of every page kept for v4 (train, val, test, unseen, clean, pilot); with
    `everything`, also those of T5 and of any other survey (the English survey must not repeat anything)."""
    index, keys = DuplicateIndex(), set()
    for set_dir in sorted(
        p for p in (data_dir / "selection").iterdir() if p.is_dir() and (everything or "5" not in p.name)
    ):
        add_kept(set_dir, index, keys)
    return index, keys


def add_kept(set_dir: Path, index: DuplicateIndex, keys: set[str]) -> None:
    for _, record in kept_pages(set_dir):
        index.add(shingles(w[1] for w in record["words"]))
        keys.add(record["key"])


def used_documents(sources: Path, split: Path) -> tuple[set[str], set[str]]:
    """Document ids and SHA-256 hashes already used by v4 (SOURCES.csv and split.json)."""
    ids, hashes = set(), set()
    with open(sources, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            ids.add(row["doc_id"])
            hashes.add(row["sha256"])
    for entry in json.loads(split.read_text(encoding="utf-8"))["pages"]:
        ids.add(entry["doc_id"])
    return ids, hashes


def candidates(catalog_root: Path, ids: set[str], hashes: set[str]) -> tuple[list[catalog.Document], dict[str, int]]:
    """Unused public-administration documents in the seeded order of v5, and how many were left out and why."""
    documents, skipped = catalog.list_documents(catalog_root)
    seen, unused, dropped = set(), [], {"used_by_v4": 0, "same_file_twice": 0}
    for document in documents:
        if document.doc_id in ids or document.sha256 in hashes:
            dropped["used_by_v4"] += 1
        elif document.doc_id in seen:
            dropped["same_file_twice"] += 1
        else:
            seen.add(document.doc_id)
            unused.append(document)
    order = stable_rng(SEED, ORDER_SALT).permutation(len(unused))
    return [unused[i] for i in order], {**skipped, **dropped}


def check_document(
    document, index: DuplicateIndex, keys: set[str], rejected: Counter, store: Path, name: str = "survey5"
) -> list[dict]:
    """Up to SPARE_PAGES eligible pages of a document, each kept in `store` and added to the duplicate index."""
    try:
        count = catalog._page_count(document.path)
    except pdfium.PdfiumError:
        rejected["unopenable"] += 1
        return []
    order = stable_rng(SEED, "pages", document.doc_id).permutation(count)[: catalog.CANDIDATES_PER_DOCUMENT]
    found = []
    for number in order + 1:
        page, reason = catalog.check_page(document, int(number), keys)
        if page is None:
            rejected[reason] += 1
            continue
        page_shingles = shingles(page.words)
        if index.max_jaccard(page_shingles) >= MAX_JACCARD:
            rejected["near_duplicate"] += 1
            continue
        keys.add(page.key)
        index.add(page_shingles)
        kept = store / name / document.doc_id / f"{page.number}.json"
        kept.parent.mkdir(parents=True, exist_ok=True)
        words = [[w.id, w.text, list(w.box)] for w in page.words]
        kept.write_text(
            json.dumps(
                {"width_pt": page.width_pt, "height_pt": page.height_pt, "key": page.key, "words": words},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        found.append({"page": page.number, "words": len(page.words)})
        if len(found) == SPARE_PAGES:
            break
    return found


def survey(
    data_dir: Path,
    catalog_root: Path,
    sources: Path,
    split: Path,
    cap: int,
    log=print,
    name: str = "survey5",
    everything: bool = False,
) -> dict:
    """`name` is the folder of the survey's rows and of the truth it keeps: `survey5` for the Spanish T5, `survey5en` for
    the English corpus, whose duplicate index (`everything`) holds every page kept so far, T5 included."""
    ids, hashes = used_documents(sources, split)
    ordered, left_out = candidates(catalog_root, ids, hashes)
    index, keys = development_index(data_dir, everything)
    out = data_dir / name / "survey.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not everything:
        add_kept(data_dir / "selection" / name, index, keys)
    rejected: Counter = Counter()
    rows = _load(out)
    eligible = sum(1 for row in rows if len(row["pages"]) >= MIN_PAGES)
    done = {row["doc_id"] for row in rows}
    began = time.monotonic()
    with open(out, "a", encoding="utf-8") as handle:
        for document in ordered:
            if eligible >= cap:
                break
            if document.doc_id in done:
                continue
            pages = check_document(document, index, keys, rejected, data_dir / "selection", name)
            row = {"doc_id": document.doc_id, "catalog_id": document.catalog_id, "pages": pages}
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            rows.append(row)
            eligible += len(pages) >= MIN_PAGES
            log(f"{eligible}/{cap} eligible, {len(rows)} surveyed, {time.monotonic() - began:.0f}s")
    return report(rows, ordered, left_out, rejected, cap)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


def report(rows: list[dict], ordered: list, left_out: dict, rejected: Counter, cap: int) -> dict:
    eligible = [r for r in rows if len(r["pages"]) >= MIN_PAGES]
    surveyed = len(rows)
    share = len(eligible) / surveyed if surveyed else 0.0
    return {
        "candidate_documents": len(ordered),
        "left_out": left_out,
        "surveyed": surveyed,
        "with_one_page": sum(1 for r in rows if r["pages"]),
        "eligible_two_pages": len(eligible),
        "eligible_three_pages": sum(1 for r in rows if len(r["pages"]) >= 3),
        "with_a_300_word_page": sum(1 for r in rows if any(p["words"] >= LONG_PAGE_WORDS for p in r["pages"])),
        "stopped_at_cap": len(eligible) >= cap,
        "estimated_eligible_in_stock": round(share * len(ordered)),
        "rejected_pages": dict(rejected),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cap", type=int, default=150)
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--catalog-root", type=Path, default=catalog.default_catalog_root())
    parser.add_argument("--sources", type=Path, default=Path("bench/SOURCES.csv"))
    parser.add_argument("--split", type=Path, default=Path("bench/corpus/split.json"))
    args = parser.parse_args()
    result = survey(
        args.data_dir,
        args.catalog_root,
        args.sources,
        args.split,
        args.cap,
        log=lambda line: print(line, file=sys.stderr, flush=True),
    )
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
