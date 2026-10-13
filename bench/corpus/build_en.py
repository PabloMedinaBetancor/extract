"""The English part of the v5 corpus: development pages and the sealed test T5-EN (good with English text
too). The same recipe as `build5.py`, over the English catalogue that `discover_en.py` fetched.

    python -m bench.corpus.build_en survey     # eligible documents of the English catalogue (no OCR, no damaged render)
    python -m bench.corpus.build_en plan       # the frozen list: T5-EN candidates first, then the development documents
    python -m bench.corpus.build_en pages      # render, damage and label the pages of both
    python -m bench.corpus.build_en manifest   # `language` on every page, EN pages and folds, the `t5en` block, SOURCES_EN split
    python -m bench.corpus.build_en verify     # no repeated or near page against anything, ES or EN, and no document in both
    python -m bench.corpus.build_en freeze     # hashes of T5-EN before any English model (bench/results/test-freeze-v5en.json)

What is fixed here:

* the documents are walked in a seeded order and the first `T5_EN_CANDIDATES` eligible ones (at least two eligible pages)
  are the T5-EN list, the next `DEV_DOCUMENTS` the development documents; the order of the list is frozen with its
  fingerprint before any English model exists, and `n` (60 documents at least) is chosen later by M7 from the data,
  reading the first `n` by `position`;
* three pages a document (four when fewer than `MIN_TEST_DOCUMENTS` documents qualify), the same bands of damage;
* a development document gets a fold by its hash, with the seed that balances the bands best within the English stratum:
  the Spanish folds do not move (a test pins the hash of the Spanish entries);
* a page repeated (first 500 characters) or near another (5-shingle Jaccard 0.5) of ANY page kept so far, Spanish or
  English, is dropped, and a document that is a translation of another (same `unit_id`) would go on the same side;
  this catalogue has no translations and no bilingual PDF (US and UK English documents only);
* the sealed pages are the set `test5en`, read only through `sealed.read_test_pages(..., sets=sealed.EN_TEST_SETS)`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from bench.corpus import build5, catalog, discover_en, sealed, survey5
from bench.corpus.degrade import draw_band

T5_EN_CANDIDATES = 100
DEV_DOCUMENTS = 100
MIN_TEST_DOCUMENTS = 60
PAGES_PER_DOCUMENT = 3
SET_DEV, SET_TEST, SET_READBACK = "dev5en", "test5en", "clean5en"
READBACK_PAGES = 30
SURVEY = "survey5en"
FREEZE_EN = Path("bench/results/test-freeze-v5en.json")
CHECKS_EN = Path("bench/results/split-v5en-checks.json")
LANGUAGE = "en"
ENGLISH_SETS = (SET_DEV, SET_TEST, SET_READBACK)


def plan(rows: list[dict]) -> dict:
    """The frozen order of the English documents, from the survey rows in their seeded order: the first documents with
    two eligible pages are T5-EN's candidates (`position` = their place in the list), the following ones development.

    With fewer than 160 eligible documents the test keeps its floor of `MIN_TEST_DOCUMENTS` and development gets the rest;
    with fewer than 100 the pages a document gives rise to are four, so that the pages are not fewer."""
    eligible = [r for r in rows if len(r["pages"]) >= survey5.MIN_PAGES]
    n_test = min(T5_EN_CANDIDATES, max(MIN_TEST_DOCUMENTS, len(eligible) - DEV_DOCUMENTS))
    n_test = min(n_test, len(eligible))
    test, development = eligible[:n_test], eligible[n_test : n_test + DEV_DOCUMENTS]
    per_document = PAGES_PER_DOCUMENT if len(eligible) >= build5.MIN_CATALOGUE_DOCUMENTS else PAGES_PER_DOCUMENT + 1

    def document(position: int, row: dict, kind: str) -> dict:
        return {
            "doc_id": row["doc_id"],
            "unit_id": row["doc_id"],
            "position": position,
            "language": LANGUAGE,
            "slice": kind,
            "pages": [p["page"] for p in row["pages"][:per_document]],
        }

    spare = [
        {"doc_id": r["doc_id"], "page": r["pages"][per_document]["page"]}
        for r in development
        if len(r["pages"]) > per_document
    ]
    return {
        "language": LANGUAGE,
        "readback": spare[:READBACK_PAGES],
        "eligible_documents": len(eligible),
        "pages_per_document": per_document,
        "test": [document(i, r, "pages") for i, r in enumerate(test)],
        "development": [document(i, r, "development") for i, r in enumerate(development)],
    }


def development_units(plan_: dict) -> dict[str, dict[str, int]]:
    units: dict[str, Counter] = {}
    for document in plan_["development"]:
        units[document["doc_id"]] = Counter(
            draw_band(f"{document['doc_id']}/{number}", survey5.SEED) for number in document["pages"]
        )
    return {unit: dict(bands) for unit, bands in units.items()}


def entries(plan_: dict, folds: dict) -> list[dict]:
    """Manifest entries of T5-EN's pages and of the development pages, every one with its `language`."""
    out = []
    for document in plan_["test"]:
        for number in document["pages"]:
            band = draw_band(f"{document['doc_id']}/{number}", survey5.SEED)
            out.append(
                build5.entry(
                    SET_TEST,
                    document["doc_id"],
                    number,
                    band,
                    "test5en",
                    position=document["position"],
                    source="catalog",
                    language=LANGUAGE,
                )
            )
    for document in plan_["development"]:
        for number in document["pages"]:
            band = draw_band(f"{document['doc_id']}/{number}", survey5.SEED)
            out.append(
                build5.entry(
                    SET_DEV,
                    document["doc_id"],
                    number,
                    band,
                    "dev",
                    fold=folds["assignment"][document["doc_id"]],
                    source="catalog",
                    language=LANGUAGE,
                )
            )
    for item in plan_.get("readback", []):
        out.append(
            build5.entry(
                SET_READBACK, item["doc_id"], item["page"], "clean", "readback", source="catalog", language=LANGUAGE
            )
        )
    return out


def render(data_dir: Path, catalog_root: Path, plan_: dict, progress=print, sets=None) -> int:
    """Render, damage and label every English page from the truth the survey kept (the PDF layer)."""
    by_id = {d.doc_id: d for d in catalog.list_documents(catalog_root)[0]}
    folds = build5.assign_folds(development_units(plan_))
    count = 0
    for item in entries(plan_, folds):
        if sets and item["set"] not in sets:
            continue
        kept = json.loads(
            (data_dir / "selection" / SURVEY / item["doc_id"] / f"{item['page']}.json").read_text(encoding="utf-8")
        )
        page = catalog.EligiblePage(
            item["page"],
            kept["width_pt"],
            kept["height_pt"],
            [catalog.labels.TruthWord(i, t, tuple(box)) for i, t, box in kept["words"]],
            kept["key"],
        )
        catalog.build_page(by_id[item["doc_id"]], page, item["set"], item["band"], data_dir, survey5.SEED)
        count += 1
        if count % 25 == 0:
            progress(f"{count} pages")
    return count


def invoice_languages(data_dir: Path) -> dict[str, str]:
    """Language of every generated invoice, from its sidecar."""
    out = {}
    for path in sorted((data_dir / "invoices").glob("*/*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(record, dict) and "language" in record:
            out[path.stem] = record["language"]
    return out


def add_languages(pages: list[dict], data_dir: Path) -> list[dict]:
    """Every page without a `language` gets one: the language of its invoice, `es` for a Spanish catalogue page. No
    other field of an entry changes."""
    languages = invoice_languages(data_dir)
    out = []
    for page in pages:
        if "language" in page:
            out.append(page)
        elif page["source"] == "invoice":
            out.append({**page, "language": languages[page["doc_id"]]})
        else:
            out.append({**page, "language": "es"})
    return out


def spanish_fingerprint(pages: list[dict]) -> str:
    """Hash of every entry that is not of the English catalogue, without its `language`: the pin that the Spanish folds,
    roles and bands did not move when the language was added (the invoices of either language are among them)."""
    kept = [
        {k: v for k, v in p.items() if k != "language"}
        for p in pages
        if not (p["source"] == "catalog" and p["language"] == LANGUAGE)
    ]
    return hashlib.sha256(json.dumps(kept, sort_keys=True).encode()).hexdigest()


def write_manifest(
    data_dir: Path, plan_: dict, split: Path = build5.SPLIT_V5, sources: Path = discover_en.SOURCES_EN
) -> dict:
    body = json.loads(split.read_text(encoding="utf-8"))
    kept = [p for p in body["pages"] if p.get("language") != LANGUAGE or p["set"] not in ENGLISH_SETS]
    pages = add_languages(kept, data_dir)
    folds = build5.assign_folds(development_units(plan_))
    english = entries(plan_, folds)
    body["pages"] = sorted(pages + english, key=lambda e: (e["set"], e["doc_id"], e["page"]))
    body["spanish_entries_fingerprint"] = spanish_fingerprint(pages)
    body["folds_en"] = {k: v for k, v in folds.items() if k != "assignment"}
    documents = [
        {k: d[k] for k in ("doc_id", "unit_id", "position", "language", "slice", "pages")} for d in plan_["test"]
    ]
    before = body.get("t5en", {})
    body["t5en"] = {
        "pages_per_document": plan_["pages_per_document"],
        "documents": documents,
        "list_fingerprint": build5.fingerprint_of(documents),
        "minimum_documents": MIN_TEST_DOCUMENTS,
    }
    if before.get("list_fingerprint") == body["t5en"]["list_fingerprint"] and "hashes" in before:
        body["t5en"]["hashes"], body["t5en"]["pages_hashed"] = before["hashes"], before["pages_hashed"]
    body["fingerprint"] = build5.fingerprint_of({k: v for k, v in body.items() if k != "fingerprint"})
    split.write_text(json.dumps(body, indent=1), encoding="utf-8")
    sides = {d["doc_id"]: SET_TEST for d in plan_["test"]} | {d["doc_id"]: SET_DEV for d in plan_["development"]}
    _write_split_column(sources, sides)
    return body


def _write_split_column(sources: Path, sides: dict[str, str]) -> None:
    with open(sources, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["split"] = sides.get(row["doc_id"], "")
    with open(sources, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=discover_en.SOURCES_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def summary(body: dict, plan_: dict) -> dict:
    english = [p for p in body["pages"] if p["language"] == LANGUAGE and p["source"] == "catalog"]
    invoices = [p for p in body["pages"] if p["source"] == "invoice" and p["page"] == 1]
    return {
        "eligible_documents": plan_["eligible_documents"],
        "pages_per_document": plan_["pages_per_document"],
        "t5en_documents": len(plan_["test"]),
        "dev_documents": len(plan_["development"]),
        "english_pages_by_set": dict(Counter(p["set"] for p in english)),
        "english_pages_by_band": dict(Counter(f"{p['set']}/{p['band']}" for p in english)),
        "invoices_by_language_and_role": dict(Counter(f"{p['language']}/{p['role']}" for p in invoices)),
        "english_invoices_in_t5": sum(1 for p in invoices if p["language"] == "en" and p["role"] == "test5"),
        "folds_en": body["folds_en"],
        "spanish_entries_fingerprint": body["spanish_entries_fingerprint"],
    }


def verify(data_dir: Path, split: Path = build5.SPLIT_V5, sources: Path = discover_en.SOURCES_EN) -> dict:
    """The data side of the no-repeat and read-back checks for the English pages, from the rendered pages: none repeats (first 500 characters) or
    comes near (5-shingle Jaccard) a page of ANY other set, Spanish or English, none of T5-EN is near a page of the
    development documents, and no document is on both sides. Statistics only."""
    body = json.loads(split.read_text(encoding="utf-8"))
    others = ("train", "val", "test", "unseen", "clean", "pilot", "test5", "unseen5", "clean5")
    index, keys = survey5.DuplicateIndex(), set()
    for _, words in build5.page_texts(data_dir, others):
        index.add(survey5.shingles(words))
        keys.add(build5._key(words))
    report = {"pages_checked_against": len(index.sizes)}
    own = survey5.DuplicateIndex()
    worst_other = worst_own = 0.0
    repeated = near_other = near_own = 0
    for set_name in ENGLISH_SETS:
        for _doc, words in build5.page_texts(data_dir, (set_name,)):
            shingles = survey5.shingles(words)
            score = index.max_jaccard(shingles)
            worst_other = max(worst_other, score)
            repeated += build5._key(words) in keys
            near_other += score >= survey5.MAX_JACCARD
            if own.sizes:
                inner = own.max_jaccard(shingles)
                worst_own = max(worst_own, inner)
                near_own += inner >= survey5.MAX_JACCARD
            own.add(shingles)
    readback = {e["doc_id"] for e in body["pages"] if e["set"] == SET_READBACK}
    dev = {e["doc_id"] for e in body["pages"] if e["set"] == SET_DEV}
    test = {e["doc_id"] for e in body["pages"] if e["set"] == SET_TEST}
    units = defaultdict(set)
    for row in discover_en.read_sources_file(sources):
        units[row["unit_id"]].add(row["doc_id"])
    split_units = {u for u, ids in units.items() if ids & dev and ids & test}
    return {
        **report,
        "english_pages_checked": len(own.sizes),
        "development_documents": len(dev),
        "test_documents": len(test),
        "documents_in_both": len(dev & test),
        "readback_documents_in_test": len(readback & test),
        "readback_pages": sum(1 for e in body["pages"] if e["set"] == SET_READBACK),
        "units_on_both_sides": len(split_units),
        "pages_repeating_another_page": repeated,
        "pages_near_a_page_of_another_set": near_other,
        "largest_jaccard_against_other_sets": round(worst_other, 4),
        "english_pages_near_an_earlier_english_page": near_own,
        "largest_jaccard_between_english_pages": round(worst_own, 4),
        "threshold": survey5.MAX_JACCARD,
    }


def freeze(
    data_dir: Path, split: Path = build5.SPLIT_V5, out: Path = FREEZE_EN, reads: Path = sealed.EN_READS_PATH
) -> dict:
    """Fingerprint T5-EN's images, truth and ungrounded patches before any English model exists, in `out` and in the
    manifest (one logged read of the sealed pages, purpose "ocr", for hashing only)."""
    record = sealed.freeze_record(
        data_dir, out, "T5-EN as built by bench.corpus.build_en, before any English model", reads, sealed.EN_TEST_SETS
    )
    body = json.loads(split.read_text(encoding="utf-8"))
    body["t5en"]["hashes"] = {k: record[k] for k in ("images", "truth", "ungrounded")}
    body["t5en"]["pages_hashed"] = record["pages"]
    body["fingerprint"] = build5.fingerprint_of({k: v for k, v in body.items() if k != "fingerprint"})
    split.write_text(json.dumps(body, indent=1), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["survey", "plan", "pages", "manifest", "verify", "freeze"])
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--catalog-root", type=Path)
    parser.add_argument("--cap", type=int, default=T5_EN_CANDIDATES + DEV_DOCUMENTS)
    parser.add_argument("--sets", nargs="*", help="render only these sets")
    args = parser.parse_args()
    root = args.catalog_root or args.data_dir / "en" / "catalog"
    plan_path = args.data_dir / SURVEY / "plan.json"
    rows_path = args.data_dir / SURVEY / "survey.jsonl"
    if args.action == "survey":
        import sys

        result = survey5.survey(
            args.data_dir,
            root,
            Path("bench/SOURCES.csv"),
            Path("bench/corpus/split.json"),
            args.cap,
            log=lambda line: print(line, file=sys.stderr, flush=True),
            name=SURVEY,
            everything=True,
        )
        print(json.dumps(result, indent=1))
    elif args.action == "plan":
        rows = [json.loads(x) for x in rows_path.read_text("utf-8").splitlines()]
        frozen = plan(rows)
        plan_path.write_text(json.dumps(frozen, indent=1), encoding="utf-8")
        print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in frozen.items()}))
    else:
        frozen = json.loads(plan_path.read_text(encoding="utf-8"))
        if args.action == "pages":
            print(render(args.data_dir, root, frozen, sets=args.sets), "pages built")
        elif args.action == "manifest":
            print(json.dumps(summary(write_manifest(args.data_dir, frozen), frozen), indent=1))
        elif args.action == "verify":
            report = verify(args.data_dir)
            CHECKS_EN.write_text(json.dumps(report, indent=1), encoding="utf-8")
            print(json.dumps(report, indent=1))
        else:
            print(json.dumps(freeze(args.data_dir), indent=1))


if __name__ == "__main__":
    main()
