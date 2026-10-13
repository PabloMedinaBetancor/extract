"""Builds what the v5 experiments stand on: the sealed test T5, its manifest `split-v5.json` and the development folds.

    python -m bench.corpus.build5 plan        # the frozen list of T5 documents, from the survey
    python -m bench.corpus.build5 pages       # render, damage and label T5's catalogue pages and the read-back pages
    python -m bench.corpus.build5 invoices    # the 60 sealed invoices and their pages
    python -m bench.corpus.build5 manifest    # split-v5.json, bench/SOURCES.csv, the freeze record of T5

Everything is a pure function of the survey (`survey5.py`), the v4 plan (`split.json`) and fixed seeds. The development
pool D is the v4 train, validation and test (the v4 test is retired as a test); the 30 pages of the
v4 unseen-damage slice are a shift check that never trains; T5 is documents that no other set has.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from bench.corpus import catalog, invoices, sealed, survey5
from bench.corpus.degrade import draw_band
from bench.corpus.labels import TruthWord, label_words
from bench.corpus.split import assign_split, stable_rng

T5_DOCUMENTS = 150
UNSEEN_DOCUMENTS = 15
READBACK_PAGES = 30
MIN_CATALOGUE_DOCUMENTS = 100
FOLDS = 5
FOLD_SEEDS = 400
PILOT_BANDS = {"heavy": 25, "medium": 25, "light": 10}
PILOT_PER_DOCUMENT = 2
PILOT_SEED = 20261005
SET_MAIN, SET_UNSEEN, SET_READBACK, SET_INVOICES = "test5", "unseen5", "clean5", "test5"
DEVELOPMENT_SETS = ("train", "val", "test")
SPLIT_V5 = Path("bench/corpus/split-v5.json")
FREEZE_V5 = Path("bench/results/test-freeze-v5.json")


def plan_t5(rows: list[dict], cap: int = T5_DOCUMENTS, unseen: int = UNSEEN_DOCUMENTS, readback: int = READBACK_PAGES):
    """The frozen order of T5, from the survey rows in their seeded order.

    Documents with at least two eligible pages, the first `cap` of them. Three pages per document, four when fewer
    than 100 documents qualify (a document that has fewer keeps what it has). The last `unseen` documents of the list
    carry the damage no development page has. The read-back pages (the undamaged read-back of T5's documents)
    are the first spare eligible page after T5's pages, of the first `readback` standard documents that have one, so no
    page is in T5 and in the read-back and no development document is touched."""
    eligible = [r for r in rows if len(r["pages"]) >= survey5.MIN_PAGES][:cap]
    per_document = 3 if len(eligible) >= MIN_CATALOGUE_DOCUMENTS else 4
    unseen = min(unseen, len(eligible) // 8)
    split_at = len(eligible) - unseen
    documents = []
    for position, row in enumerate(eligible):
        pages = [p["page"] for p in row["pages"][:per_document]]
        slice_name = "pages" if position < split_at else "unseen"
        documents.append({"doc_id": row["doc_id"], "position": position, "slice": slice_name, "pages": pages})
    spare = [
        {"doc_id": r["doc_id"], "page": r["pages"][per_document]["page"]}
        for r in eligible[:split_at]
        if len(r["pages"]) > per_document
    ]
    readback_pages = spare[:readback]
    return {"pages_per_document": per_document, "documents": documents, "readback": readback_pages}


def assign_folds(units: dict[str, dict[str, int]], folds: int = FOLDS, tries: int = FOLD_SEEDS) -> dict:
    """Fold of every development unit by hash, with the hash seed that balances the bands best.

    `units` maps a unit id (a catalogue document or an invoice) to its page count per band. The seed (0 to
    `tries - 1`) whose worst fold deviates least from an even share of every band is kept, so the choice depends on the
    data only through a published search and every unit still lands by its own hash."""
    names = {f"fold{k}": 1.0 / folds for k in range(folds)}
    totals: Counter = Counter()
    for bands in units.values():
        totals.update(bands)
    best = None
    for seed in range(tries):
        per_fold: dict[str, Counter] = defaultdict(Counter)
        for unit, bands in units.items():
            per_fold[assign_split(unit, seed, names)].update(bands)
        worst = max(
            abs(per_fold[name][band] - totals[band] / folds) / max(1.0, totals[band] / folds)
            for name in names
            for band in totals
        )
        if best is None or worst < best[0]:
            best = (worst, seed, {f: dict(c) for f, c in per_fold.items()})
    worst, seed, per_fold = best
    return {
        "seed": seed,
        "worst_band_deviation": round(worst, 4),
        "pages_per_fold_and_band": per_fold,
        "assignment": {unit: assign_split(unit, seed, names) for unit in sorted(units)},
    }


def pilot_sample(errors: dict[tuple[str, str, int], int], bands: dict[tuple[str, str, int], str]) -> list[dict]:
    """60 development pages stratified by errors: per band the pages with at least one wrong word, in three strata of
    the wrong-word count (low, middle, high thirds), the same number from each, at most two pages per document in all.

    `errors` maps (set, document, page) to the wrong words of the v4 reading; the draw is seeded."""
    rng = stable_rng(PILOT_SEED, "pilot")
    chosen = []
    per_document: Counter = Counter()
    for band, want in PILOT_BANDS.items():
        pool = sorted(key for key, band_of in bands.items() if band_of == band and errors.get(key, 0) > 0)
        pool.sort(key=lambda key: (errors[key], key))
        thirds = [pool[len(pool) * k // 3 : len(pool) * (k + 1) // 3] for k in range(3)]
        shares = [want // 3 + (1 if k < want % 3 else 0) for k in range(3)]
        for third, share in zip(thirds, shares, strict=True):
            order = rng.permutation(len(third))
            taken = 0
            for index in order:
                key = third[index]
                if per_document[key[1]] >= PILOT_PER_DOCUMENT:
                    continue
                per_document[key[1]] += 1
                chosen.append({"set": key[0], "doc_id": key[1], "page": key[2], "band": band, "wrong": errors[key]})
                taken += 1
                if taken == share:
                    break
    return sorted(chosen, key=lambda e: (e["band"], e["set"], e["doc_id"], e["page"]))


def fingerprint_of(body: dict) -> str:
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def entry(set_name: str, doc_id: str, page: int, band: str, role: str, seed: int = survey5.SEED, **more) -> dict:
    page_seed = catalog._seed_text(catalog._page_seed(doc_id, page, seed))
    return {"set": set_name, "doc_id": doc_id, "page": page, "band": band, "seed": page_seed, "role": role, **more}


def t5_entries(plan: dict) -> list[dict]:
    """Manifest entries of T5's catalogue pages and of the read-back pages."""
    entries = []
    for document in plan["documents"]:
        unseen = document["slice"] == "unseen"
        for number in document["pages"]:
            band = "unseen" if unseen else draw_band(f"{document['doc_id']}/{number}", survey5.SEED)
            entries.append(
                entry(
                    SET_UNSEEN if unseen else SET_MAIN,
                    document["doc_id"],
                    number,
                    band,
                    "unseen5" if unseen else "test5",
                    position=document["position"],
                    source="catalog",
                )
            )
    for item in plan["readback"]:
        entries.append(entry(SET_READBACK, item["doc_id"], item["page"], "clean", "readback", source="catalog"))
    return entries


def render_t5(data_dir: Path, catalog_root: Path, plan: dict, progress=print, sets=None) -> int:
    """Render, damage and label every page of `t5_entries` (those of `sets` if given), from the truth the survey kept."""
    by_id = {d.doc_id: d for d in catalog.list_documents(catalog_root)[0]}
    count = 0
    for item in t5_entries(plan):
        if sets and item["set"] not in sets:
            continue
        kept = json.loads(
            (data_dir / "selection" / "survey5" / item["doc_id"] / f"{item['page']}.json").read_text(encoding="utf-8")
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


def v4_development(split: Path, data_dir: Path) -> tuple[list[dict], dict[str, dict]]:
    """Every page of the v4 development sets with its band, and the sidecar of each development invoice."""
    plan = json.loads(split.read_text(encoding="utf-8"))["pages"]
    sidecars = {}
    for set_name in DEVELOPMENT_SETS:
        for path in sorted((data_dir / "invoices" / set_name).glob("*.json")):
            sidecars[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    pages = [dict(e, source="catalog") for e in plan]
    for e in json.loads((data_dir / "invoices" / "pages.json").read_text(encoding="utf-8"))["pages"]:
        pages.append(
            dict(e, source="invoice", seed=catalog._seed_text(catalog._page_seed(e["doc_id"], e["page"], 20261002)))
        )
    return pages, sidecars


def v5_entries(pages: list[dict], sidecars: dict[str, dict]) -> tuple[list[dict], dict]:
    """Roles of the v4 pages for v5: development with a fold, aside (invoices of the held-out layouts and fonts),
    shift (the v4 unseen slice) and aux (the v4 undamaged and pilot pages, which belong to development documents but
    are not evaluated)."""
    aside = {doc for doc, side in sidecars.items() if invoices.held_out_in_v5(side["layout"], side["font"])}
    units: dict[str, Counter] = defaultdict(Counter)
    for page in pages:
        if page["set"] in DEVELOPMENT_SETS and page["doc_id"] not in aside:
            units[page["doc_id"]][page["band"]] += 1
    folds = assign_folds({unit: dict(bands) for unit, bands in units.items()})
    out = []
    for page in sorted(pages, key=lambda e: (e["set"], e["doc_id"], e["page"])):
        keep = {k: page[k] for k in ("set", "doc_id", "page", "band", "seed", "source")}
        if page["set"] == "unseen":
            role = "shift"
        elif page["set"] in ("clean", "pilot"):
            role = "aux"
        elif page["doc_id"] in aside:
            role = "aside"
        else:
            role = "dev"
            keep["fold"] = folds["assignment"][page["doc_id"]]
        out.append({**keep, "role": role})
    return out, folds


def sources_rows(plan: dict, documents: dict) -> list[list]:
    """SOURCES.csv rows of the new catalogue documents, all of T5 (`test5`); the read-back pages are pages of them."""
    rows = [(document["doc_id"], "test5") for document in plan["documents"]]
    out = []
    for doc_id, split in rows:
        d = documents[doc_id]
        out.append([d.doc_id, "catalog", d.url, d.sha256, d.licence, split, d.catalog_id])
    return out


def append_sources(path: Path, rows: list[list]) -> int:
    """Add the rows that are not in the file yet; the v4 rows are untouched."""
    with open(path, newline="", encoding="utf-8") as handle:
        header, *existing = list(csv.reader(handle))
    known = {row[0] for row in existing}
    new = [row for row in rows if row[0] not in known]
    with open(path, "a", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerows(new)
    return len(new)


def invoice_entries(data_dir: Path) -> list[dict]:
    """Manifest entries of the 60 sealed invoices and their sidecar facts."""
    out = []
    listed = json.loads((data_dir / "invoices" / "pages-t5.json").read_text(encoding="utf-8"))["pages"]
    for page in listed:
        side = json.loads((data_dir / "invoices" / "test5" / f"{page['doc_id']}.json").read_text(encoding="utf-8"))
        seed = catalog._seed_text(catalog._page_seed(page["doc_id"], page["page"], survey5.SEED))
        out.append(
            entry(
                SET_INVOICES,
                page["doc_id"],
                page["page"],
                page["band"],
                "test5",
                source="invoice",
                layout=side["layout"],
                font=side["font"],
            )
            | {"seed": seed}
        )
    return out


def wrong_words(record: dict, reading) -> int:
    """Grounded words of the reading that differ from their truth (words on ink the text layer does not describe are
    not counted)."""
    truth = [TruthWord(w["id"], w["text"], tuple(w["box"])) for w in record["truth"]["words"]]
    patches = [tuple(box) for box in record.get("ungrounded", [])]
    alignment = label_words([(w.id, w.text, w.box) for w in reading.words], truth, patches)
    return sum(1 for label in alignment.labels if label.wrong)


def development_errors(data_dir: Path, pages: list[dict]) -> tuple[dict, dict]:
    """Wrong words of every development page in its v4 reading, and its band."""
    from extract.engine.cache import reading_from_dict

    errors, bands = {}, {}
    for page in pages:
        path = data_dir / "pages" / page["set"] / page["doc_id"] / f"{page['page']}.json"
        record = json.loads(path.read_text(encoding="utf-8"))
        key = (page["set"], page["doc_id"], page["page"])
        errors[key] = wrong_words(record, reading_from_dict(record["ocr"]["reading"]))
        bands[key] = page["band"]
    return errors, bands


def font_report(data_dir: Path, catalog_root: Path, plan: dict, development: list[dict]) -> dict:
    """How many font families the T5 catalogue documents have and how many development never saw (one `pdffonts` a page)."""
    from bench.corpus import fonts

    by_id = {d.doc_id: d for d in catalog.list_documents(catalog_root)[0]}
    seen: set[str] = set()
    for page in development:
        if page["source"] == "catalog" and page["role"] in ("dev", "aux"):
            seen |= fonts.page_families(by_id[page["doc_id"]].path, page["page"])
    test: set[str] = set()
    for document in plan["documents"]:
        for number in document["pages"]:
            test |= fonts.page_families(by_id[document["doc_id"]].path, number)
    return {
        "development_families": len(seen),
        "t5_families": len(test),
        "t5_families_never_in_development": len(test - seen),
        "t5_families_never_in_development_share": round(len(test - seen) / max(1, len(test)), 3),
    }


def write_manifest(data_dir: Path, catalog_root: Path, split: Path, sources: Path, out: Path) -> dict:
    """Write `split-v5.json`: every page with its role, the folds, T5's frozen list and the pilot sample."""
    plan = json.loads((data_dir / "survey5" / "t5-plan.json").read_text(encoding="utf-8"))
    pages, sidecars = v4_development(split, data_dir)
    development, folds = v5_entries(pages, sidecars)
    errors, bands = development_errors(data_dir, [e for e in development if e["role"] == "dev"])
    pilot = pilot_sample(errors, bands)
    new_pages = t5_entries(plan) + invoice_entries(data_dir)
    t5_documents = [{k: d[k] for k in ("doc_id", "position", "slice", "pages")} for d in plan["documents"]] + [
        {"doc_id": e["doc_id"], "slice": "invoices", "layout": e["layout"], "font": e["font"]}
        for e in new_pages
        if e["source"] == "invoice" and e["page"] == 1
    ]
    body = {
        "version": 5,
        "seed": survey5.SEED,
        "folds": {k: v for k, v in folds.items() if k != "assignment"},
        "held_out": {"layouts": list(invoices.V5_HELD_LAYOUTS), "fonts": list(invoices.V5_HELD_FONTS)},
        "pages": sorted(development + new_pages, key=lambda e: (e["set"], e["doc_id"], e["page"])),
        "t5": {
            "pages_per_document": plan["pages_per_document"],
            "documents": t5_documents,
            "list_fingerprint": fingerprint_of(t5_documents),
        },
        "pilot": pilot,
    }
    if out.is_file():
        before = json.loads(out.read_text(encoding="utf-8"))["t5"]
        if before["list_fingerprint"] == body["t5"]["list_fingerprint"] and "hashes" in before:
            body["t5"]["hashes"], body["t5"]["pages_hashed"] = before["hashes"], before["pages_hashed"]
    body["fingerprint"] = fingerprint_of({k: v for k, v in body.items() if k != "fingerprint"})
    out.write_text(json.dumps(body, indent=1), encoding="utf-8")
    append_sources(sources, sources_rows(plan, {d.doc_id: d for d in catalog.list_documents(catalog_root)[0]}))
    return body


def summary(body: dict) -> dict:
    """Counts of the manifest, no text: pages and documents by role, band and slice."""
    pages = body["pages"]
    docs = defaultdict(set)
    for page in pages:
        docs[page["role"]].add(page["doc_id"])
    return {
        "pages_by_role": dict(Counter(p["role"] for p in pages)),
        "documents_by_role": {role: len(ids) for role, ids in sorted(docs.items())},
        "t5_pages_by_band": dict(Counter(p["band"] for p in pages if p["role"] == "test5")),
        "pilot": dict(Counter(p["band"] for p in body["pilot"])),
        "folds": body["folds"],
    }


def page_texts(data_dir: Path, sets) -> list[tuple[str, list[str]]]:
    """(doc id, truth words) of every rendered page of the sets, from the page records (invoices have none that matter)."""
    out = []
    for set_name in sets:
        for path in sorted((data_dir / "pages" / set_name).glob("*/*.json")):
            if not path.stem.isdigit():  # the second readings keep `<n>.rr2-<cell>.json` beside the page
                continue
            record = json.loads(path.read_text(encoding="utf-8"))
            if record.get("source") != "invoice":
                out.append((record["doc_id"], [w["text"] for w in record["truth"]["words"]]))
    return out


def verify(data_dir: Path, split: Path = SPLIT_V5) -> dict:
    """The data side of the no-repeat and read-back checks, redone from the rendered pages and not from the survey: no page of T5 repeats a
    development page (first 500 characters) or comes near one (5-shingle Jaccard), no T5 document has a page of another
    T5 document that does, and no document is on both sides. Statistics only."""
    body = json.loads(split.read_text(encoding="utf-8"))
    development = {e["doc_id"] for e in body["pages"] if e["role"] in ("dev", "aside", "aux", "shift")}
    t5_sets = ("test5", "unseen5", "clean5")
    index = survey5.DuplicateIndex()
    keys = set()
    for _, words in page_texts(data_dir, ("train", "val", "test", "unseen", "clean", "pilot")):
        index.add(survey5.shingles(words))
        keys.add(_key(words))
    t5 = page_texts(data_dir, t5_sets)
    worst, repeated, near = 0.0, 0, 0
    own = survey5.DuplicateIndex()
    own_owner: list[str] = []
    cross = 0
    for doc, words in t5:
        shingles = survey5.shingles(words)
        score = index.max_jaccard(shingles)
        worst = max(worst, score)
        repeated += _key(words) in keys
        near += score >= survey5.MAX_JACCARD
        if own_owner and own.max_jaccard(shingles) >= survey5.MAX_JACCARD:
            cross += 1
        own.add(shingles)
        own_owner.append(doc)
    t5_docs = {d for d, _ in t5}
    return {
        "development_pages": len(index.sizes),
        "t5_pages_checked": len(t5),
        "t5_catalogue_documents": len(t5_docs),
        "development_catalogue_documents": len(
            development - {e["doc_id"] for e in body["pages"] if e["source"] == "invoice"}
        ),
        "documents_in_both": len(t5_docs & development),
        "pages_repeating_a_development_page": repeated,
        "pages_with_jaccard_at_or_over_threshold": near,
        "largest_jaccard_against_development": round(worst, 4),
        "threshold": survey5.MAX_JACCARD,
        "t5_pages_near_an_earlier_t5_page": cross,
    }


def _key(words: list[str]) -> str:
    import re

    return hashlib.sha256(re.sub(r"\W+", "", " ".join(words)[:500].lower()).encode()).hexdigest()


def freeze(data_dir: Path, split: Path = SPLIT_V5, out: Path = FREEZE_V5, reads: Path = sealed.READS_PATH) -> dict:
    """Fingerprint T5's images, truth and ungrounded patches before any v5 model exists, in `out` and in the manifest.

    One logged read of the sealed pages (purpose "ocr", for hashing only)."""
    record = sealed.freeze_record(data_dir, out, "T5 as built by bench.corpus.build5, before any v5 model", reads)
    body = json.loads(split.read_text(encoding="utf-8"))
    body["t5"]["hashes"] = {k: record[k] for k in ("images", "truth", "ungrounded")}
    body["t5"]["pages_hashed"] = record["pages"]
    body["fingerprint"] = fingerprint_of({k: v for k, v in body.items() if k != "fingerprint"})
    split.write_text(json.dumps(body, indent=1), encoding="utf-8")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["plan", "pages", "manifest", "fonts", "verify", "freeze"])
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--catalog-root", type=Path, default=catalog.default_catalog_root())
    parser.add_argument("--split", type=Path, default=Path("bench/corpus/split.json"))
    parser.add_argument("--sources", type=Path, default=Path("bench/SOURCES.csv"))
    parser.add_argument("--out", type=Path, default=SPLIT_V5)
    parser.add_argument("--sets", nargs="*", help="render only these sets")
    args = parser.parse_args()
    plan_path = args.data_dir / "survey5" / "t5-plan.json"
    if args.action == "plan":
        rows = [json.loads(x) for x in (args.data_dir / "survey5" / "survey.jsonl").read_text("utf-8").splitlines()]
        plan = plan_t5(rows)
        plan_path.write_text(json.dumps(plan, indent=1), encoding="utf-8")
        print(json.dumps({"documents": len(plan["documents"]), "pages_per_document": plan["pages_per_document"]}))
    elif args.action == "pages":
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        print(render_t5(args.data_dir, args.catalog_root, plan, progress=print, sets=args.sets), "pages built")
    elif args.action == "manifest":
        body = write_manifest(args.data_dir, args.catalog_root, args.split, args.sources, args.out)
        print(json.dumps(summary(body), indent=1))
    elif args.action == "verify":
        report = verify(args.data_dir, args.out)
        Path("bench/results/split-v5-checks.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print(json.dumps(report, indent=1))
    elif args.action == "freeze":
        print(json.dumps(freeze(args.data_dir, args.out), indent=1))
    else:
        body = json.loads(args.out.read_text(encoding="utf-8"))
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        print(json.dumps(font_report(args.data_dir, args.catalog_root, plan, body["pages"]), indent=1))


if __name__ == "__main__":
    main()
