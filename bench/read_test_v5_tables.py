"""The second read of the sealed v5 tests: ONLY the pre-registered tables the first read lacked.

    python -m bench.read_test_v5_tables dry-run --sets val dev5en --documents 5   # development pages only, never T5
    python -m bench.read_test_v5_tables read --documents-es 130 --documents-en 100 # the second read of T5 and T5-EN

What the first read (`bench/results/read-test-v5.json`, `bench.read_test_v5`) did not keep, and this one adds (the review of the first read):

* per language, per slice and band, per page source and per component kind (pair, merge, split, mixed, invented): the
  same table as the first read (`read_test_v5.table_for`: d_WER, d_CER, precision, net per 100 wrong, the paired
  difference thorough - default) with the number of documents behind each;
* per language, model and mode, the applied fixes by the family of the edit, `op` (text, join, segment, drop), `kind` and
  candidate `source`: fixed, broken, changed still wrong, precision, broken as a share of fixed and net fixed per
  100 wrong words, each with the 95 % cluster-bootstrap interval over the same documents of the language. Every
  component that an applied edit touches is credited to the first applied edit that touches it, so these tables add up to
  the totals of the first read (asserted page by page);
* the retrained detector against the v4 detector at the 5 % review budget on the first readings, and the raw
  OCR's WER of the same pages (the function is `bench.guard.detector_eval.detector_rows`).

It ADDS and never replaces. The same pages are judged by the same code (`read_test_v5.judge_page` and `analyse` run as
they ran), and before anything is written the whole report of the first read is recomputed and compared, number by
number, with the stored `read-test-v5.json`: any difference stops the run with nothing printed or written and no read
registered. The reads are registered (with the reason "second-read-tables") only once the tables exist. The headline numbers stay
those of the first read. The gates of `read_test_v5` apply unchanged: frozen files, freeze records, frozen engine, an
exact prefix of documents; and the second read needs the first one in both registers.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Callable
from unittest import mock

import numpy as np

from bench import read_test_v5 as reader
from bench.corpus import sealed
from bench.guard import accounting, dataset, pairs
from extract import fixes, pipeline
from extract.engine.cache import reading_from_dict
from extract.guard import readings
from extract.options import Options

OUT = Path("bench/results/read-test-v5-tables.json")
FIRST = Path("bench/results/read-test-v5.json")
REASON = "second-read-tables"
KINDS = ("pair", "merge", "split", "mixed", "invented")
FAMILIES = ("op", "kind", "source")
OUTCOMES = ("fixed", "broken", "changed_still_wrong", "left_right")


class TablesError(RuntimeError):
    pass


# ------------------------------------------------------------------------------------------- one page, attributed


def edits_with_fixes(original, applied) -> list[tuple[accounting.Edit, object]]:
    """`read_test_v5.edits_of` with the `Fix` that made each edit (so its `op`, `kind` and `source` are known)."""
    index = {w.id: i for i, w in enumerate(original.words)}
    out = []
    for word in applied.words:
        if not fixes.is_applied(word):
            continue
        fix = word.fix
        if fix.op == "join":
            out.append((accounting.Edit("join", tuple(index[i] for i in fix.ids), word.text), fix))
        elif fix.op == "drop":
            out.append((accounting.Edit("drop", (index[word.id],)), fix))
        else:
            out.append((accounting.Edit(fix.op, (index[word.id],), word.text), fix))
    return out


def attribute(components: list, texts: list[str], pairs_: list[tuple[accounting.Edit, object]]) -> list[dict]:
    """The outcome of every component an applied edit touches, credited to the first applied edit that touches it. An
    edit that overlaps an earlier one is skipped, as `accounting.final_texts` and the application skip it."""
    final = accounting.final_texts(texts, [edit for edit, _ in pairs_])
    used: set[int] = set()
    owner: dict[int, int] = {}
    for number, (edit, _) in enumerate(pairs_):
        if used & set(edit.words):
            continue
        used.update(edit.words)
        for index in edit.words:
            owner[index] = number
    rows = []
    for component in components:
        owners = [owner[i] for i in component.words if i in owner]
        if not owners or not any(final[i] != texts[i] for i in component.words):
            continue
        fix = pairs_[min(owners)][1]
        right_after = accounting.is_right(component, final)
        if component.right_before and right_after:
            outcome = "left_right"
        elif component.right_before:
            outcome = "broken"
        elif right_after:
            outcome = "fixed"
        else:
            outcome = "changed_still_wrong"
        rows.append(
            {"op": fix.op, "kind": fix.kind, "source": fix.source, "outcome": outcome, "weight": component.weight}
        )
    return rows


def attribute_page(path: Path, record: dict, guards: dict, judged: dict) -> dict:
    """The extra series of one page: tallies by component kind and the attribution rows, each per model and mode. The
    page's own tallies, recomputed here, must equal those of `judge_page` (asserted)."""
    if judged.get("unreadable"):
        return {}
    reading = reading_from_dict(record["ocr"]["reading"])
    page = pairs.page_from_reading(record, reading)
    texts = [w.text for w in reading.words]
    options = Options()
    languages, budget = options.languages, pipeline.REVIEW_BUDGETS[options.review]
    spec = readings.product_spec(True)
    by_kind = {"none": {}}
    rows: dict[str, list[dict]] = {}
    for kind in KINDS:
        mine = [c for c in page.components if c.kind == kind]
        by_kind["none"][kind] = reader.tally_dict(accounting.tally(mine, texts, []))
    _check_kinds(judged["tallies"]["none"], by_kind["none"], "none", record)
    for name, guard in guards.items():
        shown = reading
        if name == "thorough":
            shown = pairs.with_cells(reading, path, reader.CELLS, spec)
        scored = pipeline.score_page(shown, guard, languages, budget, lambda w: None)
        for mode in reader.MODES:
            applied, _ = fixes.apply_fixes(scored, mode, lambda w: None)
            key = f"{name}|{mode}"
            pairs_ = edits_with_fixes(reading, applied)
            edits = [edit for edit, _ in pairs_]
            if edits != reader.edits_of(reading, applied):
                raise TablesError(f"{record['doc_id']} p{record['page']} {key}: the edits differ from the first read's")
            by_kind[key] = {
                kind: reader.tally_dict(accounting.tally([c for c in page.components if c.kind == kind], texts, edits))
                for kind in KINDS
            }
            rows[key] = attribute(page.components, texts, pairs_)
            _check_page(judged["tallies"][key], rows[key], by_kind[key], key, record)
    return {"by_kind": by_kind, "rows": rows}


def _check_kinds(tallied: dict, kinds: dict, key: str, record: dict) -> None:
    """The kinds' tallies add up to the whole page's, word weights included."""
    where = f"{record['doc_id']} p{record['page']} {key}"
    for field in ("truth_words", "truth_characters", "distance_before", "distance_after"):
        got = sum(t[field] for t in kinds.values())
        if got != tallied[field]:
            raise TablesError(f"{where}: {field} by kind is {got}, the first read has {tallied[field]}")
    for outcome in ("fixed", "broken", "changed_still_wrong", "left_right", "left_wrong"):
        got = sum(t["weights"].get(outcome, 0) for t in kinds.values())
        if got != tallied["weights"].get(outcome, 0):
            raise TablesError(
                f"{where}: {outcome} by kind is {got}, the first read has {tallied['weights'].get(outcome, 0)}"
            )


def _check_page(tallied: dict, rows: list[dict], kinds: dict, key: str, record: dict) -> None:
    """The attribution adds up to the page's tally, and the kinds add up to the whole page (what the first read has)."""
    where = f"{record['doc_id']} p{record['page']} {key}"
    for outcome in ("fixed", "broken", "changed_still_wrong"):
        got = sum(r["weight"] for r in rows if r["outcome"] == outcome)
        if got != tallied["weights"].get(outcome, 0):
            raise TablesError(
                f"{where}: {outcome} by edit is {got}, the first read has {tallied['weights'].get(outcome, 0)}"
            )
    _check_kinds(tallied, kinds, key, record)


# ----------------------------------------------------------------------------------------------------- the tables


def kind_pages(pages: list[dict], extras: list[dict], kind: str) -> list[dict]:
    """Pseudo pages (document, tallies) of one component kind, to give to `read_test_v5.table_for`."""
    out = []
    for page, extra in zip(pages, extras, strict=True):
        if page.get("unreadable") or not extra:
            continue
        if not extra["by_kind"]["none"][kind]["truth_words"]:
            continue  # no component of this kind on the page: the document is not behind this table
        out.append({"doc": page["doc"], "tallies": {key: kinds[kind] for key, kinds in extra["by_kind"].items()}})
    return out


def families_report(pages: list[dict], extras: list[dict], resamples: int) -> dict:
    """The applied fixes by `op`, `kind` and `source`, per model and mode, each with intervals over the same documents."""
    readable = [(p, e) for p, e in zip(pages, extras, strict=True) if not p.get("unreadable") and e]
    docs = sorted({p["doc"] for p, _ in readable})
    position = {d: i for i, d in enumerate(docs)}
    _, none = reader.per_document([p for p, _ in readable], "none")
    wrong, words = none["left_wrong"], none["truth_words"]
    out: dict = {}
    for name in reader.MODELS:
        for mode in reader.MODES:
            key = f"{name}|{mode}"
            cells: dict[str, dict[str, np.ndarray]] = {}
            for page, extra in readable:
                i = position[page["doc"]]
                for row in extra["rows"][key]:
                    for family in FAMILIES:
                        group = cells.setdefault(f"{family}={row[family]}", {o: np.zeros(len(docs)) for o in OUTCOMES})
                        group[row["outcome"]][i] += row["weight"]
            block = {}
            for group, d in sorted(cells.items()):
                # the same denominator as the first read's (`series_report`): a component an edit changed between two right
                # forms (`left_right`) is reported apart and is not an applied fix
                applied = d["fixed"] + d["broken"] + d["changed_still_wrong"]
                block[group] = {
                    "documents": len(docs),
                    "fixed": int(d["fixed"].sum()),
                    "broken": int(d["broken"].sum()),
                    "changed_still_wrong": int(d["changed_still_wrong"].sum()),
                    "left_right": int(d["left_right"].sum()),
                    "precision": reader.interval(d["fixed"], applied, resamples),
                    "broken_share_of_fixed": reader.interval(d["broken"], d["fixed"], resamples),
                    "net_fixed_per_100_wrong": reader._per_100(
                        reader.interval(d["fixed"] - d["broken"], wrong, resamples)
                    ),
                    "d_wer": reader.interval(d["broken"] - d["fixed"], words, resamples),
                }
            out[key] = block
    return out


def tables_for(language: str, judged: list[dict], extras: list[dict], resamples: int) -> dict:
    """Every added table of one language."""
    mine = [(p, e) for p, e in zip(judged, extras, strict=True) if p["language"] == language]
    if not mine:
        return {}
    pages, ex = [p for p, _ in mine], [e for _, e in mine]
    slices = [(name, [i for i, p in enumerate(pages) if p["slice"] == name]) for name in reader.SLICES]
    slices.append(("all", list(range(len(pages)))))
    block: dict = {"by_band": {}, "by_page_source": {}, "by_component_kind": {}}
    for name, picked in slices:
        if not picked:
            continue
        part = [pages[i] for i in picked]
        bands = sorted({str(p.get("band")) for p in part})
        block["by_band"][name] = {
            band: reader.table_for([p for p in part if str(p.get("band")) == band], resamples) for band in bands
        }
        sources = sorted({p.get("page_source") or "unknown" for p in part})
        block["by_page_source"][name] = {
            source: reader.table_for([p for p in part if (p.get("page_source") or "unknown") == source], resamples)
            for source in sources
        }
        block["by_component_kind"][name] = {
            kind: reader.table_for(kind_pages(part, [ex[i] for i in picked], kind), resamples) for kind in KINDS
        }
    block["by_fix"] = families_report(pages, ex, resamples)
    return block


def check_fix_totals(first_block: dict, by_fix: dict, language: str) -> None:
    """Every family of fixes (`op`, `kind`, `source`) adds up, per model and mode, to the first read's table of the
    language (its `all` slice): fixed, broken and changed still wrong. A component credited twice or never stops the run."""
    if not first_block["all"].get("models"):
        return  # no readable page: nothing was applied
    for name in reader.MODELS:
        for mode in reader.MODES:
            theirs = first_block["all"]["models"][name][mode]
            groups = by_fix[f"{name}|{mode}"]
            for family in FAMILIES:
                for field in ("fixed", "broken", "changed_still_wrong"):
                    got = sum(v[field] for k, v in groups.items() if k.startswith(f"{family}="))
                    if got != theirs[field]:
                        raise TablesError(
                            f"{language} {name}|{mode}: {field} by {family} is {got}, the first read has {theirs[field]}"
                        )


# --------------------------------------------------------------------------------------------- the assertion


def differences(stored, fresh, where: str = "") -> list[str]:
    """Every place where two JSON values differ (exactly: the code is deterministic, same pages, same seed)."""
    if isinstance(stored, dict) and isinstance(fresh, dict):
        out = []
        for key in sorted(set(stored) | set(fresh)):
            if key not in stored or key not in fresh:
                out.append(f"{where}/{key}: present in only one")
            else:
                out += differences(stored[key], fresh[key], f"{where}/{key}")
        return out
    if isinstance(stored, list) and isinstance(fresh, list):
        if len(stored) != len(fresh):
            return [f"{where}: length {len(stored)} against {len(fresh)}"]
        return [
            d for i, (a, b) in enumerate(zip(stored, fresh, strict=True)) for d in differences(a, b, f"{where}[{i}]")
        ]
    return [] if stored == fresh and type(stored) is type(fresh) else [f"{where}: {stored!r} against {fresh!r}"]


def assert_first_read(stored: dict, report: dict) -> None:
    """The report that `read_test_v5.analyse` just made must be the stored first read, number for number. `read_pages`
    is the only key the read adds after `analyse`."""
    fresh = json.loads(reader.render(report))
    expected = {k: v for k, v in stored.items() if k != "read_pages"}
    found = differences(expected, fresh)
    if found:
        raise TablesError(f"the first read is not reproduced ({len(found)} differences), first: {found[:5]}")


# -------------------------------------------------------------------------------------------------------- the run


def analyse(
    records: list[tuple[Path, dict]],
    data: Path,
    guards: dict,
    resamples: int,
    detector: Callable[[list[tuple[Path, dict]], Path, dict, int], dict] | None,
    stored: dict | None,
) -> dict:
    """The first read's report (to be asserted against `stored`) and the added tables, on one judging of each page."""
    cache: dict[int, dict] = {}
    original = reader.judge_page

    def judge(path, record, guards_, data_):
        out = original(path, record, guards_, data_)
        out["page_source"] = record.get("source")
        cache[len(cache)] = out
        return out

    with mock.patch.object(reader, "judge_page", judge):
        report = reader.analyse(records, data, guards, resamples)
    if stored is not None:
        assert_first_read(stored, report)
    judged = [cache[i] for i in range(len(records))]
    extras = [attribute_page(path, record, guards, page) for (path, record), page in zip(records, judged, strict=True)]
    tables = {
        "first_read_reproduced": stored is not None,
        "engines": report["engines"],
        "models": report["models"],
        "pages": report["pages"],
        "by_language": {},
    }
    for language in reader.LANGUAGES:
        block = tables_for(language, judged, extras, resamples)
        if block:
            check_fix_totals(report["by_language"][language], block["by_fix"], language)
            tables["by_language"][language] = block
    tables["g31_m9"] = detector(records, data, guards, resamples) if detector else {"status": "not computed"}
    if detector:  # the raw OCR's WER on the same pages, as the first read has it
        tables["g31_m9"]["raw_ocr_wer_as_read"] = {
            language: block["all"]["none"]["wer_as_read"]
            for language, block in report["by_language"].items()
            if block["all"].get("none")
        }
    return tables


def detector_tables(records: list[tuple[Path, dict]], data: Path, guards: dict, resamples: int) -> dict:
    """The non-inferiority check and the raw WER (`bench.guard.detector_compare`): the retrained detector of each product model against
    the frozen v4 detector at the 5 % review budget, on the first readings, per language and per language and slice;
    never pooled across languages, so the library's `all` group is left out. The raw OCR's WER is the first
    read's `wer_as_read` of the same pages and is added by `analyse`."""
    from bench.guard import detector_compare

    candidates = {name: guard.model for name, guard in guards.items()}
    out = {}
    for label, key in (
        ("by_language", lambda record: reader.language_of(record, data)),
        ("by_language_and_slice", lambda record: f"{reader.language_of(record, data)}|{reader.slice_of(record)}"),
    ):
        groups = detector_compare.rows_by_group(records, key)
        groups.pop("all", None)  # the library no longer pools the languages; drop it if it ever does
        out[label] = detector_compare.compare(groups, candidates, resamples=resamples)
    out["note"] = (
        "v4 recall is measured on the rows of the v5 labelling (ungrounded wrong words are not rows), so it differs "
        "from the v4 test's registered recall; both detectors score the same rows"
    )
    return out


def load_detector() -> Callable:
    try:
        from bench.guard import detector_compare  # noqa: F401 - the detector comparison module
    except ImportError as error:
        raise TablesError("bench.guard.detector_compare is not there") from error
    return detector_tables


def sealed_read_tables(
    root: Path,
    data: Path,
    guards_paths: tuple[Path, Path],
    resamples: int,
    documents: dict[str, int],
    detector: Callable,
    sources: tuple[reader.Source, ...] = reader.SOURCES,
    out: Path = OUT,
    first: Path = FIRST,
) -> str:
    """The second read. The gates are those of `read_test_v5.sealed_read`, except that the register must already hold
    exactly one metrics read of each test; the new one is registered, with its reason, after the tables exist."""
    if detector is None:
        raise TablesError("the second read needs the detector comparison: nothing is read without it")
    engine = reader.verify(root)
    manifest = json.loads((root / reader.MANIFEST).read_text(encoding="utf-8"))
    stored = json.loads((root / first).read_text(encoding="utf-8"))
    for source in sources:
        if sealed.metric_reads(root / source.reads) != 1:
            raise TablesError(f"{source.reads} must hold exactly one read for metrics before the second one")
    loaded = {}
    for source in sources:
        pages = sealed.read_test_pages(data, "ocr", root / source.reads, sets=source.sets)
        previous = json.loads((root / source.freeze).read_text(encoding="utf-8"))[-1]
        parts = sealed.fingerprint_parts(pages)
        if (parts["images"], parts["truth"]) != (previous["images"], previous["truth"]):
            raise reader.FrozenError(
                f"the images or the truth of the {source.language} sealed pages differ from the freeze"
            )
        reader.check_prefix(manifest, pages, source.key, documents[source.language])
        loaded[source.language] = reader.only_read(pages)
    records = [item for source in sources for item in loaded[source.language]]
    wrong = {r["ocr"]["engine"] for _, r in records if not r["ocr"]["engine"].startswith(engine)}
    if wrong:
        raise reader.FrozenError(f"the pages were read by {sorted(wrong)}, not by the frozen engine {engine}")
    tables = analyse(records, data, reader.load_guards(*guards_paths), resamples, detector, stored)
    tables["read_pages"] = {language: len(pages) for language, pages in loaded.items()}
    tables["reason"] = REASON
    tables["first_read_sha256"] = hashlib.sha256((root / first).read_bytes()).hexdigest()
    text = reader.render(tables)
    target = root / out
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    for source in sources:
        sealed.read_test_pages(data, "metrics", root / source.reads, reason=REASON, sets=source.sets)
    os.replace(temporary, target)
    return text


def dry_run(
    data: Path,
    sets: list[str],
    guards_paths: tuple[Path, Path],
    limit: int | None,
    resamples: int,
    detector: Callable | None,
) -> str:
    """The same code on development pages. With nothing stored the output says `first_read_reproduced: false`: there is no
    first read to compare with (the assertion is exercised by the tests, with mutations)."""
    records = []
    for name in sets:
        seen: set[str] = set()
        for path, record in dataset.iter_page_records(data, name):
            if limit and record["doc_id"] not in seen and len(seen) >= limit:
                continue
            seen.add(record["doc_id"])
            records.append((path, record))
    return reader.render(analyse(records, data, reader.load_guards(*guards_paths), resamples, detector, None))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["dry-run", "read"])
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--default", type=Path, default=Path("models/guard.model"))
    parser.add_argument("--thorough", type=Path, default=Path("models/guard-thorough.model"))
    parser.add_argument("--resamples", type=int, default=reader.RESAMPLES)
    parser.add_argument("--sets", nargs="+", default=["val", "dev5en"], help="dry-run: development sets (never T5)")
    parser.add_argument("--documents", type=int, help="dry-run: the first N documents of each set")
    parser.add_argument("--documents-es", type=int)
    parser.add_argument("--documents-en", type=int)
    parser.add_argument("--out", type=Path, default=None, help="dry-run: where it is written")
    args = parser.parse_args(argv)
    if args.action == "dry-run":
        try:
            detector = load_detector()
        except TablesError:
            detector = None
        text = dry_run(args.data, args.sets, (args.default, args.thorough), args.documents, args.resamples, detector)
        target = args.out or Path("/tmp/read_test_v5_tables_dry.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    else:
        if args.documents_es is None or args.documents_en is None:
            raise reader.FrozenError("read needs --documents-es and --documents-en, the n of the first read")
        text = sealed_read_tables(
            Path.cwd(),
            args.data,
            (args.default, args.thorough),
            args.resamples,
            {"es": args.documents_es, "en": args.documents_en},
            load_detector(),
        )
    sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except (TablesError, reader.FrozenError, sealed.SealedTestError) as error:
        sys.exit(str(error))
