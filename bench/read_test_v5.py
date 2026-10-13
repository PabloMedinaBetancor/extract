"""The one read of the sealed v5 tests, T5 (Spanish) and T5-EN: the fixes of both final models, from ONE reading per page.

    python -m bench.read_test_v5 freeze      # once, when nothing is left to change: models/FROZEN-v5.sha256
    python -m bench.read_test_v5 dry-run     # the same code on development pages (never on T5), written under /tmp
    python -m bench.read_test_v5 read --documents-es N --documents-en N   # the read of T5 and T5-EN (the n of each test is fixed beforehand);
                                             # a second one needs a written reason

Every page is read ONCE by the engine (RUN-T: first reading, and the cells L, Z and ZC cached beside it) and nobody
opens metrics before. Here each page is judged by the two models of the product on that single reading, with the
product's own code (`pipeline.score_page`, `fixes.apply_fixes`):

* `default`: `models/guard.model` on the first reading, with no second readings (the product's default mode);
* `thorough`: `models/guard-thorough.model` on the same reading with the readings of L and Z put together (the cell of
  the thorough mode, `readings.product_spec(True)`), through `Guard.load(path, thorough=True)`.

Each model is applied in `safe` and in `more`, and the accounting of `bench/guard/accounting.py` (the one the
out-of-fold report uses) counts fixed, broken, changed-still-wrong and left words against the truth of the text layer
(the same normalisation as `bench.readback`). The report is per language, never pooled: for each language and
slice (`pages`, `invoices`, `unseen`, `all`) and each model and mode: the table, `d_WER` and `d_CER` with the 95 %
cluster-bootstrap interval over documents (each mode against doing nothing), and the paired difference
"thorough - default" per mode, resampling the same documents. N10 is on the invoices: the arithmetic suggestions that
`safe` leaves, by rule, and the fraction of them that is the true value. F1-F3 and N6 of the invoices go with it.

`read` refuses to start if a frozen file changed, if the images or the truth of a sealed page differ from the freeze
records of `sealed.freeze_record`, if the engine that read them is not the frozen one, if a read document is not a
whole prefix of exactly n documents of the frozen list or lacks a cell file, or if the test was already read for metrics. The metrics read is
written in the register of `sealed.py` once the numbers exist (they are computed in memory, written to a temporary file,
the two reads registered, the file renamed), and what comes out is printed as it is written, in
`bench/results/read-test-v5.json`. After that no threshold, model or rule moves.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench import measure_extraction, read_test
from bench.corpus import sealed
from bench.guard import accounting, dataset, pairs
from extract import checks, fixes, pipeline
from extract.engine.cache import reading_from_dict
from extract.fields import extract_invoice
from extract.guard import Guard, readings
from extract.options import Options

FROZEN = Path("models/FROZEN-v5.sha256")
OUT = Path("bench/results/read-test-v5.json")
MANIFEST = Path("bench/corpus/split-v5.json")
FROZEN_GLOBS = (  # everything that runs in the read: the whole product, the bench code it calls, the models and the lists
    "models/*.model",
    "src/**/*.py",
    "src/extract/data/*",
    "src/extract/guard/data/*",
    "bench/*.py",
    "bench/guard/*.py",
    "bench/corpus/*.py",
    "bench/corpus/split-v5.json",
    "bench/SOURCES.csv",
    "bench/SOURCES_EN.csv",
)
MODELS = ("default", "thorough")
MODES = ("safe", "more")
CELLS = ("L", "Z")
SLICES = ("pages", "invoices", "unseen")
LANGUAGES = ("es", "en")  # the languages of the report; the product's own are `Options().languages`
RESAMPLES = 2000
SEED = 1
ARITHMETIC = {
    f"checks.rule.{name}": name for name in ("line_amount", "lines_sum", "vat_amount", "withholding_amount", "total")
}


class FrozenError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------- the freeze


def frozen_files(root: Path) -> list[Path]:
    found = {path for pattern in FROZEN_GLOBS for path in root.glob(pattern) if path.is_file()}
    return sorted(found, key=lambda path: path.relative_to(root).as_posix())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def freeze(root: Path, engine: str) -> int:
    target = root / FROZEN
    if target.exists():
        raise FrozenError(f"{FROZEN} exists: the frozen set is written once")
    lines = [f"{digest(path)}  {path.relative_to(root).as_posix()}" for path in frozen_files(root)]
    target.write_text(f"# engine {engine}\n" + "\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def verify(root: Path) -> str:
    """The engine of the freeze, after checking that every listed file is as it was and no listed glob has a new one."""
    target = root / FROZEN
    if not target.is_file():
        raise FrozenError(f"{FROZEN} is missing: freeze before reading the test")
    listed, engine = {}, None
    for line in target.read_text(encoding="utf-8").splitlines():
        if line.startswith("# engine "):
            engine = line.removeprefix("# engine ")
            continue
        hashed, name = line.split("  ", 1)
        listed[name] = hashed
    changed = [name for name, hashed in listed.items() if not (root / name).is_file() or digest(root / name) != hashed]
    added = [name for name in (path.relative_to(root).as_posix() for path in frozen_files(root)) if name not in listed]
    if changed or added:
        raise FrozenError(f"changed since the freeze: {changed}; not in the freeze: {added}")
    if engine is None:
        raise FrozenError(f"{FROZEN} does not name the engine")
    return engine


# ----------------------------------------------------------------------------------------------- one page, judged


def language_of(record: dict, data: Path) -> str:
    """The language of a page: that of the sidecar for an invoice, else that of its set (`...en` sets are English)."""
    if record.get("source") == "invoice":
        sidecar = data / "invoices" / record["set"] / f"{record['doc_id']}.json"
        return json.loads(sidecar.read_text(encoding="utf-8"))["language"]
    return "en" if record["set"].endswith("en") else "es"


def slice_of(record: dict) -> str:
    if record["set"].startswith("unseen"):
        return "unseen"
    return "invoices" if record.get("source") == "invoice" else "pages"


def edits_of(original, applied) -> list[accounting.Edit]:
    """The edits `fixes.apply_fixes` applied, as the accounting's edits over the word indexes of the first reading."""
    index = {w.id: i for i, w in enumerate(original.words)}
    edits = []
    for word in applied.words:
        if not fixes.is_applied(word):
            continue
        fix = word.fix
        if fix.op == "join":
            edits.append(accounting.Edit("join", tuple(index[i] for i in fix.ids), word.text))
        elif fix.op == "drop":
            edits.append(accounting.Edit("drop", (index[word.id],)))
        else:
            edits.append(accounting.Edit(fix.op, (index[word.id],), word.text))
    return edits


def cell_files(path: Path) -> list[Path]:
    from bench.run_pass2 import cell_path

    return [cell_path(path, cell) for cell in CELLS]


def cells_nothing_doubtful(path: Path) -> bool:
    """True when every cell file of the page exists, did not fail and chose no line to read again (`units_line` 0): a page
    with nothing doubtful has no second readings by construction, which is not a missing cell."""
    for cell in cell_files(path):
        if not cell.is_file():
            return False
        report = json.loads(cell.read_text(encoding="utf-8")).get("report") or {}
        if report.get("units_line") or report.get("reader_failed"):
            return False
    return True


def tally_dict(tallied: dict) -> dict:
    return {
        "weights": dict(tallied["weights"]),
        "components": dict(tallied["components"]),
        "truth_words": tallied["truth_words"],
        "truth_characters": tallied["truth_characters"],
        "distance_before": tallied["distance_before"],
        "distance_after": tallied["distance_after"],
    }


def judge_page(path: Path, record: dict, guards: dict[str, Guard], data: Path) -> dict:
    """One page: its tally with no edit (`none`) and for each model and mode, plus the invoice's N10 suggestions."""
    out = {
        "doc": record["doc_id"],
        "page": record["page"],
        "language": language_of(record, data),
        "slice": slice_of(record),
        "band": record.get("band"),
    }
    reading = reading_from_dict(record["ocr"]["reading"])
    if reading.status != "ocr" or not reading.words:
        return {**out, "unreadable": True}
    page = pairs.page_from_reading(record, reading)
    texts = [w.text for w in reading.words]
    out["tallies"] = {"none": tally_dict(accounting.tally(page.components, texts, []))}
    warnings: Counter = Counter()
    options = Options()  # the product's default languages and review budget, not literals of the reader
    languages, budget = options.languages, pipeline.REVIEW_BUDGETS[options.review]
    spec = readings.product_spec(True)
    out["without_cells"] = False
    out["without_doubtful_units"] = False
    for name, guard in guards.items():
        shown = reading
        if name == "thorough":
            shown = pairs.with_cells(reading, path, CELLS, spec)
            if not any(w.readings for w in shown.words):
                nothing = cells_nothing_doubtful(path)
                out["without_cells"] = not nothing
                out["without_doubtful_units"] = nothing
        scored = pipeline.score_page(shown, guard, languages, budget, lambda w: warnings.update([w.code]))
        for mode in MODES:
            applied, _ = fixes.apply_fixes(scored, mode, lambda w: warnings.update([w.code]))
            out["tallies"][f"{name}|{mode}"] = tally_dict(
                accounting.tally(page.components, texts, edits_of(reading, applied))
            )
            if mode == "safe" and record.get("source") == "invoice":
                out.setdefault("n10", {})[name] = arithmetic_suggestions(applied, page, record, data)
    out["warnings"] = dict(warnings)
    return out


def arithmetic_suggestions(applied, page, record: dict, data: Path) -> list[dict]:
    """N10 on one invoice page of the read: see `suggestions_of`."""
    truth = json.loads((data / "invoices" / record["set"] / f"{record['doc_id']}.json").read_text(encoding="utf-8"))
    return suggestions_of(applied.words, extract_invoice(applied), truth["language"], page.labels)


def suggestions_of(words, invoice, language: str, labels) -> list[dict]:
    """The one-digit suggestions by an arithmetic rule that `safe` leaves on an invoice (N10): the rule, and whether the
    suggested text is the true text of that word (`unaligned` when the word has no one-to-one truth word)."""
    index = {w.id: i for i, w in enumerate(words)}
    found = []
    for result in checks.run(words, invoice, language, suggest=True):
        suggestion = result.suggestion
        if (
            suggestion is None
            or not suggestion.verified
            or suggestion.rule not in ARITHMETIC
            or len(result.word_ids) != 1
        ):
            continue
        i = index.get(result.word_ids[0])
        if i is None:
            continue
        text = words[i].text
        suggested = text[: suggestion.position] + suggestion.new + text[suggestion.position + 1 :]
        label = labels[i]
        if label.wrong is None or label.truth_text is None or label.kind in ("merge", "split", "mixed", "invented"):
            verdict = "unaligned"
        else:
            verdict = (
                "true"
                if measure_extraction._plain(suggested) == measure_extraction._plain(label.truth_text)
                else "false"
            )
        found.append({"rule": ARITHMETIC[suggestion.rule], "verdict": verdict})
    return found


# ----------------------------------------------------------------------------------------------------- statistics


def interval(num: np.ndarray, den: np.ndarray, resamples: int = RESAMPLES, seed: int = SEED) -> dict:
    """Ratio of sums over documents with its 95 % cluster bootstrap interval (whole documents are resampled; the same
    seed and the same documents give the same resamples, so two calls on the same documents are paired)."""
    count = len(num)
    total = float(den.sum())
    if not count or not total:
        return {"value": None, "low": None, "high": None, "documents": int(count)}
    chosen = np.random.default_rng(seed).integers(0, count, size=(resamples, count))
    sums = den[chosen].sum(axis=1)
    ok = sums > 0
    stats = num[chosen].sum(axis=1)[ok] / sums[ok]
    low, high = np.percentile(stats, [2.5, 97.5])
    return {"value": float(num.sum() / total), "low": float(low), "high": float(high), "documents": int(count)}


def per_document(pages: list[dict], key: str) -> tuple[list[str], dict[str, np.ndarray]]:
    """For each document of `pages` the sums of the series `key`: fixed, broken, changed, left_wrong, truth words,
    truth characters and the edit distance change."""
    totals: dict[str, list[float]] = defaultdict(lambda: [0.0] * 7)
    for page in pages:
        tallied = page["tallies"][key]
        weights = tallied["weights"]
        row = totals[page["doc"]]
        row[0] += weights.get("fixed", 0)
        row[1] += weights.get("broken", 0)
        row[2] += weights.get("changed_still_wrong", 0)
        row[3] += weights.get("left_wrong", 0)
        row[4] += tallied["truth_words"]
        row[5] += tallied["truth_characters"]
        row[6] += tallied["distance_after"] - tallied["distance_before"]
    docs = sorted(totals)
    matrix = np.array([totals[d] for d in docs], dtype=np.float64).reshape(len(docs), 7)
    names = ("fixed", "broken", "changed", "left_wrong", "truth_words", "truth_characters", "distance_change")
    return docs, {name: matrix[:, i] for i, name in enumerate(names)}


def series_report(pages: list[dict], key: str, resamples: int) -> dict:
    _, d = per_document(pages, key)
    _, none = per_document(pages, "none")
    applied = d["fixed"] + d["broken"] + d["changed"]
    wrong = none["left_wrong"]  # nothing changed: the words wrong as read
    return {
        "documents": len(d["fixed"]),
        "truth_words": int(d["truth_words"].sum()),
        "fixed": int(d["fixed"].sum()),
        "broken": int(d["broken"].sum()),
        "changed_still_wrong": int(d["changed"].sum()),
        "applied": int(applied.sum()),
        "wrong_as_read": int(wrong.sum()),
        "wer_as_read": interval(wrong, d["truth_words"], resamples),
        "d_wer": interval(d["broken"] - d["fixed"], d["truth_words"], resamples),
        "d_cer": interval(d["distance_change"], d["truth_characters"], resamples),
        "precision": interval(d["fixed"], applied, resamples),
        "net_fixed_per_100_wrong": _per_100(interval(d["fixed"] - d["broken"], wrong, resamples)),
    }


def _per_100(block: dict) -> dict:
    return {k: (None if v is None else v * 100 if k != "documents" else v) for k, v in block.items()}


def paired_report(pages: list[dict], mode: str, resamples: int) -> dict:
    """thorough - default in one mode over the same documents: the change of `d_WER` (negative: thorough has fewer
    errors) and of the net fixed words per 100 wrong words."""
    docs_t, t = per_document(pages, f"thorough|{mode}")
    docs_d, d = per_document(pages, f"default|{mode}")
    _, none = per_document(pages, "none")
    assert docs_t == docs_d
    return {
        "d_wer_difference": interval(
            (t["broken"] - t["fixed"]) - (d["broken"] - d["fixed"]), t["truth_words"], resamples
        ),
        "net_fixed_per_100_wrong_difference": _per_100(
            interval((t["fixed"] - t["broken"]) - (d["fixed"] - d["broken"]), none["left_wrong"], resamples)
        ),
    }


def table_for(pages: list[dict], resamples: int) -> dict:
    readable = [p for p in pages if not p.get("unreadable")]
    block = {
        "pages": len(pages),
        "unreadable_pages": len(pages) - len(readable),
        "documents": len({p["doc"] for p in readable}),
        "none": series_report(readable, "none", resamples) if readable else None,
        "models": {},
        "paired_thorough_minus_default": {},
    }
    if not readable:
        return block
    for name in MODELS:
        block["models"][name] = {mode: series_report(readable, f"{name}|{mode}", resamples) for mode in MODES}
    for mode in MODES:
        block["paired_thorough_minus_default"][mode] = paired_report(readable, mode, resamples)
    return block


def upper_bound_95(count: int, total: int) -> float | None:
    """The 95 % upper bound of a rate with `count` failures in `total`: the rule of three when there are none, else the
    Wilson bound."""
    if not total:
        return None
    if count == 0:
        return 3.0 / total
    p, z = count / total, 1.96
    centre = p + z * z / (2 * total)
    spread = z * ((p * (1 - p) / total + z * z / (4 * total * total)) ** 0.5)
    return (centre + spread) / (1 + z * z / total)


def n10_report(pages: list[dict]) -> dict:
    """N10 per model: the arithmetic suggestions `safe` leaves on the invoices, by rule and in all."""
    out = {}
    for name in MODELS:
        counts: dict[str, Counter] = defaultdict(Counter)
        counts["all"]  # noqa: B018 - `all` is reported even when no suggestion was left (zero is a result)
        invoices = 0
        for page in pages:
            if page.get("unreadable") or page["slice"] != "invoices":
                continue
            invoices += 1
            for item in page.get("n10", {}).get(name, []):
                counts[item["rule"]][item["verdict"]] += 1
                counts["all"][item["verdict"]] += 1
        block = {}
        for rule, c in sorted(counts.items()):
            total = sum(c.values())
            decided = c["true"] + c["false"]
            block[rule] = {
                "suggestions": total,
                "true": c["true"],
                "false": c["false"],
                "unaligned": c["unaligned"],
                "true_fraction_of_decided": c["true"] / decided if decided else None,
                "false_upper_bound_95": upper_bound_95(c["false"], decided),
            }
        out[name] = {"invoice_pages": invoices, "by_rule": block}
    return out


# ---------------------------------------------------------------------------------------------------------- run


def load_guards(default: Path, thorough: Path) -> dict[str, Guard]:
    return {"default": Guard.load(default), "thorough": Guard.load(thorough, thorough=True)}


def analyse(records: list[tuple[Path, dict]], data: Path, guards: dict[str, Guard], resamples: int = RESAMPLES) -> dict:
    """Every number of the read, from the records of the pages (each with its reading and its cell files)."""
    judged = [judge_page(path, record, guards, data) for path, record in records]
    engines = {record["ocr"]["engine"] for _, record in records if record.get("ocr")}
    report = {
        "engines": sorted(engines),
        "models": {name: guard.model.model_id() for name, guard in guards.items()},
        "pages": len(judged),
        "pages_without_cells": sum(1 for p in judged if p.get("without_cells")),
        "pages_without_doubtful_units": sum(1 for p in judged if p.get("without_doubtful_units")),
        "warnings": dict(sum((Counter(p.get("warnings", {})) for p in judged), Counter())),
        "by_language": {},
    }
    for language in LANGUAGES:
        mine = [p for p in judged if p["language"] == language]
        if not mine:
            continue
        block = {
            name: table_for([p for p in mine if p["slice"] == name], resamples)
            for name in SLICES
            if any(p["slice"] == name for p in mine)
        }
        block["all"] = table_for(mine, resamples)
        block["n10"] = n10_report(mine)
        invoices = [
            (path, record)
            for path, record in records
            if record.get("source") == "invoice" and language_of(record, data) == language
        ]
        if invoices:
            block["extraction"] = read_test.extraction(invoices, data)
        report["by_language"][language] = block
    return report


def check_prefix(manifest: dict, records: list[tuple[Path, dict]], key: str, documents: int) -> None:
    """The read documents are exactly the first `documents` catalogue documents of the frozen list (the n fixed beforehand),
    each whole and with its cell files, and, for T5, every invoice of the list: the read cannot have picked
    its documents or stopped short."""
    listed = manifest[key]["documents"]
    by_doc = defaultdict(list)
    for path, record in records:
        by_doc[record["doc_id"]].append((path, record))
    with_reading = {
        doc: [(p, r) for p, r in items if (r.get("ocr") or {}).get("reading")] for doc, items in by_doc.items()
    }
    read = {doc for doc, items in with_reading.items() if items}
    catalogue = [d["doc_id"] for d in listed if d["slice"] == "pages"]
    if documents > len(catalogue):
        raise FrozenError(f"{documents} documents asked, the frozen list has {len(catalogue)}")
    if {d for d in catalogue if d in read} != set(catalogue[:documents]):
        raise FrozenError(f"the catalogue documents read are not the first {documents} of the frozen list")
    for doc in read:
        if len(with_reading[doc]) != len(by_doc[doc]):
            raise FrozenError(f"document {doc} is read in part")
        for path, record in with_reading[doc]:
            if record["ocr"]["reading"].get("status", "ocr") != "ocr":
                continue  # pass 2 does not run on an unreadable page: it has no cell files and the read counts it apart
            missing = [str(p) for p in cell_files(path) if not p.is_file()]
            if missing:
                raise FrozenError(f"the cells of {path} are missing: {missing}")
    unread_invoices = [d["doc_id"] for d in listed if d["slice"] == "invoices" and d["doc_id"] not in read]
    if unread_invoices:
        raise FrozenError(f"{len(unread_invoices)} invoices of the list have no reading")


def only_read(records: list[tuple[Path, dict]]) -> list[tuple[Path, dict]]:
    return [(path, record) for path, record in records if (record.get("ocr") or {}).get("reading")]


def render(report: dict) -> str:
    return json.dumps(report, indent=1, default=float) + "\n"


@dataclass(frozen=True)
class Source:
    """One sealed test: its sets, the register of its reads, its list in the manifest and its freeze record."""

    language: str
    sets: tuple[str, ...]
    reads: Path
    key: str
    freeze: Path


SOURCES = (
    Source("es", sealed.SEALED_SETS, sealed.READS_PATH, "t5", Path("bench/results/test-freeze-v5.json")),
    Source("en", sealed.EN_TEST_SETS, sealed.EN_READS_PATH, "t5en", Path("bench/results/test-freeze-v5en.json")),
)


def sealed_read(
    root: Path,
    data: Path,
    guards_paths: tuple[Path, Path],
    resamples: int,
    documents: dict[str, int],
    sources: tuple[Source, ...] = SOURCES,
    out: Path = OUT,
) -> str:
    """The read. Every gate comes before any metric; the numbers are computed in memory, written to a temporary file, and
    only then are the reads for metrics registered and the file put in its place: if anything fails before, nobody saw a
    number and no read of the test is spent."""
    engine = verify(root)
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    for source in sources:
        if sealed.metric_reads(root / source.reads):
            raise FrozenError(f"{source.reads} already has a read for metrics; a second one needs a written reason")
    loaded = {}
    for source in sources:
        pages = sealed.read_test_pages(data, "ocr", root / source.reads, sets=source.sets)  # logged as a hash-only read
        previous = json.loads((root / source.freeze).read_text(encoding="utf-8"))[-1]
        parts = sealed.fingerprint_parts(pages)
        if (parts["images"], parts["truth"]) != (previous["images"], previous["truth"]):
            raise FrozenError(
                f"the images or the truth of the {source.language} sealed pages differ from the freeze record"
            )
        check_prefix(manifest, pages, source.key, documents[source.language])
        loaded[source.language] = only_read(pages)
    records = [item for source in sources for item in loaded[source.language]]
    wrong = {r["ocr"]["engine"] for _, r in records if not r["ocr"]["engine"].startswith(engine)}
    if wrong:
        raise FrozenError(f"the pages were read by {sorted(wrong)}, not by the frozen engine {engine}")
    report = analyse(records, data, load_guards(*guards_paths), resamples)
    if report["pages_without_cells"]:
        raise FrozenError(f"{report['pages_without_cells']} pages have no readings of the cells L and Z")
    report["read_pages"] = {language: len(pages) for language, pages in loaded.items()}
    text = render(report)
    target = root / out
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    for source in sources:  # the metrics read is registered when the numbers exist
        sealed.read_test_pages(data, "metrics", root / source.reads, sets=source.sets)
    os.replace(temporary, target)
    return text


def dry_run(data: Path, sets: list[str], guards_paths: tuple[Path, Path], limit: int | None, resamples: int) -> str:
    records = []
    for name in sets:
        seen: set[str] = set()
        for path, record in dataset.iter_page_records(data, name):
            if limit and record["doc_id"] not in seen and len(seen) >= limit:
                continue
            seen.add(record["doc_id"])
            records.append((path, record))
    return render(analyse(records, data, load_guards(*guards_paths), resamples))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["freeze", "dry-run", "read"])
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--default", type=Path, default=Path("models/guard.model"))
    parser.add_argument("--thorough", type=Path, default=Path("models/guard-thorough.model"))
    parser.add_argument("--resamples", type=int, default=RESAMPLES)
    parser.add_argument("--sets", nargs="+", default=["val", "dev5en"], help="dry-run: development sets (never T5)")
    parser.add_argument("--documents", type=int, help="dry-run: the first N documents of each set")
    parser.add_argument("--documents-es", type=int, help="read: the n of T5")
    parser.add_argument("--documents-en", type=int, help="read: the n of T5-EN")
    parser.add_argument("--out", type=Path, default=None, help="dry-run: where it is written")
    args = parser.parse_args(argv)
    root = Path.cwd()
    if args.action == "freeze":
        from extract.engine import ocr

        print(f"{freeze(root, ocr.fingerprint())} files frozen in {FROZEN}")
        return
    if args.action == "dry-run":
        text = dry_run(args.data, args.sets, (args.default, args.thorough), args.documents, args.resamples)
        target = args.out or Path("/tmp/read_test_v5_dry.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
    else:
        if args.documents_es is None or args.documents_en is None:
            raise FrozenError("read needs --documents-es and --documents-en: the n of each test, fixed beforehand")
        documents = {"es": args.documents_es, "en": args.documents_en}
        text = sealed_read(root, args.data, (args.default, args.thorough), args.resamples, documents)
    sys.stdout.write(text)


if __name__ == "__main__":
    try:
        main()
    except (FrozenError, sealed.SealedTestError) as error:
        sys.exit(str(error))
