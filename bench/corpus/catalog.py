"""Builds the labelled page corpus from the document catalogue (an SQLite file next to the original PDFs).

Only documents of public administrations enter. The builder walks the documents in a seeded
random order, takes eligible pages of each up to the quota of its split, and stops when the
quotas are full, so it never reads the whole catalogue. No filter looks at a damaged render
and none uses OCR, so no page is dropped for being hard to read.

    python -m bench.corpus.catalog select
    python -m bench.corpus.catalog render [--sets train val ...]
    python -m bench.corpus.catalog extend-test
    python -m bench.corpus.catalog reground [--sets train val ...]
"""

import argparse
import csv
import hashlib
import json
import os
import re
import sqlite3
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import cv2
import numpy as np
import pypdfium2 as pdfium

from bench.corpus import labels
from bench.corpus.degrade import apply_damage, draw_band, draw_damage, map_box
from bench.corpus.split import assign_split, stable_rng, stable_unit

SEED = 20261002
FRACTIONS = {"train": 0.465, "val": 0.186, "test": 0.279, "unseen": 0.07}
EXTENDED_TEST_DOCUMENTS = 120
QUOTAS = {"train": (100, 3), "val": (40, 2), "test": (60, 2), "unseen": (15, 2)}
CLEAN_PAGES = 40
PILOT_PER_BAND = 10
MIN_WORDS = 150
MAX_IMAGE_COVER = 0.5
MIN_INK_SHARE = 0.9
MAX_JUNK_SHARE = 0.01
MAX_LEADER_WORDS = 2
LEADER = "." * 5
CANDIDATES_PER_DOCUMENT = 14
INK_DPI = 100
CLEAN_DPI = 200
SOURCES_COLUMNS = ["doc_id", "source", "source_url", "sha256", "licence", "split", "catalog_id"]

PUBLIC_DOMAINS = {
    "boe.es": "state official gazette",
    "govinfo.gov": "US federal government, public domain",
    "irs.gov": "US federal government, public domain",
    "nist.gov": "US federal government, public domain",
    "legislation.gov.uk": "UK government, Open Government Licence",
    "poderjudicial.es": "judiciary",
    "codigotecnico.org": "ministry, building code",
    "gobiernodecanarias.org": "regional government",
    "idecanarias.es": "regional government, spatial data",
    "grafcan.es": "regional government, spatial data",
    "tenerife.es": "island council",
    "idegrancanaria.es": "island council, spatial data",
    "aguasgrancanaria.com": "island water council",
    "boplaspalmas.net": "provincial gazette",
    "bopsantacruzdetenerife.es": "provincial gazette",
    "laspalmasgc.es": "municipality",
    "santaluciagc.com": "municipality",
    "telde.es": "municipality",
    "vegadesanmateo.es": "municipality",
    "candelaria.es": "municipality",
    "ingenio.es": "municipality",
    "galdar.es": "municipality",
    "sanmigueldeabona.es": "municipality",
    "guiadeisora.es": "municipality",
    "losrealejos.es": "municipality",
    "valsequillogc.es": "municipality",
    "santamariadeguia.es": "municipality",
    "firgas.es": "municipality",
    "teror.es": "municipality",
    "santacruzdetenerife.es": "municipality",
    "aytolalaguna.es": "municipality",
    "urbanismolalaguna.es": "municipality",
    "mogan.es": "municipality",
    "ayuntamientodetias.es": "municipality",
    "urbanismosantacruz.es": "municipality",
    "guimar.gob.es": "municipality",
    "puertodelacruz.es": "municipality",
    "arona.org": "municipality",
    "aguimes.es": "municipality",
    "pajara.es": "municipality",
    "laoliva.es": "municipality",
    "santabrigida.es": "municipality",
    "redtributarialanzarote.es": "island tax consortium",
}
DATABASE = "catalog.db"
CATALOG_TYPES = ("regulation", "planning", "case_law", "ruling")


@dataclass(frozen=True)
class Document:
    doc_id: str
    catalog_id: int
    url: str
    path: Path
    sha256: str
    licence: str


def default_catalog_root() -> Path:
    """The catalogue folder: the parent of the originals folder that `$EXTRACT_CATALOG` names."""
    return Path(os.environ.get("EXTRACT_CATALOG", "catalog/originals")).parent


def licence_of(url: str | None) -> str | None:
    """The licence class of a source, or None when its publisher is not a public administration."""
    host = (urlparse(url or "").hostname or "").lower()
    for domain, name in PUBLIC_DOMAINS.items():
        if host == domain or host.endswith("." + domain):
            return f"public administration ({name})"
    return None


def list_documents(catalog_root: Path) -> tuple[list[Document], dict[str, int]]:
    """Current PDFs of public administrations in the catalogue, read-only, without their text."""
    connection = sqlite3.connect(f"file:{catalog_root / DATABASE}?mode=ro", uri=True)
    marks = ",".join("?" * len(CATALOG_TYPES))
    rows = connection.execute(
        f"SELECT id, url, original_path, original_sha256 FROM sources "
        f"WHERE state = 'current' AND kind IN ({marks}) ORDER BY id",
        CATALOG_TYPES,
    ).fetchall()
    connection.close()
    documents, skipped = [], {"not_pdf": 0, "private_publisher": 0, "missing_file": 0}
    for catalog_id, url, relative, sha256 in rows:
        if not (relative or "").lower().endswith(".pdf"):
            skipped["not_pdf"] += 1
            continue
        klass = licence_of(url)
        if klass is None:
            skipped["private_publisher"] += 1
            continue
        path = catalog_root / relative
        if not path.is_file():
            skipped["missing_file"] += 1
            continue
        documents.append(Document(f"c{sha256[:10]}", catalog_id, url, path, sha256, klass))
    return documents, skipped


@dataclass(frozen=True)
class EligiblePage:
    number: int
    width_pt: float
    height_pt: float
    words: list
    key: str


def check_page(document: Document, number: int, seen: set[str]) -> tuple[EligiblePage | None, str]:
    """Decide whether page `number` (1-based) can serve as labelled truth. Returns (page, reason).

    A page a tool could not examine is rejected with the reason `tool_error`, never accepted.
    """
    pdf = str(document.path)
    try:
        width, height, words = labels.read_page_truth(pdf, number)
        if len(words) < MIN_WORDS:
            return None, "few_words"
        text = " ".join(w.text for w in words)
        if _junk_share(text) > MAX_JUNK_SHARE:
            return None, "junk_characters"
        if sum(LEADER in w.text for w in words) > MAX_LEADER_WORDS:
            return None, "dot_leaders"
        key = hashlib.sha256(re.sub(r"\W+", "", text[:500].lower()).encode()).hexdigest()
        if key in seen:
            return None, "repeated"
        if _image_cover(pdf, number, width * height) >= MAX_IMAGE_COVER:
            return None, "image_cover"
        if _has_ocr_layer_font(pdf, number):
            return None, "ocr_layer_font"
        if _ink_share(document.path, number, words) < MIN_INK_SHARE:
            return None, "no_ink"
    except (subprocess.SubprocessError, ValueError, OSError, pdfium.PdfiumError):
        return None, "tool_error"
    return EligiblePage(number, width, height, words, key), "ok"


def render_page(path: Path, number: int, dpi: float) -> tuple[np.ndarray, float]:
    """Grey render of a page (1-based) and the exact pixels-per-point scale of the result."""
    document = pdfium.PdfDocument(str(path))
    try:
        page = document[number - 1]
        try:
            width_pt, _ = page.get_size()
            bitmap = page.render(scale=dpi / 72.0, grayscale=True)
            try:
                image = np.array(bitmap.to_numpy(), dtype=np.uint8)
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        document.close()
    return image, image.shape[1] / width_pt


def build_page(
    document: Document,
    page: EligiblePage,
    set_name: str,
    band: str,
    data_dir: Path,
    seed: int = SEED,
    ranges: dict | None = None,
    source: str = "catalog",
    overwrite: bool = False,
) -> None:
    """Render, damage and write one page as `pages/<set>/<doc_id>/<n>.png` plus its `.json`."""
    page_seed = _page_seed(document.doc_id, page.number, seed)
    if band == "clean":
        image, scale = render_page(document.path, page.number, CLEAN_DPI)
        matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        damage = None
    else:
        damage = draw_damage(band, page_seed, ranges)
        image, scale = render_page(document.path, page.number, damage.dpi)
        image, matrix = apply_damage(image, damage)
    pixels = labels.truth_to_pixels(page.words, scale * 72.0, matrix)
    patches = _ungrounded(document.path, page)
    mapped = [[round(v, 2) for v in map_box(patch, scale * 72.0, matrix)] for patch in patches]
    folder = data_dir / "pages" / set_name / document.doc_id
    folder.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(folder / f"{page.number}.png"), image)
    record = {
        "schema": 1,
        "set": set_name,
        "doc_id": document.doc_id,
        "source": source,
        "page": page.number,
        "band": band,
        "image": {"width": int(image.shape[1]), "height": int(image.shape[0]), "dpi": scale * 72.0},
        "matrix": matrix.tolist(),
        "damage": damage.as_dict() if damage else None,
        "truth": {
            "width_pt": page.width_pt,
            "height_pt": page.height_pt,
            "words": [{"id": w.id, "text": w.text, "box": [round(v, 2) for v in w.box]} for w in pixels],
        },
        "ungrounded": mapped,
        "ocr": None,
    }
    target = folder / f"{page.number}.json"
    if not overwrite and _has_ocr(target):
        raise CorpusError(f"{target} already holds an OCR reading; rendering it again would drop it")
    target.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")


class CorpusError(RuntimeError):
    pass


def _has_ocr(path: Path) -> bool:
    return path.is_file() and json.loads(path.read_text(encoding="utf-8")).get("ocr") is not None


def _ungrounded(pdf_path: Path, page: EligiblePage) -> list[tuple[float, float, float, float]]:
    clean, scale = render_page(pdf_path, page.number, INK_DPI)
    return labels.ungrounded_boxes(clean, [w.box for w in page.words], scale)


def build_pdf_page(
    pdf_path: Path,
    page_number: int,
    doc_id: str,
    set_name: str,
    data_dir: Path,
    band: str | None = None,
    seed: int = SEED,
) -> str:
    """Build one page of a PDF that is not in the catalogue (the synthetic invoices) with the same damage and truth.

    The band is drawn from the page id unless given. Returns the band used.
    """
    width, height, words = labels.read_page_truth(str(pdf_path), page_number)
    page = EligiblePage(page_number, width, height, words, "")
    document = Document(doc_id, 0, "", pdf_path, "", "")
    band = band or draw_band(f"{doc_id}/{page_number}", seed)
    build_page(document, page, set_name, band, data_dir, seed, source="invoice")
    return band


def select(data_dir: Path, catalog_root: Path, seed: int = SEED, progress=None) -> dict:
    """Walk the catalogue and choose the pages of every set, without rendering anything.

    The chosen truth is kept in `selection/` (boxes in points) so the pages can be rendered again,
    with other damage ranges, without reading the documents once more.
    """
    documents, skipped = list_documents(catalog_root)
    order = stable_rng(seed, "document-order").permutation(len(documents))
    seen: set[str] = set()
    taken = {name: 0 for name in QUOTAS}
    entries: list[dict] = []
    used: list[tuple[Document, str]] = []
    rejected: dict[str, int] = {}
    for index in order:
        document = documents[index]
        split = assign_split(document.doc_id, seed, FRACTIONS)
        wanted_docs, per_doc = QUOTAS[split]
        if taken[split] >= wanted_docs:
            continue
        pages = _eligible(document, per_doc, seen, seed, rejected)
        if not pages:
            continue
        for page in pages:
            band = "unseen" if split == "unseen" else draw_band(f"{document.doc_id}/{page.number}", seed)
            entries.append(_keep(document, page, split, band, data_dir, seed))
        used.append((document, split))
        taken[split] += 1
        if progress:
            progress(f"{sum(taken.values())}/{sum(q[0] for q in QUOTAS.values())} {split} {document.doc_id}")
        if all(taken[name] >= QUOTAS[name][0] for name in QUOTAS):
            break
    entries += _select_extra(used, seen, data_dir, seed, rejected)
    return {"entries": entries, "documents": used, "taken": taken, "skipped": skipped, "rejected": rejected}


def render_selected(
    data_dir: Path,
    split_path: Path,
    catalog_root: Path,
    sets=None,
    ranges: dict | None = None,
    overwrite: bool = False,
) -> int:
    """Render the pages listed in split.json (those of `sets`, or all) from the kept selection."""
    documents = {d.doc_id: d for d in list_documents(catalog_root)[0]}
    plan = json.loads(split_path.read_text(encoding="utf-8"))
    entries = [entry for entry in plan["pages"] if not sets or entry["set"] in sets]
    for entry in entries:
        _render_entry(entry, documents, data_dir, plan["seed"], ranges, overwrite)
    return len(entries)


def _render_entry(entry: dict, documents: dict, data_dir: Path, seed: int, ranges, overwrite: bool) -> None:
    kept = json.loads(_selection_file(data_dir, entry["set"], entry["doc_id"], entry["page"]).read_text("utf-8"))
    page = EligiblePage(
        entry["page"],
        kept["width_pt"],
        kept["height_pt"],
        [labels.TruthWord(i, t, tuple(box)) for i, t, box in kept["words"]],
        kept["key"],
    )
    build_page(
        documents[entry["doc_id"]], page, entry["set"], entry["band"], data_dir, seed, ranges, overwrite=overwrite
    )


def extend_test(
    data_dir: Path,
    catalog_root: Path,
    split_path: Path,
    sources_path: Path,
    test_documents: int = EXTENDED_TEST_DOCUMENTS,
    progress=None,
) -> dict:
    """Grow the sealed test set to `test_documents` documents without touching a page that is already selected.

    Documents are met in the seeded order of `select`; one joins only if the hash split gives it to the test set
    and the plan does not use it yet, so no other set, no band and no page seed changes and every reading already
    in the cache stays valid. The new pages are rendered without OCR.
    """
    plan = json.loads(split_path.read_text(encoding="utf-8"))
    documents, _ = list_documents(catalog_root)
    by_id = {d.doc_id: d for d in documents}
    in_plan = {entry["doc_id"] for entry in plan["pages"]}
    missing = in_plan - set(by_id)
    if missing:
        raise CorpusError(f"{len(missing)} documents of the plan are no longer in the catalogue")
    have = len({entry["doc_id"] for entry in plan["pages"] if entry["set"] == "test"})
    seed = plan["seed"]
    seen = {json.loads(path.read_text("utf-8"))["key"] for path in sorted((data_dir / "selection").glob("*/*/*.json"))}
    order = stable_rng(seed, "document-order").permutation(len(documents))
    per_doc = QUOTAS["test"][1]
    entries, rejected = [], {}
    for index in order:
        if have + len({e["doc_id"] for e in entries}) >= test_documents:
            break
        document = documents[index]
        if document.doc_id in in_plan or assign_split(document.doc_id, seed, FRACTIONS) != "test":
            continue
        for page in _eligible(document, per_doc, seen, seed, rejected):
            band = draw_band(f"{document.doc_id}/{page.number}", seed)
            entries.append(_keep(document, page, "test", band, data_dir, seed))
        if progress:
            progress(f"{have + len({e['doc_id'] for e in entries})}/{test_documents} test {document.doc_id}")
    for entry in entries:
        _render_entry(entry, by_id, data_dir, seed, None, overwrite=False)
    merged = plan["pages"] + entries
    docs = [by_id[doc_id] for doc_id in sorted({e["doc_id"] for e in merged if e["set"] in QUOTAS})]
    write_manifests({"entries": merged, "documents": [(doc, "") for doc in docs]}, sources_path, split_path, seed)
    return {
        "test_documents": have + len({e["doc_id"] for e in entries}),
        "new_documents": len({e["doc_id"] for e in entries}),
        "new_pages": len(entries),
        "rejected": rejected,
    }


def reground(data_dir: Path, split_path: Path, catalog_root: Path, sets=None, progress=None) -> int:
    """Recompute the `ungrounded` patches of the pages in split.json, keeping everything else, the OCR reading included.

    The patches depend only on the clean render and the selected truth, so they can be redone after the rule
    that finds them changes without rendering or reading a page again.
    """
    documents = {d.doc_id: d for d in list_documents(catalog_root)[0]}
    plan = json.loads(split_path.read_text(encoding="utf-8"))
    count = 0
    for entry in plan["pages"]:
        if sets and entry["set"] not in sets:
            continue
        kept = json.loads(_selection_file(data_dir, entry["set"], entry["doc_id"], entry["page"]).read_text("utf-8"))
        page = EligiblePage(
            entry["page"],
            kept["width_pt"],
            kept["height_pt"],
            [labels.TruthWord(i, t, tuple(box)) for i, t, box in kept["words"]],
            kept["key"],
        )
        target = data_dir / "pages" / entry["set"] / entry["doc_id"] / f"{entry['page']}.json"
        record = json.loads(target.read_text(encoding="utf-8"))
        dpi, matrix = record["image"]["dpi"], np.array(record["matrix"])
        patches = _ungrounded(documents[entry["doc_id"]].path, page)
        record["ungrounded"] = [[round(v, 2) for v in map_box(patch, dpi, matrix)] for patch in patches]
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, target)
        count += 1
        if progress and count % 50 == 0:
            progress(f"{count} pages")
    return count


def write_manifests(result: dict, sources_path: Path, split_path: Path, seed: int = SEED) -> None:
    entries = sorted(result["entries"], key=lambda e: (e["set"], e["doc_id"], e["page"]))
    body = json.dumps(entries, sort_keys=True)
    split_path.write_text(
        json.dumps(
            {"seed": seed, "fingerprint": hashlib.sha256(body.encode()).hexdigest(), "pages": entries}, indent=1
        ),
        encoding="utf-8",
    )
    write_sources(sources_path, [doc for doc, _ in result["documents"]], entries)


def write_sources(path: Path, documents: list[Document], entries: list[dict]) -> None:
    """One line per catalogue document that has pages in the corpus, with its URL, hash and licence class."""
    split_of = {}
    for entry in entries:
        if entry["set"] in QUOTAS:
            split_of[entry["doc_id"]] = entry["set"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SOURCES_COLUMNS)
        for document in sorted(documents, key=lambda d: d.doc_id):
            if document.doc_id in split_of:
                writer.writerow(
                    [
                        document.doc_id,
                        "catalog",
                        document.url,
                        document.sha256,
                        document.licence,
                        split_of[document.doc_id],
                        document.catalog_id,
                    ]
                )


def _page_seed(doc_id: str, number: int, seed: int) -> int:
    return int(stable_unit(seed, "page", f"{doc_id}/{number}") * 2**31)


def _seed_text(seed: int) -> str:
    return f"{seed:08x}"


def _selection_file(data_dir: Path, set_name: str, doc_id: str, number: int) -> Path:
    return data_dir / "selection" / set_name / doc_id / f"{number}.json"


def _keep(document: Document, page: EligiblePage, set_name: str, band: str, data_dir: Path, seed: int) -> dict:
    target = _selection_file(data_dir, set_name, document.doc_id, page.number)
    target.parent.mkdir(parents=True, exist_ok=True)
    words = [[w.id, w.text, list(w.box)] for w in page.words]
    kept = {"width_pt": page.width_pt, "height_pt": page.height_pt, "key": page.key, "words": words}
    target.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    return {
        "set": set_name,
        "doc_id": document.doc_id,
        "page": page.number,
        "band": band,
        "seed": _seed_text(_page_seed(document.doc_id, page.number, seed)),
    }


def _select_extra(used, seen, data_dir, seed, rejected) -> list[dict]:
    """Undamaged pages for the read-back check (G3) and pilot pages that tune the damage bands.

    Both come from documents of train and validation, on pages no other set uses.
    """
    entries, clean, pilot = [], 0, {"light": 0, "medium": 0, "heavy": 0}
    for document, split in used:
        if split not in ("train", "val"):
            continue
        if clean >= CLEAN_PAGES and all(n >= PILOT_PER_BAND for n in pilot.values()):
            break
        pages = _eligible(document, 1, seen, seed, rejected, salt="extra")
        if not pages:
            continue
        if clean < CLEAN_PAGES:
            entries.append(_keep(document, pages[0], "clean", "clean", data_dir, seed))
            clean += 1
            continue
        band = next(b for b, n in pilot.items() if n < PILOT_PER_BAND)
        entries.append(_keep(document, pages[0], "pilot", band, data_dir, seed))
        pilot[band] += 1
    return entries


def _eligible(document, wanted, seen, seed, rejected, salt="main") -> list[EligiblePage]:
    try:
        count = _page_count(document.path)
    except pdfium.PdfiumError:
        rejected["unopenable"] = rejected.get("unopenable", 0) + 1
        return []
    order = stable_rng(seed, "pages", salt, document.doc_id).permutation(count)[:CANDIDATES_PER_DOCUMENT]
    pages: list[EligiblePage] = []
    for number in order + 1:
        page, reason = check_page(document, int(number), seen)
        if page is None:
            rejected[reason] = rejected.get(reason, 0) + 1
            continue
        seen.add(page.key)
        pages.append(page)
        if len(pages) == wanted:
            break
    return pages


def _page_count(path: Path) -> int:
    document = pdfium.PdfDocument(str(path))
    try:
        return len(document)
    finally:
        document.close()


def _junk_share(text: str) -> float:
    """Share of private-use, replacement and control characters: signs of a font without a Unicode map."""
    junk = sum(1 for c in text if 0xE000 <= ord(c) <= 0xF8FF or ord(c) == 0xFFFD or (ord(c) < 32 and c not in "\n\t"))
    return junk / max(1, len(text))


def _image_cover(pdf: str, number: int, page_area: float) -> float:
    listing = _run(["pdfimages", "-list", "-f", str(number), "-l", str(number), pdf])
    covered = 0.0
    for line in listing.splitlines()[2:]:
        fields = line.split()
        if len(fields) < 14:
            continue
        width, height, xppi, yppi = float(fields[3]), float(fields[4]), float(fields[12]), float(fields[13])
        if xppi > 0 and yppi > 0:
            covered += (width / xppi * 72.0) * (height / yppi * 72.0)
    return covered / page_area if page_area else 1.0


def _has_ocr_layer_font(pdf: str, number: int) -> bool:
    listing = _run(["pdffonts", "-f", str(number), "-l", str(number), pdf])
    for line in listing.splitlines()[2:]:
        fields = line.split()
        if "glyphless" in line.lower() or ("Type 3" in line and fields[-3] == "no"):
            return True
    return False


def _ink_share(path: Path, number: int, words) -> float:
    image, scale = render_page(path, number, INK_DPI)
    height, width = image.shape
    inked = 0
    for word in words:
        x0, y0, x1, y1 = (int(v * scale) for v in word.box)
        region = image[max(0, y0) : min(height, y1 + 1), max(0, x0) : min(width, x1 + 1)]
        inked += bool(region.size and region.min() < 128)
    return inked / len(words)


def _run(command: list[str]) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, errors="replace", timeout=60).stdout
    except FileNotFoundError:
        # A missing tool says nothing about the page; left alone, every page would be rejected as `tool_error`.
        raise CorpusError(
            f"{command[0]} is not installed; it comes with poppler-utils (sudo apt install poppler-utils)"
        ) from None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("action", choices=["select", "render", "sources", "extend-test", "reground"])
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--catalog-root", type=Path, default=default_catalog_root())
    parser.add_argument("--sources", type=Path, default=Path("bench/SOURCES.csv"))
    parser.add_argument("--split", type=Path, default=Path("bench/corpus/split.json"))
    parser.add_argument("--sets", nargs="*", help="render or reground only these sets")
    parser.add_argument("--overwrite", action="store_true", help="also replace pages that already hold an OCR reading")
    args = parser.parse_args()
    if args.action == "select":
        result = select(args.data_dir, args.catalog_root, progress=lambda line: print(line, file=sys.stderr))
        write_manifests(result, args.sources, args.split)
        print(json.dumps({k: result[k] for k in ("taken", "skipped", "rejected")}))
    elif args.action == "reground":
        print(
            reground(
                args.data_dir,
                args.split,
                args.catalog_root,
                args.sets,
                progress=lambda line: print(line, file=sys.stderr),
            ),
            "pages regrounded",
        )
    elif args.action == "extend-test":
        print(
            json.dumps(
                extend_test(
                    args.data_dir,
                    args.catalog_root,
                    args.split,
                    args.sources,
                    progress=lambda line: print(line, file=sys.stderr),
                )
            )
        )
    elif args.action == "sources":
        plan = json.loads(args.split.read_text(encoding="utf-8"))
        write_sources(args.sources, list_documents(args.catalog_root)[0], plan["pages"])
    else:
        print(
            render_selected(args.data_dir, args.split, args.catalog_root, args.sets, overwrite=args.overwrite),
            "pages rendered",
        )


if __name__ == "__main__":
    main()
