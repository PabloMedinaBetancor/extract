"""The data of the chooser: pages with their signals and labels, the sites on them, and the rows of (site, candidate).

Candidates and features are pure functions of the OCR output (`site_rows`); the label of a candidate is the only
place where the truth is read (`site_labels`). Words that no candidate of their site can repair enter with every
label at 0, so the model also learns what an unfixable word looks like.
"""

import dataclasses
import json
import logging
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bench.corpus.labels import Label, TruthWord
from bench.guard import accounting, artifacts, dataset
from extract import checks
from extract.engine.cache import reading_from_dict
from extract.guard import candidates, features, language, readings, signals, structure
from extract.guard.forms import normalise
from extract.types import Word

LOG = logging.getLogger(__name__)
LANGUAGES = ("es", "en")
PRODUCT_CELLS = ("L", "Z")  # the readings of pass 2 the product hands the guard (ZC is for the comparison)
DEFAULT_LANGUAGE = "es"  # the language of a page whose manifest entry says none (every page of v4 and of T5)
DEVELOPMENT_SETS = ("train", "val", "test", "dev5en")  # the sets that hold the development pages of the manifest (D)
MANIFEST = Path(__file__).resolve().parents[1] / "corpus" / "split-v5.json"
SITE_KINDS = ("ok", "substitution")


@dataclass
class Page:
    key: str
    doc: str
    band: str
    words: tuple[Word, ...]
    matrix: np.ndarray
    labels: list[Label]
    fragments: frozenset[int]
    fold: int | None = None
    language: str | None = None  # the language of its document, from the manifest (None when it does not say)
    source: str | None = None  # `catalog` or `invoice`, from the manifest
    sensed: str | None = None  # the sensed language of the page (`sense_pages`); None: the candidates use every list
    _context: candidates.PageContext | None = field(default=None, repr=False)
    artifacts: Counter = field(default_factory=Counter)
    _components: list[accounting.Component] | None = field(default=None, repr=False)
    _component_index: dict[int, accounting.Component] | None = field(default=None, repr=False)
    truth_words: int = 0  # the words of the page's truth (the denominator of the rates of the sample size)

    @property
    def context(self) -> candidates.PageContext:
        if self._context is None:
            languages = language.dictionaries(self.sensed, LANGUAGES)
            self._context = candidates.PageContext(self.words, languages, self.fragments)
        return self._context

    @property
    def components(self) -> list[accounting.Component]:
        """The grounded components of the page (one-to-one pairs, merges, splits, mixed groups, invented words)."""
        if self._components is None:
            self._components = accounting.components_of([w.text for w in self.words], self.labels)
        return self._components

    def component_of(self, word: int) -> accounting.Component | None:
        """The component a word belongs to, or None for an ungrounded word."""
        if self._component_index is None:
            self._component_index = {i: c for c in self.components for i in c.words}
        return self._component_index.get(word)

    def grounded(self) -> np.ndarray:
        """Indexes of the words whose label says right or wrong (the ungrounded ones are out of every set)."""
        return np.array([i for i, label in enumerate(self.labels) if label.wrong is not None], dtype=np.int64)


def dev_pages(manifest: Path) -> dict[tuple[str, str, str], dict]:
    """(set, document, page) of the development pages of the manifest -> {fold, language, source}: role `dev`, never the
    pages set aside. `fold` is the number (`fold3` is 3; None when the entry has none), `language` and `source` are what
    the entry says or None."""
    entries = json.loads(Path(manifest).read_text(encoding="utf-8"))["pages"]
    return {
        (e["set"], e["doc_id"], str(e["page"])): {
            "fold": int(e["fold"].removeprefix("fold")) if e.get("fold") else None,
            "language": e.get("language"),
            "source": e.get("source"),
        }
        for e in entries
        if e["role"] == "dev"
    }


def load_pages(
    data_dir: Path,
    set_names: Sequence[str],
    mark_artifacts: bool = True,
    manifest: Path | None = None,
    cells: Sequence[str] = (),
    spec: readings.ReadingSpec | None = None,
    caps: Mapping[str, int] | None = None,
) -> list[Page]:
    """The readable pages of the sets, with signals and labels, in a fixed order. Sealed sets are refused. The
    components that are oddities of the text layer are left ungrounded unless `mark_artifacts` is off. With a
    `manifest` only its development pages are loaded (not those set aside), each with the fold the manifest gives it; with `cells` (`L`, `Z`) every word carries
    the readings of those cells of pass 2, as the product has them, and a page a cell has not read has none of them; with a `spec`
    they are the words as `readings.apply` shows them (the readers of the cell, the units under its cap), the same function as the runtime."""
    dev = dev_pages(manifest) if manifest else None
    pages: list[Page] = []
    loaded: set = set()
    for set_name in set_names:
        for path, record in dataset.iter_page_records(data_dir, set_name):
            key = (set_name, record["doc_id"], str(record["page"]))
            if dev is not None and key not in dev:
                continue
            page = page_of(record, mark_artifacts, path, cells, spec, caps)
            if page is not None:
                if dev is not None:
                    page.fold, page.language, page.source = (dev[key][k] for k in ("fold", "language", "source"))
                    loaded.add(key)
                pages.append(page)
    if dev is not None:
        check_loaded(dev, set_names, loaded)
    return pages


MAX_MISSING_SHARE = (
    0.01  # a run that left out more of the development pages than this is half done: refuse to train on it
)


def check_loaded(dev: dict, set_names: Sequence[str], loaded: set) -> None:
    """Say how many development pages of the manifest (of the sets asked for) were not loaded, and refuse when it is over
    `MAX_MISSING_SHARE`: the missing ones are pages the engine could not read or that no run has read yet."""
    wanted = {key for key in dev if key[0] in set_names}
    missing = sorted(wanted - loaded)
    if not missing:
        return
    LOG.warning(
        "%d of the %d development pages of the manifest were not loaded: %s", len(missing), len(wanted), missing[:5]
    )
    if len(missing) > MAX_MISSING_SHARE * len(wanted):
        raise ValueError(
            f"{len(missing)} of the {len(wanted)} development pages are not loaded (over {MAX_MISSING_SHARE:.0%}): "
            f"a run that is not finished? first ones: {missing[:5]}"
        )


def page_of(
    record: dict,
    mark_artifacts: bool = True,
    path: Path | None = None,
    cells: Sequence[str] = (),
    spec: readings.ReadingSpec | None = None,
    caps: Mapping[str, int] | None = None,
) -> Page | None:
    reading = reading_from_dict(record["ocr"]["reading"])
    if reading.status != "ocr" or not reading.words:
        return None
    if cells and path is not None:
        reading = with_cells(reading, path, cells, spec, caps)
    return page_from_reading(record, reading, mark_artifacts)


def cap_by_reader(words: Sequence, caps: Mapping[str, int]) -> tuple:
    """The words with the readings of a reader hidden from the unit of rank `caps[reader]` on (an exploration of two cells
    read under different caps, `L` lines and `Z` zones: the product's `ReadingSpec` has one unit and one cap); a word left
    with no reading has `readings = None`. A reading with no rank belongs to no unit and stays."""
    out = []
    for word in words:
        if word.readings:
            kept = tuple(r for r in word.readings if r.source not in caps or r.rank is None or r.rank < caps[r.source])
            if len(kept) != len(word.readings):
                word = dataclasses.replace(word, readings=kept or None)
        out.append(word)
    return tuple(out)


def with_cells(
    reading,
    path: Path,
    cells: Sequence[str],
    spec: readings.ReadingSpec | None = None,
    caps: Mapping[str, int] | None = None,
):
    """The first reading with the readings of every cell of pass 2 that the page has, the cells in the order given; with a
    `spec`, as the runtime shows them (`readings.apply`: the readers of the cell and the units under its cap)."""
    from bench.run_pass2 import load_cell

    words = list(reading.words)
    for cell in cells:
        second = load_cell(path, cell)
        if second is None:
            continue
        for i, (word, read) in enumerate(zip(words, second.words, strict=True)):
            if read.readings:
                words[i] = dataclasses.replace(word, readings=(*(word.readings or ()), *read.readings))
    if spec is not None:
        words = readings.apply(words, spec)
    if caps:
        words = cap_by_reader(words, caps)
    return dataclasses.replace(reading, words=tuple(words))


def page_from_reading(record: dict, reading, mark_artifacts: bool = True) -> Page:
    """The page of a record with the words of `reading` (the first reading, or the same page with its second readings)."""
    truth = [TruthWord(w["id"], w["text"], tuple(w["box"])) for w in record["truth"]["words"]]
    read = [(w.id, w.text, w.box) for w in reading.words]
    alignment, found = artifacts.align(read, truth, record.get("ungrounded", []), mark_artifacts)
    labels = alignment.labels
    matrix = signals.extract(reading.words, LANGUAGES, checks.detect_locale(reading.words))
    fragments = frozenset(np.nonzero(matrix[:, signals.INDEX["hyphen_fragment"]] == 1)[0].tolist())
    return Page(
        f"{record['doc_id']}/{record['page']}",
        record["doc_id"],
        record["band"],
        reading.words,
        matrix,
        labels,
        fragments,
        artifacts=found,
        truth_words=len(truth),
    )


@dataclass
class SiteRows:
    """The candidates of one site and their feature rows, the token as read first."""

    candidates: list[candidates.Candidate]
    x: np.ndarray


def site_rows(page: Page, word: int) -> SiteRows:
    """Candidates and features of one word: a function of the OCR output and the word lists alone."""
    found = candidates.generate(page.context, word)
    return SiteRows(found, features.feature_rows(page.context, word, found, page.matrix[word]))


def site_labels(label: Label, found: Sequence[candidates.Candidate]) -> np.ndarray:
    """1 for the candidate that is the truth: the token as read when the word is right, the candidate whose normalised
    text is the truth word when it is a one-to-one substitution, none for a word that is wrong in a way no single
    token edit repairs (merge, split, mixed, invented). At most one label of a site is 1."""
    out = np.zeros(len(found), dtype=np.uint8)
    if label.kind == "ok":
        out[0] = 1
    elif label.kind == "substitution":
        for i, candidate in enumerate(found[1:], start=1):
            if normalise(candidate.text) == label.truth_text:
                out[i] = 1
                break
    return out


def structure_site_rows(page: Page, family: str, span: tuple[int, ...]) -> SiteRows:
    """Candidates and features of a structure site: a function of the OCR output and the word lists alone."""
    found = structure.generate_structure(page.context, family, span)
    return SiteRows(found, features.structure_rows(page.context, family, span, found, page.matrix[span[0]]))


def structure_labels(page: Page, span: tuple[int, ...], found: Sequence[candidates.Candidate]) -> np.ndarray | None:
    """1 for the candidate after which every component the span touches is right, KEEP's label being 1 when they
    are right as read. None when the span cannot be labelled: it holds a word the layer does not ground, or only
    part of a component (an open one: the other words of a split, a merge or a mixed group lie outside the span), ."""
    touched: dict[int, accounting.Component] = {}
    for word in span:
        component = page.component_of(word)
        if component is None:
            return None
        touched[id(component)] = component
    if any(not set(c.words) <= set(span) for c in touched.values()):
        return None
    texts = [w.text for w in page.words]
    out = np.zeros(len(found), dtype=np.uint8)
    out[0] = all(accounting.is_right(c, texts) for c in touched.values())
    if out[0]:
        return out
    for i, candidate in enumerate(found[1:], start=1):
        final = accounting.final_texts(texts, [accounting.Edit(candidate.op, span, candidate.text)])
        out[i] = all(accounting.is_right(c, final) for c in touched.values())
    return out


def structure_sites(page: Page, processed=None) -> tuple[list[tuple[str, tuple[int, ...]]], Counter]:
    """The labelled structure sites of a page as (family, span), and how many were left out as open or ungrounded.
    With `processed` (word indexes), only the sites that hold one of those words."""
    candidates_of: list[tuple[str, tuple[int, ...]]] = [("join", span) for span in structure.join_spans(page.context)]
    for index in range(len(page.words)):
        if structure.segment_ok(page.context, index):
            candidates_of.append(("segment", (index,)))
        if structure.drop_ok(page.context, index):
            candidates_of.append(("drop", (index,)))
    kept, left_out = [], Counter()
    for family, span in candidates_of:
        if processed is not None and not processed.intersection(span):
            continue
        found = structure.generate_structure(page.context, family, span)
        if len(found) == 1:
            left_out[f"{family}_without_candidates"] += 1
        elif structure_labels(page, span, found) is None:
            left_out[family] += 1
        else:
            kept.append((family, span))
    return kept, left_out


def structure_reachable(page: Page) -> set[int]:
    """The words of a wrong component that some structure candidate repairs (a join, a segmentation or a drop), found
    without building any feature row: what the generation threshold may count as repairable besides the text sites."""
    found: set[int] = set()
    ctx = page.context
    sites: list[tuple[str, tuple[int, ...]]] = [("join", span) for span in structure.join_spans(ctx)]
    for index in range(len(page.words)):
        if structure.segment_ok(ctx, index):
            sites.append(("segment", (index,)))
        if structure.drop_ok(ctx, index):
            sites.append(("drop", (index,)))
    for family, span in sites:
        candidates_of = structure.generate_structure(ctx, family, span)
        if len(candidates_of) == 1:
            continue
        labels = structure_labels(page, span, candidates_of)
        if labels is not None and not labels[0] and labels[1:].any():
            found.update(span)
    return found


@dataclass
class Table:
    """Rows of many sites, one block per site; `starts[s]` is the first row of site `s`."""

    x: np.ndarray
    y: np.ndarray
    site: np.ndarray
    starts: np.ndarray
    page: np.ndarray
    word: np.ndarray
    doc: np.ndarray
    token_right: np.ndarray
    reachable: np.ndarray
    found: list[list[candidates.Candidate]]
    token: np.ndarray
    truth: np.ndarray
    label_kind: np.ndarray
    family: np.ndarray | None = None
    span: list | None = None
    language: np.ndarray | None = None  # per site: the language of its page (`es` when the manifest says none)

    @property
    def sites(self) -> int:
        return len(self.starts)

    def block(self, s: int) -> slice:
        end = self.starts[s + 1] if s + 1 < len(self.starts) else len(self.y)
        return slice(self.starts[s], end)


class SiteCache:
    """Candidates, features and labels of the sites already built, so a site met by several folds is built once."""

    def __init__(self, pages: list[Page]) -> None:
        self.pages = pages
        self._built: dict[tuple[int, int], tuple[SiteRows, np.ndarray]] = {}

    def get(self, page: int, word: int) -> tuple[SiteRows, np.ndarray]:
        key = (page, word)
        if key not in self._built:
            rows = site_rows(self.pages[page], word)
            self._built[key] = (rows, site_labels(self.pages[page].labels[word], rows.candidates))
        return self._built[key]

    def get_structure(self, page: int, family: str, span: tuple[int, ...]) -> tuple[SiteRows, np.ndarray]:
        key = (page, family, span)
        if key not in self._built:
            rows = structure_site_rows(self.pages[page], family, span)
            labels = structure_labels(self.pages[page], span, rows.candidates)
            if labels is None:
                raise ValueError(f"the {family} site {span} of page {page} cannot be labelled")
            self._built[key] = (rows, labels)
        return self._built[key]

    def table(self, sites: Sequence[tuple]) -> Table:
        """The rows of the sites: `(page, word)` for a text site, `(page, word, family, span)` for a structure site
        (`word` is the span's first word)."""
        blocks, found, info = [], [], []
        for site in sites:
            page, word = site[0], site[1]
            if len(site) == 2:
                rows, labels = self.get(page, word)
                label = self.pages[page].labels[word]
                token, truth, kind, family, span = (
                    self.pages[page].words[word].text,
                    label.truth_text,
                    label.kind,
                    "text",
                    (word,),
                )
                right = label.kind == "ok"
            else:
                _, _, family, span = site
                rows, labels = self.get_structure(page, family, span)
                comps = {id(c): c for c in map(self.pages[page].component_of, span)}.values()
                token = " ".join(self.pages[page].words[i].text for i in span)
                truth = " ".join(c.truth for c in sorted(comps, key=lambda c: c.words[0]))
                kind, right = family, bool(labels[0])
            blocks.append((rows.x, labels))
            found.append(rows.candidates)
            info.append((page, word, self.pages[page].doc, token, truth or "", kind, right, family, span))
        sizes = np.array([len(b[1]) for b in blocks], dtype=np.int64)
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]) if blocks else np.zeros(0, dtype=np.int64)
        width = len(features.FEATURES)
        return Table(
            x=np.vstack([b[0] for b in blocks]) if blocks else np.empty((0, width), dtype=np.float32),
            y=np.concatenate([b[1] for b in blocks]) if blocks else np.empty(0, dtype=np.uint8),
            site=np.repeat(np.arange(len(blocks)), sizes),
            starts=starts,
            page=np.array([i[0] for i in info], dtype=np.int64),
            word=np.array([i[1] for i in info], dtype=np.int64),
            doc=np.array([i[2] for i in info], dtype=object),
            token_right=np.array([i[6] for i in info], dtype=bool),
            reachable=np.array([bool(b[1][1:].any()) for b in blocks], dtype=bool),
            found=found,
            token=np.array([i[3] for i in info], dtype=object),
            truth=np.array([i[4] for i in info], dtype=object),
            label_kind=np.array([i[5] for i in info], dtype=object),
            family=np.array([i[7] for i in info], dtype=object),
            span=[i[8] for i in info],
            language=np.array([self.pages[i[0]].language or DEFAULT_LANGUAGE for i in info], dtype=object),
        )


def is_site(page: Page, word: int) -> bool:
    """A grounded word that the candidate sources may edit."""
    return page.labels[word].wrong is not None and page.context.is_site(word)


def sense_pages(pages: list[Page], sensor: "float | dict") -> list[Page]:
    """Give every page its sensed language with the settings of the sensor (a floor, or the dict of `language.sensor_settings`): its candidates then come from the word list
    of that language alone, and a page that is `unsure` keeps every list (the guard withholds its fixes at run time)."""
    for page in pages:
        page.sensed = language.sensed_language(page.words, sensor)
        page._context = None
    return pages


def add_data_arguments(parser) -> None:
    """The arguments every runner of the protocol shares: which pages, where, and with which second readings."""
    import os

    parser.add_argument("--sets", nargs="+", default=list(DEVELOPMENT_SETS))
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=MANIFEST,
        help="split-v5.json: only its development pages, in its folds (the default; `--no-manifest` loads every page)",
    )
    parser.add_argument("--no-manifest", action="store_const", const=None, dest="manifest")
    parser.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="cells of pass 2 whose readings the words carry. Left out: the cell of the product (unit, readers and cap of "
        "`data/second_reading.json`) and the readings as the runtime shows them. Given (`--cells L Z`): an exploration "
        "of those cells, all their readings; none (`--cells`): without second readings",
    )
    parser.add_argument(
        "--cap", type=int, default=None, help="with --cells: the units a page reads at most (view of the runtime cap)"
    )
    parser.add_argument(
        "--thorough",
        action="store_true",
        help="the thorough mode: the lines of L and the zones of Z together, all their readers and no cap; the spec "
        "of the model is `readings.product_spec(True)`",
    )
    parser.add_argument(
        "--by-language",
        action="store_true",
        help="sense the language of every page (floor fitted on the pages) and take its candidates from that list",
    )


def reading_cells(args) -> tuple[list[str], readings.ReadingSpec | None]:
    """The cells to load and the spec of the readings they are shown with: the product's when `--cells` is left out, else
    the cells named (and, with `--cap`, the cap of the view); `args.spec` keeps the product's spec for the model that is saved."""
    if getattr(args, "thorough", False):
        if args.cells is not None:
            raise ValueError("--thorough reads the cells of the mode itself: leave --cells out")
        spec = readings.product_spec(True)
        args.spec = spec
        return ["L", "Z"], spec  # pass 2 stores L and Z apart; the mode shows them together
    if args.cells is None:
        spec = readings.product_spec()
        args.spec = spec
        return [spec.cell], spec
    args.spec = None
    if args.cap is not None:
        return list(args.cells), readings.ReadingSpec("line", ("latin", "wide", "deep", "bin", "clahe"), args.cap)
    return list(args.cells), None


def pages_from_args(args) -> list[Page]:
    cells, spec = reading_cells(args)
    pages = load_pages(args.data, args.sets, manifest=args.manifest, cells=cells, spec=spec)
    if args.by_language:
        from bench.guard import language_sensor

        args.sensor = language.sensor_settings({"floor": language_sensor.fit_floors(pages)})
        sense_pages(pages, args.sensor)
    return pages
