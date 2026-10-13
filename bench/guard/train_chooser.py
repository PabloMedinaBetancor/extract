"""The out-of-fold protocol of the chooser: cross-fitted by document, nested for the detector.

Documents go to five folds. For each outer fold the chooser trains on the other four and scores the fold, and the
detector scores of its training documents are inner out-of-fold (four inner folds), so the training rows are the
words the detector flags among documents it has not seen, as at run time. The pooled out-of-fold predictions are
what calibration, thresholds and every report are made from; the test decides nothing.
"""

import hashlib
from dataclasses import dataclass

import numpy as np

from bench.guard import folds as folding
from bench.guard.pairs import Page, SiteCache, Table, is_site, structure_reachable, structure_sites
from extract.guard import model

DETECTOR_CELL = (7, 100)
CHOOSER_CELL = (31, 200)
GENERATION_RECALL = 0.97
GENERATION_SHARE = 0.20
INNER_FOLDS = 4


@dataclass
class Words:
    """The grounded words of all pages as detector rows."""

    x: np.ndarray
    y: np.ndarray
    page: np.ndarray
    word: np.ndarray
    doc: np.ndarray

    def __len__(self) -> int:
        return len(self.y)


def words_of(pages: list[Page]) -> Words:
    xs, ys, pg, wd, doc = [], [], [], [], []
    for p, page in enumerate(pages):
        grounded = page.grounded()
        xs.append(page.matrix[grounded])
        ys.append(np.array([page.labels[i].wrong for i in grounded], dtype=np.uint8))
        pg.append(np.full(len(grounded), p))
        wd.append(grounded)
        doc.append(np.full(len(grounded), page.doc, dtype=object))
    return Words(*(np.concatenate(part) for part in (xs, ys, pg, wd, doc)))


def document_table(pages: list[Page]) -> dict[str, tuple[str, int]]:
    """Document id -> (band, grounded words), for the folds."""
    table: dict[str, tuple[str, int]] = {}
    for page in pages:
        band, count = table.get(page.doc, (page.band, 0))
        table[page.doc] = (band, count + int((page.grounded()).size))
    return table


def document_folds(pages: list[Page], documents: dict, seed: int) -> dict[str, int]:
    """Document id -> outer fold. The folds recorded in the manifest when every page has one (they are decided
    once, with the sealed test, and every experiment uses them); else the greedy assignment of `folds`. A document whose
    pages are in two folds, or only some pages with a fold, is an error."""
    recorded: dict[str, set] = {}
    for page in pages:
        recorded.setdefault(page.doc, set()).add(page.fold)
    if all(folds_ == {None} for folds_ in recorded.values()):
        return folding.assign_folds(documents, folding.FOLDS, seed)
    bad = sorted(d for d, f in recorded.items() if len(f) != 1 or None in f)
    if bad:
        raise ValueError(f"documents with no fold or in several folds: {bad[:5]}")
    fold = {d: next(iter(f)) for d, f in recorded.items()}
    if not set(fold.values()) <= set(range(folding.FOLDS)):
        raise ValueError(f"folds outside 0..{folding.FOLDS - 1}: {sorted(set(fold.values()))}")
    return fold


def fit_detector(words: Words, rows: np.ndarray, seed: int, threads: int):
    leaves, trees = DETECTOR_CELL
    return model.train_detector(words.x[rows], words.y[rows], leaves, trees, seed=seed, threads=threads)


def outer_scores(words: Words, fold_of_row: np.ndarray, seed: int, threads: int) -> np.ndarray:
    """Raw detector score of every word from a detector trained on the other folds."""
    raw = np.empty(len(words))
    for k in range(folding.FOLDS):
        held_out = fold_of_row == k
        booster = fit_detector(words, np.nonzero(~held_out)[0], seed, threads)
        raw[held_out] = model.raw_scores(booster, words.x[held_out], threads)
    return raw


def inner_scores(
    words: Words, documents: dict, fold: dict[str, int], outer: int, seed: int, threads: int
) -> np.ndarray:
    """Raw detector score of every word outside the outer fold, from a detector that never saw its document: the
    training documents are split again into `INNER_FOLDS` and each is scored by the model of the others."""
    training = [d for d in documents if fold[d] != outer]
    inner = folding.assign_folds({d: documents[d] for d in training}, INNER_FOLDS, seed + outer + 1)
    rows = np.nonzero([fold[d] != outer for d in words.doc])[0]
    inner_of_row = np.array([inner[d] for d in words.doc[rows]])
    raw = np.full(len(words), np.nan)
    for j in range(INNER_FOLDS):
        held_out = inner_of_row == j
        booster = fit_detector(words, rows[~held_out], seed, threads)
        raw[rows[held_out]] = model.raw_scores(booster, words.x[rows[held_out]], threads)
    return raw


def generation_threshold(raw: np.ndarray, flaggable: np.ndarray, repairable: np.ndarray) -> float:
    """`t_gen`: the lowest raw score that keeps `GENERATION_RECALL` of the wrong words some candidate can repair
    while flagging at most `GENERATION_SHARE` of the words (the share wins if both cannot hold)."""
    keeps = np.quantile(raw[repairable], 1 - GENERATION_RECALL, method="lower")
    cap = np.quantile(raw[flaggable], 1 - GENERATION_SHARE, method="higher")
    return float(max(keeps, cap))


@dataclass
class OutOfFold:
    """The pooled held-out predictions of the protocol."""

    table: Table
    raw: dict[str, np.ndarray]
    fold_of_site: np.ndarray
    t_gen: float
    detector_raw: np.ndarray
    words: Words


def permute_within_sites(table: Table, rng: np.random.Generator) -> np.ndarray:
    """The labels with each site's block shuffled among its own candidates: the canary experiment."""
    y = table.y.copy()
    for s in range(table.sites):
        block = table.block(s)
        y[block] = rng.permutation(y[block])
    return y


def run(
    pages: list[Page],
    learners: dict,
    seed: int = 1,
    threads: int = 2,
    cache: SiteCache | None = None,
    permute_labels: bool = False,
    structure: bool = False,
    train_share: float = 1.0,
) -> OutOfFold:
    """The whole protocol. `learners` maps a name to `fit(x, y) -> score(x)` factories (`fit(x, y, groups)` when the
    factory has `needs_groups`): each is trained on the
    flagged sites of the training folds and scores those of the held-out fold. With `permute_labels` the training
    labels are shuffled inside each site, which must leave nothing to learn. With `structure` the joins, the
    segmentations and the drops are sites too: one holding a word the detector flags (`>= t_gen`) is built, and the
    wrong words that a structure candidate repairs count as repairable when `t_gen` is set. With `train_share` under 1
    only that share of the training documents of each fold (chosen by a hash of the document id, so a smaller share
    is always inside a larger one) trains the chooser: the learning curve."""
    cache = cache or SiteCache(pages)
    words = words_of(pages)
    documents = document_table(pages)
    fold = document_folds(pages, documents, seed)
    fold_of_row = np.array([fold[d] for d in words.doc])
    outer = outer_scores(words, fold_of_row, seed, threads)
    unit = {d: document_unit(d) for d in documents}
    in_share = np.array([unit[d] < train_share for d in words.doc])
    site_row = np.array([is_site(pages[p], w) for p, w in zip(words.page, words.word, strict=True)])
    repairable = np.array(
        [bool(site and words.y[i] and _reachable(cache, words, i)) for i, site in enumerate(site_row)], dtype=bool
    )
    if structure:
        reached = [structure_reachable(page) for page in pages]
        repairable |= np.array(
            [bool(words.y[i] and int(words.word[i]) in reached[int(words.page[i])]) for i in range(len(words))],
            dtype=bool,
        )
    t_gen = generation_threshold(outer, np.ones(len(words), dtype=bool), repairable)
    held_sites: list[tuple] = []
    held_fold: list[int] = []
    scores: dict[str, list[np.ndarray]] = {name: [] for name in learners}
    for k in range(folding.FOLDS):
        inner = inner_scores(words, documents, fold, k, seed, threads)
        train_flag = (fold_of_row != k) & (inner >= t_gen) & in_share
        test_flag = (fold_of_row == k) & (outer >= t_gen)
        train_sites = sites_of(words, np.nonzero(site_row & train_flag)[0])
        test_sites = sites_of(words, np.nonzero(site_row & test_flag)[0])
        if structure:
            train_sites = with_structure(pages, words, np.nonzero(train_flag)[0], train_sites)
            test_sites = with_structure(pages, words, np.nonzero(test_flag)[0], test_sites)
        train = cache.table(train_sites)
        test = cache.table(test_sites)
        labels = permute_within_sites(train, np.random.default_rng(seed + k)) if permute_labels else train.y
        groups = np.diff(np.append(train.starts, len(train.y)))
        for name, factory in learners.items():
            # a ranking learner (`needs_groups`) is told how many rows each site has: its queries are the sites
            scorer = (
                factory(train.x, labels, groups)
                if getattr(factory, "needs_groups", False)
                else factory(train.x, labels)
            )
            scores[name].append(scorer(test.x))
        held_sites.extend(test_sites)
        held_fold.extend([k] * len(test_sites))
    table = cache.table(held_sites)
    return OutOfFold(
        table, {name: np.concatenate(parts) for name, parts in scores.items()}, np.array(held_fold), t_gen, outer, words
    )


def document_unit(doc: str) -> float:
    """A number in [0, 1) that depends only on the document id: the share of documents a learning-curve point keeps."""
    return int(hashlib.sha256(doc.encode()).hexdigest()[:8], 16) / 16**8


def sites_of(words: Words, rows: np.ndarray) -> list[tuple]:
    return [(int(words.page[i]), int(words.word[i])) for i in rows]


def with_structure(pages: list[Page], words: Words, rows: np.ndarray, sites: list[tuple]) -> list[tuple]:
    """`sites` and the structure sites of the pages that hold a flagged word (a row of `rows`), as
    `(page, first word, family, span)`."""
    flagged: dict[int, set[int]] = {}
    for i in rows:
        flagged.setdefault(int(words.page[i]), set()).add(int(words.word[i]))
    out = list(sites)
    for page_number in sorted(flagged):
        kept, _ = structure_sites(pages[page_number], flagged[page_number])
        out.extend((page_number, span[0], family, span) for family, span in kept)
    return out


def _reachable(cache: SiteCache, words: Words, i: int) -> bool:
    page, word = int(words.page[i]), int(words.word[i])
    return bool(cache.get(page, word)[1][1:].any())


def lightgbm_learner(cell=CHOOSER_CELL, seed: int = 1, threads: int = 2):
    """Factory for `run`: a LightGBM chooser of the cell, fitted on the training rows."""

    def fit(x, y):
        booster = model.train_chooser(x, y, cell[0], cell[1], seed=seed, threads=threads)
        return lambda rows: model.raw_scores(booster, rows, threads)

    return fit
