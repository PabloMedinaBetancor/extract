import unittest
from unittest import mock

import numpy as np

from bench.guard import baselines5, folds, operating, pairs, train_chooser
from extract.guard import model
from tests import guard_pages


def with_document_numbers(pages):
    """Detector rows whose only feature is the number of their document, so a fake model can say whom it trained on."""
    words = train_chooser.words_of(pages)
    names = sorted(set(words.doc))
    words.x = np.array([[names.index(d)] for d in words.doc], dtype=np.float32)
    return words


def recording_models(trained_on):
    """A fake detector fit that remembers its training documents and a fake scorer that returns 1.0 for the rows of
    a document the model was trained on and 0.0 for the others."""

    def fit(words, rows, seed, threads):
        booster = object()
        trained_on[id(booster)] = {int(x) for x in words.x[rows, 0]}
        return booster

    def score(booster, x, threads=1):
        return np.array([float(int(row[0]) in trained_on[id(booster)]) for row in x])

    return fit, score


class GenerationThresholdTest(unittest.TestCase):
    def test_G17_the_threshold_keeps_the_recall_and_flags_fewer_words_than_the_cap_when_it_can(self):
        raw = np.linspace(0, 10, 1000)
        repairable = raw > 9.5
        threshold = train_chooser.generation_threshold(raw, np.ones(1000, bool), repairable)
        self.assertLess((raw >= threshold).mean(), 0.1)
        self.assertGreaterEqual(repairable[raw >= threshold].sum() / repairable.sum(), 0.97)

    def test_G17_the_share_cap_wins_when_the_recall_would_need_more_words(self):
        raw = np.linspace(0, 10, 1000)
        repairable = raw > 1
        threshold = train_chooser.generation_threshold(raw, np.ones(1000, bool), repairable)
        self.assertAlmostEqual((raw >= threshold).mean(), 0.2, places=2)


class DetectorScoresTest(unittest.TestCase):
    def setUp(self):
        self.pages = guard_pages.corpus(20)
        self.words = with_document_numbers(self.pages)
        self.documents = train_chooser.document_table(self.pages)
        self.fold = folds.assign_folds(self.documents, 5, 1)

    def test_G17_the_inner_scores_of_a_training_document_come_from_a_model_that_never_saw_it(self):
        fit, score = recording_models({})
        with mock.patch.object(train_chooser, "fit_detector", fit), mock.patch.object(model, "raw_scores", score):
            inner = train_chooser.inner_scores(self.words, self.documents, self.fold, 0, 1, 1)
        in_training = np.array([self.fold[d] != 0 for d in self.words.doc])
        self.assertTrue(np.isnan(inner[~in_training]).all())
        self.assertTrue((inner[in_training] == 0.0).all())

    def test_G17_the_outer_scores_of_a_fold_come_from_a_model_trained_on_the_other_four(self):
        fit, score = recording_models({})
        fold_of_row = np.array([self.fold[d] for d in self.words.doc])
        with mock.patch.object(train_chooser, "fit_detector", fit), mock.patch.object(model, "raw_scores", score):
            raw = train_chooser.outer_scores(self.words, fold_of_row, 1, 1)
        self.assertEqual(len(raw), len(self.words))
        self.assertTrue((raw == 0.0).all())


class ProtocolTest(unittest.TestCase):
    cell = (3, 10)

    def run_protocol(self, **kwargs):
        pages = guard_pages.corpus(40, errors=0.3)
        learners = {"lightgbm": train_chooser.lightgbm_learner(self.cell), "logistic": baselines5.logistic_learner()}
        with mock.patch.object(train_chooser, "DETECTOR_CELL", self.cell):
            return train_chooser.run(pages, learners, threads=1, **kwargs)

    def test_G17_each_flagged_site_is_scored_once_and_every_fold_has_sites(self):
        run = self.run_protocol()
        table = run.table
        self.assertGreater(table.sites, 100)
        for name, raw in run.raw.items():
            self.assertEqual(len(raw), len(table.y), name)
        keys = list(zip(table.page.tolist(), table.word.tolist(), strict=True))
        self.assertEqual(len(keys), len(set(keys)))
        self.assertGreaterEqual(len(set(run.fold_of_site.tolist())), 3)

    def test_G29_labels_shuffled_inside_each_site_leave_nothing_to_apply_where_true_labels_do_find_fixes(self):
        floor, minimum = 0.9, 30
        found = {}
        for name, permute in (("true", False), ("shuffled", True)):
            run = self.run_protocol(permute_labels=permute)
            probability = operating.calibrate_rows(run.table, run.raw["lightgbm"])
            choice = operating.choose(run.table, probability)
            found[name] = (
                operating.safe_threshold(choice.p_fix, choice.outcome, floor, minimum),
                operating.counts(
                    choice, operating.applied(choice, "more", operating.more_margin(choice.margin, choice.outcome))
                ),
            )
        self.assertIsNotNone(found["true"][0])
        self.assertIsNone(found["shuffled"][0])
        self.assertGreater(found["true"][1]["precision"], 0.8)
        self.assertLess(found["shuffled"][1]["precision"], 0.5)


class RecordingCache(pairs.SiteCache):
    """A cache that remembers the sites of every table asked for, in order: the training table of a fold first."""

    def __init__(self, pages):
        super().__init__(pages)
        self.asked = []

    def table(self, sites):
        self.asked.append(list(sites))
        return super().table(sites)


def noop_learner(x, y):
    return lambda rows: np.zeros(len(rows))


class LeakTest(unittest.TestCase):
    """What `run` hands the chooser: the rows it trains on are chosen with the inner detector, never the outer one,
    and no document of the fold it scores is among them."""

    def setUp(self):
        self.pages = guard_pages.corpus(30, errors=0.3)
        self.documents = train_chooser.document_table(self.pages)
        self.fold = folds.assign_folds(self.documents, 5, 1)

    def run_with(self, outer, inner):
        cache = RecordingCache(self.pages)
        with (
            mock.patch.object(train_chooser, "outer_scores", lambda words, fold_of_row, seed, threads: outer(words)),
            mock.patch.object(
                train_chooser, "inner_scores", lambda words, documents, fold, k, seed, threads: inner(words, fold, k)
            ),
        ):
            train_chooser.run(self.pages, {"fake": noop_learner}, cache=cache, threads=1)
        return cache

    def test_G17_the_training_rows_of_a_fold_are_picked_by_the_inner_scores_not_the_outer_ones(self):
        cache = self.run_with(
            lambda words: np.full(len(words), 5.0),
            lambda words, fold, k: np.where([fold[d] != k for d in words.doc], -5.0, np.nan),
        )
        trained = [sites for sites in cache.asked[0:-1:2]]
        self.assertTrue(trained)
        self.assertTrue(all(len(sites) == 0 for sites in trained))

    def test_G17_no_document_of_the_scored_fold_is_in_the_training_sites_and_every_fold_trains_on_some(self):
        cache = self.run_with(
            lambda words: np.full(len(words), 5.0),
            lambda words, fold, k: np.where([fold[d] != k for d in words.doc], 10.0, np.nan),
        )
        pairs_of_tables = list(zip(cache.asked[0:-1:2], cache.asked[1:-1:2], strict=True))
        self.assertEqual(len(pairs_of_tables), 5)
        for train_sites, test_sites in pairs_of_tables:
            train_docs = {self.pages[p].doc for p, _ in train_sites}
            test_docs = {self.pages[p].doc for p, _ in test_sites}
            self.assertTrue(train_docs)
            self.assertTrue(test_docs)
            self.assertFalse(train_docs & test_docs)
            self.assertEqual(len({self.fold[d] for d in test_docs}), 1)


class StructureProtocolTest(unittest.TestCase):
    """With `structure` the joins, segmentations and drops are sites of the protocol, chosen by the flagged words."""

    def setUp(self):
        self.pages = guard_pages.corpus(30, errors=0.3)
        self.documents = train_chooser.document_table(self.pages)
        self.fold = folds.assign_folds(self.documents, 5, 1)

    def run_with(self, inner_value, structure, pages=None):
        pages = pages or self.pages
        cache = RecordingCache(pages)
        with (
            mock.patch.object(
                train_chooser, "outer_scores", lambda words, fold_of_row, seed, threads: np.full(len(words), 5.0)
            ),
            mock.patch.object(
                train_chooser,
                "inner_scores",
                lambda words, documents, fold, k, seed, threads: np.where(
                    [fold[d] != k for d in words.doc], inner_value, np.nan
                ),
            ),
        ):
            train_chooser.run(pages, {"fake": noop_learner}, cache=cache, threads=1, structure=structure)
        return cache

    def test_G33_without_the_option_every_site_is_a_text_site(self):
        cache = self.run_with(10.0, structure=False)
        self.assertTrue(all(len(site) == 2 for sites in cache.asked for site in sites))

    def test_G33_with_it_the_training_and_held_out_sites_hold_structure_sites(self):
        cache = self.run_with(10.0, structure=True)
        families = {site[2] for sites in cache.asked for site in sites if len(site) == 4}
        self.assertIn("segment", families)
        for site in (s for sites in cache.asked for s in sites if len(s) == 4):
            self.assertEqual(site[1], site[3][0])

    def test_G33_a_structure_site_needs_a_flagged_word_so_no_inner_flag_means_no_training_site(self):
        cache = self.run_with(-5.0, structure=True)
        self.assertTrue(all(len(sites) == 0 for sites in cache.asked[0:-1:2]))

    def test_G33_no_document_of_the_scored_fold_is_in_the_training_structure_sites(self):
        cache = self.run_with(10.0, structure=True)
        for train_sites, test_sites in zip(cache.asked[0:-1:2], cache.asked[1:-1:2], strict=True):
            train_docs = {self.pages[s[0]].doc for s in train_sites}
            test_docs = {self.pages[s[0]].doc for s in test_sites}
            self.assertTrue(train_docs and test_docs)
            self.assertFalse(train_docs & test_docs)

    def test_G17_the_words_a_structure_candidate_repairs_count_as_repairable_for_the_generation_threshold(self):
        import dataclasses

        split = guard_pages.labelled_page(
            ["la", "orde", "nanza", "del", "plan"],
            [("la", [0], None), ("ordenanza", [1, 2], None), ("del", [3], None), ("plan", [4], None)],
        )
        pages = [*self.pages, *[dataclasses.replace(split, doc=f"s{i}") for i in range(6)]]
        seen = {}
        real = train_chooser.generation_threshold

        def spy(raw, flaggable, repairable):
            seen["repairable"] = int(repairable.sum())
            return real(raw, flaggable, repairable)

        for structure in (False, True):
            with mock.patch.object(train_chooser, "generation_threshold", spy):
                cache = self.run_with(10.0, structure, pages)
            seen[structure] = seen["repairable"]
        self.assertIn("join", {site[2] for sites in cache.asked for site in sites if len(site) == 4})
        self.assertGreaterEqual(seen[True] - seen[False], 12)  # two words of the split on each of the six pages


class LearningCurveTest(unittest.TestCase):
    """`train_share` keeps only a stable share of the training documents of every fold."""

    def setUp(self):
        self.pages = guard_pages.corpus(30, errors=0.3)

    def training_docs(self, share):
        cache = RecordingCache(self.pages)
        with (
            mock.patch.object(
                train_chooser, "outer_scores", lambda words, fold_of_row, seed, threads: np.full(len(words), 5.0)
            ),
            mock.patch.object(
                train_chooser,
                "inner_scores",
                lambda words, documents, fold, k, seed, threads: np.where(
                    [fold[d] != k for d in words.doc], 10.0, np.nan
                ),
            ),
        ):
            train_chooser.run(self.pages, {"fake": noop_learner}, cache=cache, threads=1, train_share=share)
        return [{self.pages[s[0]].doc for s in sites} for sites in cache.asked[0:-1:2]]

    def test_G12_a_smaller_share_trains_on_fewer_documents_of_every_fold_and_inside_the_larger_share(self):
        full, half, quarter = (self.training_docs(s) for s in (1.0, 0.5, 0.25))
        for a, b, c in zip(full, half, quarter, strict=True):
            self.assertLess(len(b), len(a))
            self.assertLess(len(c), len(b))
            self.assertTrue(c <= b <= a)

    def test_G12_the_share_is_decided_by_the_document_id_alone_so_it_is_the_same_every_run(self):
        self.assertEqual(self.training_docs(0.5), self.training_docs(0.5))
        self.assertTrue(all(0 <= train_chooser.document_unit(f"d{i}") < 1 for i in range(50)))
        self.assertEqual(train_chooser.document_unit("d7"), train_chooser.document_unit("d7"))

    def test_G12_a_share_of_one_is_the_whole_protocol(self):
        self.assertEqual(self.training_docs(1.0), self.training_docs(1.0))
        self.assertTrue(all(len(docs) > 15 for docs in self.training_docs(1.0)))


class ManifestFoldsTest(unittest.TestCase):
    """The outer folds are the ones the manifest records, not a new assignment."""

    def setUp(self):
        self.pages = guard_pages.corpus(30, errors=0.3)
        for number, page in enumerate(self.pages):
            page.fold = (number * 3 + 1) % 5  # nothing like the greedy assignment
        self.documents = train_chooser.document_table(self.pages)

    def test_G17_the_folds_are_those_of_the_pages_when_they_have_them(self):
        fold = train_chooser.document_folds(self.pages, self.documents, 1)
        self.assertEqual(fold, {p.doc: p.fold for p in self.pages})
        self.assertNotEqual(fold, folds.assign_folds(self.documents, 5, 1))

    def test_G17_pages_with_no_fold_get_the_greedy_assignment(self):
        for page in self.pages:
            page.fold = None
        self.assertEqual(
            train_chooser.document_folds(self.pages, self.documents, 1), folds.assign_folds(self.documents, 5, 1)
        )

    def test_G17_a_document_in_two_folds_or_with_a_page_that_has_none_is_an_error(self):
        other = guard_pages.corpus(1, errors=0.3)[0]
        other.doc, other.fold = self.pages[0].doc, (self.pages[0].fold + 1) % 5
        with self.assertRaises(ValueError):
            train_chooser.document_folds([*self.pages, other], self.documents, 1)
        other.fold = None
        with self.assertRaises(ValueError):
            train_chooser.document_folds([*self.pages, other], self.documents, 1)

    def test_G17_a_fold_the_protocol_has_no_place_for_is_an_error(self):
        self.pages[0].fold = 7
        with self.assertRaises(ValueError):
            train_chooser.document_folds(self.pages, self.documents, 1)

    def test_G17_the_sites_a_fold_scores_are_those_of_the_documents_the_manifest_puts_in_it(self):
        cache = RecordingCache(self.pages)
        with (
            mock.patch.object(
                train_chooser, "outer_scores", lambda words, fold_of_row, seed, threads: np.full(len(words), 5.0)
            ),
            mock.patch.object(
                train_chooser,
                "inner_scores",
                lambda words, documents, fold, k, seed, threads: np.where(
                    [fold[d] != k for d in words.doc], 10.0, np.nan
                ),
            ),
        ):
            result = train_chooser.run(self.pages, {"fake": noop_learner}, cache=cache, threads=1)
        by_doc = {p.doc: p.fold for p in self.pages}
        self.assertGreater(result.table.sites, 0)
        for doc, k in zip(result.table.doc, result.fold_of_site, strict=True):
            self.assertEqual(by_doc[doc], int(k))
