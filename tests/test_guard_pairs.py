import copy
import dataclasses
import random
import unittest

import numpy as np

from bench.corpus.labels import Label
from bench.guard import pairs
from extract.guard.features import FEATURES
from tests import guard_pages


class LabelsTest(unittest.TestCase):
    def found(self, texts):
        from extract.guard.candidates import Candidate

        return [
            Candidate(texts[0], "keep", "keep", ("keep",)),
            *[Candidate(t, "text", "letters", ("dictionary",)) for t in texts[1:]],
        ]

    def label(self, kind, truth, wrong):
        return Label("w", wrong, kind, None, truth, ("t0",))

    def test_G34_a_right_word_labels_the_token_as_read(self):
        labels = pairs.site_labels(self.label("ok", "obra", False), self.found(["obra", "obras"]))
        self.assertEqual(labels.tolist(), [1, 0])

    def test_G34_a_substitution_labels_the_candidate_that_is_the_truth(self):
        labels = pairs.site_labels(self.label("substitution", "obra", True), self.found(["obrx", "obra", "obre"]))
        self.assertEqual(labels.tolist(), [0, 1, 0])

    def test_G34_a_word_wrong_in_a_way_no_candidate_repairs_has_every_label_at_zero(self):
        for kind in ("merge", "split", "mixed", "invented"):
            labels = pairs.site_labels(self.label(kind, "obra", True), self.found(["obrx", "obra"]))
            self.assertEqual(labels.tolist(), [0, 0], kind)

    def test_G34_a_substitution_whose_truth_is_not_among_the_candidates_is_all_zero(self):
        labels = pairs.site_labels(self.label("substitution", "plan", True), self.found(["obrx", "obra"]))
        self.assertEqual(labels.tolist(), [0, 0])

    def test_G34_the_labels_of_a_site_are_exclusive_on_generated_pages(self):
        pages = guard_pages.corpus(8)
        cache = pairs.SiteCache(pages)
        total = 0
        for p, page in enumerate(pages):
            for w in range(len(page.words)):
                if pairs.is_site(page, w):
                    _, labels = cache.get(p, w)
                    self.assertLessEqual(int(labels.sum()), 1)
                    total += 1
        self.assertGreater(total, 100)


class LabelEdgesTest(unittest.TestCase):
    found = LabelsTest.found

    def test_G34_a_candidate_is_the_truth_when_only_its_typography_differs(self):
        label = Label("w", True, "substitution", None, "obra'", ("t0",))
        labels = pairs.site_labels(label, self.found(["obrx", "obra\u2019"]))
        self.assertEqual(labels.tolist(), [0, 1])

    def test_G34_a_word_the_layer_does_not_ground_is_not_a_site(self):
        page = guard_pages.corpus(2)[0]
        ungrounded = dataclasses.replace(page.labels[3], wrong=None, kind="ungrounded")
        page.labels[3] = ungrounded
        self.assertTrue(page.context.is_site(3))
        self.assertFalse(pairs.is_site(page, 3))
        self.assertNotIn(3, page.grounded().tolist())


class PurityTest(unittest.TestCase):
    def test_G14_features_and_candidates_are_the_same_whatever_the_truth_says(self):
        rng = random.Random(3)
        record = guard_pages.record("d1", 1, rng)
        garbage = copy.deepcopy(record)
        for word in garbage["truth"]["words"]:
            word["text"] = "".join(rng.choice("abcdefgh") for _ in range(7))
        real, other = pairs.page_of(record), pairs.page_of(garbage)
        checked = 0
        for w in range(len(real.words)):
            if real.labels[w].wrong is None or other.labels[w].wrong is None or not real.context.is_site(w):
                continue
            first, second = pairs.site_rows(real, w), pairs.site_rows(other, w)
            self.assertEqual([c.text for c in first.candidates], [c.text for c in second.candidates])
            self.assertEqual(first.x.tobytes(), second.x.tobytes())
            checked += 1
        self.assertGreater(checked, 20)
        self.assertNotEqual([label.wrong for label in real.labels], [label.wrong for label in other.labels])


class TableTest(unittest.TestCase):
    def test_G17_a_table_has_one_block_per_site_with_the_token_first(self):
        pages = guard_pages.corpus(4)
        cache = pairs.SiteCache(pages)
        sites = [(p, w) for p, page in enumerate(pages) for w in range(len(page.words)) if pairs.is_site(page, w)][:60]
        table = cache.table(sites)
        self.assertEqual(table.sites, 60)
        self.assertEqual(table.x.shape, (len(table.y), len(FEATURES)))
        for s in range(table.sites):
            block = table.block(s)
            self.assertEqual(table.found[s][0].op, "keep")
            self.assertEqual(block.stop - block.start, len(table.found[s]))
            self.assertTrue((table.site[block] == s).all())
        self.assertEqual(len(np.unique(table.site)), 60)

    def test_G17_a_site_built_twice_comes_from_the_cache(self):
        pages = guard_pages.corpus(2)
        cache = pairs.SiteCache(pages)
        first = cache.get(0, 3)
        self.assertIs(first, cache.get(0, 3))
