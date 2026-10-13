import random
import unittest
from collections import Counter

from bench.guard import folds


def documents(n=100, seed=1):
    rng = random.Random(seed)
    return {f"doc{i}": (rng.choice(("light", "medium", "heavy")), rng.randrange(200, 900)) for i in range(n)}


class FoldsTest(unittest.TestCase):
    def test_G17_every_document_is_in_exactly_one_fold(self):
        assigned = folds.assign_folds(documents(), 5, 1)
        self.assertEqual(set(assigned), set(documents()))
        self.assertEqual(set(assigned.values()), set(range(5)))

    def test_G17_the_assignment_does_not_depend_on_the_order_of_the_documents(self):
        docs = documents()
        shuffled = dict(random.Random(9).sample(sorted(docs.items()), len(docs)))
        self.assertEqual(folds.assign_folds(docs, 5, 1), folds.assign_folds(shuffled, 5, 1))

    def test_G17_the_seed_changes_the_assignment(self):
        self.assertNotEqual(folds.assign_folds(documents(), 5, 1), folds.assign_folds(documents(), 5, 2))

    def test_G17_each_band_is_spread_over_the_folds_by_words(self):
        docs = documents(200)
        assigned = folds.assign_folds(docs, 5, 1)
        for band in ("light", "medium", "heavy"):
            words = Counter()
            for doc, (b, count) in docs.items():
                if b == band:
                    words[assigned[doc]] += count
            self.assertLess(max(words.values()) - min(words.values()), 900, band)

    def test_G17_fewer_than_two_folds_is_refused(self):
        with self.assertRaises(ValueError):
            folds.assign_folds(documents(), 1)
