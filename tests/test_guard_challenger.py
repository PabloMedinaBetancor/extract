import unittest
from unittest import mock

import numpy as np

from bench.guard import challenger, train_chooser
from extract.guard import features
from tests import guard_pages
from tests.test_guard_train_chooser import RecordingCache


def sites_matrix(n_sites, rows=3, seed=1):
    """`rows` candidates a site, one of them the truth: the column of the signal is high for it."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_sites * rows, len(features.FEATURES))).astype(np.float32)
    y = np.zeros(n_sites * rows, dtype=np.uint8)
    for s in range(n_sites):
        y[s * rows + int(rng.integers(rows))] = 1
    x[:, 5] += 3.0 * y
    return x, y, np.full(n_sites, rows)


class LambdaRankTest(unittest.TestCase):
    def test_B4_the_ranker_puts_the_true_candidate_of_a_site_first(self):
        x, y, groups = sites_matrix(300)
        score = challenger.lambdarank_learner((7, 30), threads=1)(x, y, groups)(x)
        top = score.reshape(-1, 3).argmax(axis=1)
        truth = y.reshape(-1, 3).argmax(axis=1)
        self.assertGreater(float((top == truth).mean()), 0.95)

    def test_B4_the_factory_asks_for_the_groups(self):
        self.assertTrue(challenger.lambdarank_learner().needs_groups)


class GroupsInTheProtocolTest(unittest.TestCase):
    def test_B4_a_learner_that_needs_groups_gets_the_size_of_every_training_site(self):
        pages = guard_pages.corpus(30, errors=0.3)
        seen = []

        def spy(x, y, groups):
            seen.append((len(y), int(groups.sum()), int(groups.min()), len(groups)))
            return lambda rows: np.zeros(len(rows))

        spy.needs_groups = True
        plain = []

        def ordinary(x, y):
            plain.append(len(y))
            return lambda rows: np.zeros(len(rows))

        with (
            mock.patch.object(
                train_chooser, "outer_scores", lambda words, fold_of_row, seed, threads: np.full(len(words), 5.0)
            ),
            mock.patch.object(
                train_chooser,
                "inner_scores",
                lambda words, documents, fold_, k, seed, threads: np.where(
                    [fold_[d] != k for d in words.doc], 10.0, np.nan
                ),
            ),
        ):
            train_chooser.run(pages, {"spy": spy, "plain": ordinary}, cache=RecordingCache(pages), threads=1)
        self.assertEqual(len(seen), 5)
        for rows, total, smallest, sites in seen:
            self.assertEqual(rows, total)
            self.assertGreaterEqual(smallest, 2)  # KEEP and at least one candidate
            self.assertGreater(sites, 10)
        self.assertEqual(plain, [rows for rows, *_ in seen])


if __name__ == "__main__":
    unittest.main()
