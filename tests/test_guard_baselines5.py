import unittest

import numpy as np

from bench.guard import baselines5, operating, pairs
from extract.guard.features import FEATURES, INDEX, KINDS
from tests import guard_pages


class OneHotTest(unittest.TestCase):
    def test_G18_the_kind_and_the_operation_become_indicators(self):
        x = np.full((2, len(FEATURES)), np.nan, dtype=np.float32)
        x[:, INDEX["kind"]] = [KINDS.index("accent"), KINDS.index("keep")]
        out = baselines5.one_hot(x)
        self.assertEqual(out.shape[1], len(FEATURES) - 2 + len(KINDS) + 4)
        indicators = out[:, -(len(KINDS) + 4) : -4]
        self.assertEqual(indicators.argmax(axis=1).tolist(), [KINDS.index("accent"), KINDS.index("keep")])


class NearestWordTest(unittest.TestCase):
    def table(self):
        pages = guard_pages.corpus(6, errors=0.4)
        cache = pairs.SiteCache(pages)
        sites = [(p, w) for p, page in enumerate(pages) for w in range(len(page.words)) if pairs.is_site(page, w)]
        return cache.table(sites)

    def test_G18_the_rule_only_touches_words_outside_the_dictionary(self):
        table = self.table()
        choice = baselines5.nearest_word_choice(table)
        applied = np.isfinite(choice.p_fix)
        self.assertGreater(applied.sum(), 5)
        self.assertTrue((table.x[table.starts[applied], INDEX["in_dictionary"]] == 0).all())

    def test_G18_the_rule_scores_its_replacements_like_the_chooser_does(self):
        table = self.table()
        choice = baselines5.nearest_word_choice(table)
        mask = np.isfinite(choice.p_fix)
        counts = operating.counts(choice, mask)
        self.assertEqual(counts["applied"], counts["fixed"] + counts["broken"] + counts["changed_still_wrong"])
        self.assertGreater(counts["fixed"], 0)


class LogisticTest(unittest.TestCase):
    def test_G18_the_logistic_baseline_learns_a_separable_signal_and_handles_missing_values(self):
        rng = np.random.default_rng(0)
        x = np.full((400, len(FEATURES)), np.nan, dtype=np.float32)
        x[:, INDEX["kind"]] = rng.integers(0, len(KINDS), 400)
        x[:, 0] = rng.normal(size=400)
        y = (x[:, 0] > 0).astype(np.uint8)
        score = baselines5.logistic_learner()(x, y)(x)
        self.assertGreater(np.corrcoef(score, y)[0, 1], 0.7)
