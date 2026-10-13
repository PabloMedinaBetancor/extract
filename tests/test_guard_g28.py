"""The source-dropping and canary experiments: the pool without a source, the rule that keeps a source, the learning curve and the canary."""

import unittest

from bench.guard import g28
from extract.guard import candidates
from tests.guard_words import Reading, page


def generated(dropped=()):
    words = page("la ordeu del plan", **{"1": {"readings": (Reading("latin", "orden", rank=0, group=("p1w1",)),)}})
    ctx = candidates.PageContext(words, ("es", "en"), frozenset())
    with g28.without_sources(set(dropped)):
        return {source for c in candidates.generate(ctx, 1) for source in c.sources}


class WithoutSourcesTest(unittest.TestCase):
    def test_G28_the_pool_has_every_source_and_dropping_one_takes_only_that_one_out(self):
        everything = generated()
        self.assertIn("dictionary", everything)
        self.assertIn("reading:latin", everything)
        without = generated({"dictionary"})
        self.assertNotIn("dictionary", without)
        self.assertIn("reading:latin", without)

    def test_G28_reading_drops_every_reader_and_a_named_reader_only_itself(self):
        self.assertFalse({s for s in generated({"reading"}) if s.startswith("reading:")})
        self.assertNotIn("reading:latin", generated({"reading:latin"}))
        self.assertIn("reading:latin", generated({"reading:deep"}))

    def test_G28_the_pool_is_restored_afterwards(self):
        generated({"dictionary"})
        self.assertIn("dictionary", generated())


class VerdictTest(unittest.TestCase):
    base = {"safe_fixed": 0.0, "more_net": 2600.0}

    def row(self, safe, net):
        return {"safe_fixed": safe, "more_net": net, "reachable_sites": 3000}

    def test_G28_a_source_stays_if_dropping_it_costs_two_per_hundred_wrong_words_at_safe_or_in_more(self):
        out = g28.source_verdicts(
            self.base,
            {
                "page": self.row(0.0, 2600 - 100),
                "ctc_alt": self.row(0.0, 2600 - 199),
                "fold": self.row(-1.0, 2600 - 201),
            },
            wrong=10_000,
        )
        self.assertFalse(out["page"]["keeps"])  # 1 per 100 wrong words
        self.assertFalse(out["ctc_alt"]["keeps"])  # 1.99
        self.assertTrue(out["fold"]["keeps"])  # 2.01 in more
        self.assertAlmostEqual(out["fold"]["more_net_lost_per_100_wrong"], 2.01)
        safe_base = {"safe_fixed": 400.0, "more_net": 2600.0}
        out = g28.source_verdicts(safe_base, {"punct": self.row(400 - 200, 2600)}, wrong=10_000)
        self.assertTrue(out["punct"]["keeps"])  # 2 correct fixes per 100 wrong at safe

    def test_G28_a_source_that_helps_when_removed_is_not_kept(self):
        out = g28.source_verdicts(self.base, {"page": self.row(0.0, 2700)}, wrong=10_000)
        self.assertFalse(out["page"]["keeps"])
        self.assertLess(out["page"]["more_net_lost_per_100_wrong"], 0)

    def test_G28_the_curve_has_the_net_and_its_se_at_each_share(self):
        points = {
            "0.25": {"more_net": 2000, "more_net_se": 250, "safe_fixed": 0},
            "1.0": {"more_net": 2600, "more_net_se": 280, "safe_fixed": 0},
        }
        self.assertEqual(g28.curve_points(points)["0.25"], {"more_net": 2000, "se": 250, "safe_fixed": 0})

    def test_G29_the_canary_passes_with_nothing_applied_at_safe_and_a_net_within_noise(self):
        self.assertTrue(g28.canary_verdict({"safe_applied": 0, "more_net": 12, "more_net_se": 40})["passes"])
        self.assertTrue(g28.canary_verdict({"safe_applied": 0, "more_net": -300, "more_net_se": 40})["passes"])
        self.assertFalse(g28.canary_verdict({"safe_applied": 3, "more_net": 0, "more_net_se": 40})["passes"])
        self.assertFalse(g28.canary_verdict({"safe_applied": 0, "more_net": 200, "more_net_se": 40})["passes"])


if __name__ == "__main__":
    unittest.main()
