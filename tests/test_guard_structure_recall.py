import unittest

from bench.guard import structure_recall as sr
from extract.guard.candidates import PageContext
from tests.guard_words import page

LANGUAGES = ("es", "en")


class VerdictTest(unittest.TestCase):
    def test_G36_a_floor_over_fewer_than_twenty_eligible_components_is_not_conclusive(self):
        self.assertIsNone(sr.verdict(1.0, 1, 0.55))
        self.assertIsNone(sr.verdict(0.0, sr.MIN_CONCLUSIVE - 1, 0.55))
        self.assertIsNone(sr.verdict(None, 100, 0.55))

    def test_G36_with_enough_components_the_floor_passes_or_fails(self):
        self.assertTrue(sr.verdict(0.55, sr.MIN_CONCLUSIVE, 0.55))
        self.assertFalse(sr.verdict(0.54, 100, 0.55))


class NotPlainReasonTest(unittest.TestCase):
    def reason(self, token):
        return sr.not_plain_reason(PageContext(page(f"la {token} casa", gap=4), LANGUAGES), 1)

    def test_G36_the_reason_a_token_is_not_a_flat_one_is_named(self):
        self.assertIsNone(self.reason("suelo"))
        self.assertEqual(self.reason("art.12"), "digit")
        self.assertEqual(self.reason("a1"), "digit")
        self.assertEqual(self.reason("so(lo"), "internal_marks")
        self.assertEqual(self.reason("obra-"), "hyphen_end")
        self.assertEqual(self.reason("SueLo"), "mixed_capitals")
        self.assertEqual(self.reason("a" * 41), "blocked_or_long")

    def test_G36_the_reasons_are_the_ones_the_report_lists(self):
        for token in ("art.12", "so(lo", "obra-", "SueLo", "a" * 41):
            self.assertIn(self.reason(token), sr.NOT_PLAIN_REASONS)


class TallyTest(unittest.TestCase):
    def test_G36_tallies_add_up_across_sets(self):
        from collections import Counter

        a = {k: Counter({"x": 1}) for k in ("join", "segment", "drop", "cost", "wrong", "excluded")}
        a["pages"], a["documents"] = 1, {"d1"}
        b = {k: Counter({"x": 2}) for k in a if k not in ("pages", "documents")}
        b["pages"], b["documents"] = 2, {"d1", "d2"}
        total = sr.add(a, b)
        self.assertEqual((total["join"]["x"], total["pages"], len(total["documents"])), (3, 3, 2))


if __name__ == "__main__":
    unittest.main()
