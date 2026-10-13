"""Proper names: the class of a token by code, and `safe` leaving them alone in the sweep, the report and at run time."""

import unittest

import numpy as np

from bench.guard import operating
from extract.guard import proper
from tests.test_guard_fixing import propose_with, scripted
from tests.test_guard_operating_ops import table_of


class ClassTest(unittest.TestCase):
    def test_a_capitalised_token_that_no_list_knows_is_a_proper_name_and_the_rest_is_not(self):
        self.assertEqual(proper.proper_class("Bracuetas"), proper.TITLE)
        self.assertEqual(proper.proper_class("(Bracuetas),"), proper.TITLE)
        self.assertEqual(proper.proper_class("BRACUETAS"), proper.UPPER)
        self.assertEqual(proper.proper_class("Ayuntamiento"), proper.TITLE_LISTED)
        self.assertEqual(proper.proper_class("XQZ"), proper.UPPER)
        for other in ("AYUNTAMIENTO", "ayuntamiento", "ZQ", "1.250,00", "·", "bracuetas"):
            self.assertEqual(proper.proper_class(other), proper.OTHER, other)

    def test_only_the_two_classes_that_no_list_knows_are_protected_and_any_word_of_a_span_counts(self):
        self.assertEqual(proper.PROTECTED_IN_SAFE, {proper.TITLE, proper.UPPER})
        self.assertTrue(proper.protected_in_safe(["la", "Bracuetas"]))
        self.assertTrue(proper.protected_in_safe(["BRACUETAS"]))
        self.assertFalse(proper.protected_in_safe(["Ayuntamiento", "de", "la"]))
        self.assertFalse(proper.protected_in_safe([]))


def choice_and_table(tokens):
    table = table_of([("text", 1, 0, (i,), f"d{i % 5}") for i in range(len(tokens))])
    table.token = np.array(tokens, dtype=object)
    n = len(tokens)
    choice = operating.Choice(
        np.ones(n, dtype=np.int64), np.linspace(1.0, 0.5, n), np.zeros(n), np.full(n, operating.FIXED, dtype=np.int8)
    )
    return table, choice


class SweepTest(unittest.TestCase):
    def test_the_proper_names_have_no_safe_probability_but_keep_the_real_one_for_more(self):
        table, choice = choice_and_table(["casa", "Bracuetas", "plan", "BRACUETAS", "Ayuntamiento"])
        protected = operating.protect_proper(table, choice)
        self.assertEqual(np.isfinite(protected.safe_score).tolist(), [True, False, True, False, True])
        self.assertTrue(np.isfinite(protected.p_fix).all())
        self.assertEqual(protected.margin.tolist(), choice.margin.tolist())
        self.assertEqual(operating.applied(protected, "safe", 0.0).tolist(), [True, False, True, False, True])
        self.assertTrue(operating.applied(protected, "more", 0.0).all())

    def test_a_proper_name_is_not_in_the_sweep_of_safe(self):
        # a proper name that is wrong sits at the top; swept with it the threshold fails, without it the others pass
        table, choice = choice_and_table(["Bracuetas"] + ["casa"] * 120)
        choice.outcome[0] = operating.STILL_WRONG
        ops = operating.site_ops(table)
        minima = {"text": 100, "join": 10, "segment": 10, "drop": 10}
        plain = operating.thresholds_by_op(choice, ops, floor=0.995, minima=minima)
        protected = operating.thresholds_by_op(operating.protect_proper(table, choice), ops, floor=0.995, minima=minima)
        self.assertIsNone(plain["text"]["t_safe"])
        self.assertIsNotNone(protected["text"]["t_safe"])

    def test_the_slices_of_a_choice_keep_what_safe_sees(self):
        table, choice = choice_and_table(["Bracuetas", "casa", "plan"])
        sliced = operating._members(operating.protect_proper(table, choice), np.array([True, False, True]))
        self.assertEqual(np.isfinite(sliced.safe_score).tolist(), [False, True])

    def test_a_span_with_a_proper_name_in_it_is_protected(self):
        table, choice = choice_and_table(["la Bracuetas", "de la casa"])
        self.assertEqual(np.isfinite(operating.protect_proper(table, choice).safe_score).tolist(), [False, True])


class RuntimeTest(unittest.TestCase):
    def test_a_join_or_a_segmentation_is_protected_when_any_word_of_its_span_is_a_proper_name(self):
        words, matrix, raw, model, sites = scripted("el plan de la orde Nanzaq del municipio es el que rige la ciudad")
        joins = [s for s in sites if s.op == "join" and s.span == (4, 5)]
        self.assertTrue(joins)
        self.assertTrue(all(s.proper for s in joins))
        words, matrix, raw, model, sites = scripted("el plan de la orde nanza del municipio es el que rige la ciudad")
        self.assertFalse([s for s in sites if s.op == "join" and s.proper])

    def modes(self, text):
        words, matrix, raw, model, sites = scripted(text)
        site = next(s for s in sites if s.op == "text")

        def fake(s):
            return [0.05] + [0.999] * (len(s.found) - 1)

        fixes, _ = propose_with(sites, fake, words, matrix, raw, model)
        return fixes[4].op, fixes[4].modes, site.proper

    def test_a_proper_name_is_never_applied_in_safe_but_is_still_proposed_for_more(self):
        lower = "el plan de la ordenaza del municipio es el que rige la ciudad"
        capital = "el plan de la Ordenaza del municipio es el que rige la ciudad"
        self.assertEqual(self.modes(lower)[1], ("safe", "more"))
        op, modes, proper_flag = self.modes(capital)
        self.assertTrue(proper_flag)
        self.assertEqual(modes, ("more",))


if __name__ == "__main__":
    unittest.main()
