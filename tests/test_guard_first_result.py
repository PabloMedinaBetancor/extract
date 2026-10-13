"""The report of the first out-of-fold result: the errors among the best-ranked fixes are listed so that they can be read."""

import unittest
from types import SimpleNamespace

import numpy as np

from bench.guard import first_result, operating
from tests.test_guard_operating_ops import table_of


class TopErrorsTest(unittest.TestCase):
    def test_the_errors_among_the_best_ranked_sites_are_counted_by_outcome_op_and_kind_and_listed_in_rank_order(self):
        sites = [("text", 1, 0, (i,), f"d{i}") for i in range(6)]
        table = table_of(sites)
        table.found = [[None, SimpleNamespace(kind="accent", text=f"fix{i}")] for i in range(6)]
        table.token = np.array([f"tok{i}" for i in range(6)], dtype=object)
        table.truth = np.array([f"truth{i}" for i in range(6)], dtype=object)
        # p_fix by site: 5 4 3 2 1 0 -> the order of the sites is 5, 4, 3, 2, 1, 0
        choice = operating.Choice(
            np.ones(6, dtype=np.int64),
            np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),
            np.zeros(6),
            np.array(
                [
                    operating.FIXED,
                    operating.FIXED,
                    operating.STILL_WRONG,
                    operating.FIXED,
                    operating.BROKEN,
                    operating.FIXED,
                ],
                dtype=np.int8,
            ),
        )
        out = first_result.top_errors(table, choice, top=5)
        self.assertEqual(out["errors"], 2)  # site 0 (the sixth) is outside the top 5, and was fixed anyway
        self.assertEqual(out["by_outcome_op_kind"], {"2|text|accent": 1, "3|text|accent": 1})
        self.assertEqual(
            [(e["rank"], e["outcome"], e["token"], e["applied"]) for e in out["first"]],
            [(1, "broken", "tok4", "fix4"), (3, "still_wrong", "tok2", "fix2")],
        )
        self.assertEqual(first_result.top_errors(table, choice, top=1)["errors"], 0)

    def test_the_applied_edits_that_stay_wrong_are_told_apart_by_the_letters_and_the_marks(self):
        cases = [
            ("CARACTERISTICAS", "CARACTERÍSTICAS.", "CARACTERÍSTICAS"),  # letters right, the final mark is not
            ("naturnles", "naturales.", "naturales"),
            ("cicto", "cielo", "ciclo"),  # the letters themselves
            ("a b", "ab cd", "ab"),  # two words of the token: not one word
        ]
        table = table_of([("text", 1, 0, (i,), f"d{i}") for i in range(len(cases))])
        table.token = np.array([c[0] for c in cases], dtype=object)
        table.truth = np.array([c[1] for c in cases], dtype=object)
        table.found = [[None, SimpleNamespace(kind="letters", text=c[2])] for c in cases]
        n = len(cases)
        choice = operating.Choice(
            np.ones(n, dtype=np.int64), np.ones(n), np.zeros(n), np.full(n, operating.STILL_WRONG, dtype=np.int8)
        )
        mask = np.array([True, True, True, True])
        out = first_result.still_wrong_classes(table, choice, mask)
        self.assertEqual(out["counts"], {"core_right": 2, "core_wrong": 1, "other": 1})
        self.assertEqual(out["examples"]["core_right"][0], ["CARACTERISTICAS", "CARACTERÍSTICAS", "CARACTERÍSTICAS."])
        none = first_result.still_wrong_classes(table, choice, np.zeros(4, dtype=bool))
        self.assertEqual(none["counts"], {})

    def test_the_class_of_a_token_as_read_is_told_by_its_capitals_and_the_word_lists(self):
        self.assertEqual(first_result.proper_class("Bracuetas"), "title_not_listed")
        self.assertEqual(first_result.proper_class("Zxqvbn,"), "title_not_listed")
        self.assertEqual(first_result.proper_class("BRACUETAS"), "upper_not_listed")
        self.assertEqual(first_result.proper_class("Ayuntamiento"), "title_listed")
        self.assertEqual(first_result.proper_class("AYUNTAMIENTO"), "other")  # capitals but listed
        self.assertEqual(first_result.proper_class("ayuntamiento"), "other")
        self.assertEqual(first_result.proper_class("AB"), "other")  # two capitals: an acronym, not a surname
        self.assertEqual(first_result.proper_class("1.250,00"), "other")
        self.assertEqual(first_result.proper_class("·"), "other")

    def test_the_applied_edits_are_counted_by_the_class_of_the_token_they_changed(self):
        cases = [
            ("Bracuetas", "Bracuetas", "Bracuetaz"),
            ("Ordenanza", "Ordenanza", "Ordenanza"),
            ("zapato", "zapatos", "zapatos"),
        ]
        table = table_of([("text", 1, 0, (i,), f"d{i}") for i in range(3)])
        table.token = np.array([c[0] for c in cases], dtype=object)
        table.truth = np.array([c[1] for c in cases], dtype=object)
        table.found = [[None, SimpleNamespace(kind="letters", text=c[2])] for c in cases]
        choice = operating.Choice(
            np.ones(3, dtype=np.int64),
            np.ones(3),
            np.zeros(3),
            np.array([operating.BROKEN, operating.FIXED, operating.FIXED], dtype=np.int8),
        )
        out = first_result.by_proper_class(table, choice, {"more": np.ones(3, dtype=bool)}, resamples=5)
        self.assertEqual(out["sites"]["title_not_listed"], 1)
        self.assertEqual(
            (out["more"]["title_not_listed"]["applied"], out["more"]["title_not_listed"]["broken"]), (1, 1)
        )
        self.assertEqual(out["more"]["title_listed"]["fixed"], 1)
        self.assertEqual(out["more"]["upper_not_listed"], {"applied": 0})

    def test_a_site_with_no_candidate_is_never_listed(self):
        table = table_of([("text", 1, 0, (0,), "d0")])
        table.found = [[None, SimpleNamespace(kind="accent", text="x")]]
        choice = operating.Choice(
            np.zeros(1, dtype=np.int64), np.array([-np.inf]), np.zeros(1), np.array([operating.BROKEN], dtype=np.int8)
        )
        self.assertEqual(first_result.top_errors(table, choice)["errors"], 0)


if __name__ == "__main__":
    unittest.main()
