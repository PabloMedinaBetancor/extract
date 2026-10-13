import copy
import random
import unittest
from decimal import Decimal
from unittest import mock

from bench import inject
from bench.corpus.sidecar_inputs import build, generated_truths, reprice
from extract import checks


class InjectionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = inject.run_harness(generated_truths(), trials=1500, seed=3)

    def test_N6_one_wrong_digit_never_leaves_a_number_checked(self):
        one = self.report["one_digit"]
        self.assertGreater(one["trials"], 1400)
        self.assertEqual(one["false_checked"], 0)
        self.assertIsNotNone(one["upper_bound_95"])

    def test_N6_any_one_character_edit_never_leaves_a_wrong_number_checked(self):
        one = self.report["one_edit"]
        self.assertGreater(one["trials"], 1400)
        self.assertEqual(one["false_checked"], 0)

    def test_N6_the_sweep_over_every_invoice_and_every_checked_digit_finds_no_false_checked(self):
        sweep = self.report["one_digit_sweep"]
        self.assertEqual(sweep["invoices"], len(generated_truths()))
        self.assertGreater(sweep["trials"], 15_000)
        self.assertEqual(sweep["false_checked"], 0)

    def test_N6_two_wrong_digits_are_measured_and_reported_with_their_rate(self):
        for mode in ("two_digits_one_number", "two_digits_same_equation"):
            result = self.report[mode]
            self.assertGreater(result["trials"], 1000, mode)
            self.assertGreaterEqual(result["rate"], 0.0)
            self.assertLess(result["rate"], 0.5)
            self.assertEqual(result["rate"], result["false_checked"] / result["trials"])

    def test_N6_two_digits_can_compensate_and_the_harness_sees_it(self):
        """Without a quantity column nothing ties a line amount to anything but the sum: swapping 7,00 and 1,00
        between two lines keeps the sum, and both wrong numbers come out checked. This is the known limit."""
        truth = copy.deepcopy(
            next(t for t in generated_truths() if t["layout"] == "L3" and t["language"] == "es" and len(t["vat"]) == 1)
        )
        truth["lines"] = truth["lines"][:2]
        truth["lines"][0].update(unit_price="7,00", amount="7,00")
        truth["lines"][1].update(unit_price="1,00", amount="1,00")
        reprice(truth)
        inputs = build(truth)
        first, second = inputs.ids[("line_amount", 0)], inputs.ids[("line_amount", 1)]
        baseline = {r.word_ids[0]: r.status for r in checks.run(inputs.words, inputs.fields, "es")}
        self.assertEqual((baseline[first], baseline[second]), ("checked", "checked"))
        swapped = inputs.with_texts({first: "1,00", second: "7,00"})
        after = {r.word_ids[0]: r.status for r in checks.run(swapped.words, swapped.fields, "es")}
        self.assertEqual((after[first], after[second]), ("checked", "checked"))

    def test_N6_the_harness_counts_a_false_checked_when_one_exists(self):
        truth = next(t for t in generated_truths() if t["layout"] == "L3")
        inputs = build(truth)
        baseline = inject.checked_results(inputs, truth["language"])
        self.assertGreater(len(baseline), 5)

        def everything_checked(words, fields, locale="auto", **kwargs):
            return [
                type("Result", (), {"word_ids": (w.id,), "status": "checked", "value": Decimal(-1), "text": "x"})()
                for w in words
            ]

        with mock.patch.object(checks, "run", everything_checked):
            outcome = inject.trial(inputs, truth["language"], baseline, "one_digit", random.Random(1))
        self.assertTrue(outcome)


if __name__ == "__main__":
    unittest.main()
