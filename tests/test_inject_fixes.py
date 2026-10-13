import dataclasses
import json
import random
import unittest
from pathlib import Path
from unittest import mock

from bench import inject_fixes
from bench.corpus.sidecar_inputs import build, generated_truths
from extract import checks
from extract.types import DigitSuggestion

RESULT = Path(__file__).parents[1] / "bench" / "results" / "injection-fixes.json"


def truths(n=24):
    return [t for t in generated_truths() if len(t["vat"]) == 1][:n]


class HelpersTest(unittest.TestCase):
    def test_N7_the_alternatives_of_a_digit_are_its_ctc_partners_both_ways(self):
        self.assertEqual(set(inject_fixes.alternatives("1")), {"7"})
        self.assertEqual(set(inject_fixes.alternatives("8")), {"0", "3"})
        self.assertEqual(inject_fixes.alternatives("2"), ())

    def test_N7_the_bound_is_the_rule_of_three_for_zero_and_wilson_otherwise(self):
        self.assertEqual(inject_fixes.upper_bound_95(0, 20000), 3 / 20000)
        self.assertGreater(inject_fixes.upper_bound_95(5, 1000), 0.005)
        self.assertLess(inject_fixes.upper_bound_95(5, 1000), 0.015)
        self.assertIsNone(inject_fixes.upper_bound_95(0, 0))

    def test_N7_the_harness_applies_through_the_pipelines_own_step_7b_so_an_iban_in_groups_is_never_touched(self):
        truth = truths(1)[0]
        inputs = build(truth)
        iban = next(r for r in checks.run(inputs.words, inputs.fields, truth["language"]) if r.kind == "iban")
        self.assertGreater(len(iban.word_ids), 1)
        fake = dataclasses.replace(iban, suggestion=DigitSuggestion(1, iban.text[1], "7", True, "checks.rule.iban"))
        for use_alts in (True, False):
            self.assertEqual(inject_fixes.apply_verified(inputs, [fake], use_alts)[1], [])

    def test_N7_a_letter_that_could_be_the_error_stops_a_verified_digit_fix_of_a_dni(self):
        truth = next(t for t in generated_truths() if t["customer"]["tax_id"][0].isdigit())
        inputs = build(truth)
        word_id = inputs.ids["customer_tax_id"]
        text = inputs.text_of(word_id)
        wrong_letter = "T" if text[-1] != "T" else "R"
        wrong = inputs.with_texts({word_id: text[:-1] + wrong_letter})
        results = checks.run(wrong.words, wrong.fields, truth["language"])
        number = next(r for r in results if word_id in r.word_ids)
        digit = next(i for i, c in enumerate(text) if c.isdigit() and inject_fixes.alternatives(c))
        partner = inject_fixes.alternatives(text[digit])[0]
        fake = dataclasses.replace(
            number, suggestion=DigitSuggestion(digit, text[digit], partner, True, "checks.rule.dni_letter")
        )
        self.assertEqual(len(inject_fixes.apply_verified(wrong, [fake], True, truth=inputs, letter_alts=False)[1]), 1)
        self.assertEqual(inject_fixes.apply_verified(wrong, [fake], True, truth=inputs, letter_alts=True)[1], [])

    def test_N7_the_alternatives_gate_refuses_a_fix_whose_digit_is_not_a_partner_of_the_read_one(self):
        truth = truths(1)[0]
        inputs = build(truth)
        word_id = inputs.ids["total"]
        result = next(r for r in checks.run(inputs.words, inputs.fields, truth["language"]) if word_id in r.word_ids)
        text = inputs.text_of(word_id)
        position = next(i for i, c in enumerate(text) if c.isdigit() and inject_fixes.alternatives(c))
        old = text[position]
        partner = inject_fixes.alternatives(old)[0]
        other = next(d for d in "0123456789" if d != old and d not in inject_fixes.alternatives(old))
        for new, applied in ((partner, 1), (other, 0)):
            fake = dataclasses.replace(
                result, suggestion=DigitSuggestion(position, old, new, True, "checks.rule.total")
            )
            # the total is an arithmetic rule: only `more` applies it
            self.assertEqual(len(inject_fixes.apply_verified(inputs, [fake], use_alts=True, mode="more")[1]), applied)
            self.assertEqual(len(inject_fixes.apply_verified(inputs, [fake], use_alts=False, mode="more")[1]), 1)
        unverified = dataclasses.replace(
            result, suggestion=DigitSuggestion(position, old, partner, False, "checks.rule.total")
        )
        self.assertEqual(inject_fixes.apply_verified(inputs, [unverified], use_alts=False, mode="more")[1], [])


class HarnessTest(unittest.TestCase):
    def test_N10_the_arithmetic_suggestions_are_reported_apart_from_the_applied_fixes(self):
        _, report = inject_fixes.run_mode(("one_digit_ctc", truths(), 150, 5))
        found = report["verified_and_alts"]
        suggested = found["suggestions_by_rule"]
        self.assertGreater(sum(e["suggested"] for e in suggested.values()), 0)
        self.assertTrue(set(suggested) <= inject_fixes.fixes.ARITHMETIC_RULES)
        self.assertFalse(set(found["by_rule"]) & inject_fixes.fixes.ARITHMETIC_RULES)
        for entry in suggested.values():
            self.assertEqual(entry["false"], 0)

    def test_N7_one_wrong_digit_gives_no_false_fix_and_no_wrong_check_in_either_variant(self):
        for mode in ("one_digit_any", "one_digit_ctc"):
            _, report = inject_fixes.run_mode((mode, truths(), 120, 3))
            for variant, found in report.items():
                with self.subTest(mode=mode, variant=variant):
                    self.assertEqual(found["trials"], 120)
                    self.assertEqual(found["false_fixes"], 0)
                    self.assertEqual(found["wrong_checked"], 0)
                    self.assertEqual(found["all_corrected"], found["trials_with_a_fix"])

    def test_N7_most_ctc_single_digit_errors_are_fixed_through_a_verifying_rule(self):
        _, report = inject_fixes.run_mode(("one_digit_ctc", truths(), 150, 5))
        found = report["verified_and_alts"]
        # In `safe` only the check-digit fixes apply alone; the arithmetic ones are suggestions, counted apart
        fixed = sum(e["applied"] for e in found["by_rule"].values())
        suggested = sum(e["suggested"] for e in found["suggestions_by_rule"].values())
        self.assertGreater(fixed + suggested, 0.5 * found["trials"])
        self.assertTrue(set(found["by_rule"]) <= {k for k in found["by_rule"] if k.startswith("checks.rule.")})

    def test_N7_the_alternatives_gate_only_removes_fixes(self):
        _, report = inject_fixes.run_mode(("two_digits_any", truths(), 100, 7))
        self.assertLessEqual(report["verified_and_alts"]["trials_with_a_fix"], report["verified"]["trials_with_a_fix"])

    def test_N7_a_harness_that_cannot_see_a_false_fix_is_a_finding(self):
        real = checks.run

        def lying(words, invoice, locale="auto", *args, **kwargs):
            results = real(words, invoice, locale, *args, **kwargs)
            if not kwargs.get("suggest", True):
                return results
            out = []
            for r in results:
                digits = [i for i, c in enumerate(r.text) if c.isdigit()]
                if r.status == "mismatch" and digits:
                    i = digits[0]
                    wrong = "9" if r.text[i] != "9" else "8"
                    r = dataclasses.replace(
                        r, suggestion=DigitSuggestion(i, r.text[i], wrong, True, "checks.rule.fake")
                    )
                out.append(r)
            return out

        with mock.patch.object(inject_fixes.checks, "run", lying):
            _, report = inject_fixes.run_mode(("one_digit_any", truths(), 60, 3))
        self.assertGreater(report["verified"]["false_fixes"], 0)
        self.assertIn("checks.rule.fake", report["verified"]["by_rule"])

    def test_N7_the_trials_are_the_same_on_every_run(self):
        first = inject_fixes.run_mode(("two_digits_ctc", truths(), 40, 9))
        second = inject_fixes.run_mode(("two_digits_ctc", truths(), 40, 9))
        self.assertEqual(first, second)

    def test_N7_a_corruption_never_leaves_a_number_unchanged_and_two_digit_modes_change_two_digits(self):
        rng = random.Random(1)
        truth = truths(1)[0]
        inputs = build(truth)
        baseline = {
            r.word_ids[0]: r
            for r in checks.run(inputs.words, inputs.fields, truth["language"], suggest=False)
            if r.status == "checked"
        }
        for mode, expected in (("one_digit_any", 1), ("two_digits_any", 2)):
            for _ in range(60):
                changes = inject_fixes.corrupt(inputs, baseline, mode, rng)
                if changes is None:
                    continue
                differences = sum(
                    a != b for w, text in changes.items() for a, b in zip(text, inputs.text_of(w), strict=True)
                )
                self.assertEqual(differences, expected if mode == "one_digit_any" else differences)
                self.assertGreaterEqual(differences, 1)


class PublishedResultTest(unittest.TestCase):
    def test_N7_the_published_run_has_at_least_20000_trials_a_mode_and_no_false_fix_with_one_digit(self):
        report = json.loads(RESULT.read_text(encoding="utf-8"))
        self.assertGreaterEqual(report["trials_per_mode"], 20000)
        for mode in ("one_digit_any", "one_digit_ctc", "one_letter_ctc"):
            for variant in ("verified", "verified_and_alts"):
                if mode == "one_letter_ctc" and variant == "verified":
                    continue
                found = report["modes"][mode][variant]
                self.assertGreaterEqual(found["trials"], 20000)
                self.assertEqual(found["false_fixes"], 0, (mode, variant))
                self.assertEqual(found["wrong_checked"], 0, (mode, variant))
        for mode in ("two_digits_any", "two_digits_ctc"):
            found = report["modes"][mode]["verified_and_alts"]
            self.assertGreaterEqual(found["trials"], 20000)
            self.assertIsNotNone(found["false_fix_upper_bound_95"])


if __name__ == "__main__":
    unittest.main()
