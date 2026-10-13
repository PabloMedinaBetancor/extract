import copy
import dataclasses
import random
import unittest
from decimal import Decimal
from unittest import mock

from bench.corpus.sidecar_inputs import CONFUSABLE, build, generated_truths, reprice
from extract import checks
from extract.types import InvoiceFields
from tests import checks_oracle as oracle

SEED = 20261003
NONE = InvoiceFields()


def identifier(text, kind=None):
    """checks.run over a document that is only the identifier, and the one result it gives."""
    results = checks.run(oracle.words(*text.split()), NONE, "es")
    return results[0]


def with_digit(text, position, new):
    return text[:position] + new + text[position + 1 :]


def other_digit(char):
    return "3" if char != "3" else "8"


def number_results(inputs, locale, suggest=True):
    return {r.word_ids[0]: r for r in checks.run(inputs.words, inputs.fields, locale, suggest=suggest)}


def single_digit_fixes(text, check):
    """Every one-digit change that makes `text` pass `check`, by brute force over positions and digit values."""
    return [
        (i, new)
        for i, char in enumerate(text)
        if char.isdigit()
        for new in "0123456789"
        if new != char and check(with_digit(text, i, new))
    ]


class IdentifierTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(SEED)

    def dnis(self, n=400):
        return [oracle.make_dni(self.rng) for _ in range(n)]

    def test_N8_a_dni_with_one_wrong_digit_and_one_possible_fix_is_verified_under_the_letter_rule(self):
        seen = 0
        for dni in self.dnis(4000):
            position = self.rng.randrange(8)
            wrong = with_digit(dni, position, other_digit(dni[position]))
            if checks.check_dni(wrong) or len(single_digit_fixes(wrong, checks.check_dni)) != 1:
                continue
            fix = identifier(wrong).suggestion
            seen += 1
            self.assertEqual((fix.position, fix.new), (position, dni[position]))
            self.assertTrue(fix.verified)
            self.assertEqual(fix.rule, "checks.rule.dni_letter")
        self.assertGreater(seen, 20)

    def test_N8_a_dni_with_two_possible_fixes_gets_no_suggestion_so_nothing_is_ever_verified(self):
        ties = 0
        for dni in self.dnis(600):
            position = self.rng.randrange(8)
            wrong = with_digit(dni, position, other_digit(dni[position]))
            if checks.check_dni(wrong) or len(single_digit_fixes(wrong, checks.check_dni)) < 2:
                continue
            ties += 1
            self.assertIsNone(identifier(wrong).suggestion)
        self.assertGreater(ties, 20)

    def test_N8_a_nie_is_verified_in_the_same_way(self):
        found = 0
        for _ in range(4000):
            nie = oracle.make_nie(self.rng)
            position = self.rng.randrange(1, 8)
            wrong = with_digit(nie, position, other_digit(nie[position]))
            if checks.check_nie(wrong) or len(single_digit_fixes(wrong, checks.check_nie)) != 1:
                continue
            fix = identifier(wrong).suggestion
            found += 1
            self.assertTrue(fix.verified)
            self.assertEqual(fix.rule, "checks.rule.nie_letter")
        self.assertGreater(found, 10)

    def test_N8_a_cif_with_a_wrong_digit_has_several_fixes_so_it_gets_no_suggestion_and_is_never_verified(self):
        for _ in range(1500):
            cif = oracle.make_cif(self.rng)
            position = self.rng.randrange(1, 9)
            if not cif[position].isdigit():
                continue
            wrong = with_digit(cif, position, other_digit(cif[position]))
            if checks.check_cif(wrong):
                continue
            self.assertGreater(len(single_digit_fixes(wrong, checks.check_cif)), 1)
            self.assertIsNone(identifier(wrong).suggestion)

    def test_N8_a_digits_only_iban_fix_is_verified_under_mod_97(self):
        found = 0
        for _ in range(400):
            iban = oracle.make_iban(self.rng, "ES")
            compact = iban.replace(" ", "")
            position = self.rng.randrange(2, len(compact))
            wrong_compact = with_digit(compact, position, other_digit(compact[position]))
            wrong = " ".join(wrong_compact[i : i + 4] for i in range(0, len(wrong_compact), 4))
            result = identifier(wrong)
            if result.status != "mismatch" or result.suggestion is None:
                continue
            found += 1
            self.assertTrue(result.suggestion.verified)
            self.assertEqual(result.suggestion.rule, "checks.rule.iban_mod97")
        self.assertGreater(found, 15)

    def test_N8_an_iban_with_letters_in_its_account_part_never_gets_a_verified_fix(self):
        found = 0
        for country in ("FR", "IT", "CH"):
            for _ in range(60):
                iban = oracle.make_iban(self.rng, country)
                compact = iban.replace(" ", "")
                digit_positions = [i for i, c in enumerate(compact) if c.isdigit() and i >= 2]
                position = self.rng.choice(digit_positions)
                wrong_compact = with_digit(compact, position, other_digit(compact[position]))
                wrong = " ".join(wrong_compact[i : i + 4] for i in range(0, len(wrong_compact), 4))
                result = identifier(wrong)
                if result.suggestion is not None:
                    found += 1
                    self.assertFalse(result.suggestion.verified, wrong)
        self.assertGreater(found, 5)

    def test_N8_the_suggestion_defaults_say_not_verified(self):
        fix = checks.DigitSuggestion(0, "1", "2")
        self.assertEqual((fix.verified, fix.rule), (False, ""))


class ArithmeticTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(SEED)
        self.truths = [t for t in generated_truths() if len(t["vat"]) == 1]

    def error_case(self, role):
        truth = self.rng.choice([t for t in self.truths if t["layout"] in ("L1", "L2")])
        inputs = build(truth)
        word_id = inputs.ids[role] if role in inputs.ids else inputs.ids[(role, 0)]
        text = inputs.text_of(word_id)
        position = self.rng.choice([i for i, c in enumerate(text) if c.isdigit()])
        wrong = inputs.with_text(word_id, with_digit(text, position, CONFUSABLE[text[position]][0]))
        return truth, inputs, wrong, word_id, position

    def test_N8_a_wrong_total_gets_its_one_fix_verified_under_the_total_rule(self):
        truth, inputs, wrong, word_id, position = self.error_case("total")
        fix = number_results(wrong, truth["language"])[word_id].suggestion
        self.assertEqual((fix.position, fix.new), (position, inputs.text_of(word_id)[position]))
        self.assertTrue(fix.verified)
        self.assertEqual(fix.rule, "checks.rule.total")

    def test_N8_a_wrong_line_amount_names_a_rule_that_covers_it(self):
        truth, inputs, wrong, word_id, _ = self.error_case("line_amount")
        fix = number_results(wrong, truth["language"])[word_id].suggestion
        if fix is None:
            self.skipTest("this draw has a tie")
        self.assertTrue(fix.verified)
        self.assertIn(fix.rule, ("checks.rule.line_amount", "checks.rule.lines_sum"))

    def test_N8_a_verified_fix_is_the_only_one_that_closes_the_failing_rules_over_every_number(self):
        """Brute force with the public API only: every digit of every number, every digit value."""
        checked = 0
        for _ in range(6):
            truth = self.rng.choice(self.truths)
            inputs = build(truth)
            locale = truth["language"]
            baseline = number_results(inputs, locale, suggest=False)
            target = self.rng.choice([w for w, r in baseline.items() if r.status == "checked" and r.kind != "iban"])
            text = inputs.text_of(target)
            position = self.rng.choice([i for i, c in enumerate(text) if c.isdigit()])
            wrong = inputs.with_text(target, with_digit(text, position, other_digit(text[position])))
            as_read = number_results(wrong, locale)
            fix = as_read[target].suggestion
            statuses = {w: r.status for w, r in as_read.items()}
            failing = {w for w, s in statuses.items() if s == "mismatch"}
            if not failing:
                continue
            passing_changes = []
            for word_id, result in as_read.items():
                for i, char in enumerate(result.text):
                    for new in "0123456789" if char.isdigit() else "":
                        if new == char:
                            continue
                        trial = wrong.with_text(word_id, with_digit(result.text, i, new))
                        after = number_results(trial, locale, suggest=False)
                        if not any(r.status == "mismatch" for r in after.values()):
                            passing_changes.append((word_id, i, new))
            checked += 1
            if fix is not None and fix.verified:
                self.assertEqual(passing_changes, [(target, fix.position, fix.new)])
            elif len(passing_changes) == 1 and fix is not None:
                self.assertFalse(fix.verified)
        self.assertGreaterEqual(checked, 3)

    def test_N8_a_wrong_withholding_amount_gets_a_fix_that_is_never_verified_because_its_sign_cannot_be_covered(self):
        offered = 0
        for truth in [t for t in self.truths if t["withholding"]][:40]:
            inputs = build(truth)
            word_id = inputs.ids["withholding"]
            text = inputs.text_of(word_id)
            for position in [i for i, c in enumerate(text) if c.isdigit()]:
                wrong = inputs.with_text(word_id, with_digit(text, position, other_digit(text[position])))
                fix = number_results(wrong, truth["language"])[word_id].suggestion
                if fix is None:
                    continue
                offered += 1
                self.assertFalse(fix.verified)
                self.assertIn(fix.rule, ("checks.rule.withholding_amount", "checks.rule.total"))
        self.assertGreater(offered, 10)

    def test_N8_a_tie_between_two_numbers_gives_no_suggestion_for_either(self):
        truth = next(
            t for t in self.truths if any(line["quantity"] == "1" for line in t["lines"]) and t["layout"] == "L1"
        )
        index = next(i for i, line in enumerate(truth["lines"]) if line["quantity"] == "1")
        inputs = build(truth)
        word_id = inputs.ids[("line_unit_price", index)]
        text = inputs.text_of(word_id)
        wrong = inputs.with_text(word_id, with_digit(text, len(text) - 1, str((int(text[-1]) + 3) % 10)))
        alone = dataclasses.replace(
            wrong.fields, subtotal=None, total=None, vat=(), withholding=None, withholding_rate=None
        )
        results = checks.run(wrong.words, alone, truth["language"])
        self.assertEqual([r.suggestion for r in results if r.suggestion is not None], [])

    def implausible_rate_case(self):
        truth = copy.deepcopy(next(t for t in self.truths if t["layout"] == "L1" and not t["withholding"]))
        truth["vat"][0]["rate"] = "22 %"
        reprice(truth)
        inputs = build(truth)
        rate_id = inputs.ids[("vat_rate", 0)]
        return truth, inputs.with_text(rate_id, "23"), rate_id

    def test_N8_a_vat_rate_that_would_become_implausible_is_never_suggested(self):
        """The amounts are those of a 22 % rate and the rate is read 23: the one digit that closes every equation turns
        it into 22, which is no rate. Checked against the same case with 22 made plausible, which does suggest it."""
        truth, wrong, rate_id = self.implausible_rate_case()
        self.assertIsNone(number_results(wrong, truth["language"])[rate_id].suggestion)
        with mock.patch.object(checks, "PLAUSIBLE_RATES", checks.PLAUSIBLE_RATES | {Decimal(22)}):
            fix = number_results(wrong, truth["language"])[rate_id].suggestion
        self.assertEqual((fix.old, fix.new), ("3", "2"))

    def test_N8_a_change_after_which_a_rule_of_the_group_is_gone_is_not_a_fix(self):
        """A digit whose change makes a failing rule vanish (the number is no longer read as that kind of number) has not
        fixed it. The seam drops the failing line rule from every re-evaluation; the same case without it is fixed."""
        truth = next(t for t in self.truths if t["layout"] == "L1" and len(t["vat"]) == 1)
        inputs = build(truth)
        word_id = inputs.ids[("line_amount", 0)]
        text = inputs.text_of(word_id)
        position = next(i for i, c in enumerate(text) if c.isdigit())
        wrong = inputs.with_text(word_id, with_digit(text, position, other_digit(text[position])))
        plain = number_results(wrong, truth["language"])[word_id].suggestion
        self.assertEqual((plain.position, plain.new), (position, text[position]))
        real = checks._equations

        def without_the_line_rule(doc, only=None):
            found = real(doc, only)
            return found if only is None else [eq for eq in found if eq.ident != ("line_amount", 0)]

        with mock.patch.object(checks, "_equations", without_the_line_rule):
            self.assertIsNone(number_results(wrong, truth["language"])[word_id].suggestion)

    def test_N8_the_rule_of_a_fix_is_the_one_that_verifies_the_number_not_the_first_that_passes(self):
        first = checks.Equation("checks.rule.a_first", ("a",), ("w1",), True, {})
        verifying = checks.Equation("checks.rule.b_second", ("b",), ("w1",), True, {}, frozenset({"w1"}))
        idents = {("a",), ("b",)}
        fix = checks._arithmetic_fix(0, "1", "2", "w1", idents, [first, verifying])
        self.assertEqual((fix.verified, fix.rule), (True, "checks.rule.b_second"))
        reversed_order = checks._arithmetic_fix(0, "1", "2", "w1", idents, [verifying, first])
        self.assertEqual((reversed_order.verified, reversed_order.rule), (True, "checks.rule.b_second"))
        only_unverified = checks._arithmetic_fix(0, "1", "2", "w1", {("a",)}, [first])
        self.assertEqual((only_unverified.verified, only_unverified.rule), (False, "checks.rule.a_first"))

    def test_N8_confidence_and_other_word_fields_do_not_change_the_verified_flag(self):
        truth, _, wrong, word_id, _ = self.error_case("total")
        plain = number_results(wrong, truth["language"])[word_id].suggestion
        low = [0.99] * len(wrong.text_of(word_id))
        low[0] = 0.2
        confident = wrong.with_conf(word_id, 0.99, tuple(low))
        self.assertEqual(plain, number_results(confident, truth["language"])[word_id].suggestion)


class AfterTheFixTest(unittest.TestCase):
    """N9: the rules run again on the changed number; `checked` comes only from a verifying rule that now passes."""

    def setUp(self):
        self.rng = random.Random(SEED + 1)

    def apply(self, words, results):
        changes = {}
        for result in results:
            fix = result.suggestion
            if fix is not None and fix.verified:
                word_id = result.word_ids[0]
                text = changes.get(word_id, next(w.text for w in words if w.id == word_id))
                changes[word_id] = with_digit(text, fix.position, fix.new)
        return [dataclasses.replace(w, text=changes.get(w.id, w.text)) for w in words], changes

    def test_N9_a_dni_fixed_by_its_letter_rule_becomes_checked_through_that_rule(self):
        done = 0
        for _ in range(5000):
            dni = oracle.make_dni(self.rng)
            position = self.rng.randrange(8)
            wrong = with_digit(dni, position, other_digit(dni[position]))
            first = identifier(wrong)
            if first.suggestion is None or not first.suggestion.verified:
                continue
            words = oracle.words(wrong)
            fixed, changes = self.apply(words, [first])
            after = checks.run(fixed, NONE, "es")[0]
            self.assertEqual(after.text, dni)
            self.assertEqual(after.status, "checked")
            self.assertTrue(
                any(r.key == first.suggestion.rule and r.verifying and r.result == "pass" for r in after.rules)
            )
            done += 1
        self.assertGreater(done, 10)

    def test_N9_a_cif_that_passes_its_control_stays_unverified(self):
        for _ in range(50):
            after = identifier(oracle.make_cif(self.rng))
            self.assertEqual(after.status, "unverified")
            self.assertTrue(all(not r.verifying for r in after.rules))

    def test_N9_one_wrong_digit_in_an_invoice_never_ends_with_a_wrong_value_checked(self):
        truths = [t for t in generated_truths() if len(t["vat"]) == 1]
        done = 0
        for _ in range(60):
            truth = self.rng.choice(truths)
            inputs = build(truth)
            locale = truth["language"]
            baseline = number_results(inputs, locale, suggest=False)
            target = self.rng.choice([w for w, r in baseline.items() if r.status == "checked" and r.kind != "iban"])
            text = inputs.text_of(target)
            position = self.rng.choice([i for i, c in enumerate(text) if c.isdigit()])
            wrong = inputs.with_text(target, with_digit(text, position, CONFUSABLE[text[position]][0]))
            results = checks.run(wrong.words, wrong.fields, locale)
            fixed, changes = self.apply(wrong.words, results)
            after = {r.word_ids[0]: r for r in checks.run(fixed, wrong.fields, locale, suggest=False)}
            for word_id, result in after.items():
                if result.status == "checked":
                    self.assertEqual(result.text, baseline[word_id].text, f"{word_id} is checked with a wrong value")
            if changes:
                self.assertEqual(changes, {target: text})
            done += 1
        self.assertEqual(done, 60)

    def test_N9_a_changed_number_is_judged_by_the_rules_not_by_having_been_changed(self):
        truth = next(t for t in generated_truths() if len(t["vat"]) == 1 and t["layout"] == "L1")
        inputs = build(truth)
        locale = truth["language"]
        word_id = inputs.ids["total"]
        text = inputs.text_of(word_id)
        wrong = inputs.with_text(word_id, with_digit(text, 0, other_digit(text[0])))
        broken = number_results(wrong, locale, suggest=False)[word_id]
        self.assertEqual(broken.status, "mismatch")
        no_equation = dataclasses.replace(wrong.fields, subtotal=None, lines=())
        fixed_without_rule = number_results(inputs.with_text(word_id, text), locale, suggest=False)[word_id]
        self.assertEqual(fixed_without_rule.status, "checked")
        alone = checks.run(inputs.words, no_equation, locale, suggest=False)
        self.assertEqual({r.word_ids[0]: r.status for r in alone}[word_id], "unverified")


if __name__ == "__main__":
    unittest.main()
