import dataclasses
import unittest

from bench.corpus.sidecar_inputs import build, generated_truths
from extract import checks


def run_by_id(inputs, locale):
    by_id = {}
    for result in checks.run(inputs.words, inputs.fields, locale):
        for word_id in result.word_ids:
            by_id[word_id] = result
    return by_id


def break_digit(text, position, new):
    return text[:position] + new + text[position + 1 :]


class SuggestionTest(unittest.TestCase):
    def setUp(self):
        self.truth = next(
            t for t in generated_truths() if t["layout"] == "L1" and t["language"] == "es" and len(t["vat"]) == 1
        )
        self.inputs = build(self.truth)

    def test_N4_a_single_digit_error_in_the_total_gets_exactly_that_fix(self):
        word_id = self.inputs.ids["total"]
        text = self.inputs.text_of(word_id)
        position = max(i for i, c in enumerate(text) if c.isdigit() and i < len(text) - 3)
        original = text[position]
        wrong = "3" if original != "3" else "8"
        by_id = run_by_id(self.inputs.with_text(word_id, break_digit(text, position, wrong)), "es")
        suggestion = by_id[word_id].suggestion
        self.assertIsNotNone(suggestion)
        self.assertEqual((suggestion.position, suggestion.old, suggestion.new), (position, wrong, original))

    def test_N4_a_suggestion_is_offered_never_applied(self):
        word_id = self.inputs.ids["total"]
        text = self.inputs.text_of(word_id)
        wrong_text = break_digit(text, 0, "3" if text[0] != "3" else "8")
        wrong = self.inputs.with_text(word_id, wrong_text)
        result = run_by_id(wrong, "es")[word_id]
        self.assertEqual(result.status, "mismatch")
        self.assertEqual(result.text, wrong_text)
        self.assertIsNotNone(result.suggestion)

    def _quantity_one_case(self):
        truth = next(
            t for t in generated_truths() if t["layout"] == "L1" and any(line["quantity"] == "1" for line in t["lines"])
        )
        i = next(i for i, line in enumerate(truth["lines"]) if line["quantity"] == "1")
        inputs = build(truth)
        word_id = inputs.ids[("line_unit_price", i)]
        text = inputs.text_of(word_id)
        wrong = inputs.with_text(word_id, break_digit(text, len(text) - 1, str((int(text[-1]) + 3) % 10)))
        return truth, inputs, wrong, i, word_id

    def test_N4_two_equally_good_fixes_give_no_suggestion(self):
        truth, inputs, wrong, i, word_id = self._quantity_one_case()
        alone = dataclasses.replace(
            wrong.fields, subtotal=None, total=None, vat=(), withholding=None, withholding_rate=None
        )
        by_id = run_by_id(type(wrong)(wrong.words, alone, wrong.ids), truth["language"])
        self.assertEqual(by_id[word_id].status, "mismatch")
        for role in (("line_unit_price", i), ("line_amount", i), ("line_quantity", i)):
            self.assertIsNone(by_id[inputs.ids[role]].suggestion, role)

    def test_N4_a_rule_that_still_holds_on_a_partner_rules_it_out_as_a_fix(self):
        truth, inputs, wrong, i, word_id = self._quantity_one_case()
        by_id = run_by_id(wrong, truth["language"])
        suggestion = by_id[word_id].suggestion
        self.assertIsNotNone(
            suggestion, "the line amount is held by the sum of lines, so only the price can be the fix"
        )
        self.assertEqual(suggestion.new, inputs.text_of(word_id)[suggestion.position])
        self.assertIsNone(by_id[inputs.ids[("line_amount", i)]].suggestion)

    def test_N4_two_wrong_digits_in_one_equation_give_no_suggestion(self):
        word_id = self.inputs.ids["total"]
        text = self.inputs.text_of(word_id)
        wrong = self.inputs.with_text(
            word_id, break_digit(break_digit(text, 0, "3" if text[0] != "3" else "8"), 1, "4")
        )
        by_id = run_by_id(wrong, "es")
        self.assertIsNone(by_id[word_id].suggestion)

    def test_N4_only_a_digit_is_ever_suggested_never_a_separator_or_a_letter(self):
        for truth in generated_truths()[::11]:
            inputs = build(truth)
            word_id = inputs.ids["subtotal"]
            text = inputs.text_of(word_id)
            wrong = inputs.with_text(word_id, break_digit(text, 0, "3" if text[0] != "3" else "8"))
            for result in run_by_id(wrong, truth["language"]).values():
                if result.suggestion:
                    self.assertTrue(result.suggestion.old.isdigit() and result.suggestion.new.isdigit())
                    self.assertTrue(result.text[result.suggestion.position].isdigit())

    def test_N4_a_suggestion_is_attached_to_the_number_it_changes_and_to_no_other(self):
        word_id = self.inputs.ids["total"]
        text = self.inputs.text_of(word_id)
        wrong = self.inputs.with_text(word_id, break_digit(text, 0, "3" if text[0] != "3" else "8"))
        by_id = run_by_id(wrong, "es")
        holders = [wid for wid, result in by_id.items() if result.suggestion]
        self.assertEqual(holders, [word_id])

    def test_N4_a_number_that_passes_has_no_suggestion(self):
        for result in run_by_id(self.inputs, "es").values():
            self.assertIsNone(result.suggestion)

    def test_N4_a_plausibility_failure_alone_has_no_suggestion(self):
        by_id = run_by_id(self.inputs.with_text(self.inputs.ids["date"], "45/13/2026"), "es")
        result = by_id[self.inputs.ids["date"]]
        self.assertEqual(result.status, "mismatch")
        self.assertIsNone(result.suggestion)

    def test_N4_confidence_does_not_change_which_fix_is_offered(self):
        word_id = self.inputs.ids["total"]
        text = self.inputs.text_of(word_id)
        wrong = self.inputs.with_text(word_id, break_digit(text, 0, "3" if text[0] != "3" else "8"))
        plain = run_by_id(wrong, "es")[word_id].suggestion
        low = [0.99] * len(wrong.text_of(word_id))
        low[0] = 0.3
        word_index = next(i for i, w in enumerate(wrong.words) if w.id == word_id)
        words = list(wrong.words)
        words[word_index] = dataclasses.replace(words[word_index], char_conf=tuple(low), conf=0.9)
        with_conf = {r.word_ids[0]: r for r in checks.run(words, wrong.fields, "es")}[word_id].suggestion
        self.assertEqual(plain, with_conf)


if __name__ == "__main__":
    unittest.main()
