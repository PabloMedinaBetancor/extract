import dataclasses
import unittest
from decimal import Decimal

from bench.corpus.sidecar_inputs import Builder, Inputs, build, generated_truths
from extract import checks
from extract.types import InvoiceFields, VatPair
from tests.checks_oracle import GOLDEN, GOLDEN_INVALID, words


def one(results, text):
    matches = [r for r in results if r.text == text]
    assert len(matches) == 1, (text, [r.text for r in results])
    return matches[0]


class KindsTest(unittest.TestCase):
    def run_words(self, *texts, locale="es"):
        row = words(*texts)
        return checks.run(row, InvoiceFields(), locale)

    def test_N5_each_kind_of_token_is_recognised_and_checked_by_its_own_rule(self):
        results = self.run_words(GOLDEN["dni"][0], GOLDEN["nie"][0], GOLDEN["cif"][2])
        self.assertEqual(
            [(r.kind, r.status, r.rules[0].key, r.rules[0].verifying) for r in results],
            [
                ("dni", "checked", "checks.rule.dni_letter", True),
                ("nie", "checked", "checks.rule.nie_letter", True),
                ("cif", "unverified", "checks.rule.cif_control", False),
            ],
        )

    def test_N5_a_valid_cif_passes_its_rule_without_verifying_anything(self):
        for example in GOLDEN["cif"]:
            result = self.run_words(example)[0]
            self.assertEqual((result.kind, result.status), ("cif", "unverified"), example)
            self.assertEqual(
                [(r.key, r.result, r.verifying) for r in result.rules], [("checks.rule.cif_control", "pass", False)]
            )

    def test_N5_a_cif_with_a_changed_digit_is_a_mismatch(self):
        for example in GOLDEN["cif"]:
            for position in range(1, 8):
                changed = example[:position] + str((int(example[position]) + 1) % 10) + example[position + 1 :]
                result = self.run_words(changed)[0]
                self.assertEqual(result.status, "mismatch", changed)
                self.assertEqual(result.rules[0].result, "fail")

    def test_N5_a_wrong_check_digit_is_a_mismatch(self):
        results = self.run_words(GOLDEN_INVALID["dni"], GOLDEN_INVALID["nie"], GOLDEN_INVALID["cif"])
        self.assertEqual([r.status for r in results], ["mismatch"] * 3)
        self.assertEqual([r.rules[0].result for r in results], ["fail"] * 3)

    def test_N5_an_iban_in_several_words_is_one_number_with_all_their_ids(self):
        row = words("IBAN:", *GOLDEN["iban"][0].split(), "Forma", "de", "pago")
        results = checks.run(row, InvoiceFields(), "es")
        iban = [r for r in results if r.kind == "iban"]
        self.assertEqual(len(iban), 1)
        self.assertEqual(iban[0].word_ids, tuple(w.id for w in row[1:7]))
        self.assertEqual(iban[0].text, GOLDEN["iban"][0])
        self.assertEqual(iban[0].status, "checked")
        self.assertEqual(len(results), 1, "the groups of the IBAN are not also reported one by one")

    def test_N5_a_wrong_iban_is_a_mismatch_and_a_wrong_length_says_so(self):
        bad = checks.run(words(*GOLDEN_INVALID["iban"].split()), InvoiceFields(), "es")
        self.assertEqual((bad[0].kind, bad[0].status), ("iban", "mismatch"))
        short = checks.run(words(*GOLDEN["iban"][0].split()[:-1]), InvoiceFields(), "es")
        keys = {r.key for r in short[0].rules}
        self.assertEqual(short[0].status, "mismatch")
        self.assertIn("checks.rule.iban_length", keys)

    def test_N5_an_iban_with_alphanumeric_characters_passes_but_is_not_checked(self):
        french = GOLDEN["iban"][3]
        results = checks.run(words(*french.split()), InvoiceFields(), "es")
        self.assertEqual(results[0].status, "unverified")
        mod = next(r for r in results[0].rules if r.key == "checks.rule.iban_mod97")
        self.assertEqual((mod.result, mod.verifying), ("pass", False))
        self.assertEqual(mod.params["country"], "FR")

    def test_N5_dates(self):
        results = self.run_words("18/03/2026", "31/02/2026", "13/13/2026")
        self.assertEqual(
            [(r.kind, r.status) for r in results], [("date", "unverified"), ("date", "mismatch"), ("date", "mismatch")]
        )
        self.assertEqual(results[0].rules[0].key, "checks.rule.date_valid")
        self.assertFalse(results[0].rules[0].verifying)
        self.assertEqual(results[0].rules[0].result, "pass")
        english = checks.run(words("03/18/2026"), InvoiceFields(), "en")
        self.assertEqual(english[0].status, "unverified")

    def test_N5_amounts_percentages_integers_and_codes(self):
        results = self.run_words("1.234,56", "21%", "2026", "F-2025-3318", "-5,00")
        kinds = {r.text: (r.kind, r.status) for r in results}
        self.assertEqual(kinds["1.234,56"], ("amount", "unverified"))
        self.assertEqual(kinds["21%"], ("percent", "unverified"))
        self.assertEqual(kinds["2026"], ("integer", "unverified"))
        self.assertEqual(kinds["F-2025-3318"], ("other", "unverified"))
        self.assertEqual(kinds["-5,00"], ("amount", "unverified"))
        self.assertEqual(one(results, "1.234,56").value, Decimal("1234.56"))
        self.assertEqual(one(results, "21%").value, Decimal("21"))
        self.assertEqual(one(results, "2026").value, Decimal("2026"))
        self.assertIsNone(one(results, "F-2025-3318").value)

    def test_N5_a_number_in_the_wrong_notation_for_the_document_fails_its_format_rule(self):
        results = self.run_words("1.234,56", "12.5", "3,20", "1,50")
        bad = one(results, "12.5")
        self.assertEqual(bad.status, "mismatch")
        self.assertEqual(bad.rules[0].key, "checks.rule.number_format")
        self.assertEqual(bad.rules[0].params["locale"], "es")
        self.assertEqual(one(results, "1.234,56").status, "unverified")

    def test_N5_a_house_number_with_a_comma_after_it_is_an_integer_not_a_bad_amount(self):
        results = self.run_words("Mayor", "33,", "Madrid", "12.", "5;")
        self.assertEqual(
            [(r.text, r.kind, r.status) for r in results],
            [("33,", "integer", "unverified"), ("12.", "integer", "unverified"), ("5;", "integer", "unverified")],
        )
        self.assertEqual([r.value for r in results], [Decimal(33), Decimal(12), Decimal(5)])

    def test_N5_a_word_without_a_digit_is_not_a_number(self):
        self.assertEqual(self.run_words("Total", "IVA", "%", "€"), [])

    def test_N5_the_locale_must_be_auto_es_or_en(self):
        with self.assertRaises(ValueError):
            checks.run(words("1"), InvoiceFields(), "fr")

    def test_N5_each_word_belongs_to_at_most_one_result_and_inputs_are_not_changed(self):
        truth = next(t for t in generated_truths() if t["language"] == "es")
        inputs = build(truth)
        before = list(inputs.words)
        results = checks.run(inputs.words, inputs.fields, "es")
        seen = [wid for r in results for wid in r.word_ids]
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(before, inputs.words)


class ConfidenceTest(unittest.TestCase):
    def setUp(self):
        self.truth = next(t for t in generated_truths() if t["layout"] == "L2" and t["language"] == "en")
        self.inputs = build(self.truth)

    def statuses(self, inputs, **kwargs):
        return {r.word_ids[0]: r.status for r in checks.run(inputs.words, inputs.fields, "en", **kwargs)}

    def test_N5_the_invoice_number_with_perfect_confidence_stays_unverified(self):
        word_id = self.inputs.ids["number"]
        text = self.inputs.text_of(word_id)
        sure = self.inputs.with_conf(word_id, 1.0, (1.0,) * len(text))
        self.assertEqual(self.statuses(sure)[word_id], "unverified")

    def _scored(self, conf, status, p_wrong):
        words_ = [
            dataclasses.replace(w, conf=conf, char_conf=(conf,) * len(w.text), status=status, p_wrong=p_wrong)
            for w in self.inputs.words
        ]
        return Inputs(words_, self.inputs.fields, self.inputs.ids)

    def test_N5_scores_and_confidence_never_change_who_is_checked(self):
        baseline = self.statuses(self.inputs)
        self.assertIn("checked", baseline.values())
        self.assertEqual(self.statuses(self._scored(1.0, "ok", 0.0001)), baseline)
        self.assertEqual(self.statuses(self._scored(0.1, "doubtful", 0.95)), baseline)

    def test_N5_a_misread_number_with_perfect_confidence_is_still_caught(self):
        word_id = self.inputs.ids["total"]
        text = self.inputs.text_of(word_id)
        wrong = self.inputs.with_text(word_id, text[:-1] + str((int(text[-1]) + 1) % 10)).with_conf(
            word_id, 1.0, (1.0,) * len(text)
        )
        self.assertEqual(self.statuses(wrong)[word_id], "mismatch")

    def test_N5_a_vat_rate_that_is_plausible_but_has_no_amount_is_unverified(self):
        builder = Builder()
        rate, _ = builder.cell("21", "%")
        fields = InvoiceFields(vat=(VatPair(rate, None),))
        results = checks.run(builder.words, fields, "es")
        first = results[0]
        self.assertEqual(first.status, "unverified")
        by_key = {r.key: r for r in first.rules}
        self.assertEqual(set(by_key), {"checks.rule.vat_rate_plausible", "checks.rule.number_format"})
        plausible = by_key["checks.rule.vat_rate_plausible"]
        self.assertEqual((plausible.result, plausible.verifying), ("pass", False))
        self.assertEqual(plausible.params["rate"], Decimal("21"))

    def test_N5_a_vat_rate_nobody_uses_is_a_mismatch(self):
        builder = Builder()
        rate, _ = builder.cell("13", "%")
        results = checks.run(builder.words, InvoiceFields(vat=(VatPair(rate, None),)), "es")
        self.assertEqual(results[0].status, "mismatch")
        failed = [r.key for r in results[0].rules if r.result == "fail"]
        self.assertEqual(failed, ["checks.rule.vat_rate_plausible"])

    def test_N5_nothing_is_checked_on_an_unreadable_page(self):
        baseline = self.statuses(self.inputs)
        self.assertIn("checked", baseline.values())
        dead = self.statuses(self.inputs, unreadable_pages={1})
        self.assertNotIn("checked", dead.values())
        self.assertEqual(set(dead), set(baseline))
        self.assertEqual(sum(1 for s in dead.values() if s == "mismatch"), 0)

    def test_N5_an_unreadable_page_keeps_its_failures(self):
        word_id = self.inputs.ids["total"]
        text = self.inputs.text_of(word_id)
        wrong = self.inputs.with_text(word_id, text[:-1] + str((int(text[-1]) + 1) % 10))
        self.assertEqual(self.statuses(wrong, unreadable_pages={1})[word_id], "mismatch")


if __name__ == "__main__":
    unittest.main()
