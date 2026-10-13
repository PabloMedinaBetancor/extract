import copy
import dataclasses
import unittest
from decimal import ROUND_HALF_EVEN, ROUND_HALF_UP, Decimal

from bench.corpus.sidecar_inputs import Builder, Inputs, build, generated_truths, reprice
from extract import checks
from extract.types import InvoiceFields

D = Decimal
CENT = D("0.01")


def results_by_id(inputs, locale):
    results = checks.run(inputs.words, inputs.fields, locale)
    by_id = {}
    for result in results:
        for word_id in result.word_ids:
            by_id[word_id] = result
    return by_id


def parse(text, lang):
    plain = text.replace(".", "").replace(",", ".") if lang == "es" else text.replace(",", "")
    return D(plain)


def step(text, lang):
    """The value of the last printed decimal place: 0.01 for "12,34", 1 for "21"."""
    separator = "," if lang == "es" else "."
    decimals = len(text.rsplit(separator, 1)[1]) if separator in text else 0
    return D(1).scaleb(-decimals)


def quantise(value, rounding):
    return value.quantize(CENT, rounding=rounding)


def agree(*values):
    return len(set(values)) == 1


def expected_checked(truth):
    """The roles that must come out `checked` on a consistent generator invoice, stated again from the rules of the number checks
    (coefficient times last-place step of at least one cent, and every accepted rounding agreeing)."""
    lang = truth["language"]
    expected = {"iban", "subtotal", "total"}
    if truth["customer"]["tax_id_kind"] == "dni":
        expected.add("customer_tax_id")
    for i, line in enumerate(truth["lines"]):
        expected.add(("line_amount", i))
        if line["quantity"] is None:
            continue
        qty, price = parse(line["quantity"], lang), parse(line["unit_price"], lang)
        product = qty * price
        if qty != 0 and agree(quantise(product, ROUND_HALF_UP), quantise(product, ROUND_HALF_EVEN)):
            if price * step(line["quantity"], lang) >= CENT:
                expected.add(("line_quantity", i))
            if qty * step(line["unit_price"], lang) >= CENT:
                expected.add(("line_unit_price", i))
    subtotal = parse(truth["subtotal"], lang)
    several = len(truth["vat"]) > 1
    for j, vat in enumerate(truth["vat"]):
        expected.add(("vat_amount", j))
        rate = parse(vat["rate"].replace(" %", ""), lang)
        base = parse(vat["base"], lang) if several else subtotal
        variants = {quantise(base * rate / 100, ROUND_HALF_UP), quantise(base * rate / 100, ROUND_HALF_EVEN)}
        if not several:
            for rounding in (ROUND_HALF_UP, ROUND_HALF_EVEN):
                variants.add(
                    sum((quantise(parse(line["amount"], lang) * rate / 100, rounding) for line in truth["lines"]), D(0))
                )
        if len(variants) == 1 and base * step(vat["rate"].replace(" %", ""), lang) / 100 >= CENT:
            expected.add(("vat_rate", j))
    if truth["withholding"]:
        rate_text = truth["withholding"]["rate"].replace(" %", "")
        rate = parse(rate_text, lang)
        product = subtotal * rate / 100
        if (
            agree(quantise(product, ROUND_HALF_UP), quantise(product, ROUND_HALF_EVEN))
            and subtotal * step(rate_text, lang) / 100 >= CENT
        ):
            expected.add("withholding_rate")
    return expected


class TruthInvoicesTest(unittest.TestCase):
    def test_N3_a_consistent_invoice_never_has_a_mismatch(self):
        for truth in generated_truths():
            inputs = build(truth)
            for word_id, result in results_by_id(inputs, truth["language"]).items():
                self.assertNotEqual(result.status, "mismatch", (truth["id"], word_id, result.text, result.rules))

    def test_N3_exactly_the_covered_numbers_are_checked(self):
        for truth in generated_truths()[::3]:
            inputs = build(truth)
            by_id = results_by_id(inputs, truth["language"])
            checked_ids = {word_id for word_id, result in by_id.items() if result.status == "checked"}
            expected_ids = {inputs.ids[role] for role in expected_checked(truth) if role != "iban"}
            expected_ids |= set(inputs.ids["iban"])
            self.assertEqual(checked_ids, expected_ids, truth["id"])

    def test_N3_the_withholding_amount_is_never_checked_whatever_its_sign(self):
        seen = 0
        for truth in generated_truths():
            if not truth["withholding"]:
                continue
            seen += 1
            inputs = build(truth)
            word_id = inputs.ids["withholding"]
            locale = truth["language"]
            text = inputs.text_of(word_id)
            self.assertTrue(text.startswith("-"))
            for shown in (text, text[1:], f"({text[1:]})"):
                by_id = results_by_id(inputs.with_text(word_id, shown), locale)
                self.assertNotEqual(by_id[word_id].status, "checked", (truth["id"], shown))
                self.assertNotEqual(by_id[word_id].status, "mismatch", (truth["id"], shown))
                rule_flags = {rule.key: rule.verifying for rule in by_id[word_id].rules}
                self.assertFalse(rule_flags["checks.rule.total"])
                self.assertFalse(rule_flags["checks.rule.withholding_amount"])
            wrong = text[:-1] + str((int(text[-1]) + 1) % 10)
            self.assertEqual(results_by_id(inputs.with_text(word_id, wrong), locale)[word_id].status, "mismatch")
        self.assertGreater(seen, 40)

    def test_N3_the_invoice_number_and_a_numeric_date_are_never_checked(self):
        for truth in generated_truths()[::5]:
            inputs = build(truth)
            by_id = results_by_id(inputs, truth["language"])
            self.assertEqual(by_id[inputs.ids["number"]].status, "unverified")
            self.assertEqual(by_id[inputs.ids["date"]].status, "unverified")

    def test_N3_locale_auto_agrees_with_the_documents_locale(self):
        for truth in generated_truths()[::9]:
            inputs = build(truth)
            auto = results_by_id(inputs, "auto")
            explicit = results_by_id(inputs, truth["language"])
            self.assertEqual(
                {k: v.status for k, v in auto.items()}, {k: v.status for k, v in explicit.items()}, truth["id"]
            )


class EquationsTest(unittest.TestCase):
    def setUp(self):
        self.truth = next(
            t for t in generated_truths() if t["layout"] == "L1" and t["language"] == "es" and t["withholding"]
        )
        self.inputs = build(self.truth)
        self.lang = "es"

    def run_with(self, inputs):
        return results_by_id(inputs, self.lang)

    def test_N3_the_rules_attached_to_each_number_use_the_closed_keys(self):
        by_id = self.run_with(self.inputs)
        keys = {rule.key for result in by_id.values() for rule in result.rules}
        allowed = {
            "checks.rule.dni_letter", "checks.rule.nie_letter", "checks.rule.cif_control", "checks.rule.iban_mod97",
            "checks.rule.line_amount", "checks.rule.lines_sum", "checks.rule.vat_amount", "checks.rule.withholding_amount",
            "checks.rule.total", "checks.rule.date_valid", "checks.rule.iban_length", "checks.rule.vat_rate_plausible",
            "checks.rule.number_format",
        }  # fmt: skip
        self.assertLessEqual(keys, allowed)
        self.assertIn("checks.rule.lines_sum", keys)
        self.assertIn("checks.rule.total", keys)
        self.assertIn("checks.rule.withholding_amount", keys)

    def test_N3_a_failing_equation_marks_every_operand_mismatch(self):
        amount_id = self.inputs.ids[("line_amount", 0)]
        text = self.inputs.text_of(amount_id)
        wrong = self.inputs.with_text(amount_id, text[:-1] + str((int(text[-1]) + 1) % 10))
        by_id = self.run_with(wrong)
        for role in (("line_quantity", 0), ("line_unit_price", 0), ("line_amount", 0), "subtotal"):
            self.assertEqual(by_id[self.inputs.ids[role]].status, "mismatch", role)
        for i in range(1, len(self.truth["lines"])):
            self.assertEqual(by_id[self.inputs.ids[("line_amount", i)]].status, "mismatch")

    def test_N3_numbers_outside_the_failing_equation_keep_their_status(self):
        amount_id = self.inputs.ids[("line_amount", 0)]
        text = self.inputs.text_of(amount_id)
        wrong = self.inputs.with_text(amount_id, text[:-1] + str((int(text[-1]) + 1) % 10))
        by_id = self.run_with(wrong)
        self.assertEqual(by_id[self.inputs.ids["total"]].status, "checked")
        self.assertEqual(by_id[self.inputs.ids[("vat_amount", 0)]].status, "checked")

    def test_N3_the_failing_rule_carries_what_was_expected_and_found(self):
        amount_id = self.inputs.ids[("line_amount", 0)]
        text = self.inputs.text_of(amount_id)
        wrong = self.inputs.with_text(amount_id, text[:-1] + str((int(text[-1]) + 1) % 10))
        rule = next(r for r in self.run_with(wrong)[amount_id].rules if r.key == "checks.rule.line_amount")
        self.assertEqual(rule.result, "fail")
        self.assertTrue(rule.verifying)
        self.assertEqual(rule.params["found"], parse(wrong.text_of(amount_id), "es"))
        self.assertEqual(rule.params["expected"], parse(text, "es"))

    def test_N3_the_base_under_a_vat_rule_alone_is_not_verified_by_it(self):
        by_id = self.run_with(self.inputs)
        subtotal = by_id[self.inputs.ids["subtotal"]]
        vat_rules = [r for r in subtotal.rules if r.key == "checks.rule.vat_amount"]
        self.assertEqual(len(vat_rules), 1)
        self.assertFalse(vat_rules[0].verifying)
        self.assertEqual(vat_rules[0].result, "pass")

    def test_N3_several_rates_keep_their_bases_unverified(self):
        truth = next(t for t in generated_truths() if len(t["vat"]) > 1 and t["language"] == "es")
        inputs = build(truth)
        by_id = results_by_id(inputs, "es")
        for j in range(len(truth["vat"])):
            self.assertEqual(by_id[inputs.ids[("vat_base", j)]].status, "unverified")
            self.assertEqual(by_id[inputs.ids[("vat_amount", j)]].status, "checked")

    def test_N3_a_missing_operand_means_the_rule_is_not_applied(self):
        fields = self.inputs.fields.__class__(**{**self.inputs.fields.__dict__, "total": None})
        by_id = results_by_id(type(self.inputs)(self.inputs.words, fields, self.inputs.ids), "es")
        total_word = by_id[self.inputs.ids["total"]]
        self.assertEqual(total_word.status, "unverified")
        for result in by_id.values():
            self.assertNotIn("checks.rule.total", {r.key for r in result.rules if r.result != "not_applicable"})
        self.assertEqual(by_id[self.inputs.ids["subtotal"]].status, "checked")

    def test_N3_without_a_quantity_column_the_line_rule_does_not_assume_one(self):
        truth = next(t for t in generated_truths() if t["layout"] == "L3" and t["language"] == "es")
        self.assertTrue(all(line["quantity"] is None for line in truth["lines"]))
        self.assertTrue(any(line["unit_price"] != line["amount"] for line in truth["lines"]))
        inputs = build(truth)
        by_id = results_by_id(inputs, "es")
        for i in range(len(truth["lines"])):
            price = by_id[inputs.ids[("line_unit_price", i)]]
            amount = by_id[inputs.ids[("line_amount", i)]]
            self.assertEqual(price.status, "unverified")
            self.assertEqual(amount.status, "checked")
            for result in (price, amount):
                line_rules = [r for r in result.rules if r.key == "checks.rule.line_amount"]
                self.assertEqual([r.result for r in line_rules], ["not_applicable"])

    def test_N3_a_quantity_of_zero_covers_nothing_of_its_line_but_the_sum_still_covers_the_amount(self):
        truth = copy.deepcopy(self.truth)
        truth["lines"][0].update(quantity="0", unit_price="5,00", amount="0,00")
        self._reprice(truth)
        inputs = build(truth)
        by_id = results_by_id(inputs, "es")
        self.assertEqual(by_id[inputs.ids[("line_quantity", 0)]].status, "unverified")
        self.assertEqual(by_id[inputs.ids[("line_unit_price", 0)]].status, "unverified")
        self.assertEqual(by_id[inputs.ids[("line_amount", 0)]].status, "checked")
        for result in by_id.values():
            self.assertNotEqual(result.status, "mismatch")

    def test_N3_a_unit_price_with_more_decimals_than_cents_is_covered_only_if_the_quantity_makes_it_sensitive(self):
        truth = copy.deepcopy(self.truth)
        truth["lines"][0].update(quantity="3", unit_price="1,234", amount="3,70")
        self._reprice(truth)
        inputs = build(truth)
        by_id = results_by_id(inputs, "es")
        self.assertEqual(by_id[inputs.ids[("line_unit_price", 0)]].status, "unverified")
        self.assertEqual(by_id[inputs.ids[("line_quantity", 0)]].status, "checked")
        truth["lines"][0].update(quantity="30", unit_price="1,234", amount="37,02")
        self._reprice(truth)
        inputs = build(truth)
        by_id = results_by_id(inputs, "es")
        self.assertEqual(by_id[inputs.ids[("line_unit_price", 0)]].status, "checked")

    def test_N3_both_roundings_of_vat_are_accepted_and_the_rate_stays_unverified(self):
        truth = self._tiny_invoice(["0,05", "0,05", "0,05"], "10")
        for vat_amount in ("0,02", "0,03"):
            truth["vat"][0]["amount"] = vat_amount
            truth["total"] = "0,17" if vat_amount == "0,02" else "0,18"
            inputs = build(truth)
            by_id = results_by_id(inputs, "es")
            for result in by_id.values():
                self.assertNotEqual(result.status, "mismatch", vat_amount)
            self.assertEqual(by_id[inputs.ids[("vat_rate", 0)]].status, "unverified", vat_amount)
            self.assertEqual(by_id[inputs.ids[("vat_amount", 0)]].status, "checked", vat_amount)
        truth["vat"][0]["amount"] = "0,04"
        truth["total"] = "0,19"
        by_id = results_by_id(build(truth), "es")
        self.assertTrue(any(r.status == "mismatch" for r in by_id.values()))

    def test_N3_half_even_and_half_up_both_accepted_when_they_differ(self):
        truth = self._tiny_invoice(["0,50"], "21")
        for vat_amount, total in (("0,11", "0,61"), ("0,10", "0,60")):
            truth["vat"][0]["amount"] = vat_amount
            truth["total"] = total
            inputs = build(truth)
            by_id = results_by_id(inputs, "es")
            self.assertNotIn("mismatch", {r.status for r in by_id.values()}, vat_amount)
            self.assertEqual(by_id[inputs.ids[("vat_rate", 0)]].status, "unverified", vat_amount)

    def _small_base(self, amount, vat, withheld, total):
        truth = self._tiny_invoice([amount], "21")
        truth["vat"][0]["amount"] = vat
        truth["withholding"] = {"rate": "15 %", "amount": "-" + withheld}
        truth["total"] = total
        inputs = build(truth)
        return inputs, results_by_id(inputs, "es")

    def test_N3_a_tiny_base_leaves_the_rates_unverified_but_not_the_amounts(self):
        inputs, by_id = self._small_base("0,40", "0,08", "0,06", "0,42")
        for result in by_id.values():
            self.assertNotEqual(result.status, "mismatch")
        self.assertEqual(by_id[inputs.ids[("vat_amount", 0)]].status, "checked")
        self.assertEqual(by_id[inputs.ids[("vat_rate", 0)]].status, "unverified")
        self.assertEqual(by_id[inputs.ids["withholding_rate"]].status, "unverified")
        vat_rule = next(r for r in by_id[inputs.ids[("vat_rate", 0)]].rules if r.key == "checks.rule.vat_amount")
        self.assertEqual((vat_rule.result, vat_rule.verifying), ("pass", False))

    def test_N3_a_base_of_one_euro_is_the_smallest_that_covers_a_whole_number_rate(self):
        inputs, by_id = self._small_base("1,00", "0,21", "0,15", "1,06")
        self.assertEqual(by_id[inputs.ids[("vat_rate", 0)]].status, "checked")
        self.assertEqual(by_id[inputs.ids["withholding_rate"]].status, "checked")
        inputs, by_id = self._small_base("0,99", "0,21", "0,15", "1,05")
        self.assertEqual(by_id[inputs.ids[("vat_rate", 0)]].status, "unverified")
        self.assertEqual(by_id[inputs.ids["withholding_rate"]].status, "unverified")

    def test_N3_no_tolerance_a_cent_off_is_a_mismatch(self):
        for field_name in ("subtotal", "total"):
            text = self.truth[field_name]
            cents = parse(text, "es") + CENT
            wrong = f"{cents:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
            inputs = self.inputs.with_text(self.inputs.ids[field_name], wrong)
            by_id = self.run_with(inputs)
            self.assertEqual(by_id[self.inputs.ids[field_name]].status, "mismatch", field_name)

    def test_N3_a_written_out_date_is_several_words_and_has_no_date_rule(self):
        row = Builder()
        row.add("5", "de", "marzo", "de", "2026")
        by_id = results_by_id(Inputs(row.words, InvoiceFields(), {}), "es")
        for word in ("5", "2026"):
            result = next(r for r in by_id.values() if r.text == word)
            self.assertEqual(result.status, "unverified")
            self.assertEqual(result.kind, "integer")
            self.assertNotIn("checks.rule.date_valid", {r.key for r in result.rules})
        self.assertEqual(len(by_id), 2)

    def test_N3_a_thousands_separator_written_as_a_space_is_never_checked(self):
        index = next(i for i, line in enumerate(self.truth["lines"]) if len(line["amount"]) >= 8)
        amount_id = self.inputs.ids[("line_amount", index)]
        head, tail = self.inputs.text_of(amount_id).replace(".", " ", 1).split(" ", 1)
        words = list(self.inputs.words)
        position = next(i for i, w in enumerate(words) if w.id == amount_id)
        words[position] = dataclasses.replace(words[position], text=head)
        words.insert(position + 1, dataclasses.replace(words[position], id="p1w9999", text=tail))
        for result in checks.run(words, self.inputs.fields, "es"):
            if set(result.word_ids) & {amount_id, "p1w9999"}:
                self.assertNotEqual(result.status, "checked", result)

    def _reprice(self, truth):
        reprice(truth)

    @staticmethod
    def _format(value):
        return f"{value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")

    def _tiny_invoice(self, amounts, rate):
        truth = copy.deepcopy(self.truth)
        truth["lines"] = [{"description": "x", "quantity": "1", "unit_price": a, "amount": a} for a in amounts]
        truth["subtotal"] = self._format(sum(parse(a, "es") for a in amounts))
        truth["vat"] = [{"rate": f"{rate} %", "base": truth["subtotal"], "amount": "0,00"}]
        truth["withholding"] = None
        return truth


if __name__ == "__main__":
    unittest.main()
