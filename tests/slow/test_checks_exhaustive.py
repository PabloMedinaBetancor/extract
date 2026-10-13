import unittest
from decimal import Decimal

from bench import inject
from bench.corpus.sidecar_inputs import build, generated_truths
from extract import checks

CENT = Decimal("0.01")
SAMPLE_PER_COMBINATION = 1


def sample():
    """One invoice per layout, language and withholding case: every template and font family appears."""
    seen = {}
    for truth in generated_truths():
        key = (truth["layout"], truth["language"], truth["withholding"] is not None, len(truth["vat"]) > 1)
        seen.setdefault(key, []).append(truth)
    return [group[0] for _, group in sorted(seen.items())]


def status_of(inputs, word_id, locale):
    for result in checks.run(inputs.words, inputs.fields, locale, suggest=False):
        if word_id in result.word_ids:
            return result.status
    raise AssertionError(f"no result for {word_id}")


def cents_away(text, locale, delta):
    """The same number with one cent more or less, in the same notation."""
    comma = locale == "es"
    plain = text.replace(".", "").replace(",", ".") if comma else text.replace(",", "")
    value = Decimal(plain) + delta
    out = f"{value:,.2f}"
    return out.replace(",", "_").replace(".", ",").replace("_", ".") if comma else out


class ExhaustiveTest(unittest.TestCase):
    def test_N3_every_one_digit_substitution_of_every_checked_number_is_caught(self):
        total = 0
        for truth in sample():
            locale = truth["language"]
            inputs = build(truth)
            baseline = checks.run(inputs.words, inputs.fields, locale, suggest=False)
            for result in baseline:
                if result.status != "checked":
                    continue
                for word_id in result.word_ids:
                    text = inputs.text_of(word_id)
                    for position, char in enumerate(text):
                        if not char.isdigit():
                            continue
                        for other in "0123456789":
                            if other == char:
                                continue
                            total += 1
                            mutated = inputs.with_text(word_id, text[:position] + other + text[position + 1 :])
                            status = status_of(mutated, word_id, locale)
                            self.assertNotEqual(status, "checked", (truth["id"], text, position, other))
        self.assertGreater(total, 20_000)

    def test_N3_every_one_character_edit_of_every_checked_number_is_caught_or_keeps_the_value(self):
        """Substitution by any character (digits, separators, signs, brackets), deletion and insertion. A number may
        stay checked only if it still reads as the true value: a deleted thousands dot is no error, a lost minus is."""
        total = changed_value_checked = 0
        for truth in sample()[::4]:
            locale = truth["language"]
            inputs = build(truth)
            for before in inject.checked_results(inputs, locale).values():
                for word_id in before.word_ids:
                    text = inputs.text_of(word_id)
                    for edited in set(inject.edits(text)) - {text}:
                        total += 1
                        after = checks.run(
                            inputs.with_text(word_id, edited).words, inputs.fields, locale, suggest=False
                        )
                        result = inject.result_of(after, word_id)
                        self.assertFalse(inject.is_false_checked(before, result), (truth["id"], text, edited))
                        changed_value_checked += result is not None and result.status == "checked" and edited != text
        self.assertGreater(total, 30_000)
        self.assertGreater(
            changed_value_checked, 0, "some edits keep the value (leading zero), so the criterion is not vacuous"
        )

    def test_N3_every_one_cent_misread_of_every_checked_amount_is_caught(self):
        total = 0
        for truth in sample():
            locale = truth["language"]
            inputs = build(truth)
            for result in checks.run(inputs.words, inputs.fields, locale, suggest=False):
                if result.status != "checked" or result.kind not in ("amount", "quantity"):
                    continue
                text = inputs.text_of(result.word_ids[0])
                if "," not in text and "." not in text:
                    continue
                for delta in (CENT, -CENT):
                    total += 1
                    mutated = inputs.with_text(result.word_ids[0], cents_away(text, locale, delta))
                    self.assertNotEqual(
                        status_of(mutated, result.word_ids[0], locale), "checked", (truth["id"], text, delta)
                    )
        self.assertGreater(total, 100)

    def test_N3_the_sample_covers_every_layout_and_both_languages(self):
        picked = sample()
        self.assertEqual({t["layout"] for t in picked}, {"L1", "L2", "L3", "L4", "L5", "L6"})
        self.assertEqual({t["language"] for t in picked}, {"es", "en"})
        self.assertGreaterEqual(len(picked), 12)


if __name__ == "__main__":
    unittest.main()
