import unittest

import numpy as np

from extract.guard import reasons, signals
from extract.guard.signals import COLUMNS, INDEX
from extract.types import Word

SPEC_KEYS = {
    "guard.reason.low_char_conf",
    "guard.reason.not_in_dictionary",
    "guard.reason.close_to",
    "guard.reason.mixed_letters_digits",
    "guard.reason.odd_chars",
    "guard.reason.unusual_height",
    "guard.reason.unusual_width",
    "guard.reason.bad_number_format",
    "guard.reason.low_page_quality",
    "guard.reason.neighbours_unusual",
}


def contributions(**pushes):
    row = np.zeros(len(COLUMNS) + 1)
    for name, value in pushes.items():
        row[INDEX[name]] = value
    return row


def word(text="ordenansa", chars=(0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.35, 0.9)):
    return Word("p1w0", 1, text, (0, 0, 70, 12), sum(chars) / len(chars), chars, 0, 0)


class ReasonKeysTest(unittest.TestCase):
    def test_G9_the_reason_keys_are_the_closed_list_of_the_design(self):
        self.assertEqual(set(reasons.KEYS), SPEC_KEYS)

    def test_G9_every_signal_that_has_a_reason_is_in_the_schema_once(self):
        named = [name for names in reasons.GROUPS.values() for name in names]
        self.assertEqual(len(named), len(set(named)))
        self.assertTrue(set(named) <= set(signals.COLUMNS))


class ExplainTest(unittest.TestCase):
    def explain(self, **pushes):
        return reasons.explain(contributions(**pushes), word(), np.zeros(len(COLUMNS)), ("es", "en"))

    def test_G9_the_largest_pushes_towards_wrong_become_reasons_in_order(self):
        got = self.explain(conf_mean=0.4, conf_min=0.3, in_dictionary=0.5, odd_chars=0.1)
        self.assertEqual(
            [r.key for r in got],
            ["guard.reason.low_char_conf", "guard.reason.not_in_dictionary", "guard.reason.odd_chars"],
        )

    def test_G9_signals_of_one_reason_add_up(self):
        got = self.explain(conf_mean=0.2, conf_min=0.2, low_conf_chars=0.2, in_dictionary=0.5)
        self.assertEqual(got[0].key, "guard.reason.low_char_conf")

    def test_G9_at_most_three_reasons(self):
        got = self.explain(conf_mean=1, in_dictionary=1, odd_chars=1, height_ratio=1, width_per_char_ratio=1)
        self.assertEqual(len(got), 3)

    def test_G9_a_signal_that_pulls_towards_right_is_never_a_reason(self):
        got = self.explain(conf_mean=-2.0, in_dictionary=0.3)
        self.assertEqual([r.key for r in got], ["guard.reason.not_in_dictionary"])

    def test_a_word_without_digits_is_never_a_badly_written_number(self):
        plain = word()
        got = reasons.explain(
            contributions(number_format_ok=2.0, has_digit=1.5, in_dictionary=0.5),
            plain,
            np.zeros(len(COLUMNS)),
            ("es", "en"),
        )
        self.assertEqual([r.key for r in got], ["guard.reason.not_in_dictionary"])

    def test_a_word_with_digits_keeps_the_number_reasons(self):
        number = word("12,5O", (0.9,) * 5)
        got = reasons.explain(
            contributions(number_format_ok=2.0, has_digit=1.5, in_dictionary=0.5),
            number,
            np.zeros(len(COLUMNS)),
            ("es", "en"),
        )
        self.assertEqual(
            [r.key for r in got][:2], ["guard.reason.bad_number_format", "guard.reason.mixed_letters_digits"]
        )

    def test_G9_nothing_pushes_no_reasons(self):
        self.assertEqual(self.explain(), ())

    def test_G9_signals_without_a_reason_are_ignored(self):
        self.assertEqual(self.explain(length=3.0, case_pattern=2.0, hyphen_fragment=1.0), ())

    def test_G9_the_low_confidence_reason_names_the_weakest_character(self):
        got = self.explain(conf_min=1.0)[0]
        self.assertEqual((got.key, got.params["char"], got.params["conf"]), ("guard.reason.low_char_conf", "s", 0.35))

    def test_G9_the_close_to_reason_names_the_nearest_listed_word(self):
        got = self.explain(dict_distance=1.0)[0]
        self.assertEqual((got.key, got.params["word"]), ("guard.reason.close_to", "ordenanza"))

    def test_G9_close_to_is_left_out_when_no_word_is_near(self):
        far = reasons.explain(contributions(dict_distance=1.0), word("xqzvbnwk"), np.zeros(len(COLUMNS)), ("es",))
        self.assertEqual(far, ())

    def test_G9_a_word_without_character_confidences_gets_no_weakest_character_reason(self):
        bare = Word("p1w0", 1, "obra", (0, 0, 32, 12), 0.5)
        got = reasons.explain(contributions(conf_mean=1.0, in_dictionary=1.0), bare, np.zeros(len(COLUMNS)), ("es",))
        self.assertNotIn("guard.reason.low_char_conf", [reason.key for reason in got])
        self.assertTrue(all(reason.params.get("char") != "" for reason in got))


if __name__ == "__main__":
    unittest.main()
