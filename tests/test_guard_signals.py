import math
import random
import time
import unittest
from unittest import mock

import numpy as np

from extract.guard import dictionary, signals
from extract.guard.signals import COLUMNS, INDEX, extract, schema_fingerprint
from extract.types import Word
from tests.checks_oracle import DNI, IBAN_ES
from tests.words import WORDS


def word(text, conf=0.99, chars=None, box=None, line=0, idx=0, number=0):
    box = box or (idx * 60.0, line * 30.0, idx * 60.0 + 8.0 * len(text), line * 30.0 + 20.0)
    chars = chars if chars is not None else tuple([conf] * len(text))
    return Word(f"p1w{number}", 1, text, box, conf, chars, line, idx)


def signal(text, name, languages=("es", "en"), locale="es", **options):
    return float(extract([word(text, **options)], languages, locale)[0, INDEX[name]])


class ConfidenceSignalsTest(unittest.TestCase):
    def test_G1_confidence_signals_come_from_the_characters(self):
        w = word("obra", conf=0.8, chars=(0.99, 0.5, 0.99, 0.72))
        row = extract([w])[0]
        self.assertAlmostEqual(row[INDEX["conf_mean"]], 0.8, places=4)
        self.assertAlmostEqual(row[INDEX["conf_min"]], 0.5, places=4)
        self.assertEqual(row[INDEX["low_conf_chars"]], 2)
        self.assertEqual(row[INDEX["length"]], 4)

    def test_G1_a_character_counts_as_low_below_0_90_only(self):
        w = word("abcd", conf=0.9, chars=(0.85, 0.95, 0.90, 0.89))
        self.assertEqual(extract([w])[0, INDEX["low_conf_chars"]], 2)

    def test_G1_a_word_without_confidences_has_nan_confidence_signals(self):
        w = Word("p1w0", 1, "obra", (0, 0, 40, 20))
        row = extract([w])[0]
        self.assertTrue(all(math.isnan(row[INDEX[n]]) for n in ("conf_mean", "conf_min", "low_conf_chars")))
        self.assertEqual(row[INDEX["length"]], 4)


class DictionarySignalsTest(unittest.TestCase):
    def test_G1_listed_words_are_found_with_punctuation_and_case(self):
        for text in ("casa", "Casa", "obra,", "(casa)", "CASA."):
            with self.subTest(text=text):
                self.assertEqual(signal(text, "in_dictionary"), 1.0)
                self.assertEqual(signal(text, "dict_distance"), 0.0)

    def test_G1_a_lost_accent_is_found_only_when_folded(self):
        self.assertEqual(signal("informacion", "in_dictionary", ("es",)), 0.0)
        self.assertEqual(signal("informacion", "in_dictionary_folded", ("es",)), 1.0)
        self.assertEqual(signal(WORDS["informacion_acc"], "in_dictionary", ("es",)), 1.0)

    def test_G1_noise_is_in_no_list(self):
        self.assertEqual(signal("xqzvbn", "in_dictionary"), 0.0)
        self.assertEqual(signal("xqzvbn", "in_dictionary_folded"), 0.0)
        self.assertEqual(signal("xqzvbn", "log_frequency"), 0.0)

    def test_G1_hyphenated_and_apostrophe_words_count_when_each_part_is_listed(self):
        self.assertEqual(signal("Contencioso-Administrativa", "in_dictionary", ("es",)), 1.0)
        self.assertEqual(signal("don't", "in_dictionary", ("en",)), 1.0)
        self.assertEqual(signal("casa-xqzvbn", "in_dictionary", ("es",)), 0.0)

    def test_G1_a_hyphenated_word_is_as_far_from_the_list_as_its_farthest_part(self):
        self.assertEqual(signal("casa-ordenansaa", "dict_distance", ("es",)), 2.0)
        self.assertEqual(signal("casa-ordenansa", "dict_distance", ("es",)), 1.0)

    def test_G1_frequency_orders_common_words_above_rare_ones(self):
        self.assertGreater(signal("casa", "log_frequency"), signal("catastral", "log_frequency"))
        self.assertGreater(signal("catastral", "log_frequency"), 0.0)

    def test_G1_distance_is_capped_at_three(self):
        self.assertEqual(signal("ordenansa", "dict_distance", ("es",)), 1.0)
        self.assertEqual(signal("xqzvbnwk", "dict_distance", ("es",)), 3.0)

    def test_G1_the_character_model_scores_words_above_noise(self):
        self.assertGreater(signal("construcciones", "char_lm_logp"), signal("xqzkvjw", "char_lm_logp") + 1.0)

    def test_G1_a_token_with_digits_has_no_dictionary_signals(self):
        for name in ("in_dictionary", "in_dictionary_folded", "log_frequency", "dict_distance", "char_lm_logp"):
            self.assertTrue(math.isnan(signal("A12b", name)), name)

    def test_G13_a_language_without_a_list_gives_nan_dictionary_signals_and_still_scores(self):
        row = extract([word("casa")], ("fr",))[0]
        for name in ("in_dictionary", "in_dictionary_folded", "log_frequency", "dict_distance", "char_lm_logp"):
            self.assertTrue(math.isnan(row[INDEX[name]]), name)
        self.assertEqual(row[INDEX["length"]], 4)

    def test_G1_the_page_languages_decide_which_lists_count(self):
        self.assertEqual(signal("invoice", "in_dictionary", ("es",)), 0.0)
        self.assertEqual(signal("invoice", "in_dictionary", ("es", "en")), 1.0)


class ShapeSignalsTest(unittest.TestCase):
    def test_G1_digits_and_mixed_tokens(self):
        self.assertEqual(signal("A12b", "has_digit"), 1.0)
        self.assertEqual(signal("A12b", "mixed_letters_digits"), 1.0)
        self.assertEqual(signal("1234", "mixed_letters_digits"), 0.0)
        self.assertEqual(signal("casa", "has_digit"), 0.0)
        self.assertEqual(signal("casa", "mixed_letters_digits"), 0.0)

    def test_G1_number_format_follows_the_locale(self):
        self.assertEqual(signal("1.234,56", "number_format_ok", locale="es"), 1.0)
        self.assertEqual(signal("1.234,56", "number_format_ok", locale="en"), 0.0)
        self.assertEqual(signal("1,234.56", "number_format_ok", locale="en"), 1.0)
        self.assertEqual(signal("1,234.56", "number_format_ok", locale="es"), 0.0)

    def test_G1_number_format_accepts_dates_percentages_and_ids(self):
        for text in (
            "12/03/2024",
            "2024-03-12",
            "21%",
            "21,5%",
            WORDS["ordinal_one"],
            "12:30",
            DNI,
            IBAN_ES.replace(" ", ""),
            "70.2",
        ):
            with self.subTest(text=text):
                self.assertEqual(signal(text, "number_format_ok"), 1.0)

    def test_G1_number_format_refuses_impossible_dates_and_broken_numbers(self):
        for text in ("31/31/2024", "1.23.4,5,6", "12,,5", "A-123"):
            with self.subTest(text=text):
                self.assertEqual(signal(text, "number_format_ok"), 0.0)

    def test_G1_number_format_is_nan_without_digits(self):
        self.assertTrue(math.isnan(signal("casa", "number_format_ok")))

    def test_G1_case_patterns(self):
        for text, expected in (
            ("hola", 0),
            ("Hola", 1),
            ("HOLA", 2),
            ("hOla", 3),
            ("Contencioso-Administrativa", 1),
            ("Ab-cd", 3),
        ):
            with self.subTest(text=text):
                self.assertEqual(signal(text, "case_pattern"), expected)
        self.assertTrue(math.isnan(signal("123", "case_pattern")))

    def test_G1_odd_characters_are_counted(self):
        self.assertEqual(signal("ab|c", "odd_chars"), 1.0)
        self.assertEqual(signal("a~b^", "odd_chars"), 2.0)
        self.assertEqual(signal("obra,", "odd_chars"), 0.0)
        self.assertEqual(signal(WORDS["question"], "odd_chars"), 0.0)


class LayoutSignalsTest(unittest.TestCase):
    def test_G1_hyphenated_halves_across_two_lines_are_both_fragments(self):
        words = [
            word("el", line=0, idx=0, number=0),
            word(WORDS["urbanis_hyphen"], line=0, idx=1, number=1),
            word("tico", line=1, idx=0, number=2),
            word("plan", line=1, idx=1, number=3),
        ]
        column = extract(words)[:, INDEX["hyphen_fragment"]]
        self.assertEqual(column.tolist(), [0.0, 1.0, 1.0, 0.0])

    def test_G1_a_lone_hyphen_or_a_hyphen_inside_a_line_is_no_fragment(self):
        words = [
            word("-", line=0, idx=0, number=0),
            word("a-b", line=0, idx=1, number=1),
            word("x", line=1, idx=0, number=2),
        ]
        self.assertEqual(extract(words)[:, INDEX["hyphen_fragment"]].tolist(), [0.0, 0.0, 0.0])

    def test_G1_height_and_width_ratios_are_relative_to_the_line(self):
        words = [
            word("aaaa", box=(0, 0, 40, 10), idx=0, number=0),
            word("bbbb", box=(50, 0, 90, 10), idx=1, number=1),
            word("cccc", box=(100, 0, 180, 20), idx=2, number=2),
        ]
        matrix = extract(words)
        self.assertEqual(matrix[:, INDEX["height_ratio"]].tolist(), [1.0, 1.0, 2.0])
        self.assertEqual(matrix[:, INDEX["width_per_char_ratio"]].tolist(), [1.0, 1.0, 2.0])

    def test_G1_line_and_page_confidence_are_means(self):
        words = [
            word("aa", conf=0.9, line=0, idx=0, number=0),
            word("bb", conf=0.7, line=0, idx=1, number=1),
            word("cc", conf=0.5, line=1, idx=0, number=2),
        ]
        matrix = extract(words)
        np.testing.assert_array_equal(matrix[:, INDEX["line_conf_mean"]], np.float32([0.8, 0.8, 0.5]))
        np.testing.assert_array_equal(matrix[:, INDEX["page_conf_mean"]], np.float32([0.7, 0.7, 0.7]))

    def test_G1_neighbours_report_their_dictionary_state_and_nan_at_line_ends(self):
        words = [word("casa", idx=0, number=0), word("xqzvbn", idx=1, number=1), word("obra", idx=2, number=2)]
        matrix = extract(words)
        prev, nxt = matrix[:, INDEX["prev_in_dictionary"]], matrix[:, INDEX["next_in_dictionary"]]
        self.assertTrue(math.isnan(prev[0]) and math.isnan(nxt[2]))
        self.assertEqual((prev[1], prev[2]), (1.0, 0.0))
        self.assertEqual((nxt[0], nxt[1]), (0.0, 1.0))

    def test_G1_words_on_a_line_are_read_by_their_position_not_their_input_order(self):
        words = [word("obra", idx=1, number=0), word("casa", idx=0, number=1)]
        self.assertEqual(extract(words)[0, INDEX["prev_in_dictionary"]], 1.0)


class SpeedTest(unittest.TestCase):
    def test_G10_signals_for_a_600_word_page_take_at_most_60_ms_of_cpu_with_cold_caches(self):
        rng = random.Random(3)
        book = dictionary.load("es")
        dictionary.load("en")
        pool = book.words[:6000]
        texts = []
        for _ in range(600):
            text = pool[min(int(rng.paretovariate(0.8)) * 7, 5999)]
            if rng.random() < 0.08:
                position = rng.randrange(len(text))
                text = text[:position] + rng.choice("abcdeilmnorstu") + text[position + 1 :]
            texts.append(text)
        words = [word(t, line=i // 10, idx=i % 10, number=i) for i, t in enumerate(texts)]
        self.assertLess(len(set(texts)) / 600, 0.7)
        extract(words[:30])
        for language in ("es", "en"):
            dictionary.load(language).clear_caches()
        started = time.process_time()
        matrix = extract(words)
        elapsed = time.process_time() - started
        self.assertEqual(matrix.shape, (600, 21))
        self.assertLess(elapsed, 0.060)


class SchemaTest(unittest.TestCase):
    def test_G1_the_schema_has_21_named_signals_with_the_signs_of_the_design(self):
        self.assertEqual(len(COLUMNS), 21)
        self.assertEqual(len(set(COLUMNS)), 21)
        signs = dict(signals.SCHEMA)
        self.assertEqual(signs["conf_mean"], -1)
        self.assertEqual(signs["dict_distance"], 1)
        self.assertEqual(signs["length"], 0)
        self.assertEqual(extract([word("casa")]).shape, (1, 21))
        self.assertEqual(extract([]).shape, (0, 21))

    def test_G1_the_fingerprint_changes_with_a_signal_its_order_its_sign_or_the_version(self):
        base = schema_fingerprint()
        self.assertRegex(base, "^[0-9a-f]{64}$")
        swapped = (signals.SCHEMA[1], signals.SCHEMA[0], *signals.SCHEMA[2:])
        renamed = (("conf_avg", -1), *signals.SCHEMA[1:])
        flipped = ((signals.SCHEMA[0][0], 1), *signals.SCHEMA[1:])
        for label, changed in (
            ("order", swapped),
            ("name", renamed),
            ("sign", flipped),
            ("shorter", signals.SCHEMA[:-1]),
        ):
            with self.subTest(change=label), mock.patch.object(signals, "SCHEMA", changed):
                self.assertNotEqual(schema_fingerprint(), base)
        with mock.patch.object(signals, "EXTRACTOR_VERSION", 2):
            self.assertNotEqual(schema_fingerprint(), base)

    def test_G4_signals_are_a_pure_function_of_the_words_and_the_cache_changes_nothing(self):
        words = [word("casa", idx=0, number=0), word("xqzvbn", idx=1, number=1), word("1.234,56", idx=2, number=2)]
        first = extract(words)
        cache = {}
        warm = extract(words, cache=cache)
        again = extract(words, cache=cache)
        self.assertEqual(first.tobytes(), warm.tobytes())
        self.assertEqual(first.tobytes(), again.tobytes())
        self.assertGreater(len(cache), 0)

    def test_G1_values_are_float32_rounded_to_four_decimals(self):
        row = extract([word("construcciones", conf=0.123456789)])[0]
        self.assertEqual(row.dtype, np.float32)
        self.assertEqual(row[INDEX["conf_mean"]], np.float32(0.1235))


if __name__ == "__main__":
    unittest.main()
