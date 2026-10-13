import math
import unittest
from unittest import mock

import numpy as np

from extract.guard import features, signals
from extract.guard.candidates import PageContext, generate
from extract.guard.features import FEATURES, INDEX, feature_fingerprint, feature_rows
from tests.guard_words import Reading, page

LANGUAGES = ("es", "en")


def rows_of(words, index, reader=None):
    ctx = PageContext(words, LANGUAGES)
    listed = generate(ctx, index)
    matrix = signals.extract(words)
    return listed, feature_rows(ctx, index, listed, matrix[index], reader)


def column(rows, name):
    return rows[:, INDEX[name]]


class SchemaTest(unittest.TestCase):
    def test_G14_the_fingerprint_is_stable_and_changes_with_the_strong_reader(self):
        self.assertEqual(feature_fingerprint(), feature_fingerprint())
        with mock.patch.object(features, "STRONG_READER", "strong"):
            changed = feature_fingerprint()
        self.assertNotEqual(changed, feature_fingerprint())

    def test_G14_every_feature_has_one_column_and_the_signals_come_first(self):
        self.assertEqual(len(set(FEATURES)), len(FEATURES))
        self.assertEqual(FEATURES[: len(signals.COLUMNS)], signals.COLUMNS)


class RowsTest(unittest.TestCase):
    def test_G14_there_is_one_float32_row_per_candidate_and_the_token_comes_first(self):
        listed, rows = rows_of(page("la ordenansa del"), 1)
        self.assertEqual(rows.shape, (len(listed), len(FEATURES)))
        self.assertEqual(rows.dtype, np.float32)
        keep = rows[0]
        self.assertEqual(keep[INDEX["kind"]], features.KINDS.index("keep"))
        self.assertEqual((keep[INDEX["n_edits"]], keep[INDEX["distance"]]), (0.0, 0.0))

    def test_G14_the_signals_of_the_token_repeat_on_every_row(self):
        words = page("la ordenansa del")
        _, rows = rows_of(words, 1)
        expected = signals.extract(words)[1]
        for row in rows:
            np.testing.assert_array_equal(row[: len(signals.COLUMNS)], expected)

    def test_G14_the_same_page_gives_the_same_bytes(self):
        words = page("la ordenansa del", "y la ordenanza ordenanza")
        self.assertEqual(rows_of(words, 1)[1].tobytes(), rows_of(words, 1)[1].tobytes())

    def test_G14_values_are_rounded_to_four_decimals(self):
        _, rows = rows_of(page("la ordenansa del"), 1)
        finite = rows[np.isfinite(rows)]
        np.testing.assert_array_equal(finite, np.round(finite.astype(np.float64), 4).astype(np.float32))

    def test_G14_an_edit_that_changes_one_character_names_the_confusion_and_the_confidence(self):
        words = page("la caea del", conf=0.9)
        listed, rows = rows_of(words, 1)
        wanted = next(i for i, c in enumerate(listed) if c.text == "casa")
        self.assertEqual(rows[wanted, INDEX["n_edits"]], 1.0)
        self.assertAlmostEqual(rows[wanted, INDEX["changed_chars_min_conf"]], 0.9, places=4)
        self.assertFalse(math.isnan(rows[wanted, INDEX["confusion_id"]]))

    def test_G14_a_candidate_that_is_a_dictionary_word_and_the_token_that_is_not_say_so(self):
        listed, rows = rows_of(page("la ordenansa del"), 1)
        wanted = next(i for i, c in enumerate(listed) if c.text == "ordenanza")
        self.assertEqual(rows[wanted, INDEX["cand_in_dictionary"]], 1.0)
        self.assertEqual(rows[0, INDEX["cand_in_dictionary"]], 0.0)
        self.assertEqual(rows[wanted, INDEX["in_dictionary"]], 0.0)

    def test_G14_page_counts_tell_how_often_the_token_and_the_candidate_appear(self):
        listed, rows = rows_of(page("ordenanza ordenansa ordenanza ordenanza"), 1)
        wanted = next(i for i, c in enumerate(listed) if c.text == "ordenanza")
        self.assertEqual(rows[wanted, INDEX["page_count_cand"]], 3.0)
        self.assertEqual(rows[wanted, INDEX["page_count_token"]], 1.0)
        self.assertEqual(rows[wanted, INDEX["page_vocab_proposed"]], 1.0)

    def test_G14_the_context_features_know_the_end_of_the_line(self):
        _, rows = rows_of(page("el gran plan", "Otra linea"), 2)
        self.assertEqual(rows[0, INDEX["ends_line"]], 1.0)
        self.assertEqual(rows[0, INDEX["punct_context"]], 1.0)
        self.assertEqual(rows[0, INDEX["starts_line"]], 0.0)


class PunctuationFeaturesTest(unittest.TestCase):
    def wanted(self, token, text):
        listed, rows = rows_of(page(f"el {token} de"), 1)
        return rows[next(i for i, c in enumerate(listed) if c.text == text)]

    def test_G14_the_mark_that_an_edit_removes_is_a_feature_so_a_bracket_is_not_a_stray_quote(self):
        bracket, quote = self.wanted("(plan", "plan"), self.wanted("'plan", "plan")
        self.assertNotEqual(bracket[INDEX["edit_punct_id"]], quote[INDEX["edit_punct_id"]])
        self.assertTrue(np.isnan(self.wanted("plan", "Plan")[INDEX["edit_punct_id"]]))

    def test_G14_written_counts_are_case_sensitive_and_folded_counts_are_not(self):
        listed, rows = rows_of(page("Plan plan plan Plan"), 1)
        lower = rows[next(i for i, c in enumerate(listed) if c.text == "Plan")]
        self.assertEqual(lower[INDEX["page_count_token_written"]], 2.0)
        self.assertEqual(lower[INDEX["page_count_cand_written"]], 2.0)
        self.assertEqual(lower[INDEX["page_count_cand"]], 4.0)


class EngineFeaturesTest(unittest.TestCase):
    def alternatives(self):
        return (
            (("c", 0.99),),
            (("a", 0.99),),
            (("e", 0.45), ("s", 0.4)),
            (("a", 0.99),),
        )

    def test_G14_without_alternatives_the_engine_features_are_missing_not_zero(self):
        _, rows = rows_of(page("la caea"), 1)
        for name in features.ENGINE:
            self.assertTrue(np.isnan(column(rows, name)).all(), name)

    def test_G14_the_probability_the_engine_gave_to_the_candidates_character_is_a_feature(self):
        words = page("la caea", **{"1": {"char_alts": self.alternatives()}})
        listed, rows = rows_of(words, 1)
        wanted = next(i for i, c in enumerate(listed) if c.text == "casa")
        self.assertAlmostEqual(rows[wanted, INDEX["alt_prob_min"]], 0.4, places=4)
        self.assertEqual(rows[wanted, INDEX["alt_positions"]], 1.0)
        self.assertEqual(rows[wanted, INDEX["ctc_proposed"]], 1.0)
        self.assertEqual(rows[0, INDEX["ctc_proposed"]], 0.0)

    def test_G14_a_candidate_the_engine_never_considered_has_probability_zero(self):
        words = page("la caea", **{"1": {"char_alts": self.alternatives()}})
        listed, rows = rows_of(words, 1)
        other = next(i for i, c in enumerate(listed) if c.text in ("caca", "cada"))
        self.assertEqual(rows[other, INDEX["alt_prob_min"]], 0.0)

    def test_G14_masked_differs_is_one_when_the_latin_set_decode_changed_the_word(self):
        words = page(
            "la cbra",
            **{"1": {"char_alts": ((("c", 0.9),), (("b", 0.9),), (("r", 0.9),), (("a", 0.9),)), "masked": "cbra"}},
        )
        _, rows = rows_of(words, 1)
        self.assertEqual(rows[0, INDEX["masked_differs"]], 1.0)


class ReadingFeaturesTest(unittest.TestCase):
    def test_G14_a_word_that_was_not_read_again_has_missing_reading_features_not_zeros(self):
        _, rows = rows_of(page("la cbra"), 1)
        for name in features.READINGS:
            self.assertTrue(np.isnan(column(rows, name)).all(), name)

    def test_G14_a_reader_that_ran_and_proposed_something_else_counts_as_zero_agreement(self):
        words = page("la cbra", **{"1": {"readings": (Reading("latin", "cabra"), Reading("bin", "cbra"))}})
        listed, rows = rows_of(words, 1)
        wanted = next(i for i, c in enumerate(listed) if c.text == "obra")
        self.assertEqual(rows[wanted, INDEX["readers_run"]], 2.0)
        self.assertEqual(rows[wanted, INDEX["readers_equal_candidate"]], 0.0)
        self.assertEqual(rows[wanted, INDEX["readers_equal_token"]], 1.0)

    def test_G14_an_unaligned_reader_is_run_but_not_aligned(self):
        words = page("la cbra", **{"1": {"readings": (Reading("latin", None), Reading("bin", "obra"))}})
        _, rows = rows_of(words, 1)
        self.assertEqual((rows[0, INDEX["readers_run"]], rows[0, INDEX["readers_aligned"]]), (2.0, 1.0))

    def test_G14_agreement_of_an_independent_network_is_flagged_and_the_light_ones_are_not(self):
        light = page("la cbra", **{"1": {"readings": (Reading("bin", "obra"),)}})
        independent = page("la cbra", **{"1": {"readings": (Reading("latin", "obra"),)}})
        for words, expected in ((light, 0.0), (independent, 1.0)):
            listed, rows = rows_of(words, 1)
            wanted = next(i for i, c in enumerate(listed) if c.text == "obra")
            self.assertEqual(rows[wanted, INDEX["independent_agree"]], expected)

    def test_G14_the_strong_reader_is_the_one_named_by_the_configuration(self):
        reading = Reading("strong", "obra", conf=0.8, rank=2)
        words = page("la cbra", **{"1": {"readings": (reading, Reading("latin", "cbra"))}})
        listed, rows = rows_of(words, 1, reader="strong")
        wanted = next(i for i, c in enumerate(listed) if c.text == "obra")
        self.assertEqual(rows[wanted, INDEX["strong_run"]], 1.0)
        self.assertEqual(rows[wanted, INDEX["strong_equal_candidate"]], 1.0)
        self.assertEqual(rows[wanted, INDEX["strong_equal_token"]], 0.0)
        self.assertAlmostEqual(rows[wanted, INDEX["strong_conf"]], 0.8, places=4)
        self.assertEqual(rows[wanted, INDEX["zone_rank"]], 2.0)
        _, default = rows_of(words, 1)
        self.assertEqual(default[0, INDEX["strong_run"]], 0.0)
        self.assertTrue(np.isnan(default[0, INDEX["strong_equal_candidate"]]))

    def test_G14_a_generative_reader_without_confidences_gives_missing_values_not_errors(self):
        words = page("la cbra", **{"1": {"readings": (Reading("deep", "obra", conf=None, char_conf=None),)}})
        _, rows = rows_of(words, 1)
        self.assertTrue(np.isnan(rows[0, INDEX["strong_conf"]]))
        self.assertEqual(rows[0, INDEX["strong_run"]], 1.0)

    def test_G14_a_site_in_a_zone_only_as_context_is_flagged(self):
        words = page("la obra", **{"1": {"readings": (Reading("deep", "obra", context=True),)}})
        _, rows = rows_of(words, 1)
        self.assertEqual(rows[0, INDEX["in_zone_as_context"]], 1.0)
