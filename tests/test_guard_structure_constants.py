"""Pins on the constants and branches of the structure candidates that an earlier review found unpinned."""

import dataclasses
import unittest

from bench.guard import pairs
from extract.guard import structure
from extract.guard.candidates import MAX_CANDIDATES, PageContext
from tests.guard_pages import labelled_page as built
from tests.guard_words import Reading, page

LANGUAGES = ("es", "en")


def context(*lines, **kwargs):
    return PageContext(page(*lines, **kwargs), LANGUAGES)


def with_box(ctx_words, index, height):
    words = list(ctx_words)
    box = words[index].box
    words[index] = dataclasses.replace(words[index], box=(box[0], box[1], box[2], box[1] + height))
    return PageContext(words, LANGUAGES)


class DropConstantsTest(unittest.TestCase):
    def test_S12_a_speck_read_with_doubt_is_a_site_but_one_read_with_certainty_and_full_size_is_not(self):
        self.assertTrue(structure.drop_ok(context("la x casa", conf=0.985), 1))
        self.assertFalse(structure.drop_ok(context("la x casa", conf=0.995), 1))

    def test_S13_a_box_under_seven_tenths_of_the_line_is_small_and_one_over_it_is_not(self):
        words = page("la x casa de la", conf=0.995)
        self.assertTrue(structure.drop_ok(with_box(words, 1, 7.2), 1))
        self.assertFalse(structure.drop_ok(with_box(words, 1, 9.6), 1))

    def test_S14_a_lone_protected_character_is_never_a_drop_site(self):
        for mark in ("/", "_", "#", "@", "\\"):
            ctx = context(f"la {mark} casa", conf=0.5)
            self.assertFalse(structure.drop_ok(ctx, 1), mark)

    def test_S14_a_lone_hyphen_or_apostrophe_is_left_to_the_chooser_not_proposed(self):
        for mark in ("-", "'"):
            self.assertFalse(structure.drop_ok(context(f"la {mark} casa", conf=0.5), 1), mark)


class PieceConstantsTest(unittest.TestCase):
    def test_S8_a_lone_piece_is_a_word_only_when_it_is_one_of_ayoeu_or_a_capital_roman_numeral(self):
        ctx = context("la casa")
        for piece in ("a", "y", "o", "e", "u", "A", "I", "V", "X"):
            self.assertTrue(structure.piece_words(ctx, piece), piece)
        for piece in ("b", "c", "d", "i", "v", "x", "z", "B"):
            self.assertEqual(structure.piece_words(ctx, piece), (), piece)

    def test_S23_a_piece_gets_a_dictionary_edit_from_three_letters_not_from_two(self):
        ctx = context("la casa")
        self.assertEqual(structure.piece_words(ctx, "qk"), ())
        self.assertEqual([cost for _, cost, _ in structure.piece_words(ctx, "zqv")], [1])

    def test_S22_every_edit_has_a_price_so_the_cut_as_written_beats_a_cheaper_edit_of_frequent_words(self):
        for token, first in (("anchoplanes", "ancho planes"), ("empiezanpodrías", "empiezan podrías")):
            self.assertEqual(structure.segment_texts(context(token), 0)["segment"][0], first)


class JoinConstantsTest(unittest.TestCase):
    def test_S18_boxes_that_overlap_have_no_gap_not_a_negative_one(self):
        words = page("orde nanza", gap=4)
        overlapping = dataclasses.replace(words[1], box=(words[0].box[2] - 3.0, 0.0, words[0].box[2] + 30.0, 12.0))
        ctx = PageContext([words[0], overlapping], LANGUAGES)
        self.assertEqual(structure.gap_ratio(ctx, 0, 1), 0.0)

    def test_S19_a_site_holds_at_most_the_candidate_cap_and_keep_first(self):
        readings = tuple(
            Reading(f"r{i}", "ordenanza" + chr(97 + i) * 2, group=("p1w1", "p1w2")) for i in range(MAX_CANDIDATES + 6)
        )
        ctx = PageContext(page("la orde nanza del", gap=4, **{"1": {"readings": readings}}), LANGUAGES)
        found = structure.generate_structure(ctx, "join", (1, 2))
        self.assertEqual(len(found), MAX_CANDIDATES)
        self.assertEqual(found[0].op, "keep")

    def test_S21_two_texts_that_differ_only_before_normalising_are_one_candidate_with_both_sources(self):
        readings = (
            Reading("a", "oficina", group=("p1w1", "p1w2")),
            Reading("b", "o\ufb01cina", group=("p1w1", "p1w2")),
        )
        ctx = PageContext(page("la ofi cina del", gap=4, **{"1": {"readings": readings}}), LANGUAGES)
        found = structure.generate_structure(ctx, "join", (1, 2))
        self.assertEqual([c.text for c in found if c.op == "join"], ["oficina"])
        self.assertLessEqual({"reading:a", "reading:b"}, set(found[1].sources))


class LabelBranchesTest(unittest.TestCase):
    def test_L2_a_candidate_is_right_only_when_every_component_the_span_touches_is_right(self):
        p = built(["la", "suelo", "obrx"], [("la", [0], None), ("suelo", [1], None), ("obra", [2], None)])
        found = structure.generate_structure(p.context, "join", (1, 2))
        labels = pairs.structure_labels(p, (1, 2), found)
        self.assertEqual(labels[0], 0)

    def test_L2b_a_candidate_that_makes_one_of_two_touched_components_right_is_not_right(self):
        from extract.guard.candidates import KEEP, Candidate

        p = built(["la", "xqzw", "casa"], [("la", [0], None), ("obra", [1], None), ("casa", [2], None)])
        found = [
            Candidate("la xqzw", "keep", KEEP, (KEEP,), (0, 1)),
            Candidate("la", "join", "structure", ("t",), (0, 1)),
        ]
        self.assertEqual(pairs.structure_labels(p, (0, 1), found).tolist(), [0, 0])

    def test_L3_a_span_with_a_component_partly_outside_it_has_no_labels_even_if_another_is_inside(self):
        p = built(
            ["la", "orde", "nan", "za", "del"], [("la", [0], None), ("ordenanza", [1, 2, 3], None), ("del", [4], None)]
        )
        found = structure.generate_structure(p.context, "join", (0, 1))
        self.assertIsNone(pairs.structure_labels(p, (0, 1), found))

    def test_L5_a_site_with_no_candidate_but_keep_is_left_out_not_kept(self):
        p = built(["la", "zwhqvvgqhk", "casa"], [("la", [0], None), ("zwhqvvgqhk", [1], None), ("casa", [2], None)])
        sites, left_out = pairs.structure_sites(p)
        self.assertNotIn(("segment", (1,)), sites)
        self.assertGreaterEqual(left_out["segment_without_candidates"], 1)


if __name__ == "__main__":
    unittest.main()
