import unittest
from dataclasses import replace
from unittest import mock

import numpy as np

from bench.corpus.degrade import apply_damage, draw_damage
from bench.corpus.labels import (
    TruthWord,
    label_words,
    normalise,
    truth_to_pixels,
    ungrounded_boxes,
    words_from_textpage,
)
from tests.words import WORDS


def word(i, text, x, y=0, w=40, h=12):
    return TruthWord(f"t{i}", text, (x, y, x + w, y + h))


def ocr(i, text, x, y=0, w=40, h=12):
    return (f"w{i}", text, (x, y, x + w, y + h))


def by_id(alignment):
    return {label.ocr_id: label for label in alignment.labels}


class LabelTest(unittest.TestCase):
    def test_G2_equal_words_are_right_and_substitutions_are_wrong(self):
        truth = [word(0, "obra,", 0), word(1, WORDS["publica_acc"], 50)]
        got = by_id(label_words([ocr(0, "obra,", 0), ocr(1, "publica", 50)], truth))
        self.assertFalse(got["w0"].wrong)
        self.assertTrue(got["w1"].wrong)
        self.assertEqual(
            (got["w1"].kind, got["w1"].detail, got["w1"].truth_text), ("substitution", "accent", WORDS["publica_acc"])
        )

    def test_G2_case_and_punctuation_count_as_errors(self):
        truth = [word(0, "Plan", 0), word(1, "general.", 50)]
        got = by_id(label_words([ocr(0, "plan", 0), ocr(1, "general", 50)], truth))
        self.assertEqual((got["w0"].detail, got["w1"].detail), ("case", "punctuation"))

    def test_G2_an_ocr_word_over_no_truth_is_invented(self):
        result = label_words([ocr(0, "xx", 300)], [word(0, "real", 0)])
        self.assertEqual(result.labels[0].kind, "invented")
        self.assertTrue(result.labels[0].wrong)
        self.assertEqual([w.text for w in result.missed], ["real"])

    def test_G2_one_truth_word_read_as_two_is_a_split(self):
        result = label_words(
            [ocr(0, "urba", 0, w=50), ocr(1, WORDS["nistico"], 50, w=50)], [word(0, WORDS["urbanistico"], 0, w=100)]
        )
        self.assertEqual([label.kind for label in result.labels], ["split", "split"])
        self.assertTrue(all(label.wrong for label in result.labels))
        self.assertEqual(result.boundary_only, 2)

    def test_G2_two_truth_words_read_as_one_are_a_merge(self):
        truth = [word(0, "de", 0, w=20), word(1, "la", 25, w=20)]
        result = label_words([ocr(0, "dela", 0, w=45)], truth)
        self.assertEqual(result.labels[0].kind, "merge")
        self.assertTrue(result.labels[0].wrong)
        self.assertEqual(result.labels[0].truth_text, "de la")
        self.assertEqual(result.boundary_only, 1)

    def test_G2_a_merge_that_changes_letters_is_not_a_boundary_case(self):
        truth = [word(0, "de", 0, w=20), word(1, "la", 25, w=20)]
        self.assertEqual(label_words([ocr(0, "dola", 0, w=45)], truth).boundary_only, 0)

    def test_G2_missed_words_are_counted_apart_and_never_labelled(self):
        truth = [word(0, "uno", 0), word(1, "dos", 50), word(2, "tres", 100)]
        result = label_words([ocr(0, "uno", 0), ocr(1, "tres", 100)], truth)
        self.assertEqual([w.text for w in result.missed], ["dos"])
        self.assertEqual(len(result.labels), 2)
        self.assertFalse(any(label.wrong for label in result.labels))

    def test_G2_typographic_variants_an_ocr_cannot_tell_apart_are_not_errors(self):
        pairs = (
            ('extinci\u00f3n"', "extinci\u00f3n" + chr(0x201D)),
            ("1'40", "1" + chr(0x2019) + "40"),
            ("-", chr(0x2013)),
            ('"en', chr(0x201C) + "en"),
        )
        for ocr_text, truth_text in pairs:
            with self.subTest(ocr=ocr_text):
                got = label_words([ocr(0, ocr_text, 0)], [word(0, normalise(truth_text), 0)])
                self.assertFalse(got.labels[0].wrong)

    def test_G2_truth_written_under_an_older_rule_is_normalised_before_the_comparison(self):
        got = label_words([ocr(0, '"Parque', 0)], [word(0, chr(0x201C) + "Parque", 0)])
        self.assertFalse(got.labels[0].wrong)

    def test_G2_every_bullet_is_the_same_bullet(self):
        got = label_words([ocr(0, chr(0x25CF), 0)], [word(0, chr(0xB7), 0)])
        self.assertFalse(got.labels[0].wrong)

    def test_G2_a_real_difference_next_to_a_typographic_one_is_still_an_error(self):
        got = label_words([ocr(0, '"cas', 0)], [word(0, normalise(chr(0x201C) + "casa"), 0)])
        self.assertTrue(got.labels[0].wrong)

    def test_G2_the_two_halves_of_a_hyphenated_word_are_their_own_fragments(self):
        truth = [word(0, WORDS["urbanis_hyphen"], 400, y=0), word(1, "tico", 0, y=20)]
        got = by_id(label_words([ocr(0, WORDS["urbanis_hyphen"], 400, y=0), ocr(1, "tico", 0, y=20)], truth))
        self.assertFalse(any(label.wrong for label in got.values()))

    def test_G2_ocr_words_come_back_in_input_order(self):
        truth = [word(0, "a", 0), word(1, "b", 50)]
        result = label_words([ocr(1, "b", 50), ocr(0, "a", 0)], truth)
        self.assertEqual([label.ocr_id for label in result.labels], ["w1", "w0"])

    def test_G2_no_words_on_either_side_is_not_an_error(self):
        self.assertEqual(label_words([], []).labels, [])
        self.assertEqual(len(label_words([], [word(0, "a", 0)]).missed), 1)
        self.assertEqual(label_words([ocr(0, "a", 0)], []).labels[0].kind, "invented")


class DamagedPageTest(unittest.TestCase):
    def damaged_ink(self):
        """A page with one black block far from the centre, turned 2 degrees: the block's box as found by thresholding."""
        damage = replace(
            draw_damage("heavy", 4), angle=2.0, specks=0, noise_sigma=0.0, show_through=0.0, blur_sigma=0.0
        )
        page = np.full((2000, 800), 255, np.uint8)
        page[1900:1920, 700:730] = 0
        damaged, matrix = apply_damage(page, damage)
        ys, xs = np.nonzero(damaged < 128)
        return matrix, (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))

    def test_G2_the_truth_box_moved_with_the_damage_links_to_the_ink_it_describes(self):
        matrix, ink = self.damaged_ink()
        truth = truth_to_pixels([TruthWord("t0", "bloque", (700, 1900, 730, 1920))], 72.0, matrix)
        self.assertEqual(label_words([("w0", "bloque", ink)], truth).labels[0].kind, "ok")

    def test_G2_the_unmoved_truth_box_does_not_link_to_the_turned_ink(self):
        _, ink = self.damaged_ink()
        truth = [TruthWord("t0", "bloque", (700, 1900, 730, 1920))]
        result = label_words([("w0", "bloque", ink)], truth)
        self.assertEqual(result.labels[0].kind, "invented")
        self.assertEqual(len(result.missed), 1)


class OverlapTest(unittest.TestCase):
    def test_G2_boxes_link_from_half_of_the_smaller_one_and_not_below(self):
        truth = [TruthWord("t0", "uno", (0, 0, 100, 10))]
        for share, kind in ((0.4, "invented"), (0.6, "ok")):
            with self.subTest(share=share):
                ocr_box = (100 - 100 * share, 0, 200 - 100 * share, 10)
                self.assertEqual(label_words([("w0", "uno", ocr_box)], truth).labels[0].kind, kind)


class GroupTest(unittest.TestCase):
    def chain(self, ocr_texts):
        truth = [TruthWord("t0", "uno", (0, 0, 40, 10)), TruthWord("t1", "dos", (42, 0, 82, 10))]
        boxes = [(0, 0, 66, 10), (30, 0, 82, 10), (70, 0, 90, 10)]
        return label_words([(f"w{i}", t, boxes[i]) for i, t in enumerate(ocr_texts)], truth)

    def test_G2_a_chain_with_as_many_words_each_side_is_paired_left_to_right_by_text(self):
        got = self.chain(["uno", "dos"])
        self.assertEqual([label.kind for label in got.labels], ["ok", "ok"])
        wrong = self.chain(["uno", "dox"])
        self.assertEqual([label.kind for label in wrong.labels], ["ok", "substitution"])

    def test_G2_an_extra_ocr_word_that_only_half_covers_a_truth_word_is_invented_and_the_others_stay_right(self):
        got = self.chain(["uno", "dos", "xx"])
        self.assertEqual([label.kind for label in got.labels], ["ok", "ok", "invented"])

    def test_G2_two_words_read_as_one_box_over_both_are_a_merge_on_each_side(self):
        truth = [TruthWord("t0", "uno", (0, 0, 40, 10)), TruthWord("t1", "dos", (42, 0, 82, 10))]
        got = label_words([("w0", "unodos", (0, 0, 82, 10))], truth)
        self.assertEqual((got.labels[0].kind, got.labels[0].wrong), ("merge", True))

    def test_G2_in_a_tangle_over_two_lines_a_word_equal_to_one_truth_word_is_right_and_the_others_are_wrong(self):
        truth = [TruthWord("t0", "uno", (0, 0, 40, 10)), TruthWord("t1", "dos", (0, 12, 40, 22))]
        ocr = [("w0", "unx", (0, 0, 40, 22)), ("w1", "dos", (0, 8, 40, 22))]
        got = label_words(ocr, truth)
        self.assertEqual([(label.kind, label.wrong) for label in got.labels], [("mixed", True), ("ok", False)])

    def test_G2_a_ligature_in_the_ocr_text_is_normalised_too(self):
        truth = [TruthWord("t0", "final", (0, 0, 40, 10))]
        self.assertFalse(label_words([("w0", chr(0xFB01) + "nal", (0, 0, 40, 10))], truth).labels[0].wrong)


class UngroundedTest(unittest.TestCase):
    truth = [TruthWord("t0", "real", (0, 0, 40, 10))]

    def test_G2_a_word_on_ink_the_text_layer_does_not_describe_is_ungrounded_and_unscored(self):
        got = label_words([("w0", "Gobierno", (100, 0, 160, 10))], self.truth, ungrounded=[(95, 0, 170, 12)])
        self.assertEqual((got.labels[0].kind, got.labels[0].wrong), ("ungrounded", None))

    def test_G2_a_word_on_blank_paper_is_invented_and_wrong(self):
        got = label_words([("w0", "fantasma", (100, 0, 160, 10))], self.truth, ungrounded=[(300, 0, 340, 12)])
        self.assertEqual((got.labels[0].kind, got.labels[0].wrong), ("invented", True))

    def test_G2_a_word_only_partly_on_that_ink_is_invented(self):
        got = label_words([("w0", "mixto", (100, 0, 160, 10))], self.truth, ungrounded=[(100, 0, 120, 10)])
        self.assertEqual(got.labels[0].kind, "invented")

    def test_G2_ungrounded_words_are_counted_in_their_own_kind(self):
        got = label_words(
            [("w0", "real", (0, 0, 40, 10)), ("w1", "logo", (100, 0, 160, 10))],
            self.truth,
            ungrounded=[(95, 0, 170, 12)],
        )
        self.assertEqual(dict(got.kinds), {"ok": 1, "ungrounded": 1})


class NeighbouringLineTest(unittest.TestCase):
    def test_G2_a_word_of_the_line_above_is_not_linked_to_a_tall_ocr_box_that_reaches_into_it(self):
        truth = [TruthWord("t0", "haya", (100, 0, 140, 12)), TruthWord("t1", "economica", (100, 20, 180, 32))]
        read = [("w0", "economica", (98, 6, 182, 34))]
        got = label_words(read, truth)
        self.assertEqual((got.labels[0].wrong, got.labels[0].kind), (False, "ok"))
        self.assertEqual([word.text for word in got.missed], ["haya"])

    def test_G2_a_word_on_the_same_line_with_a_taller_ocr_box_is_still_linked(self):
        truth = [TruthWord("t0", "en", (100, 10, 118, 20))]
        got = label_words([("w0", "en", (96, 2, 122, 28))], truth)
        self.assertEqual(got.labels[0].kind, "ok")


class TurnedWordTest(unittest.TestCase):
    def test_G2_pieces_read_over_a_word_turned_a_quarter_turn_are_ungrounded_not_wrong(self):
        truth = [TruthWord("t0", "SECTOR-3", (100, 0, 130, 160))]
        read = [("w0", "S", (100, 0, 130, 30)), ("w1", "3", (100, 130, 130, 160))]
        got = label_words(read, truth)
        self.assertEqual([(label.kind, label.wrong) for label in got.labels], [("ungrounded", None)] * 2)

    def test_G2_a_tall_single_letter_is_still_a_word_of_the_layer(self):
        truth = [TruthWord("t0", "l", (100, 0, 104, 12))]
        got = label_words([("w0", "l", (99, 0, 105, 12))], truth)
        self.assertEqual(got.labels[0].kind, "ok")


class UngroundedBoxesTest(unittest.TestCase):
    def page(self):
        image = np.full((400, 400), 255, np.uint8)
        image[20:40, 20:120] = 0
        image[200:240, 250:330] = 0
        image[100, 0:400] = 0
        image[300:360, 20:80] = 0
        image[300:360:2, 20:80] = 255
        return image

    def test_G2_a_logo_is_a_patch_and_the_word_next_to_it_is_not(self):
        patches = ungrounded_boxes(self.page(), [(20.0, 20.0, 120.0, 40.0)], 1.0)
        self.assertEqual(len(patches), 2)
        logo = min(patches, key=lambda b: abs(b[0] - 250))
        self.assertTrue(
            250 - 5 <= logo[0] <= 250 and 200 - 5 <= logo[1] <= 200 and 330 <= logo[2] <= 335 and 240 <= logo[3] <= 245
        )

    def test_G2_ink_within_a_few_pixels_of_a_truth_word_belongs_to_it(self):
        patches = ungrounded_boxes(self.page(), [(20.0, 20.0, 120.0, 40.0), (250.0, 200.0, 330.0, 240.0)], 1.0)
        self.assertEqual([b for b in patches if b[0] < 100 and b[1] < 150], [])

    def test_G2_a_thin_rule_is_not_a_patch(self):
        image = np.full((400, 400), 255, np.uint8)
        image[100, 0:400] = 0
        image[0:400, 50] = 0
        self.assertEqual(ungrounded_boxes(image, [], 1.0), [])

    def test_G2_the_outline_of_a_table_is_not_a_patch(self):
        image = np.full((400, 400), 255, np.uint8)
        image[50:350, 50:52] = 0
        image[50:350, 348:350] = 0
        image[50:52, 50:350] = 0
        image[348:350, 50:350] = 0
        self.assertEqual(ungrounded_boxes(image, [], 1.0), [])

    def test_G2_a_sparse_area_of_letter_sized_ink_is_a_patch_where_a_sparse_outline_is_not(self):
        stamp = np.full((400, 400), 255, np.uint8)
        for row in range(3):
            for column in range(10):
                stamp[60 + row * 14 : 66 + row * 14, 40 + column * 10 : 45 + column * 10] = 0
        outline = np.full((400, 400), 255, np.uint8)
        outline[50:350, 50:52] = 0
        outline[50:350, 348:350] = 0
        outline[50:52, 50:350] = 0
        outline[348:350, 50:350] = 0
        with mock.patch("bench.corpus.labels.MIN_PATCH_FILL", 0.6):
            (box,) = ungrounded_boxes(stamp, [], 1.0)
            self.assertEqual(ungrounded_boxes(outline, [], 1.0), [])
        self.assertTrue(box[0] <= 40 and box[1] <= 60 and box[2] >= 135 and box[3] >= 94)

    def test_G2_boxes_come_back_in_points_when_the_render_is_not_72_dpi(self):
        image = np.full((400, 400), 255, np.uint8)
        image[200:240, 250:330] = 0
        (box,) = ungrounded_boxes(image, [], 2.0)
        self.assertAlmostEqual(box[0], 125 - 2.0, delta=3.0)
        self.assertAlmostEqual(box[2], 165 + 2.0, delta=3.0)


class FakePage:
    def __init__(self, cropbox=(0.0, 0.0, 200.0, 100.0), rotation=0):
        self.cropbox, self.rotation = cropbox, rotation

    def get_cropbox(self):
        return self.cropbox

    def get_rotation(self):
        return self.rotation


class FakeText:
    """A text page of fixed-width characters: each char of the lines is 6 wide and 8 high, lines 12 apart."""

    def __init__(self, lines, top=90.0):
        self.text, self.boxes = "", []
        for row, line in enumerate(lines):
            for col, char in enumerate(line + ("\r\n" if row < len(lines) - 1 else "")):
                y = top - 12 * row
                size = 0.0 if char in "\r\n " else 1.0
                self.boxes.append((6.0 * col, y - 8 * size, 6.0 * col + 6 * size, y))
                self.text += char

    def get_text_range(self):
        return self.text

    def count_chars(self):
        return len(self.text)

    def get_charbox(self, index):
        return self.boxes[index]


def words_of(lines, **page):
    return [(w.text, w.box) for w in words_from_textpage(FakePage(**page), FakeText(lines))[2]]


class TextLayerTest(unittest.TestCase):
    def test_G2_words_split_at_spaces_and_line_breaks_with_tight_boxes(self):
        got = words_of(["ab cd", "ef"])
        self.assertEqual([t for t, _ in got], ["ab", "cd", "ef"])
        self.assertEqual(got[0][1], (0.0, 10.0, 12.0, 18.0))
        self.assertEqual(got[2][1][1], 22.0)

    def test_G2_a_ligature_is_normalised(self):
        self.assertEqual([t for t, _ in words_of(["o\ufb01cial"])], ["oficial"])
        self.assertEqual(normalise("\ufb01"), "fi")
        self.assertEqual(normalise("ofi\u00adcial"), "oficial")

    def test_G2_a_hyphen_mark_inside_a_line_stays_in_the_word(self):
        self.assertEqual([t for t, _ in words_of(["Contencioso\ufffeAdministrativa"])], ["Contencioso-Administrativa"])

    def test_G2_a_hyphen_mark_at_the_end_of_a_line_splits_the_word_into_its_two_halves(self):
        self.assertEqual([t for t, _ in words_of(["con\ufffe", "tribuir"])], ["con-", "tribuir"])

    def test_G2_control_characters_are_dropped_from_words(self):
        self.assertEqual([t for t, _ in words_of(["a\x01b"])], ["ab"])

    def test_G2_text_and_boxes_that_disagree_are_refused(self):
        text = FakeText(["ab"])
        text.count_chars = lambda: 5
        with self.assertRaises(ValueError):
            words_from_textpage(FakePage(), text)

    def test_G2_rotation_and_crop_box_move_the_boxes_into_the_render(self):
        base = words_of(["ab"])[0][1]
        self.assertEqual(base, (0.0, 10.0, 12.0, 18.0))
        shifted = words_of(["ab"], cropbox=(10.0, 20.0, 210.0, 120.0))[0][1]
        self.assertEqual(shifted, (-10.0, 30.0, 2.0, 38.0))
        for rotation, size, box in (
            (180, (200.0, 100.0), (188.0, 82.0, 200.0, 90.0)),
            (270, (100.0, 200.0), (10.0, 188.0, 18.0, 200.0)),
        ):
            with self.subTest(rotation=rotation):
                turned = words_from_textpage(FakePage(rotation=rotation), FakeText(["ab"]))
                self.assertEqual((turned[0], turned[1]), size)
                self.assertEqual(turned[2][0].box, box)
        turned = words_from_textpage(FakePage(rotation=90), FakeText(["ab"]))
        self.assertEqual((turned[0], turned[1]), (100.0, 200.0))
        self.assertEqual(turned[2][0].box, (82.0, 0.0, 90.0, 12.0))


if __name__ == "__main__":
    unittest.main()
