import random
import unittest

import numpy as np

from extract.engine import ocr
from extract.engine.ocr import CharConf, build_latin_set, line_alternatives, pair_confidences

CHARACTERS = ["blank", *"abeinouáéñúüAÁ.,;:0O1lm²°·–çÇ“”•丨〇", " "]
CLASS = {c: i for i, c in enumerate(CHARACTERS)}
LATIN = build_latin_set(CHARACTERS)


def softmax_line(steps):
    """A (steps, classes) array whose row t holds the probabilities named in steps[t], a dict {char: p}; what is left
    goes to the blank."""
    probs = np.zeros((len(steps), len(CHARACTERS)))
    for t, spec in enumerate(steps):
        for char, p in spec.items():
            probs[t, CLASS[char]] = p
        probs[t, 0] = 1.0 - sum(spec.values())
    return probs


def with_best(probs):
    return probs, probs.argmax(axis=1)


def alternatives_of(steps):
    alts, masked = line_alternatives(*with_best(softmax_line(steps)), CHARACTERS, LATIN)
    return alts, masked


class LatinSetTest(unittest.TestCase):
    def test_R2_the_set_is_what_written_spanish_catalan_and_english_use_not_only_letters(self):
        self.assertTrue({"²", "°", "·", "–", "ç", "Ç", "“", "”", "•"} <= set(LATIN.chars))
        self.assertTrue({"丨", "〇"}.isdisjoint(LATIN.chars))

    def test_R3_a_square_metre_or_a_degree_sign_is_not_masked(self):
        _, masked = alternatives_of([{"m": 0.9}, {"²": 0.9}, {" ": 0.9}, {"°": 0.9}])
        self.assertIsNone(masked)

    def test_R3_the_set_is_kept_by_the_content_of_the_character_list(self):
        from types import SimpleNamespace

        from extract.engine.ocr import _latin_set

        a, b, c = (
            SimpleNamespace(character=list(CHARACTERS)),
            SimpleNamespace(character=list(CHARACTERS)),
            SimpleNamespace(character=["blank", "a", " "]),
        )
        self.assertIs(_latin_set(a), _latin_set(b))
        self.assertIsNot(_latin_set(a), _latin_set(c))
        self.assertEqual(_latin_set(c).chars, ("a",))


class AlternativesTest(unittest.TestCase):
    def test_R2_the_emitted_character_comes_first_then_the_others_by_probability(self):
        alts, _ = alternatives_of([{"a": 0.7, "e": 0.2, "o": 0.06, "i": 0.03}, {"b": 0.9}])
        self.assertEqual([c for c, _ in alts[0]], ["a", "e", "o"])
        self.assertEqual([c for c, _ in alts[1]], ["b"])
        self.assertAlmostEqual(alts[0][0][1], 0.7)

    def test_R2_at_most_three_plus_the_accent_counterpart_and_all_in_the_latin_set(self):
        alts, _ = alternatives_of([{"e": 0.5, "a": 0.2, "o": 0.15, "i": 0.1, "u": 0.04, "é": 0.01}])
        self.assertEqual([c for c, _ in alts[0]], ["e", "a", "o", "é"])
        self.assertLessEqual(len(alts[0]), 4)
        for char, _ in alts[0]:
            self.assertIn(char, LATIN.chars)

    def test_R2_the_counterpart_is_added_from_1e_minus_4_even_outside_the_top_three(self):
        alts, _ = alternatives_of([{"a": 0.6, "e": 0.2, "o": 0.1, "i": 0.0999, "á": 0.0001}])
        self.assertIn("á", [c for c, _ in alts[0]])
        alts, _ = alternatives_of([{"a": 0.6, "e": 0.2, "o": 0.1, "i": 0.0999, "á": 0.00009}])
        self.assertNotIn("á", [c for c, _ in alts[0]])

    def test_R2_a_marked_vowel_and_ntilde_have_their_plain_counterpart_and_u_has_two(self):
        alts, _ = alternatives_of([{"ñ": 0.9, "n": 0.05}, {"u": 0.9, "ú": 0.05, "ü": 0.02}, {"Á": 0.9, "A": 0.01}])
        self.assertEqual([c for c, _ in alts[0]], ["ñ", "n"])
        self.assertEqual([c for c, _ in alts[1]], ["u", "ú", "ü"])
        self.assertEqual([c for c, _ in alts[2]], ["Á", "A"])

    def test_R2_alternatives_under_a_thousandth_are_dropped_so_a_sure_character_has_one_entry(self):
        alts, _ = alternatives_of([{"b": 0.995, "l": 0.0005}])
        self.assertEqual(alts[0], (("b", 0.995),))

    def test_R2_classes_outside_the_latin_set_are_never_named(self):
        alts, _ = alternatives_of([{"0": 0.5, "丨": 0.3, "O": 0.15, "〇": 0.05}])
        self.assertEqual([c for c, _ in alts[0]], ["0", "O"])

    def test_R2_the_peak_step_of_the_run_is_used_not_the_first_or_the_mean(self):
        alts, _ = alternatives_of([{"a": 0.45, "e": 0.40}, {"a": 0.97, "e": 0.02}, {"a": 0.5, "e": 0.3}, {"b": 0.9}])
        self.assertEqual(len(alts), 2)
        self.assertAlmostEqual(alts[0][0][1], 0.97)
        self.assertAlmostEqual(dict(alts[0])["e"], 0.02)

    def test_R2_one_entry_per_emitted_character_with_blanks_and_repeats_as_ctc_does(self):
        alts, _ = alternatives_of([{"l": 0.9}, {"l": 0.9}, {}, {"l": 0.8}, {" ": 0.9}, {"a": 0.9}])
        self.assertEqual([a[0][0] for a in alts], ["l", "l", " ", "a"])

    def test_R2_a_line_with_nothing_emitted_has_no_alternatives_and_no_masked_text(self):
        self.assertEqual(alternatives_of([{}, {}, {}]), ([], None))

    def test_R3_the_masked_text_replaces_a_character_outside_the_latin_set(self):
        _, masked = alternatives_of([{"a": 0.9}, {"丨": 0.5, "l": 0.4}, {"b": 0.9}])
        self.assertEqual(masked, "alb")

    def test_R3_there_is_no_masked_text_when_the_recogniser_stayed_in_the_latin_set(self):
        _, masked = alternatives_of([{"a": 0.9}, {"l": 0.9}, {" ": 0.9}, {"b": 0.9}])
        self.assertIsNone(masked)

    def test_R3_a_character_outside_the_set_with_no_latin_rival_just_disappears_from_the_masked_text(self):
        _, masked = alternatives_of([{"a": 0.9}, {"丨": 0.6}, {"b": 0.9}])
        self.assertEqual(masked, "ab")

    def test_R3_property_a_line_with_no_character_outside_the_set_gives_the_same_text(self):
        rng = random.Random(7)
        latin_classes = [CLASS[c] for c in LATIN.chars] + [CLASS[" "]]
        for _ in range(60):
            steps = rng.randrange(1, 40)
            probs = np.array([[rng.random() * 0.2 for _ in CHARACTERS] for _ in range(steps)])
            for t in range(steps):
                probs[t, rng.choice([0, *latin_classes])] += 1.0
            probs /= probs.sum(axis=1, keepdims=True)
            _, masked = line_alternatives(*with_best(probs), CHARACTERS, LATIN)
            self.assertIsNone(masked)


class PairedAlternativesTest(unittest.TestCase):
    def pair(self, text, words, alts, masked_line=None):
        confs = [0.9 - 0.01 * i for i in range(len(text))]
        entries = [((c, 0.5), ("x", 0.1)) for c in text]
        return pair_confidences(text, words, confs, entries if alts else None, masked_line), confs

    def test_R2_each_word_gets_the_alternatives_of_its_own_characters_and_the_first_keeps_char_conf(self):
        paired, confs = self.pair("ab cd", ["ab", "cd"], True)
        self.assertEqual(paired[1].alts, ((("c", confs[3]), ("x", 0.1)), (("d", confs[4]), ("x", 0.1))))
        self.assertEqual([entries[0][1] for entries in paired[0].alts], list(paired[0].chars))

    def test_R2_a_line_that_cannot_be_aligned_gives_no_alternatives(self):
        paired = pair_confidences("ab cd", ["ab", "cd"], [0.2, 0.4], [(("a", 1.0),)] * 5)
        self.assertEqual([p.alts for p in paired], [None, None])

    def test_R2_without_alternatives_the_words_carry_none(self):
        paired, _ = self.pair("ab cd", ["ab", "cd"], False)
        self.assertEqual([p.alts for p in paired], [None, None])

    def test_R3_masked_is_kept_only_for_the_words_that_differ(self):
        paired, _ = self.pair("ab cd", ["ab", "cd"], True, "ab cl")
        self.assertEqual([p.masked for p in paired], [None, "cl"])

    def test_R3_a_line_whose_masked_text_has_another_word_count_gives_no_masked_words(self):
        paired, _ = self.pair("ab cd", ["ab", "cd"], True, "abcd")
        self.assertEqual([p.masked for p in paired], [None, None])
        self.assertIsInstance(paired[0], CharConf)


class FingerprintTest(unittest.TestCase):
    def test_R8_the_version_of_the_alternatives_is_in_the_fingerprint(self):
        self.assertTrue(ocr.fingerprint().endswith("|alts2"))


if __name__ == "__main__":
    unittest.main()
