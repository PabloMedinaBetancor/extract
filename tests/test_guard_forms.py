import unittest

from bench.corpus import labels
from extract.guard import forms


class NormaliseTest(unittest.TestCase):
    def test_G14_quotes_dashes_bullets_and_invisible_marks_are_folded(self):
        cases = {
            "“obra”": '"obra"',
            "d’aquí": "d'aquí",
            "a–b—c": "a-b-c",
            "ob­ra": "obra",
            "·": "•",
            "ﬁn": "fin",
            "5°": "5o",
        }
        for text, expected in cases.items():
            with self.subTest(text=text):
                self.assertEqual(forms.normalise(text), expected)

    def test_G14_the_labelling_uses_the_same_function(self):
        self.assertIs(labels.normalise, forms.normalise)


class WrittenFormTest(unittest.TestCase):
    def test_G16_case_patterns_and_their_application(self):
        self.assertEqual(
            [forms.case_pattern(w) for w in ("obra", "Obra", "OBRA", "oBra", "A")],
            ["lower", "title", "upper", "irregular", "upper"],
        )
        self.assertEqual(forms.apply_case("obra", "Xyz"), "Obra")
        self.assertEqual(forms.apply_case("obra", "XYZ"), "OBRA")
        self.assertEqual(forms.apply_case("obra", "xYz"), "obra")

    def test_G16_a_plain_word_splits_into_its_punctuation_and_letters(self):
        self.assertEqual(forms.split_word("¿qué?"), ("¿", "qué", "?"))
        self.assertIsNone(forms.split_word("a-b"))

    def test_G16_stripping_accents_keeps_the_case(self):
        self.assertEqual(forms.strip_accents("ÁRBOL año"), "ARBOL ano")
