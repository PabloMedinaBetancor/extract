"""The page-language sensor: a pure function of the tokens, disjoint lists, and the floors of the criterion."""

import math
import unittest
from types import SimpleNamespace

from bench.guard import language_sensor
from extract.guard import language
from tests.guard_words import page

ES = "El plan de ordenación de la ciudad establece que los usos del suelo serán los que se indican en la normativa para cada zona del municipio"
EN = "The committee approved the budget for the following year and the members agreed that the proposal should be reviewed by the council"


def words(text):
    return page(text, gap=4)


class LanguageTest(unittest.TestCase):
    def test_G40_the_two_function_word_lists_are_disjoint_and_have_none_of_the_shared_words(self):
        self.assertEqual(language.ES & language.EN, frozenset())
        self.assertGreater(len(language.ES), 50)
        self.assertGreater(len(language.EN), 50)
        for shared in ("a", "no", "me", "he", "son", "has"):
            self.assertNotIn(shared, language.ES)
            self.assertNotIn(shared, language.EN)

    def test_G40_a_spanish_page_an_english_page_and_a_page_of_non_words(self):
        self.assertEqual(language.page_language(words(ES), min_tokens=5), "es")
        self.assertEqual(language.page_language(words(EN), min_tokens=5), "en")
        self.assertEqual(language.page_language(words("xqz vbn kkr " * 12), second_stage=False), "unsure")

    def test_G40_a_page_with_one_quotation_in_the_other_language_keeps_its_own(self):
        quoted = ES + " " + "as the law says that the"
        self.assertEqual(language.page_language(words(quoted), min_tokens=5), "es")
        self.assertEqual(language.page_language(words(EN + " según la norma de la"), min_tokens=5), "en")

    def test_G40_an_invoice_that_is_mostly_numbers_and_a_page_with_too_few_words_are_unsure(self):
        invoice = "Factura 2026 0415 de 1.250,00 EUR IVA 21 % 262,50 Total 1.512,50 " * 3
        self.assertEqual(language.page_language(words(invoice), second_stage=False), "unsure")
        self.assertEqual(language.page_language(words("el plan de la ciudad")), "unsure")
        self.assertEqual(language.page_language([]), "unsure")

    def test_G40_a_bilingual_letterhead_over_spanish_text_is_spanish(self):
        self.assertEqual(language.page_language(words("Ayuntamiento Town Hall " + ES), min_tokens=5), "es")

    def test_G40_both_languages_at_once_is_unsure_not_a_guess(self):
        self.assertEqual(language.page_language(words(ES + " " + EN), min_tokens=5), "unsure")

    def test_G40_the_floor_and_the_ratio_decide_at_the_edge(self):
        text = ["la"] * 3 + ["xx"] * 17  # 15 % Spanish function words in 20 tokens
        self.assertEqual(language.page_language(words(" ".join(text)), floor=0.15, second_stage=False), "es")
        self.assertEqual(language.page_language(words(" ".join(text)), floor=0.16, second_stage=False), "unsure")
        both = ["la"] * 4 + ["the"] * 2 + ["xx"] * 14  # Spanish is exactly twice English
        self.assertEqual(language.page_language(words(" ".join(both)), floor=0.1, second_stage=False), "es")
        both = ["la"] * 4 + ["the"] * 3 + ["xx"] * 13
        self.assertEqual(language.page_language(words(" ".join(both)), floor=0.1, second_stage=False), "unsure")

    def test_G40_punctuation_digits_and_case_do_not_change_the_label(self):
        noisy = " ".join(f"{w.upper()}," if i % 3 == 0 else f"({w})" for i, w in enumerate(ES.split()))
        self.assertEqual(language.page_language(words(noisy), min_tokens=5), "es")
        self.assertEqual(language.tokens(words("12 3,4 -- Año")), ["año"])

    def test_G40_it_never_looks_at_the_truth_or_at_the_confidences(self):
        low = page(ES, gap=4, conf=0.1)
        self.assertEqual(language.page_language(low, min_tokens=5), language.page_language(words(ES), min_tokens=5))


def sensed(text, language_of, source="catalog"):
    return SimpleNamespace(words=words(text), language=language_of, source=source)


LABELS_EN = (
    "Invoice No 2026 0147 Date of issue Bill To Qty Description Unit price Total Subtotal VAT Amount due Payment terms"
)
LABELS_ES = "Factura Nº 2026 0147 Fecha Cliente Cant Descripción Precio unidad Total Subtotal IVA Importe a pagar Condiciones de pago"
FR = "Le comité a approuvé le budget de l'année prochaine et les membres ont convenu que la proposition devrait être examinée par le conseil avant que la décision finale ne soit prise"
CA = "El comitè va aprovar el pressupost de l'any vinent i els membres van acordar que la proposta s'havia de revisar pel consell abans que es prengui la decisió final el mes vinent"


class CharacterStageTest(unittest.TestCase):
    """An invoice of labels has no function words; the character models of the two lists say what it is, and veto
    the pages of another language that the function words took for Spanish."""

    def test_V5B35_an_invoice_of_labels_gets_its_language_from_the_character_models(self):
        self.assertEqual(language.function_language(words(LABELS_EN)), "unsure")
        self.assertEqual(language.page_language(words(LABELS_EN)), "en")
        self.assertEqual(language.page_language(words(LABELS_ES)), "es")
        self.assertEqual(language.page_language(words(LABELS_EN), second_stage=False), "unsure")

    def test_V5B35_the_difference_is_positive_for_english_negative_for_spanish_and_none_without_words(self):
        self.assertGreater(language.char_difference(words(EN)), language.LM_EN)
        self.assertLess(language.char_difference(words(ES)), language.LM_ES)
        self.assertIsNone(language.char_difference(words("12 3,4 -- 5")))
        self.assertIsNone(language.char_difference([]))
        self.assertIsNone(language.char_difference(words("a b c")))  # one letter: no word counts

    def test_V5B35_french_and_catalan_that_the_function_words_take_for_spanish_are_unsure(self):
        self.assertEqual(language.function_language(words(FR)), "es")
        self.assertEqual(language.function_language(words(CA)), "es")
        self.assertEqual(language.page_language(words(FR)), "unsure")
        self.assertEqual(language.page_language(words(CA)), "unsure")

    def test_V5B35_a_long_page_the_function_words_cannot_place_stays_unsure_but_a_short_one_asks_the_models(self):
        prose = " ".join(["Invoice", "payment", "total"] * 40)  # 120 words of labels
        self.assertEqual(language.page_language(words(prose)), "unsure")
        self.assertEqual(language.page_language(words(prose), alone_below=1000), "en")
        self.assertEqual(language.page_language(words("el plan de la ciudad")), "unsure")  # 5 words: nothing is said

    def test_V5B35_the_size_that_makes_a_page_prose_and_the_edges_of_the_two_thresholds(self):
        labels = ["Invoice", "payment", "total"]
        self.assertEqual(language.page_language(words(" ".join(labels * 33 + ["total"])), alone_below=100), "unsure")
        self.assertEqual(language.page_language(words(" ".join(labels * 33)), alone_below=100), "en")  # 99 words
        self.assertEqual(language.character_language(language.LM_EN), "en")
        self.assertEqual(language.character_language(language.LM_ES), "es")
        self.assertEqual(language.character_language((language.LM_EN + language.LM_ES) / 2), "unsure")
        self.assertEqual(language.character_language(None), "unsure")

    def test_V5B35_words_with_a_digit_do_not_count_for_the_models_and_a_page_they_cannot_read_keeps_its_first_stage(
        self,
    ):
        base = "Invoice payment total"
        self.assertEqual(
            language.char_difference(words(base)), language.char_difference(words(base + " Invoice2026 Total1 xx9yy"))
        )
        single = "y o u " * 10  # Spanish function words of one letter: the models have no word to read
        self.assertEqual(language.function_language(words(single)), "es")
        self.assertIsNone(language.char_difference(words(single)))
        self.assertEqual(language.page_language(words(single)), "es")

    def test_V5B35_a_language_the_function_words_give_needs_the_models_to_agree(self):
        wrong = language.page_language(words(ES), lm_es=-99)  # the models would never say Spanish
        self.assertEqual(wrong, "unsure")
        self.assertEqual(language.page_language(words(ES), lm_es=0.0), "es")
        self.assertEqual(language.page_language(words(EN), lm_en=99), "unsure")

    def test_V5B35_the_floor_may_be_one_per_language_and_the_settings_have_defaults(self):
        text = ["la"] * 3 + ["xx"] * 17
        self.assertEqual(language.function_language(words(" ".join(text)), {"es": 0.15, "en": 0.9}), "es")
        self.assertEqual(language.function_language(words(" ".join(text)), {"es": 0.16}), "unsure")
        tenth = ["la"] * 2 + ["xx"] * 18  # 10 % Spanish: under the default floor of 0.12
        self.assertEqual(language.function_language(words(" ".join(tenth)), {"en": 0.0}), "unsure")
        self.assertEqual(language.function_language(words(" ".join(tenth)), {"es": 0.1}), "es")
        settings = language.sensor_settings(0.3)
        self.assertEqual(
            (settings["floor"], settings["lm_en"], settings["lm_es"]), (0.3, language.LM_EN, language.LM_ES)
        )
        self.assertEqual(language.sensor_settings({"lm_en": 0.5, "floor": {"es": 0.1}})["lm_en"], 0.5)
        self.assertEqual(language.sensed_language(words(ES), {"second_stage": False}), "es")


class SensorFloorsTest(unittest.TestCase):
    def test_G40_the_floor_is_a_percentile_of_the_right_languages_share_over_long_catalogue_pages_only(self):
        long_es = " ".join(["la"] * 20 + ["xx"] * 80)  # 20 % over 100 tokens
        short = " ".join(["la"] * 3)  # too short to count
        pages = [sensed(long_es, "es"), sensed(short, "es"), sensed(long_es, "es", "invoice")]
        self.assertAlmostEqual(language_sensor.fit_floor(pages), 0.2)
        self.assertEqual(language_sensor.fit_floor([]), language.DEFAULT_FLOOR)

    def test_G40_the_confusion_counts_by_source_and_the_catalogue_rates_are_the_ones_of_the_criterion(self):
        long_es = " ".join(["la"] * 30 + ["xx"] * 70)
        long_en = " ".join(["the"] * 30 + ["xx"] * 70)
        pages = [sensed(long_es, "es")] * 8 + [sensed(long_en, "en")] * 2 + [sensed(long_en, "es")]
        result = language_sensor.confusion(pages, {"floor": 0.2, "second_stage": False})
        self.assertEqual(result["matrix"]["catalog"]["es"], {"es": 8, "en": 1})
        rates = result["rates"]["catalog"]
        self.assertEqual(rates["pages"], 11)
        self.assertTrue(math.isclose(rates["wrong"], 1 / 11))
        self.assertTrue(math.isclose(rates["right"], 10 / 11))

    def test_G40_an_invoice_with_no_known_language_is_left_out_until_the_manifest_gives_one(self):
        invoice = sensed(" ".join(["la"] * 30 + ["xx"] * 70), None, "invoice")
        self.assertEqual(language_sensor.confusion([invoice], {"floor": 0.2, "second_stage": False})["matrix"], {})
        invoice.language = "es"
        self.assertEqual(
            language_sensor.confusion([invoice], {"floor": 0.2, "second_stage": False})["matrix"]["invoice"]["es"],
            {"es": 1},
        )


if __name__ == "__main__":
    unittest.main()
