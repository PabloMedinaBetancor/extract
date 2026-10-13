import ast
import unittest
from datetime import date
from decimal import Decimal
from pathlib import Path

from extract import checks
from tests.checks_oracle import words

D = Decimal


class ParseAmountTest(unittest.TestCase):
    def test_N1_spanish_amounts(self):
        cases = {
            "1.234,56": D("1234.56"),
            "12,5": D("12.5"),
            "0,05": D("0.05"),
            "1.234.567,89": D("1234567.89"),
            "1234,56": D("1234.56"),
            "7": D("7"),
            "1.234": D("1234"),
        }
        for text, value in cases.items():
            with self.subTest(text=text):
                self.assertEqual(checks.parse_amount(text, "es"), value)

    def test_N1_english_amounts(self):
        cases = {
            "1,234.56": D("1234.56"),
            "12.5": D("12.5"),
            "0.05": D("0.05"),
            "1,234,567.89": D("1234567.89"),
            "1234.56": D("1234.56"),
            "1,234": D("1234"),
        }
        for text, value in cases.items():
            with self.subTest(text=text):
                self.assertEqual(checks.parse_amount(text, "en"), value)

    def test_N1_an_ambiguous_single_separator_follows_the_locale(self):
        self.assertEqual(checks.parse_amount("1.234", "es"), D("1234"))
        self.assertEqual(checks.parse_amount("1.234", "en"), D("1.234"))
        self.assertEqual(checks.parse_amount("1,234", "es"), D("1.234"))
        self.assertEqual(checks.parse_amount("1,234", "en"), D("1234"))

    def test_N1_a_format_that_is_not_the_locales_is_refused_not_guessed(self):
        for text in (
            "12.5",
            "1.2345,6",
            "1.23,4",
            "1,234.56",
            "1.234.56",
            "1,,5",
            "",
            ",5",
            "5,",
            "1 234,56",
            "abc",
            "1e5",
        ):
            with self.subTest(text=text, locale="es"):
                self.assertIsNone(checks.parse_amount(text, "es"))
        for text in ("12,5", "1,2345.6", "1,23.4", "1.234,56", "1,234,56", "1..5", ".5", "5.", "1 234.56"):
            with self.subTest(text=text, locale="en"):
                self.assertIsNone(checks.parse_amount(text, "en"))

    def test_N1_a_space_between_thousands_is_refused_in_every_width(self):
        for space in (" ", "\u00a0", "\u202f", "\u2009"):
            with self.subTest(space=repr(space)):
                self.assertIsNone(checks.parse_amount(f"1{space}234,56", "es"))
                self.assertIsNone(checks.parse_amount(f"1{space}234.56", "en"))
                self.assertIsNone(checks.parse_integer(f"1{space}234", "es"))

    def test_N1_digits_that_are_not_ascii_are_refused_by_every_reader(self):
        fullwidth, arabic, superscript = "１２,５", "١٢,٥", "²"
        for text in (fullwidth, arabic, superscript, "1²,50", "१२,५"):
            with self.subTest(text=text):
                self.assertIsNone(checks.parse_amount(text, "es"))
                self.assertIsNone(checks.parse_integer(text, "es"))
        self.assertIsNone(checks.parse_percent("２１%", "es"))
        self.assertIsNone(checks.parse_date("１８/０３/２０２６", "es"))
        self.assertIsNone(checks.parse_date("18/03/٢٠٢٦", "es"))
        self.assertEqual(checks.detect_locale(words("１２,５"), first_page_language="en"), "en")

    def test_N1_negatives_in_three_notations(self):
        for text in ("-5", "(5)", "5-"):
            with self.subTest(text=text):
                self.assertEqual(checks.parse_amount(text, "es"), D("-5"))
        self.assertEqual(checks.parse_amount("-1.234,50", "es"), D("-1234.50"))
        self.assertEqual(checks.parse_amount("(1,234.50)", "en"), D("-1234.50"))
        self.assertEqual(checks.parse_amount("1.234,50-", "es"), D("-1234.50"))
        for text in ("-(5)", "(5-)", "--5", "(5", "5)", "-5-"):
            with self.subTest(text=text):
                self.assertIsNone(checks.parse_amount(text, "es"))

    def test_N1_currency_symbols_and_codes_are_stripped(self):
        self.assertEqual(checks.parse_amount("12,50€", "es"), D("12.50"))
        self.assertEqual(checks.parse_amount("€1.234,56", "es"), D("1234.56"))
        self.assertEqual(checks.parse_amount("$1,234.56", "en"), D("1234.56"))
        self.assertEqual(checks.parse_amount("£5.00", "en"), D("5.00"))
        self.assertEqual(checks.parse_amount("1.234,56EUR", "es"), D("1234.56"))
        self.assertEqual(checks.parse_amount("USD12.5", "en"), D("12.5"))
        self.assertIsNone(checks.parse_amount("€", "es"))
        self.assertIsNone(checks.parse_amount("12,50€€", "es"))

    def test_N1_the_value_keeps_its_exact_decimals(self):
        value = checks.parse_amount("0,10", "es")
        self.assertEqual(str(value), "0.10")
        self.assertEqual(checks.parse_amount("12,500", "es"), D("12.500"))

    def test_N1_every_result_is_a_decimal(self):
        for text, locale in (("1,5", "es"), ("1.5", "en"), ("3", "es"), ("(2,00)", "es")):
            self.assertIsInstance(checks.parse_amount(text, locale), Decimal)
        self.assertIsInstance(checks.parse_percent("21%", "es"), Decimal)


class ParsePercentTest(unittest.TestCase):
    def test_N1_percentages(self):
        self.assertEqual(checks.parse_percent("21%", "es"), D("21"))
        self.assertEqual(checks.parse_percent("21,5%", "es"), D("21.5"))
        self.assertEqual(checks.parse_percent("21.5%", "en"), D("21.5"))
        self.assertEqual(checks.parse_percent("7,5 %", "es"), D("7.5"))
        self.assertEqual(checks.parse_percent("10 %", "en"), D("10"))
        self.assertEqual(checks.parse_percent("0%", "es"), D("0"))

    def test_N1_not_percentages(self):
        for text in ("21", "%", "21,5", "x%", "21%%", "1.234%", "-%", ""):
            with self.subTest(text=text):
                self.assertIsNone(checks.parse_percent(text, "es"))
        self.assertIsNone(checks.parse_percent("21,5%", "en"))


class ParseDateTest(unittest.TestCase):
    def test_N1_numeric_dates_follow_the_locale_order(self):
        self.assertEqual(checks.parse_date("18/03/2026", "es"), date(2026, 3, 18))
        self.assertEqual(checks.parse_date("03/18/2026", "en"), date(2026, 3, 18))
        self.assertEqual(checks.parse_date("04/05/2026", "es"), date(2026, 5, 4))
        self.assertEqual(checks.parse_date("04/05/2026", "en"), date(2026, 4, 5))
        self.assertEqual(checks.parse_date("18-03-2026", "es"), date(2026, 3, 18))
        self.assertEqual(checks.parse_date("18.03.2026", "es"), date(2026, 3, 18))
        self.assertEqual(checks.parse_date("5/3/2026", "es"), date(2026, 3, 5))

    def test_N1_iso_dates_in_either_locale(self):
        for locale in ("es", "en"):
            self.assertEqual(checks.parse_date("2026-03-18", locale), date(2026, 3, 18))

    def test_N1_month_names_in_both_languages(self):
        cases = [
            ("18 de marzo de 2026", "es"),
            ("18 marzo 2026", "es"),
            ("18 Marzo 2026", "es"),
            ("March 18, 2026", "en"),
            ("18 March 2026", "en"),
            ("Mar 18, 2026", "en"),
            ("18 mar 2026", "es"),
            ("18-mar-2026", "es"),
        ]
        for text, locale in cases:
            with self.subTest(text=text):
                self.assertEqual(checks.parse_date(text, locale), date(2026, 3, 18))
        self.assertEqual(checks.parse_date("1 de septiembre de 2025", "es"), date(2025, 9, 1))
        self.assertEqual(checks.parse_date("September 1, 2025", "en"), date(2025, 9, 1))

    def test_N1_the_calendar_decides_validity(self):
        self.assertEqual(checks.parse_date("29/02/2024", "es"), date(2024, 2, 29))
        for text in ("29/02/2025", "31/04/2026", "00/01/2026", "32/01/2026", "15/13/2026", "2026-13-01", "2026-02-30"):
            with self.subTest(text=text):
                self.assertIsNone(checks.parse_date(text, "es"))
        self.assertIsNone(checks.parse_date("13/01/2026", "en"))

    def test_N1_not_dates(self):
        for text in (
            "",
            "18/03",
            "18/03/26",
            "marzo",
            "18 de febrer de 2026",
            "March 32, 2026",
            "12345678",
            "1.234,56",
        ):
            with self.subTest(text=text):
                self.assertIsNone(checks.parse_date(text, "es"))

    def test_N1_a_misread_digit_that_is_still_a_date_stays_a_date(self):
        self.assertEqual(checks.parse_date("18/03/2026", "es"), date(2026, 3, 18))
        self.assertEqual(checks.parse_date("13/03/2026", "es"), date(2026, 3, 13))


class ParseIntegerTest(unittest.TestCase):
    def test_N1_integers(self):
        self.assertEqual(checks.parse_integer("42", "es"), 42)
        self.assertEqual(checks.parse_integer("-3", "es"), -3)
        self.assertEqual(checks.parse_integer("0", "en"), 0)
        self.assertEqual(checks.parse_integer("1.234", "es"), 1234)
        self.assertEqual(checks.parse_integer("1,234", "en"), 1234)
        self.assertIsInstance(checks.parse_integer("42", "es"), int)

    def test_N1_not_integers(self):
        for text, locale in (
            ("4,2", "es"),
            ("1.234", "en"),
            ("1,234", "es"),
            ("4.2", "en"),
            ("", "es"),
            ("a1", "es"),
            ("1 2", "es"),
        ):
            with self.subTest(text=text, locale=locale):
                self.assertIsNone(checks.parse_integer(text, locale))


class DetectLocaleTest(unittest.TestCase):
    def test_N1_unambiguous_spanish_and_english_tokens(self):
        self.assertEqual(checks.detect_locale(words("Total", "1.234,56")), "es")
        self.assertEqual(checks.detect_locale(words("Total", "1,234.56")), "en")
        self.assertEqual(checks.detect_locale(words("12,5")), "es")
        self.assertEqual(checks.detect_locale(words("12.5")), "en")
        self.assertEqual(checks.detect_locale(words("1.234.567")), "es")
        self.assertEqual(checks.detect_locale(words("1,234,567")), "en")

    def test_N1_majority_of_the_unambiguous_tokens_wins(self):
        mixed = words("12,5", "3,20", "1.234,56", "7.5", "9,99")
        self.assertEqual(checks.detect_locale(mixed), "es")
        mixed = words("12.5", "3.20", "1,234.56", "7,5", "9.99")
        self.assertEqual(checks.detect_locale(mixed), "en")

    def test_N1_ambiguous_tokens_do_not_vote(self):
        self.assertEqual(checks.detect_locale(words("1.234", "2.345", "3,456", "12,5")), "es")
        self.assertEqual(checks.detect_locale(words("1.234", "2,345", "3,456", "12.5")), "en")

    def test_N1_a_tie_goes_to_the_first_page_language(self):
        tie = words("12,5", "12.5")
        self.assertEqual(checks.detect_locale(tie, first_page_language="en"), "en")
        self.assertEqual(checks.detect_locale(tie, first_page_language="es"), "es")
        self.assertEqual(checks.detect_locale(words("1.234"), first_page_language="en"), "en")
        self.assertEqual(checks.detect_locale(words("Nothing", "here"), first_page_language="es"), "es")

    def test_N1_words_that_are_not_numbers_do_not_vote(self):
        voters = words("2026-03-18", "18/03/2026", "ES91", "12.5.3", "A-1,5")
        self.assertEqual(checks.detect_locale(voters, first_page_language="es"), "es")
        self.assertEqual(checks.detect_locale(voters, first_page_language="en"), "en")


class NoFloatTest(unittest.TestCase):
    def test_N1_the_module_has_no_float_in_it(self):
        source = Path(checks.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        found = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and node.id == "float":
                found.append(("float name", node.lineno))
            if isinstance(node, ast.Constant) and isinstance(node.value, float):
                found.append(("float literal", node.lineno))
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + ([node.module] if isinstance(node, ast.ImportFrom) else [])
                if any(n and n.split(".")[0] in ("math", "statistics", "numpy", "random") for n in names):
                    found.append(("import", node.lineno))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "from_float":
                found.append(("from_float", node.lineno))
        self.assertEqual(found, [])
        self.assertGreater(len(tree.body), 5)

    def test_N1_integer_division_and_true_division_are_not_mixed_into_money(self):
        source = Path(checks.__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        divisions = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Div)]
        self.assertEqual(divisions, [], "true division of ints gives a float; divide Decimals through Decimal methods")


if __name__ == "__main__":
    unittest.main()
