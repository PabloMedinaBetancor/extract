import unittest
from decimal import Decimal

from bench import inject
from extract.types import NumberResult
from tests.checks_oracle import DNI


def result(text, status="checked", value=None):
    return NumberResult(("w1",), text, value, "dni", status)


class FalseCheckedTest(unittest.TestCase):
    def test_a_different_identifier_that_is_still_checked_is_false(self):
        self.assertTrue(inject.is_false_checked(result(DNI), result(DNI[:7] + "9" + DNI[8:])))

    def test_the_punctuation_that_ends_a_sentence_does_not_change_an_identifier(self):
        self.assertFalse(inject.is_false_checked(result(DNI), result(DNI + ",")))
        self.assertFalse(inject.is_false_checked(result(DNI), result(DNI.lower() + ".")))

    def test_a_comma_inside_an_identifier_does_change_it(self):
        self.assertTrue(inject.is_false_checked(result(DNI), result(DNI[:4] + "," + DNI[4:])))

    def test_a_number_that_is_no_longer_checked_is_never_false(self):
        self.assertFalse(inject.is_false_checked(result(DNI), result(DNI[:7] + "9" + DNI[8:], status="mismatch")))
        self.assertFalse(inject.is_false_checked(result(DNI), None))

    def test_an_amount_is_false_when_its_value_changes(self):
        before = result("10,00", value=Decimal("10.00"))
        self.assertTrue(inject.is_false_checked(before, result("10,01", value=Decimal("10.01"))))
        self.assertFalse(inject.is_false_checked(before, result("010,00", value=Decimal("10.00"))))


if __name__ == "__main__":
    unittest.main()
