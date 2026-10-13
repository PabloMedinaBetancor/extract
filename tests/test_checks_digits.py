import random
import unittest

from extract import checks
from tests import checks_oracle as oracle

SEED = 20261002
ALPHABET = oracle.ALPHABET
CHECKERS = {
    "dni": lambda text: checks.check_dni(text),
    "nie": lambda text: checks.check_nie(text),
    "cif": lambda text: checks.check_cif(text),
    "iban": lambda text: checks.check_iban(text),
}


def substitutions(text: str, skip=" "):
    """Every string that differs from `text` in exactly one character, over the whole alphabet, with its position."""
    for i, char in enumerate(text):
        if char in skip:
            continue
        for other in ALPHABET:
            if other != char:
                yield i, other, text[:i] + other + text[i + 1 :]


class GoldenTest(unittest.TestCase):
    def test_N2_the_golden_valid_cases_pass(self):
        for kind, examples in oracle.GOLDEN.items():
            for example in examples:
                with self.subTest(kind=kind, example=example):
                    self.assertTrue(CHECKERS[kind](example))

    def test_N2_the_golden_invalid_cases_fail(self):
        for kind, example in oracle.GOLDEN_INVALID.items():
            with self.subTest(kind=kind):
                self.assertFalse(CHECKERS[kind](example))

    def test_N2_the_oracle_builds_a_valid_iban_for_every_country_of_the_table(self):
        self.assertEqual(len(oracle.valid_registry()), len(oracle.STRUCTURE))

    def test_N2_each_checker_refuses_the_others_formats_and_junk(self):
        for kind, checker in CHECKERS.items():
            for junk in (
                "",
                " ",
                "1",
                "ABCDEFGHI",
                "123456789012345678901234567",
                "12345678",
                "12345678ZZ",
                "\u00e92345678Z",
            ):
                with self.subTest(kind=kind, junk=junk):
                    self.assertFalse(checker(junk))
        self.assertFalse(checks.check_dni(oracle.NIE))
        self.assertFalse(checks.check_nie(oracle.DNI))
        self.assertFalse(checks.check_cif(oracle.DNI))

    def test_N2_digits_that_are_not_ascii_are_never_valid(self):
        for kind, example in (
            ("dni", oracle.fullwidth(oracle.DNI[:8]) + oracle.DNI[8:]),
            ("dni", oracle.DNI[:8].translate({ord(d): ord(d) + 0x0660 for d in "0123456789"}) + oracle.DNI[8:]),
            ("nie", oracle.NIE[0] + oracle.fullwidth(oracle.NIE[1:8]) + oracle.NIE[8:]),
            ("cif", oracle.GOLDEN["cif"][0][0] + oracle.fullwidth(oracle.GOLDEN["cif"][0][1:])),
            ("iban", oracle.IBAN_ES[:5] + oracle.fullwidth(oracle.IBAN_ES[5:9]) + oracle.IBAN_ES[9:]),
            ("iban", oracle.IBAN_ES[:-1] + "\u00b2"),
        ):
            with self.subTest(kind=kind, example=example):
                self.assertFalse(CHECKERS[kind](example))

    def test_N2_lowercase_and_spaces_are_normalised_but_nothing_else(self):
        dni, iban = oracle.DNI, oracle.IBAN_ES
        self.assertTrue(checks.check_dni(dni.lower()))
        self.assertTrue(checks.check_iban(iban.lower()))
        self.assertTrue(checks.check_iban(iban.replace(" ", "")))
        self.assertFalse(checks.check_dni(f"{dni[:2]} {dni[2:5]} {dni[5:8]} {dni[8]}"))
        self.assertFalse(checks.check_dni(f"{dni[:8]}-{dni[8]}"))
        self.assertFalse(checks.check_iban(iban.replace(" ", "-")))


class DniNieTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(SEED)

    def test_N2_generated_dni_and_nie_are_valid(self):
        for _ in range(500):
            self.assertTrue(checks.check_dni(oracle.make_dni(self.rng)))
            self.assertTrue(checks.check_nie(oracle.make_nie(self.rng)))

    def test_N2_every_one_character_substitution_of_a_dni_is_detected(self):
        examples = oracle.GOLDEN["dni"] + [oracle.make_dni(self.rng) for _ in range(150)]
        checked = 0
        for example in examples:
            for i, other, mutated in substitutions(example):
                checked += 1
                self.assertFalse(checks.check_dni(mutated), (example, i, other))
        self.assertGreater(checked, 40_000)

    def test_N2_every_one_character_substitution_of_a_nie_is_detected(self):
        examples = oracle.GOLDEN["nie"] + [oracle.make_nie(self.rng) for _ in range(150)]
        checked = 0
        for example in examples:
            for i, other, mutated in substitutions(example):
                checked += 1
                self.assertFalse(checks.check_nie(mutated), (example, i, other))
        self.assertGreater(checked, 40_000)

    def test_N2_the_dni_letters_i_o_u_n_tilde_are_never_valid_controls(self):
        for n in range(0, 230, 7):
            for letter in "IOU\u00d1":
                self.assertFalse(checks.check_dni(f"{n:08d}{letter}"))

    def test_N2_adjacent_transpositions_of_different_digits_are_detected(self):
        for example in [oracle.make_dni(self.rng) for _ in range(200)]:
            for i in range(7):
                if example[i] != example[i + 1]:
                    swapped = example[:i] + example[i + 1] + example[i] + example[i + 2 :]
                    self.assertFalse(checks.check_dni(swapped), (example, i))
        for example in [oracle.make_nie(self.rng) for _ in range(200)]:
            for i in range(1, 7):
                if example[i] != example[i + 1]:
                    swapped = example[:i] + example[i + 1] + example[i] + example[i + 2 :]
                    self.assertFalse(checks.check_nie(swapped), (example, i))


class CifTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(SEED)

    def test_N2_generated_cifs_of_every_organisation_are_valid(self):
        for org in oracle.CIF_ORGANISATIONS:
            for _ in range(60):
                self.assertTrue(checks.check_cif(oracle.make_cif(self.rng, org)), org)

    def test_N2_organisations_that_force_a_letter_or_a_digit_refuse_the_other_form(self):
        for org in oracle.LETTER_ONLY:
            number = oracle.make_cif(self.rng, org)
            digits = number[1:8]
            self.assertFalse(checks.check_cif(f"{org}{digits}{oracle.cif_control(digits)[0]}"), org)
        for org in oracle.DIGIT_ONLY:
            number = oracle.make_cif(self.rng, org)
            digits = number[1:8]
            self.assertFalse(checks.check_cif(f"{org}{digits}{oracle.cif_control(digits)[1]}"), org)

    def test_N2_the_organisation_letter_must_be_a_valid_one(self):
        digits = "1234567"
        digit = oracle.cif_control(digits)[0]
        for letter in "IKLMOTXYZ":
            self.assertFalse(checks.check_cif(f"{letter}{digits}{digit}"), letter)

    def test_N2_every_substitution_is_detected_except_the_ones_the_standard_allows(self):
        """Exhaustive over every character of every example. The undetected ones must be exactly: another valid
        organisation letter that accepts the same control character, and, for organisations that allow both forms
        of the control, its digit-letter twin."""
        examples = oracle.GOLDEN["cif"] + [oracle.make_cif(self.rng) for _ in range(300)]
        checked = 0
        for example in examples:
            for i, other, mutated in substitutions(example):
                checked += 1
                expected_valid = mutated[1:8].isdigit() and oracle.cif_allows(mutated[0], mutated[8], mutated[1:8])
                self.assertEqual(checks.check_cif(mutated), expected_valid, (example, i, other))
                if i in range(1, 8):
                    self.assertFalse(checks.check_cif(mutated), "a digit of the seven is always detected")
        self.assertTrue(checks.check_cif(oracle.GOLDEN["cif"][0]))
        self.assertGreater(checked, 90_000)

    def test_N2_the_seven_digits_are_always_detected(self):
        for example in [oracle.make_cif(self.rng) for _ in range(200)]:
            for i in range(1, 8):
                for digit in "0123456789":
                    if digit != example[i]:
                        mutated = example[:i] + digit + example[i + 1 :]
                        self.assertFalse(checks.check_cif(mutated), (example, i))

    def test_N2_adjacent_transpositions_are_detected_exactly_where_the_weights_allow(self):
        def weight(v: int, position: int) -> int:
            return sum(divmod(2 * v, 10)) if position % 2 == 0 else v

        for example in [oracle.make_cif(self.rng) for _ in range(300)]:
            for i in range(1, 7):
                a, b = int(example[i]), int(example[i + 1])
                if a == b:
                    continue
                swapped = example[:i] + example[i + 1] + example[i] + example[i + 2 :]
                same_sum = weight(a, i - 1) + weight(b, i) == weight(b, i - 1) + weight(a, i)
                self.assertEqual(checks.check_cif(swapped), same_sum, (example, i))


class IbanTest(unittest.TestCase):
    def setUp(self):
        self.rng = random.Random(SEED)
        self.examples = oracle.GOLDEN["iban"] + oracle.valid_registry()
        for country in oracle.STRUCTURE:
            self.examples += [oracle.make_iban(self.rng, country) for _ in range(4)]

    def test_N2_registry_and_generated_ibans_are_valid(self):
        for example in self.examples:
            with self.subTest(example=example):
                self.assertTrue(checks.check_iban(example))

    def test_N2_a_wrong_length_for_the_country_is_refused_even_if_mod_97_holds(self):
        compact = oracle.IBAN_ES.replace(" ", "")
        self.assertTrue(checks.check_iban(compact))
        for cut in (compact[:-1], compact + "0"):
            self.assertFalse(checks.check_iban(cut))
        self.assertFalse(checks.check_iban("XX" + compact[2:]))

    def test_N2_a_country_outside_the_table_is_never_valid(self):
        for country in ("US", "BR", "JP", "ZZ", "EA"):
            check = 98 - oracle.iban_mod97(f"{country}00" + "1" * 20)
            self.assertFalse(checks.check_iban(f"{country}{check:02d}" + "1" * 20))

    def test_N2_every_substitution_is_detected_except_digit_letter_swaps_in_alphanumeric_parts(self):
        """Exhaustive over every character of every example. Where the BBAN is digits only (ES, DE, PT, ...) or a
        letter stays a letter and a digit a digit, mod 97 over a prime catches every one; a digit-letter swap inside
        a mixed part changes the length of the number mod 97 reads, and is caught with probability 96/97 only."""
        checked = 0
        leaks = []
        for example in self.examples:
            compact = example.replace(" ", "")
            classes = "ll" + "dd" + "".join({"n": "d", "a": "l", "c": "x"}[c] for c in oracle.classes_of(compact[:2]))
            for i, other, mutated in substitutions(compact):
                checked += 1
                if checks.check_iban(mutated):
                    leaks.append((compact, i, other))
                    self.assertEqual(classes[i], "x", ("outside an alphanumeric part", compact, i, other))
                    self.assertNotEqual(other.isdigit(), compact[i].isdigit(), ("a swap of class", compact, i, other))
        self.assertGreater(checked, 100_000)
        self.assertLess(len(leaks), checked / 500)

    def test_N2_the_country_code_and_check_digits_are_covered_too(self):
        for example in oracle.GOLDEN["iban"]:
            compact = example.replace(" ", "")
            for i in range(4):
                for other, mutated in ((o, compact[:i] + o + compact[i + 1 :]) for o in ALPHABET if o != compact[i]):
                    self.assertFalse(checks.check_iban(mutated), (compact, i, other))

    def test_N2_adjacent_transpositions_of_different_digits_are_detected_in_digit_only_ibans(self):
        for country in ("ES", "DE", "PT", "PL", "SE", "NO"):
            for _ in range(60):
                compact = oracle.make_iban(self.rng, country).replace(" ", "")
                for i in range(2, len(compact) - 1):
                    if compact[i] != compact[i + 1]:
                        swapped = compact[:i] + compact[i + 1] + compact[i] + compact[i + 2 :]
                        self.assertFalse(checks.check_iban(swapped), (compact, i))

    def test_N2_which_countries_have_characters_mod_97_cannot_fully_cover(self):
        for country in ("ES", "DE", "GB", "PT", "NL", "PL"):
            self.assertTrue(checks.iban_covers_every_character(oracle.make_iban(self.rng, country)), country)
        for country in ("FR", "IT", "CH", "GR", "LU", "MT"):
            self.assertFalse(checks.iban_covers_every_character(oracle.make_iban(self.rng, country)), country)

    def test_N2_the_iban_is_taken_from_the_words_of_a_row_with_all_their_ids(self):
        row = oracle.words("IBAN:", *oracle.IBAN_ES.split(), "Forma", "de", "pago")
        found = checks.find_ibans(row)
        self.assertEqual(len(found), 1)
        ids, text = found[0]
        self.assertEqual(ids, tuple(w.id for w in row[1:7]))
        self.assertEqual(text, oracle.IBAN_ES)
        self.assertEqual(checks.find_ibans(oracle.words(*oracle.IBAN_ES.split()[:3])), [])
        self.assertEqual(len(checks.find_ibans(oracle.words("IBAN", oracle.IBAN_ES.replace(" ", "")))), 1)


if __name__ == "__main__":
    unittest.main()
