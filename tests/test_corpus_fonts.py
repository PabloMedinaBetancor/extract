import unittest

from bench.corpus.fonts import family


class FamilyTest(unittest.TestCase):
    def test_G3_subset_prefixes_and_styles_do_not_make_new_families(self):
        names = ["ABCDEF+Arial-BoldMT", "Arial,Italic", "Arial-Regular", "XYZABC+Arial", "Arial MT"]
        self.assertEqual({family(n) for n in names}, {"arial"})

    def test_G3_different_fonts_stay_different(self):
        self.assertNotEqual(family("TimesNewRomanPSMT"), family("Helvetica"))
        self.assertEqual(family("Times-Roman"), "times")


if __name__ == "__main__":
    unittest.main()
