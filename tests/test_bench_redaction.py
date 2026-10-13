import json
import unittest
from pathlib import Path

from bench.crop_roundtrip import redact_names

RESULT = Path(__file__).resolve().parents[1] / "bench" / "results" / "crop-roundtrip.json"


class RedactionTest(unittest.TestCase):
    """The crop round trip may be run again on the corpus: it must not write a person's name into the results."""

    def test_a_name_after_a_courtesy_title_is_removed(self):
        self.assertEqual(redact_names("Dª. LUCÍA DEBORAH PADILLA RAMOS (Ponente)"), "Dª. [name removed] (Ponente)")
        self.assertEqual(redact_names("Da. Lucía Padilla Ramos (Ponente)"), "Da. [name removed] (Ponente)")
        self.assertEqual(redact_names("D. Pedro Ruiz firma"), "D. [name removed] firma")

    def test_other_text_is_left_alone(self):
        for text in ("Dª. MARÍA", "Agustín Nº 6", "ILUSTRE AYUNTAMIENTO DE GUÍA DE ISORA", "Artículo 4. Afección.", ""):
            self.assertEqual(redact_names(text), text)

    def test_it_is_idempotent(self):
        once = redact_names("Dª. LUCÍA DEBORAH PADILLA RAMOS (Ponente)")
        self.assertEqual(redact_names(once), once)

    def test_the_committed_result_is_already_redacted(self):
        for failure in json.loads(RESULT.read_text(encoding="utf-8"))["failures"]:
            self.assertEqual(redact_names(failure["first"]), failure["first"])
            self.assertEqual(redact_names(failure["crop"]), failure["crop"])


if __name__ == "__main__":
    unittest.main()
