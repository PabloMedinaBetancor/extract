import json
import tempfile
import unittest
from pathlib import Path

from bench.corpus.labels import TruthWord, label_words
from bench.guard import pool_recall
from extract.types import Word
from tests.guard_words import EngineWord, Reading
from tests.test_guard_dataset import page_record
from tests.words import WORDS

RESULTS = json.loads(
    (Path(__file__).resolve().parents[1] / "bench" / "results" / "pool-recall-cached.json").read_text()
)


def line(texts, truths):
    """A page case whose words sit on one line, each over the truth word of the same position."""
    words, truth = [], []
    for i, (text, expected) in enumerate(zip(texts, truths, strict=True)):
        x = 10.0 + 100 * i
        words.append(Word(f"p1w{i}", 1, text, (x, 0.0, x + 80.0, 12.0), 0.9, (0.9,) * len(text), 0, i))
        truth.append(TruthWord(f"t{i}", expected, (x, 0.0, x + 80.0, 12.0)))
    alignment = label_words([(w.id, w.text, w.box) for w in words], truth)
    return pool_recall.PageCase("d1", 1, "medium", tuple(words), alignment.labels, truth)


class ReachedTest(unittest.TestCase):
    def test_G15_as_written_needs_the_exact_form_and_letters_ignore_case_and_punctuation(self):
        self.assertEqual(pool_recall.reached(["obra,", "Obra"], "obra,"), (True, True))
        self.assertEqual(pool_recall.reached(["obra"], "Obra,"), (False, True))
        self.assertEqual(pool_recall.reached(["obras"], "Obra,"), (False, False))

    def test_G15_an_accent_is_part_of_the_letters(self):
        self.assertEqual(pool_recall.reached(["informacion"], WORDS["informacion_acc"]), (False, False))


class V4PoolTest(unittest.TestCase):
    def test_G15_v4_proposed_the_word_in_the_tokens_case_and_never_a_new_punctuation(self):
        texts = pool_recall.v4_texts("Ordenansa,")
        self.assertIn("Ordenanza,", texts)
        self.assertNotIn("Ordenanza.", texts)

    def test_G15_v4_pool_has_the_other_casings_of_the_letters(self):
        self.assertIn("OBRA", pool_recall.v4_texts("obra"))

    def test_G15_a_token_with_a_digit_has_no_v4_candidates(self):
        self.assertEqual(pool_recall.v4_texts("ord3n"), [])


class ErrorTableTest(unittest.TestCase):
    def test_G15_digits_are_counted_apart_from_letter_errors(self):
        case = line(["obra", "ord3n", "plan,", "suelo"], ["obra", "orden", "plan.", "suelo"])
        kinds = pool_recall.error_table([case])["words"]
        self.assertEqual((kinds["digit"], kinds["punctuation"], kinds["right"]), (1, 1, 2))

    def test_G15_a_merge_that_only_misses_the_space_is_counted(self):
        counts = pool_recall._merge_repair("delacasa", "de la casa")
        self.assertEqual((counts["only_miss_the_space"], counts["within_2_edits_of_no_space"]), (1, 1))
        far = pool_recall._merge_repair("delxxxxcasa", "de la casa")
        self.assertEqual((far["only_miss_the_space"], far["within_2_edits_of_no_space"]), (0, 0))


class PoolTableTest(unittest.TestCase):
    def table(self, texts, truths):
        case = line(texts, truths)
        return pool_recall.pool_table(pool_recall.wrong_sites([case]), [case])

    def test_G15_a_lost_accent_is_reached_by_the_restoration_and_the_dictionary(self):
        table = self.table(["informacion"], [WORDS["informacion_acc"]])
        self.assertEqual(table["plain"], 1)
        self.assertEqual(table["fold"]["written"], 1.0)
        self.assertEqual(table["v4"]["written"], 1.0)

    def test_G15_a_changed_trailing_punctuation_is_reached_only_by_the_punctuation_source(self):
        table = self.table(["plan,"], ["plan."])
        self.assertEqual(table["punct"]["written"], 1.0)
        self.assertEqual(table["v4"]["written"], 0.0)

    def test_G15_the_union_never_falls_below_one_source(self):
        table = self.table(["ordenansa", "plan,"], ["ordenanza", "plan."])
        for name in ("dictionary", "fold", "punct", "page"):
            self.assertGreaterEqual(table["union"]["written"], table[name]["written"])


class BeyondPoolTest(unittest.TestCase):
    def test_G15_a_word_the_dictionary_cannot_reach_is_beyond_the_pool_until_an_alternative_has_it(self):
        case = line(["xqzvk"], ["valsequillo"])
        self.assertEqual(pool_recall.beyond_pool_ids(case), ["p1w0"])
        self.assertEqual(pool_recall.beyond_pool_ids(case, {"p1w0": ["valsequillo"]}), [])

    def test_G15_a_word_the_dictionary_reaches_is_not_beyond_the_pool(self):
        self.assertEqual(pool_recall.beyond_pool_ids(line(["ordenansa"], ["ordenanza"])), [])


class SealedTest(unittest.TestCase):
    # test_G17_the_sealed_sets_are_refused lives in tests/test_corpus_sealed.py: only that file may name the sealed sets (G4)
    def test_G15_pages_that_carry_a_reading_are_loaded_and_the_rest_ignored(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder) / "pages" / "train" / "d1"
            target.mkdir(parents=True)
            words = [Word("p1w0", 1, "obra", (0.0, 0.0, 40.0, 12.0), 0.9, (0.9,) * 4, 0, 0)]
            record = page_record("d1", 1, "light", words, [("obra", (0.0, 0.0, 40.0, 12.0))])
            (target / "1.json").write_text(json.dumps(record), encoding="utf-8")
            unread = {k: v for k, v in record.items() if k != "ocr"}
            (target / "2.json").write_text(json.dumps({**unread, "page": 2}), encoding="utf-8")
            cases, counts = pool_recall.load_cases(Path(folder), ["train"])
        self.assertEqual((len(cases), counts["pages"]), (1, 1))


class CachedResultsTest(unittest.TestCase):
    def test_G15_the_v4_pool_reproduces_the_exact_recall_of_the_design(self):
        for name, expected in (("train", 0.64), ("val", 0.66)):
            self.assertAlmostEqual(RESULTS[name]["pool"]["v4"]["written"], expected, delta=0.01, msg=name)

    def test_G15_the_dictionary_letters_ceiling_reproduces_the_design(self):
        for name, expected in (("train", 0.92), ("val", 0.89)):
            self.assertAlmostEqual(RESULTS[name]["pool"]["dictionary_letters_ceiling"], expected, delta=0.01, msg=name)

    def test_G15_the_error_table_counts_every_wrong_word_of_the_design(self):
        self.assertEqual(RESULTS["train"]["errors"]["wrong"], 3821)
        self.assertEqual(RESULTS["val"]["errors"]["wrong"], 814)


class EngineSourcesTest(unittest.TestCase):
    def test_G15_the_engine_sources_add_their_marginal_gain_when_the_words_carry_readings(self):

        case = line(["cbra", "plan"], ["obra", "plan"])
        words = list(case.words)
        words[0] = EngineWord(**{**vars(words[0]), "readings": (Reading("latin", "obra"),)})
        case = pool_recall.PageCase("d1", 1, "medium", tuple(words), case.labels, case.truth)
        sites = pool_recall.wrong_sites([case])
        table = pool_recall.pool_table(sites, [case])
        self.assertEqual(table["reading:latin"]["written"], 1.0)
        self.assertGreaterEqual(table["union_with_engine"]["written"], table["union"]["written"])
        self.assertIn("reading:latin", table["marginal_gain_written"])
