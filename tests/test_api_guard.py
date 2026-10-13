import inspect
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import extract
from bench.guard import train
from extract import pipeline
from extract.api import Reader
from extract.guard import Guard
from tests.test_api import LEXICON, ApiTest
from tests.test_guard_train import synthetic


class RealGuardTest(ApiTest):
    @classmethod
    def setUpClass(cls):
        model, _, _ = train.fit_model(synthetic(60, 1), synthetic(30, 2), {"leaves": 7, "trees": 100})
        cls.guard = Guard(model)

    def test_A5_the_guard_signature_is_what_the_pipeline_calls_by_keyword(self):
        wanted = inspect.signature(pipeline.Guard.score_page).parameters
        actual = inspect.signature(Guard.score_page).parameters
        for name in ("page", "languages", "locale", "budget"):
            self.assertIn(name, actual)
            self.assertEqual(actual[name].kind in (actual[name].POSITIONAL_OR_KEYWORD, actual[name].KEYWORD_ONLY), True)
            self.assertIn(name, wanted)
        self.assertIsInstance(self.guard.info, dict)
        self.assertIn("model_id", self.guard.info)

    def test_A5_a_read_with_the_real_guard_scores_every_ocr_word_and_records_the_model(self):
        document = self.read(self.scanned, guard=self.guard)
        words = [w for page in document.pages for w in page.words]
        self.assertTrue(words)
        self.assertEqual({w.status for w in words} - {"ok", "doubtful"}, set())
        self.assertTrue(all(w.p_wrong is not None for w in words))
        self.assertEqual(document._data.guard, self.guard.info["model_id"])

    def test_A5_a_page_over_the_word_limit_gives_one_warning_with_the_guards_own_cause(self):
        from extract.guard import MAX_WORDS
        from tests.test_guard_guard import page

        warnings = []
        reading = page(["obra"] * (MAX_WORDS + 1))
        scored = pipeline.score_page(reading, self.guard, ("es",), 0.05, warnings.append, "es")
        self.assertEqual({w.status for w in scored.words}, {"unscored"})
        self.assertEqual(len(warnings), 1)
        self.assertEqual(warnings[0].code, "guard.warning.unscored")
        self.assertEqual(warnings[0].params["cause"], "too_many_words")

    def test_A5_a_missing_model_is_a_model_mismatch_not_an_engine_error(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.dict(os.environ, {"EXTRACT_GUARD_MODEL": str(Path(folder) / "none.model")}):
                reader = Reader(self.options, lexicon=LEXICON)
                self.addCleanup(reader.close)
                with self.assertRaises(extract.ModelMismatch) as caught:
                    reader.read(self.scanned)
        self.assertEqual(caught.exception.code, "error.model_mismatch")

    def test_A5_a_model_of_another_schema_is_a_model_mismatch(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "guard.model"
            path.write_bytes(b"not a model")
            with mock.patch.dict(os.environ, {"EXTRACT_GUARD_MODEL": str(path)}):
                reader = Reader(self.options, lexicon=LEXICON)
                self.addCleanup(reader.close)
                with self.assertRaises(extract.ModelMismatch):
                    reader.read(self.scanned)

    def test_A5_missing_word_lists_stop_the_read_and_name_the_command_that_builds_them(self):
        from extract.guard import lexicon

        with mock.patch.object(lexicon, "DATA_DIR", Path(tempfile.mkdtemp())):
            reader = Reader(self.options, guard=self.guard)
            self.addCleanup(reader.close)
            with self.assertRaises(extract.EngineLoadError) as caught:
                reader.read(self.scanned)
        self.assertIn("fetch_models.py --lists", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
