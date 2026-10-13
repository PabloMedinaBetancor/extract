"""Operating points per language: the thresholds of a language come from its own rows, a language with no block
applies nothing, the page's sensed language picks the block and the word list at run time."""

import unittest
from unittest import mock

import numpy as np

from bench.guard import final_fix, operating, pairs
from extract.guard import candidates, language
from extract.guard.checker import Guard
from extract.guard.features import OPS
from tests import guard_pages
from tests.test_guard_fixing import FLOOR, MINIMA, build_model, propose_with, scripted
from tests.test_guard_operating_ops import table_of


def choice_of(languages, outcomes, p_fix):
    n = len(outcomes)
    return (
        operating.Choice(
            np.ones(n, dtype=np.int64), np.array(p_fix, dtype=float), np.zeros(n), np.array(outcomes, dtype=np.int8)
        ),
        np.array(["text"] * n, dtype=object),
        np.array(languages, dtype=object),
    )


class ThresholdsByLanguageTest(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(5)
        n = 400
        self.p = np.concatenate([np.linspace(0.5, 1.0, n), np.linspace(0.5, 1.0, n)])
        self.languages = ["es"] * n + ["en"] * n
        es = np.where(self.p[:n] > 0.55, operating.FIXED, operating.BROKEN)  # clean from 0.55 up
        en = np.where(rng.random(n) < 0.85, operating.FIXED, operating.BROKEN)  # 15 % broken at any score
        self.outcomes = np.concatenate([es, en])

    def thresholds(self, outcomes=None):
        choice, ops, languages = choice_of(self.languages, self.outcomes if outcomes is None else outcomes, self.p)
        return operating.thresholds_by_language(choice, ops, languages, floor=0.99, minima=MINIMA)

    def test_G41_a_language_that_misses_its_floor_is_off_there_and_only_there(self):
        points = self.thresholds()
        self.assertIsNotNone(points["es"]["text"]["t_safe"])
        self.assertIsNone(points["en"]["text"]["t_safe"])

    def test_G41_the_thresholds_of_a_language_do_not_depend_on_the_rows_of_the_other(self):
        before = self.thresholds()
        rng = np.random.default_rng(9)
        changed = self.outcomes.copy()
        n = len(changed) // 2
        changed[n:] = rng.permutation(changed[n:])  # the English outcomes shuffled
        self.assertEqual(self.thresholds(changed)["es"], before["es"])
        choice, ops, languages = choice_of(self.languages, self.outcomes, self.p)
        pooled = operating.thresholds_by_op(choice, ops, floor=0.99, minima=MINIMA)
        self.assertNotEqual(pooled["text"]["t_safe"], before["es"]["text"]["t_safe"])  # pooling would change Spanish

    def test_G41_a_language_with_no_block_applies_nothing_and_each_language_uses_its_own_thresholds(self):
        choice, ops, languages = choice_of(self.languages, self.outcomes, self.p)
        blocks = {"es": {"text": {"t_safe": 0.9, "m": None}}, "en": {"text": {"t_safe": 0.95, "m": None}}}
        mask = operating.applied_by_language(choice, ops, languages, "safe", blocks)
        self.assertEqual(int(mask[languages == "es"].sum()), int((self.p[:400] >= 0.9).sum()))
        self.assertEqual(int(mask[languages == "en"].sum()), int((self.p[400:] >= 0.95).sum()))
        only_es = operating.applied_by_language(choice, ops, languages, "safe", {"es": blocks["es"]})
        self.assertFalse(only_es[languages == "en"].any())
        self.assertTrue(only_es[languages == "es"].any())


class TableLanguageTest(unittest.TestCase):
    def test_G41_a_site_carries_the_language_of_its_page_and_es_when_the_manifest_says_none(self):
        pages = guard_pages.corpus(6, errors=0.3)
        for number, page in enumerate(pages):
            page.language = "en" if number % 2 else None
        cache = pairs.SiteCache(pages)
        sites = [(p, i) for p in range(6) for i in range(len(pages[p].words)) if pairs.is_site(pages[p], i)][:60]
        table = cache.table(sites)
        for site, language_of in zip(sites, table.language, strict=True):
            self.assertEqual(language_of, "en" if site[0] % 2 else "es")

    def test_G40_sensing_a_page_gives_its_context_the_list_of_its_language_even_if_it_was_built_before(self):
        page = guard_pages.corpus(1, errors=0.3)[0]
        self.assertEqual(len(page.context.books), 2)
        for sensed, books in (("es", 1), ("unsure", 2)):
            with mock.patch.object(language, "page_language", lambda *args, sensed=sensed, **kwargs: sensed):
                pairs.sense_pages([page], 0.1)
            self.assertEqual((page.sensed, len(page.context.books)), (sensed, books))

    def test_G41_the_report_is_split_by_language_with_its_own_documents_and_counts(self):
        rng = np.random.default_rng(2)
        sites = [("text", int(rng.random() < 0.97), 0, (i,), f"d{i % 10}") for i in range(300)]
        sites += [("text", int(rng.random() < 0.7), 0, (i + 1000,), f"e{i % 6}") for i in range(300)]
        table = table_of(sites)
        table.language = np.array(["es"] * 300 + ["en"] * 300, dtype=object)
        raw = np.where(table.y == 1, 0.9, 0.1) + rng.normal(0, 0.05, len(table.y))
        report = operating.report_by_language(table, raw, cross_fit=False, resamples=10, floor=0.9, minima=MINIMA)
        self.assertEqual(set(report["by_language"]), {"es", "en"})
        self.assertEqual(report["by_language"]["es"]["documents"], 10)
        self.assertEqual(report["by_language"]["en"]["documents"], 6)
        self.assertEqual(report["by_language"]["es"]["sites"], 300)
        for language_of, part in report["by_language"].items():
            self.assertEqual(part["safe"]["total"]["applied"], part["safe"]["text"]["applied"], language_of)


class RuntimeLanguageTest(unittest.TestCase):
    def setUp(self):
        text = "el plan de la orde nanza del municipio es el que rige la ciudad en este año y los usos del suelo serán los que se indican en la normativa"
        self.words, self.matrix, self.raw, self.model, self.sites = scripted(text)
        sensed = language.page_language(self.words, 0.05)
        self.assertEqual(sensed, "es")
        self.off = {"t_safe": {op: None for op in OPS}, "m": {op: None for op in OPS}}
        self.on = {"t_safe": {"text": 0.5, "join": 0.5, "segment": None, "drop": None}, "m": dict.fromkeys(OPS, 0.1)}
        self.model.fix["language_sensor"] = {"floor": 0.05, "min_tokens": 20, "ratio": 2.0}
        self.model.fix["by_language"] = {"es": self.on, "en": self.off}

    def fake(self, site):
        n = len(site.found)
        return [0.1] + [0.9] * (n - 1)

    def propose(self, languages=("es", "en")):
        return propose_with(self.sites, self.fake, self.words, self.matrix, self.raw, self.model, languages)

    def test_G40_the_block_of_the_sensed_language_decides_the_modes(self):
        fixes, cause = self.propose()
        self.assertIsNone(cause)
        self.assertTrue(any(f.modes for f in fixes.values()))
        self.model.fix["by_language"] = {"es": self.off, "en": self.on}
        fixes, cause = self.propose()
        self.assertTrue(all(f.modes == () for f in fixes.values()))
        self.assertIsNone(cause)

    def test_G41_a_language_the_model_has_no_block_for_applies_nothing(self):
        self.model.fix["by_language"] = {"en": self.on}
        fixes, _ = self.propose()
        self.assertTrue(fixes)
        self.assertTrue(all(f.modes == () for f in fixes.values()))

    def test_G40_an_unsure_page_or_an_undeclared_language_withholds_the_fixes_with_the_warning(self):
        self.model.fix["language_sensor"].update(floor=0.99, lm_es=-99, lm_en=99)  # nothing reaches it: unsure
        fixes, cause = self.propose()
        self.assertEqual(cause, "language")
        self.assertTrue(all(f.modes == () for f in fixes.values()))
        self.model.fix["language_sensor"].update(floor=0.05, lm_es=-0.1, lm_en=0.05)
        fixes, cause = self.propose(languages=("en",))  # a Spanish page declared English only
        self.assertEqual(cause, "language")
        self.assertTrue(all(f.modes == () for f in fixes.values()))

    def test_G40_the_candidates_come_from_the_list_of_the_sensed_language(self):
        seen = []
        real = candidates.PageContext

        def spy(words, languages, fragments=()):
            seen.append(tuple(languages))
            return real(words, languages, fragments)

        with mock.patch.object(candidates, "PageContext", spy):
            self.propose()
        self.assertEqual(seen, [("es",)])
        self.model.fix.pop("language_sensor")
        self.model.fix.pop("by_language")
        seen.clear()
        with mock.patch.object(candidates, "PageContext", spy):
            propose_with(self.sites, self.fake, self.words, self.matrix, self.raw, self.model)
        self.assertEqual(seen, [("es", "en")])

    def test_G40_the_dictionaries_of_a_page_are_the_sensed_one_when_declared_else_all(self):
        self.assertEqual(language.dictionaries("es", ("es", "en")), ("es",))
        self.assertEqual(language.dictionaries("en", ("es", "en")), ("en",))
        self.assertEqual(language.dictionaries("unsure", ("es", "en")), ("es", "en"))
        self.assertEqual(language.dictionaries("en", ("es",)), ("es",))
        self.assertEqual(language.dictionaries(None, ("es", "en")), ("es", "en"))


class BuildByLanguageTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = guard_pages.corpus(80, errors=0.3)
        cls.plain, _ = build_model(cls.pages)
        with mock.patch.object(final_fix.train_chooser, "DETECTOR_CELL", (3, 10)):
            cls.model, cls.report = final_fix.build(
                cls.pages,
                chooser_cell=(3, 10),
                threads=1,
                minima=MINIMA,
                min_group=50,
                min_positive=10,
                floor=FLOOR,
                sensor=0.07,
            )

    def test_G41_the_model_carries_the_sensor_and_a_block_per_language_that_the_info_shows(self):
        fix = self.model.fix
        self.assertEqual(fix["language_sensor"], language.sensor_settings(0.07))
        self.assertEqual(set(fix["by_language"]), {"es"})
        self.assertEqual(set(fix["by_language"]["es"]), {"t_safe", "m"})
        info = Guard(self.model).info["fix"]
        self.assertEqual(info["by_language"], fix["by_language"])
        self.assertEqual(info["language_sensor"], fix["language_sensor"])
        self.assertNotIn("by_language", Guard(self.plain).info["fix"])

    def test_G41_with_only_spanish_pages_the_spanish_block_is_the_pooled_one(self):
        self.assertEqual(self.model.fix["by_language"]["es"]["t_safe"], self.model.fix["t_safe"])
        self.assertEqual(self.model.fix["by_language"]["es"]["m"], self.model.fix["m"])

    def test_G44_the_report_has_the_delivered_points_and_the_split_of_both_calibrations(self):
        for key in ("delivered_by_language", "by_language_cross_fitted", "by_language_in_sample"):
            self.assertIn(key, self.report)
        self.assertEqual(set(self.report["by_language_cross_fitted"]["by_language"]), {"es"})


if __name__ == "__main__":
    unittest.main()
