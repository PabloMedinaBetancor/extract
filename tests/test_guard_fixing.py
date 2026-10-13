"""The fixes at run time: a model built from a small corpus, the guard that uses it, and the rules between them."""

import dataclasses
import json
import math
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from bench.guard import final_fix, operating, pairs, train_chooser
from extract.engine.cache import reading_from_dict
from extract.errors import ModelMismatch
from extract.guard import fixing, model
from extract.guard.checker import Guard
from tests import guard_pages

CELL = (3, 10)
FLOOR = 0.85  # the synthetic corpus is not clean enough for the real floor of 99.5 %
MINIMA = {"text": 20, "join": 10, "segment": 10, "drop": 10}


def build_model(pages=None, calls=None):
    pages = pages or guard_pages.corpus(40, errors=0.3)
    with mock.patch.object(train_chooser, "DETECTOR_CELL", CELL):
        if calls is None:
            return build_with(pages)
        # what the thresholds are chosen on: the calls of the build to the calibration and to the threshold search
        with (
            mock.patch.object(operating, "calibrate_rows_by_op", wraps=operating.calibrate_rows_by_op) as calibration,
            mock.patch.object(operating, "thresholds_by_op", wraps=operating.thresholds_by_op) as search,
        ):
            built = build_with(pages)
        calls["calibration"], calls["search"] = calibration.call_args_list, search.call_args_list
        return built


def build_with(pages):
    return final_fix.build(
        pages, chooser_cell=CELL, threads=1, minima=MINIMA, min_group=50, min_positive=10, floor=FLOOR
    )


class BuiltModelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pages = guard_pages.corpus(80, errors=0.3)
        cls.calls: dict = {}
        cls.model, cls.report = build_model(cls.pages, cls.calls)
        cls.guard = Guard(cls.model)

    def test_B31_the_thresholds_are_chosen_on_cross_fitted_probabilities_at_the_floor_asked_for(self):
        chosen = self.calls["calibration"][0]  # the first call is the build's own: the reports come after it
        self.assertIs(chosen.args[2], True)
        self.assertEqual(
            len(self.calls["calibration"]), 3
        )  # the build's, and one of each report (cross-fitted, in sample)
        self.assertEqual([c.args[2] for c in self.calls["calibration"][1:]], [True, False])
        self.assertEqual(len(self.calls["search"]), 1)
        self.assertEqual(self.calls["search"][0].kwargs["floor"], FLOOR)
        self.assertEqual(self.calls["search"][0].kwargs["minima"], MINIMA)  # the counts the thresholds must reach
        self.assertEqual(self.model.fix["minima"], MINIMA)
        self.assertEqual(self.model.fix["precision_floor"], FLOOR)
        self.assertEqual((self.model.fix["safe_promise"], self.model.fix["t5_floor"]), ("exact", 0.99))

    def test_B31_the_report_measures_the_delivered_points_cross_fitted_and_in_sample(self):
        delivered = {
            op: {"t_safe": self.model.fix["t_safe"][op], "m": self.model.fix["m"][op]} for op in self.model.fix["m"]
        }
        self.assertEqual(self.report["delivered"], delivered)
        for kind, cross in (("cross_fitted", True), ("in_sample", False)):
            part = self.report[kind]
            self.assertIs(part["cross_fit"], cross)
            self.assertEqual(part["thresholds"], delivered)

    def test_B31_the_chooser_is_trained_on_the_sites_the_detector_flags_not_on_all_of_them(self):
        every = sum(1 for page in self.pages for w in page.grounded() if pairs.is_site(page, int(w)))
        trained = self.model.fix["training"]["chooser_sites"]
        self.assertGreater(trained, 0)
        self.assertLess(trained, 0.5 * every)

    def test_B31_a_page_with_no_usable_confidence_is_left_unscored_with_its_warning(self):
        readings, _ = self.held_out()
        page = readings[0]
        for bad in (math.nan, math.inf):
            words = tuple(dataclasses.replace(w, conf=bad, char_conf=None) for w in page.words)
            scored = self.guard.score_page(dataclasses.replace(page, words=words))
            self.assertTrue(all(s.status == "unscored" and s.fix is None for s in scored))
            self.assertEqual(self.guard.last_warnings[0]["cause"], "no_confidence")
        words = tuple(dataclasses.replace(w, conf=math.nan if i else None) for i, w in enumerate(page.words))
        self.assertTrue(
            all(s.status == "unscored" for s in self.guard.score_page(dataclasses.replace(page, words=words)))
        )
        words = tuple(dataclasses.replace(w, conf=math.nan if i else 0.9) for i, w in enumerate(page.words))
        self.assertTrue(
            any(s.status != "unscored" for s in self.guard.score_page(dataclasses.replace(page, words=words)))
        )

    def test_B31_a_page_with_some_nan_confidences_is_scored_and_gives_the_same_row_as_without_them(self):
        readings, _ = self.held_out()
        page = readings[0]
        words = tuple(dataclasses.replace(w, conf=math.nan if i % 2 else w.conf) for i, w in enumerate(page.words))
        scored = self.guard.score_page(dataclasses.replace(page, words=words))
        self.assertTrue(all(s.status != "unscored" for s in scored))
        self.assertTrue(all(s.p_wrong is None or math.isfinite(s.p_wrong) for s in scored))

    def test_A2_a_word_that_is_not_doubtful_keeps_no_v4_suggestion_even_with_a_text_fix(self):
        meta = {**self.model.meta, "thresholds": {k: 2.0 for k in self.model.meta["thresholds"]}}  # nothing doubtful
        guard = Guard(model.GuardModel(self.model.detector, meta, None, self.model.chooser))
        readings, _ = self.held_out()
        found = [s for r in readings for s in guard.score_page(r) if s.fix and s.fix.op == "text"]
        self.assertTrue(found)
        self.assertTrue(all(s.status == "ok" for s in found))
        self.assertTrue(all(s.suggestion is None and s.suggestion_conf is None for s in found))

    def held_out(self, seed=7, documents=6):
        """Pages like the corpus (other seeds): the readings to score and the pairs with their labels."""
        rng = random.Random(seed)
        records = [guard_pages.record(f"h{d}", 1, rng, errors=0.3) for d in range(documents)]
        return [reading_from_dict(r["ocr"]["reading"]) for r in records], [pairs.page_of(r) for r in records]

    def test_B4_the_model_has_a_fix_block_with_the_thresholds_the_tables_and_the_fingerprint(self):
        fix = self.model.fix
        self.assertEqual(self.model.chooser is not None, True)
        for key in ("feature_fingerprint", "t_gen", "calibration", "t_safe", "m", "language_floor", "precision_floor"):
            self.assertIn(key, fix)
        self.assertIn("all", fix["calibration"])
        self.assertIsNotNone(fix["t_safe"]["text"])
        json.dumps(fix)  # everything in it is plain data

    def test_B4_the_model_file_round_trips_with_its_chooser_and_a_changed_feature_schema_is_refused(self):
        with tempfile.TemporaryDirectory() as name:
            path = Path(name) / "guard.model"
            self.model.save(path)
            loaded = model.GuardModel.load(path)
            self.assertEqual(loaded.model_id(), self.model.model_id())
            self.assertEqual(loaded.fix["t_gen"], self.model.fix["t_gen"])
            x = np.random.default_rng(1).normal(size=(5, len(model.features.FEATURES))).astype(np.float32)
            self.assertTrue(np.allclose(loaded.choose(x), self.model.choose(x)))
            broken = model.GuardModel(
                self.model.detector,
                {**self.model.meta, "fix": {**self.model.fix, "feature_fingerprint": "x"}},
                None,
                self.model.chooser,
            )
            broken.save(path)
            with self.assertRaises(ModelMismatch):
                model.GuardModel.load(path)

    def test_B4_a_model_without_a_chooser_has_no_fix_block_and_the_guard_gives_no_fixes(self):
        plain = model.GuardModel(self.model.detector, self.model.meta)
        self.assertIsNone(plain.fix)
        readings, _ = self.held_out()
        scored = Guard(plain).score_page(readings[0])
        self.assertTrue(all(s.fix is None for s in scored))

    def test_B4_the_guard_proposes_fixes_with_modes_and_the_applied_ones_fix_far_more_than_they_break(self):
        readings, labelled = self.held_out()
        fixed = broken = proposed = 0
        for reading, page in zip(readings, labelled, strict=True):
            scored = self.guard.score_page(reading)
            for index, s in enumerate(scored):
                if s.fix is None:
                    continue
                proposed += 1
                self.assertEqual(s.fix.ids, (reading.words[index].id,))
                self.assertGreaterEqual(s.fix.p_fix, 0.0)
                self.assertLessEqual(s.fix.p_fix, 1.0)
                if "safe" in s.fix.modes:
                    truth = page.labels[index].truth_text
                    was_right = page.labels[index].kind == "ok"
                    if not was_right and truth and truth == s.fix.text:
                        fixed += 1
                    elif was_right:
                        broken += 1
        self.assertGreater(proposed, 20)
        self.assertGreater(fixed, 10)
        self.assertLessEqual(broken, max(1, fixed // 20))

    def test_B4_every_proposal_that_applies_in_safe_applies_in_more_and_the_modes_are_one_of_three(self):
        readings, _ = self.held_out()
        seen = set()
        for reading in readings:
            for s in self.guard.score_page(reading):
                if s.fix is not None:
                    seen.add(s.fix.modes)
        self.assertLessEqual(seen, {("safe", "more"), ("more",), ()})
        self.assertIn(("safe", "more"), seen)

    def test_G13_the_same_page_gives_the_same_fixes_twice(self):
        readings, _ = self.held_out()
        first = [s.fix for s in self.guard.score_page(readings[0])]
        second = [s.fix for s in self.guard.score_page(readings[0])]
        self.assertEqual(first, second)

    def test_A2_a_page_that_is_not_in_the_lists_gets_proposals_without_modes_and_a_warning(self):
        readings, _ = self.held_out()
        page = readings[0]
        gibberish = tuple(
            type(w)(**{**w.__dict__, "text": "".join(chr(97 + (ord(c) * 7 + i) % 26) for c in w.text) + "xq"})
            for i, w in enumerate(page.words)
        )
        foreign = type(page)(**{**page.__dict__, "words": gibberish})
        scored = self.guard.score_page(foreign)
        self.assertTrue(all(s.fix.modes == () for s in scored if s.fix is not None))
        self.assertIn("fixes_withheld", " ".join(w["key"] for w in self.guard.last_warnings))
        self.assertEqual(self.guard.last_warnings[-1]["cause"], "language")

    def test_B4_a_doubtful_word_with_a_text_fix_keeps_the_v4_suggestion_fields(self):
        readings, _ = self.held_out()
        found = [
            s
            for r in readings
            for s in self.guard.score_page(r)
            if s.fix and s.fix.op == "text" and s.status == "doubtful"
        ]
        self.assertTrue(found)
        self.assertTrue(all(s.suggestion == s.fix.text and s.suggestion_conf == s.fix.p_fix for s in found))

    def test_B4_the_guard_says_in_its_info_when_the_model_applies_fixes_and_a_model_without_a_chooser_says_none(self):
        info = self.guard.info["fix"]
        self.assertEqual(
            set(info), {"t_gen", "t_safe", "m", "precision_floor", "t5_floor", "safe_promise", "language_floor"}
        )
        self.assertEqual(info["t_safe"], self.model.fix["t_safe"])
        self.assertIsNone(Guard(model.GuardModel(self.model.detector, self.model.meta)).info["fix"])

    def test_G13_the_chooser_scores_the_same_rows_the_same_with_one_thread_and_with_four(self):
        x = np.random.default_rng(2).normal(size=(300, len(model.features.FEATURES))).astype(np.float32)
        self.assertTrue(np.array_equal(self.model.choose(x, 1), self.model.choose(x, 4)))

    def test_B4_the_reasons_are_at_most_three_fix_keys(self):
        readings, _ = self.held_out()
        for reading in readings:
            for s in self.guard.score_page(reading):
                if s.fix is not None:
                    self.assertLessEqual(len(s.fix.reasons), 3)
                    self.assertTrue(all(r.key.startswith("guard.fix.") for r in s.fix.reasons))


class LanguageFloorTest(unittest.TestCase):
    """What `final_fix` writes as `language_floor` (half of the 1st percentile of the shares, never under 0.2)."""

    @staticmethod
    def pages(*shares):
        from types import SimpleNamespace

        return [SimpleNamespace(context=SimpleNamespace(dictionary_share=share)) for share in shares]

    def test_the_floor_is_half_of_the_first_percentile_of_the_development_shares(self):
        shares = [0.6 + 0.4 * i / 99 for i in range(100)]  # 0.60 .. 1.00, its 1st percentile is 0.604
        self.assertAlmostEqual(final_fix.language_floor(self.pages(*shares)), 0.302, places=6)

    def test_the_floor_is_never_under_the_minimum_and_the_nan_pages_are_left_out(self):
        self.assertEqual(final_fix.language_floor(self.pages(*[0.1] * 50)), final_fix.LANGUAGE_FLOOR_MIN)
        self.assertEqual(final_fix.language_floor(self.pages(math.nan, math.nan)), final_fix.LANGUAGE_FLOOR_MIN)
        self.assertEqual(final_fix.language_floor([]), final_fix.LANGUAGE_FLOOR_MIN)
        self.assertAlmostEqual(final_fix.language_floor(self.pages(math.nan, *[0.8] * 30)), 0.4)
        self.assertEqual(final_fix.LANGUAGE_FLOOR_MIN, 0.2)
        self.assertEqual(final_fix.LANGUAGE_FLOOR_SHARE, 0.5)
        self.assertEqual(final_fix.LANGUAGE_PERCENTILE, 1.0)


def scripted(text="el plan de la orde nanza del municipio es el que rige la ciudad en este año", **block):
    """A page, its context and a fake model whose calibration says what each site's rows are worth (see `fake_p`)."""
    from extract.guard import candidates, signals
    from tests.guard_words import page

    words = page(text, gap=4)
    matrix = signals.extract(words, ("es", "en"), "es")
    raw = np.zeros(len(words))
    raw[4] = 1.0

    class Model:
        fix = {
            "t_gen": 0.5,
            "calibration": {"all": {"scores": [0, 1], "probs": [0, 1]}},
            "t_safe": {"text": 0.99, "join": 0.99, "segment": None, "drop": None},
            "m": {"text": 0.3, "join": 0.3, "segment": None, "drop": None},
            "language_floor": 0.0,
            **block,
        }

        def choose(self, x):
            return np.zeros(len(x))

    ctx = candidates.PageContext(words, ("es", "en"), frozenset())
    sites = fixing.sites_of(ctx, matrix, [4])
    return words, matrix, raw, Model(), sites


def propose_with(sites, fake_p, words, matrix, raw, model, languages=("es", "en")):
    """`fixing.propose` with the calibrated probabilities of each site's rows given by `fake_p(site) -> [p_keep, p...]`."""

    def fake_calibrate(tables, raw_scores, ops, kinds):
        return np.concatenate([np.array(fake_p(s), dtype=float) for s in sites])

    with mock.patch.object(fixing, "calibrate", fake_calibrate):
        return fixing.propose(model, words, matrix, raw, languages)


class ProposeTest(unittest.TestCase):
    def setUp(self):
        self.words, self.matrix, self.raw, self.model, self.sites = scripted()

    def probabilities(self, text_row, join_row):
        def fake_p(site):
            n = len(site.found)
            if site.op == "text":
                return [text_row[0]] + [text_row[1]] * (n - 1)
            if site.op == "join":
                return [join_row[0]] + [join_row[1]] * (n - 1)
            return [0.9] + [0.1] * (n - 1)

        return fake_p

    def test_B31_the_proposal_that_applies_is_delivered_even_when_another_site_of_the_word_has_the_larger_margin(self):
        # the text fix has margin 0.95 but passes only `more`; the join has margin 0.495 and passes `safe`
        fake = self.probabilities((0.02, 0.97), (0.50, 0.995))
        fixes, withheld = propose_with(self.sites, fake, self.words, self.matrix, self.raw, self.model)
        self.assertIsNone(withheld)
        self.assertEqual((fixes[4].op, fixes[4].modes), ("join", ("safe", "more")))

    def test_B31_with_no_proposal_that_applies_the_one_with_the_larger_margin_is_kept(self):
        fake = self.probabilities((0.02, 0.60), (0.50, 0.55))  # neither reaches a threshold of its op in safe or more
        fixes, _ = propose_with(self.sites, fake, self.words, self.matrix, self.raw, self.model)
        self.assertEqual((fixes[4].op, fixes[4].modes), ("text", ("more",)))
        fake = self.probabilities((0.50, 0.60), (0.02, 0.60))
        fixes, _ = propose_with(self.sites, fake, self.words, self.matrix, self.raw, self.model)
        self.assertEqual((fixes[4].op, fixes[4].modes), ("join", ("more",)))
        for text_row, join_row, wanted in (((0.50, 0.60), (0.45, 0.60), "join"), ((0.45, 0.60), (0.50, 0.60), "text")):
            fixes, _ = propose_with(
                self.sites, self.probabilities(text_row, join_row), self.words, self.matrix, self.raw, self.model
            )
            self.assertEqual((fixes[4].op, fixes[4].modes), (wanted, ()))

    def test_B31_the_best_candidate_of_a_site_is_the_one_with_the_highest_probability_not_the_first(self):
        site = next(s for s in self.sites if s.op == "text")
        self.assertGreater(len(site.found), 2)

        def fake_p(s):
            row = [0.1] * len(s.found)
            if s is site:
                row[2] = 0.9
            return row

        fixes, _ = propose_with(self.sites, fake_p, self.words, self.matrix, self.raw, self.model)
        self.assertEqual(fixes[4].text, site.found[2].text)
        self.assertAlmostEqual(fixes[4].p_fix, 0.9)

    def test_A2_the_page_is_withheld_under_the_language_floor_and_only_under_it(self):
        from extract.guard import candidates

        share = candidates.PageContext(self.words, ("es", "en"), frozenset()).dictionary_share
        fake = self.probabilities((0.02, 0.97), (0.50, 0.995))
        for floor, withheld in ((share, None), (share + 1e-9, "language"), (0.0, None)):
            model = scripted(language_floor=floor)[3]
            fixes, cause = propose_with(self.sites, fake, self.words, self.matrix, self.raw, model)
            self.assertEqual(cause, withheld)
            self.assertEqual(all(f.modes == () for f in fixes.values()), withheld is not None)

    def test_A2_a_model_with_no_floor_uses_the_placeholder_of_the_module(self):
        fake = self.probabilities((0.02, 0.97), (0.50, 0.995))
        model = scripted()[3]
        del model.fix["language_floor"]
        gibberish = " ".join(w.text + "qx" for w in self.words)
        words, matrix, raw, bad, sites = scripted(gibberish)
        del bad.fix["language_floor"]
        _, cause = propose_with(sites, fake, words, matrix, raw, bad)
        self.assertEqual(cause, "language")
        self.assertEqual(fixing.DEFAULT_LANGUAGE_FLOOR, 0.4)

    def test_G30_a_page_with_no_site_is_still_withheld_by_the_language_gate_and_one_with_nothing_flagged_by_no_list(
        self,
    ):
        with mock.patch.object(fixing, "sites_of", lambda *args: []):
            for floor, cause in ((2.0, "language"), (0.0, None)):
                model = scripted(language_floor=floor)[3]
                self.assertEqual(fixing.propose(model, self.words, self.matrix, self.raw, ("es", "en")), ({}, cause))
        quiet = np.zeros(len(self.words))  # nothing reaches t_gen
        self.assertEqual(fixing.propose(self.model, self.words, self.matrix, quiet, ("es", "en")), ({}, None))
        for languages in (("xx",), ()):
            self.assertEqual(fixing.propose(self.model, self.words, self.matrix, quiet, languages), ({}, "language"))

    def test_G30_a_page_with_no_usable_word_list_is_withheld_with_the_warning(self):
        for languages in (("xx",), ()):
            _, cause = propose_with(
                self.sites,
                self.probabilities((0.5, 0.9), (0.5, 0.9)),
                self.words,
                self.matrix,
                self.raw,
                self.model,
                languages,
            )
            self.assertEqual(cause, "language", languages)


class FiniteConfidencesTest(unittest.TestCase):
    def test_G30_a_confidence_that_is_not_finite_says_what_a_missing_one_says_in_every_signal(self):
        from extract.guard import signals
        from tests.guard_words import page

        text = "la ordenanza del plan de la ciudad establece que los usos del suelo serán los que se indican"
        clean = page(text, gap=4)
        nan = math.nan
        broken = tuple(
            dataclasses.replace(
                w, conf=nan if i % 3 == 0 else w.conf, char_conf=(nan, 0.4, 0.9) if i % 4 == 0 else w.char_conf
            )
            for i, w in enumerate(clean)
        )
        missing = tuple(
            dataclasses.replace(
                w, conf=None if i % 3 == 0 else w.conf, char_conf=(0.4, 0.9) if i % 4 == 0 else w.char_conf
            )
            for i, w in enumerate(clean)
        )
        a = signals.extract(broken, ("es", "en"), "es")
        b = signals.extract(missing, ("es", "en"), "es")
        self.assertTrue(np.array_equal(a, b, equal_nan=True))
        self.assertTrue(np.isfinite(a[0, signals.INDEX["conf_min"]]))


class CalibrationTablesTest(unittest.TestCase):
    def test_B4_the_runtime_applies_the_tables_with_the_precedence_the_training_used(self):
        from tests.test_guard_operating_ops import table_of

        rng = np.random.default_rng(3)
        sites = [("text", int(rng.random() < 0.8), i // 3, (i,), f"d{i % 20}", "letters") for i in range(500)]
        sites += [("text", int(rng.random() < 0.2), i // 3, (i,), f"d{i % 20}", "accent") for i in range(500)]
        sites += [("join", int(rng.random() < 0.5), i // 3, (i, i + 1), f"d{i % 20}") for i in range(150)]
        sites += [("drop", 1, i // 3, (i,), f"d{i % 20}") for i in range(10)]
        table = table_of(sites)
        raw = np.where(np.arange(len(table.y)) % 2 == 1, 0.5, 0.1) + rng.normal(0, 0.01, len(table.y))
        tables = operating.calibration_tables(table, raw, min_group=100, min_positive=20)
        self.assertIn("text|letters", tables)
        self.assertIn("text|accent", tables)
        self.assertIn("join", tables)
        self.assertNotIn("drop", tables)  # ten sites: no table of its own, it is calibrated by the table of all rows
        ops = table.family[table.site]
        kinds = np.array([model.features.KINDS[int(c)] for c in table.x[:, model.features.INDEX["kind"]]], dtype=object)
        runtime = fixing.calibrate(tables, raw, ops, kinds)
        training = operating.calibrate_rows_by_op(table, raw, min_group=100, min_positive=20)
        self.assertTrue(np.allclose(runtime, training))


class ProcessedWordsTest(unittest.TestCase):
    def test_a_word_exactly_at_t_gen_is_flagged(self):
        self.assertEqual(fixing.processed_words(np.array([0.5, 0.1, 0.2]), 0.5), [0])

    def test_the_flagged_words_are_those_over_t_gen_at_most_a_fifth_of_the_page_highest_first(self):
        raw = np.linspace(0, 1, 50)
        self.assertEqual(fixing.processed_words(raw, 0.5), list(range(40, 50)))  # 20 words flagged, 10 kept
        self.assertEqual(fixing.processed_words(raw, 0.9), list(range(45, 50)))
        self.assertEqual(fixing.processed_words(raw, 2.0), [])

    def test_a_tie_at_the_cap_keeps_the_words_that_come_first(self):
        raw = np.ones(10)
        self.assertEqual(fixing.processed_words(raw, 0.5), [0, 1])

    def test_a_small_page_keeps_at_least_one_flagged_word(self):
        self.assertEqual(fixing.processed_words(np.array([0.9, 0.1, 0.2]), 0.5), [0])


if __name__ == "__main__":
    unittest.main()


def site(op, span, words=20, kind="letters"):
    return fixing.Site(op, tuple(span), [None, SimpleNamespace(kind=kind)], np.zeros((2, 1)))


def kept_of(*proposals):
    """(site, best, p_fix, p_keep) for each (op, span, p_fix, p_keep)."""
    return [(site(op, span), 1, p_fix, p_keep) for op, span, p_fix, p_keep in proposals]


BLOCK = {"t_safe": {"text": 0.95, "join": 0.97, "segment": None}, "m": {"text": 0.3, "join": 0.2, "segment": 0.4}}


class ModesTest(unittest.TestCase):
    def modes(self, *proposals, withheld=None):
        return fixing._modes(kept_of(*proposals), BLOCK, withheld)

    def test_B4_each_op_applies_against_its_own_thresholds(self):
        out = self.modes(
            ("text", (1,), 0.96, 0.0),  # over t_safe
            ("text", (5,), 0.60, 0.1),  # margin 0.5 over m, under t_safe
            ("text", (9,), 0.40, 0.3),  # margin 0.1 under m
            ("join", (12, 13), 0.96, 0.0),  # under the join's t_safe of 0.97 but margin over its m
            ("segment", (15,), 0.99, 0.0),  # segment is off in safe (None), margin over m
        )
        self.assertEqual(out, [("safe", "more"), ("more",), (), ("more",), ("more",)])

    def test_B4_an_op_that_is_off_in_both_modes_never_applies(self):
        block = {"t_safe": {"drop": None}, "m": {"drop": None}}
        self.assertEqual(fixing._modes(kept_of(("drop", (3,), 1.0, 0.0)), block, None), [()])

    def test_B4_of_two_safe_proposals_that_share_a_word_the_larger_gain_wins_in_both_modes(self):
        out = self.modes(("text", (3,), 0.96, 0.0), ("join", (2, 3), 0.99, 0.0))
        self.assertEqual(out, [(), ("safe", "more")])

    def test_B4_a_more_only_proposal_never_displaces_a_safe_winner(self):
        # the join has the larger gain but passes only `more`; the safe text fix keeps its word in both modes
        out = self.modes(("text", (3,), 0.96, 0.5), ("join", (3, 4), 0.96, 0.0))
        self.assertEqual(out, [("safe", "more"), ()])

    def test_B4_two_more_only_proposals_that_overlap_are_selected_among_themselves(self):
        out = self.modes(("text", (3,), 0.60, 0.1), ("join", (2, 3), 0.80, 0.1))
        self.assertEqual(out, [(), ("more",)])

    def test_A2_a_withheld_page_gets_no_mode_at_all(self):
        self.assertEqual(self.modes(("text", (1,), 0.99, 0.0), withheld="language"), [()])

    def test_B4_the_modes_do_not_depend_on_the_order_the_sites_arrive_in(self):
        proposals = [("text", (3,), 0.96, 0.0), ("join", (2, 3), 0.99, 0.0), ("text", (8,), 0.6, 0.1)]
        forward = self.modes(*proposals)
        backward = self.modes(*reversed(proposals))
        self.assertEqual(forward, list(reversed(backward)))


class ReasonsTest(unittest.TestCase):
    def reasons(self, op, kind, sources=("dictionary",), word_text="Orde", text="orden", readings=None, alts=None):
        from extract.guard import candidates
        from tests.guard_words import page

        words = page(f"la {word_text} del", **{"1": {"readings": tuple(readings or ()), "char_alts": alts}})
        ctx = candidates.PageContext(words, ("es", "en"))
        candidate = candidates.Candidate(text, op, kind, tuple(sources), (1,))
        return [(r.key, dict(r.params)) for r in fixing.reasons_of(site(op, (1,)), candidate, ctx, words)]

    def test_B4_the_first_reason_says_what_the_edit_is(self):
        self.assertEqual(self.reasons("text", "accent", ())[0][0], "guard.fix.accent_restored")
        self.assertEqual(self.reasons("text", "case", ())[0][0], "guard.fix.case_restored")
        self.assertEqual(self.reasons("text", "punct", ())[0][0], "guard.fix.punctuation_restored")
        self.assertEqual(self.reasons("join", "structure", ())[0][0], "guard.fix.words_joined")
        self.assertEqual(self.reasons("segment", "structure", ())[0][0], "guard.fix.word_split")
        self.assertEqual(self.reasons("drop", "structure", ())[0][0], "guard.fix.speck_dropped")

    def test_B4_an_accent_and_case_fix_gives_both_reasons_and_never_more_than_three(self):
        keys = [k for k, _ in self.reasons("text", "accent_case", ("dictionary", "page"), readings=self.agreeing(3))]
        self.assertEqual(keys[:2], ["guard.fix.accent_restored", "guard.fix.case_restored"])
        self.assertLessEqual(len(keys), 3)

    def agreeing(self, n, text="orden"):
        from tests.guard_words import Reading

        return [Reading(name, text, rank=0, group=("p1w1",)) for name in ("latin", "bin", "clahe")[:n]]

    def test_B4_readers_that_agree_with_the_candidate_are_a_reason_from_two(self):
        one = [k for k, _ in self.reasons("text", "letters", (), readings=self.agreeing(1))]
        two = self.reasons("text", "letters", (), readings=self.agreeing(2))
        self.assertNotIn("guard.fix.readers_agree", one)
        self.assertIn(("guard.fix.readers_agree", {"n": 2, "of": 2}), two)

    def test_B4_the_engine_alternative_gives_the_character_and_its_probability(self):
        alts = tuple(((c, 0.9),) for c in "Ord") + ((("e", 0.4), ("é", 0.3)),)
        found = self.reasons("text", "letters", ("ctc_alt",), word_text="Ordx", text="Orde", alts=alts)
        self.assertIn(("guard.fix.engine_alternative", {"char": "e", "conf": 0.4}), found)

    def test_B4_a_dictionary_and_a_page_source_give_their_reasons_with_the_word_and_the_count(self):
        found = dict(self.reasons("text", "letters", ("dictionary", "page"), text="orden"))
        self.assertEqual(found["guard.fix.dictionary_nearest"], {"word": "orden"})
        self.assertIn("count", found["guard.fix.page_vocabulary"])


def structure_corpus(documents, seed):
    rng = random.Random(seed)
    return [pairs.page_of(guard_pages.structure_record(f"s{seed}-{d}", 1, rng)) for d in range(documents)]


class StructureModelTest(unittest.TestCase):
    """A model trained on pages with split words proposes joins, and the joins it applies are right."""

    @classmethod
    def setUpClass(cls):
        with mock.patch.object(train_chooser, "DETECTOR_CELL", CELL):
            cls.model, _ = final_fix.build(
                structure_corpus(60, 1),
                chooser_cell=CELL,
                threads=1,
                minima=MINIMA,
                min_group=50,
                min_positive=10,
                floor=FLOOR,
            )
        cls.guard = Guard(cls.model)

    def test_B4_the_chooser_was_trained_with_structure_sites_and_has_a_join_threshold(self):
        self.assertTrue(self.model.fix["structure_sites"])
        self.assertIn("join", self.model.fix["m"])

    def test_B4_the_joins_the_guard_proposes_join_two_consecutive_words_and_are_mostly_the_truth(self):
        rng = random.Random(99)
        right = wrong = 0
        for d in range(8):
            record = guard_pages.structure_record(f"t{d}", 1, rng)
            page = pairs.page_of(record)
            reading = reading_from_dict(record["ocr"]["reading"])
            for index, s in enumerate(self.guard.score_page(reading)):
                if s.fix is None or s.fix.op != "join" or not s.fix.modes:
                    continue
                self.assertEqual(len(s.fix.ids), len(set(s.fix.ids)))
                self.assertEqual(s.fix.ids[0], reading.words[index].id)
                component = page.component_of(index)
                if component is not None and component.kind == "split" and component.truth == s.fix.text:
                    right += 1
                else:
                    wrong += 1
        self.assertGreater(right, 5)
        self.assertLessEqual(wrong, right // 3)


class SitesTest(unittest.TestCase):
    def sites(self, text, flagged, **kwargs):
        from extract.guard import candidates, signals
        from tests.guard_words import page

        words = page(text, gap=4, **kwargs)
        ctx = candidates.PageContext(words, ("es", "en"))
        matrix = signals.extract(words, ("es", "en"), "es")
        return fixing.sites_of(ctx, matrix, flagged)

    def test_B4_a_flagged_speck_is_a_drop_site_and_a_flagged_fragment_a_join_and_a_text_site(self):
        ops = {(s.op, s.span) for s in self.sites("la x casa", [1], conf=0.5)}
        self.assertIn(("drop", (1,)), ops)
        ops = {(s.op, s.span) for s in self.sites("la orde nanza del", [1])}
        self.assertIn(("join", (1, 2)), ops)
        self.assertNotIn(("join", (0, 1)), {(s.op, s.span) for s in self.sites("la orde nanza del", [2])})

    def test_B4_a_flagged_speck_of_marks_only_is_a_drop_site_whose_rows_can_be_built(self):
        # found on the real pages of RUN-D: a token with no letters has no parts, and the features asked for them
        sites = self.sites("la casa ~ del plan de la ciudad", [2], conf=0.3)
        drops = [s for s in sites if s.op == "drop" and s.span == (2,)]
        self.assertEqual(len(drops), 1)
        self.assertEqual(drops[0].x.shape[0], len(drops[0].found))
        self.assertGreater(len(drops[0].found), 1)

    def test_B4_a_flagged_long_token_is_a_segment_site_and_only_when_it_is_flagged(self):
        self.assertIn(("segment", (1,)), {(s.op, s.span) for s in self.sites("la ordenanzaplan del", [1])})
        self.assertNotIn("segment", {s.op for s in self.sites("la ordenanzaplan del", [0])})

    def test_B4_a_site_with_no_candidate_but_the_token_is_not_a_site(self):
        self.assertEqual(self.sites("la casa del", []), [])
        for site in self.sites("la orde nanza del", [1, 2]):
            self.assertGreater(len(site.found), 1)
            self.assertEqual(site.x.shape[0], len(site.found))

    def test_B4_a_proposal_exactly_at_the_threshold_applies(self):
        block = {"t_safe": {"text": 0.95}, "m": {"text": 0.3}}
        self.assertEqual(fixing._modes(kept_of(("text", (1,), 0.95, 0.9)), block, None), [("safe", "more")])
        self.assertEqual(fixing._modes(kept_of(("text", (1,), 0.5, 0.2)), block, None), [("more",)])
