"""The training rows and the rows of the product come from the same words, through the same function; a model trained on
one cell of second readings does not use another one, and a cap hides the readings of the units it leaves out."""

import dataclasses
import types
import unittest
from unittest import mock

import numpy as np

from extract.engine import reread
from extract.guard import candidates, features, readings, signals
from extract.guard.checker import Guard
from tests.guard_words import Reading, page

LANGUAGES = ("es", "en")
LINE = readings.ReadingSpec("line", ("latin", "wide"), None)


@dataclasses.dataclass(frozen=True)
class FakeReading:
    words: tuple


def with_readings(text, rank_of):
    """A page whose every word carries a reading of each reader (latin, wide, deep) in the unit of the rank `rank_of(i)`."""
    tokens = text.split()
    extras = {
        str(i): {
            "readings": tuple(
                Reading(source, f"{token}s" if i % 2 else token, 0.8, rank=rank_of(i), group=(f"p1w{i}",))
                for source in ("latin", "wide", "deep")
            )
        }
        for i, token in enumerate(tokens)
    }
    return page(text, **extras)


def rows_of(words):
    ctx = candidates.PageContext(words, LANGUAGES, frozenset())
    matrix = signals.extract(words, LANGUAGES, "es")
    out = []
    for i in range(len(words)):
        found = candidates.generate(ctx, i)
        if len(found) > 1:
            out.append(features.feature_rows(ctx, i, found, matrix[i]))
    return out


class ApplyTest(unittest.TestCase):
    def test_the_readers_of_the_cell_stay_and_the_others_go(self):
        words = with_readings("el plan de la ordenanza", lambda i: 0)
        shown = readings.apply(words, LINE)
        self.assertTrue(all({r.source for r in w.readings} == {"latin", "wide"} for w in shown))
        none = readings.apply(words, None)
        self.assertTrue(all(w.readings is None for w in none))

    def test_a_cap_hides_the_units_over_it_exactly_as_the_engine_view_does(self):
        words = with_readings("el plan de la ordenanza del municipio", lambda i: i)
        capped = readings.ReadingSpec("line", ("latin", "wide", "deep"), 3)
        shown = readings.apply(words, capped)
        self.assertEqual(shown, reread.view(words, 3))
        self.assertEqual([w.readings is not None for w in shown], [True, True, True, False, False, False, False])

    def test_the_cell_of_a_spec_is_the_cell_of_pass_2_that_reads_it(self):
        self.assertEqual(LINE.cell, "L")
        self.assertEqual(readings.ReadingSpec("zone", ("deep",)).cell, "Z")
        self.assertEqual(readings.ReadingSpec("zone", ("wide", "latin")).cell, "ZC")
        with self.assertRaises(ValueError):
            _ = readings.ReadingSpec("line", ("deep",)).cell

    def test_a_spec_survives_the_model_file(self):
        spec = readings.ReadingSpec("line", ("wide", "latin"), 4)
        self.assertEqual(
            readings.ReadingSpec.from_meta(spec.as_meta()), readings.ReadingSpec("line", ("latin", "wide"), 4)
        )
        self.assertIsNone(readings.ReadingSpec.from_meta(None))
        self.assertTrue(spec.same_cell(readings.ReadingSpec("line", ("latin", "wide"), 4)))
        self.assertFalse(spec.same_cell(readings.ReadingSpec("line", ("latin", "wide"), 5)))


class ParityTest(unittest.TestCase):
    """The rows of training (the loader) and of the runtime (the guard) are the same, byte for byte."""

    def pages(self, n=20):
        out = []
        for k in range(n):
            text = "el plan de la ordenanza del municipio es el que rige la ciudad " + ("nueva " * (k % 3)).strip()
            out.append(with_readings(text.strip(), lambda i, k=k: (i + k) % 6))
        return out

    def test_the_loader_and_the_runtime_give_the_same_rows_for_twenty_pages(self):
        from bench.guard import pairs

        spec = readings.ReadingSpec("line", ("latin", "wide"), 4)
        guard = Guard(types.SimpleNamespace(fix={"readings": spec.as_meta()}))
        for first in self.pages():
            bare = FakeReading(tuple(dataclasses.replace(w, readings=None) for w in first))
            with mock.patch("bench.run_pass2.load_cell", return_value=FakeReading(tuple(first))):
                trained = pairs.with_cells(bare, "page.json", ["L"], spec)
            with mock.patch.object(readings, "product_spec", return_value=spec):
                runtime = guard._words_as_trained(types.SimpleNamespace(words=tuple(first), number=1))
            self.assertEqual(trained.words, runtime)
            self.assertTrue(any(w.readings for w in runtime) and not all(w.readings for w in runtime))
            for a, b in zip(rows_of(list(trained.words)), rows_of(list(runtime)), strict=True):
                self.assertEqual(a.tobytes(), b.tobytes())

    def test_a_smaller_cap_leaves_the_reader_columns_empty_exactly_where_the_readings_are_hidden(self):
        words = with_readings("el plan de la ordenanza del municipio es el que rige la ciudad", lambda i: i)
        full = readings.ReadingSpec("line", ("latin", "wide"), None)
        small = readings.ReadingSpec("line", ("latin", "wide"), 5)
        index = features.INDEX["readers_run"]
        ctx_full = candidates.PageContext(readings.apply(words, full), LANGUAGES, frozenset())
        ctx_small = candidates.PageContext(readings.apply(words, small), LANGUAGES, frozenset())
        matrix = signals.extract(words, LANGUAGES, "es")
        checked = 0
        for i in range(len(words)):
            a, b = candidates.generate(ctx_full, i), candidates.generate(ctx_small, i)
            if len(a) < 2 or len(b) < 2:
                continue
            checked += 1
            x_full = features.feature_rows(ctx_full, i, a, matrix[i])[:, index]
            x_small = features.feature_rows(ctx_small, i, b, matrix[i])[:, index]
            self.assertFalse(np.isnan(x_full).any())
            self.assertEqual(bool(np.isnan(x_small).all()), i >= 5, i)
        self.assertGreater(checked, 3)


class RuntimeTest(unittest.TestCase):
    def guard(self, trained):
        return Guard(types.SimpleNamespace(fix={"readings": trained}))

    def scored(self, guard, product):
        words = with_readings("el plan de la ordenanza", lambda i: i)
        with mock.patch.object(readings, "product_spec", return_value=product):
            shown = guard._words_as_trained(types.SimpleNamespace(words=words, number=3))
        return words, shown

    def test_a_model_trained_on_the_cell_that_the_product_reads_uses_its_readings_under_its_cap(self):
        spec = readings.ReadingSpec("line", ("latin", "wide"), 3)
        guard = self.guard(spec.as_meta())
        _, shown = self.scored(guard, spec)
        self.assertEqual([w.readings is not None for w in shown], [True, True, True, False, False])
        self.assertEqual(guard.last_warnings, [])

    def test_a_product_that_reads_another_cell_than_the_model_gets_a_warning_and_no_readings(self):
        guard = self.guard(readings.ReadingSpec("line", ("latin", "wide"), 3).as_meta())
        _, shown = self.scored(guard, readings.ReadingSpec("zone", ("deep",), 40))
        self.assertTrue(all(w.readings is None for w in shown))
        warning = guard.last_warnings[-1]
        self.assertEqual(warning["key"], "guard.warning.readings_differ")
        self.assertEqual(
            (warning["model"], warning["product"], warning["page"]), ("line latin+wide @3", "zone deep @40", 3)
        )

    def test_a_model_trained_without_readings_ignores_the_ones_a_page_carries(self):
        guard = self.guard(None)
        _, shown = self.scored(guard, LINE)
        self.assertTrue(all(w.readings is None for w in shown))
        self.assertEqual(guard.last_warnings[-1]["model"], "none")

    def test_a_product_that_reads_other_readers_than_the_model_in_the_same_unit_and_cap_is_not_the_same_cell(self):
        trained = readings.ReadingSpec("line", ("latin", "wide"), 3)
        guard = self.guard(trained.as_meta())
        _, shown = self.scored(guard, readings.ReadingSpec("line", ("deep",), 3))
        self.assertTrue(all(w.readings is None for w in shown))
        self.assertEqual(guard.last_warnings[-1]["key"], "guard.warning.readings_differ")
        self.assertFalse(trained.same_cell(readings.ReadingSpec("line", ("deep",), 3)))

    def test_the_unit_tells_two_cells_with_the_same_readers_apart_but_apply_only_filters_readers(self):
        # L and ZC are read by the same readers: the spec tells them apart by the unit (and so the cell the loader asks
        # for), `apply` shows whatever readings of those readers the words carry: one cell is loaded per spec
        line, zone = readings.ReadingSpec("line", ("latin", "wide")), readings.ReadingSpec("zone", ("latin", "wide"))
        self.assertEqual((line.cell, zone.cell), ("L", "ZC"))
        self.assertFalse(line.same_cell(zone))
        words = with_readings("el plan de la ordenanza", lambda i: 0)
        self.assertEqual(readings.apply(words, line), readings.apply(words, zone))

    def test_a_page_without_readings_needs_no_comparison(self):
        guard = self.guard(None)
        words = page("el plan de la ordenanza")
        self.assertEqual(guard._words_as_trained(types.SimpleNamespace(words=words, number=1)), words)
        self.assertEqual(guard.last_warnings, [])


if __name__ == "__main__":
    unittest.main()
