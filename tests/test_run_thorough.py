"""bench/run_thorough.py: the parity check, the seconds and the memory it reports (the tool is used again for the final run
and for T5, so its own numbers are tested with fake pages and a fake reader process)."""

import dataclasses
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from bench import run_thorough
from bench.run_thorough import differences, entries_of, expected_readings, measure
from extract import second_reading
from extract.types import Reading


def reading(source, text="x"):
    return Reading(source, text, 0.9, None, (), 0, False)


def page(*per_word, failed=False):
    report = {"reader_failed": 1} if failed else {}
    return SimpleNamespace(words=[SimpleNamespace(readings=r) for r in per_word], reread_report=report)


class DifferencesTest(unittest.TestCase):
    def test_equal_readings_are_no_difference_and_a_changed_text_a_missing_reader_or_an_order_are(self):
        a, b = reading("latin"), reading("deep", "y")
        want = ((a, b), ())
        self.assertEqual(differences(page((a, b), None), want), [])
        self.assertEqual([d["word"] for d in differences(page((a,), None), want)], [0])
        self.assertEqual([d["word"] for d in differences(page((b, a), None), want)], [0])
        changed = dataclasses.replace(a, text="z")
        self.assertEqual([d["word"] for d in differences(page((changed, b), (a,)), want)], [0, 1])

    def test_a_page_with_another_number_of_words_is_an_error_and_not_a_pass(self):
        with self.assertRaises(ValueError):
            differences(page((), ()), ((),))


class ExpectedTest(unittest.TestCase):
    def cells(self, lines, zones):
        def load(path, cell):
            return {"L": lines, "Z": zones}[cell]

        return mock.patch.object(run_thorough, "load_cell", load)

    def test_the_expected_page_is_the_lines_readers_of_L_then_the_zones_reader_of_Z_and_nothing_else(self):
        words = lambda *r: SimpleNamespace(readings=r or None)  # noqa: E731
        lines = SimpleNamespace(
            words=[words(reading("latin"), reading("bin"), reading("clahe"), reading("wide")), words()]
        )
        zones = SimpleNamespace(words=[words(reading("deep"), reading("latin")), words(reading("deep", "q"))])
        with self.cells(lines, zones):
            got = expected_readings(Path("p.json"))
        self.assertEqual([[r.source for r in w] for w in got], [["latin", "wide", "deep"], ["deep"]])

    def test_a_missing_cell_is_none_and_cells_of_another_length_are_an_error(self):
        some = SimpleNamespace(words=[SimpleNamespace(readings=None)])
        with self.cells(None, some):
            self.assertIsNone(expected_readings(Path("p.json")))
        with self.cells(some, SimpleNamespace(words=[])):
            with self.assertRaises(ValueError):
                expected_readings(Path("p.json"))


class FakeProcess:
    """Answers each page at once; the clock moves by the seconds the test gives each page."""

    def __init__(self, answers, seconds):
        self.answers, self.seconds, self.now, self.jobs, self.n = list(answers), list(seconds), 0.0, {}, 0

    def submit(self, first, crops):
        self.n += 1
        time.sleep(0.12)  # long enough for the memory watcher to look once
        self.now += self.seconds.pop(0)
        answer = self.answers.pop(0)
        if answer is not None:
            self.jobs[self.n] = answer
        return self.n, None if answer is not None else None

    def collect_pending(self, wait):
        out = list(self.jobs.items())
        self.jobs.clear()
        if not out:
            self.now += 100.0  # nothing came: time passes
        return out

    def drain(self):
        return []

    def _rss_mb(self):
        return 1234


class MeasureTest(unittest.TestCase):
    def entries(self, wants):
        return [(f"d/{i}", None, {}, w) for i, w in enumerate(wants)]

    def test_seconds_parity_and_memory_of_three_pages_one_of_them_different(self):
        a, b = reading("latin"), reading("deep", "y")
        good, other = (a, b), (b, a)
        answers = [page(good), page(other), page(good)]
        process = FakeProcess(answers, [10.0, 2.0, 4.0])
        got = measure(self.entries([(good,)] * 3), process, clock=lambda: process.now)
        self.assertEqual((got["pages"], got["first_page_seconds"]), (3, 10.0))
        self.assertEqual(
            (got["seconds_per_page_mean"], got["seconds_per_page_p95"], got["seconds_per_page_max"]), (3.0, 2.0, 4.0)
        )
        self.assertEqual(got["parity"]["compared"], 3)
        self.assertEqual(got["parity"]["equal"], 2)
        self.assertEqual(got["parity"]["different_pages"], 1)
        self.assertEqual(got["parity"]["first_differences"][0]["page"], "d/1")
        self.assertEqual(got["reader_rss_peak_mb"], 1234)
        self.assertEqual([p["equal"] for p in got["per_page"]], [True, False, True])

    def test_a_page_the_reader_failed_is_a_difference_and_one_that_never_answers_is_listed_and_not_counted(self):
        a = reading("latin")
        process = FakeProcess([page((a,), failed=True), None, page((a,))], [1.0, 1.0, 1.0])
        got = measure(self.entries([((a,),)] * 3), process, clock=lambda: process.now, timeout=50.0)
        self.assertEqual((got["pages"], got["parity"]["equal"]), (2, 1))
        self.assertEqual(got["parity"]["first_differences"][0]["reader_failed"], True)
        self.assertEqual([p.get("error") for p in got["per_page"]], [None, "no answer", None])
        self.assertEqual(got["seconds_per_page_mean"], 1.0)

    def test_the_p95_is_the_value_at_the_floor_of_95_percent_of_the_ranks(self):
        a = reading("latin")
        process = FakeProcess([page((a,))] * 21, [10.0] + [float(i) for i in range(1, 21)])
        got = measure(self.entries([((a,),)] * 21), process, clock=lambda: process.now)
        self.assertEqual((got["seconds_per_page_p95"], got["seconds_per_page_mean"]), (19.0, 10.5))

    def test_the_limit_stops_the_run_after_that_many_pages(self):
        a = reading("latin")
        process = FakeProcess([page((a,))] * 3, [1.0] * 3)
        got = measure(self.entries([((a,),)] * 3), process, clock=lambda: process.now, limit=2)
        self.assertEqual(got["pages"], 2)


class EntriesTest(unittest.TestCase):
    def test_pages_without_a_pass_1_are_counted_and_left_out(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "1.json"
            path.write_text(json.dumps({"ocr": None}))
            left = {"no_pass_1": 0, "other_units": 0, "no_cells": 0}
            got = list(entries_of([("train", path)], second_reading.thorough(), left))
        self.assertEqual((got, left["no_pass_1"]), ([], 1))


if __name__ == "__main__":
    unittest.main()
