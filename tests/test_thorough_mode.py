"""The thorough second reading (16 lines with latin+wide, 40 zones with deep, its own guard model) is an
explicit option that exists in the code, the reader process, the cache key and the guard."""

import functools
import json
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import jsonschema

from extract import api, second_reading
from extract.engine import readerproc
from extract.engine.batch import BatchReader
from extract.engine.readerproc import THOROUGH_UNIT, ReaderProcess, ReaderSettings, settings_for
from extract.engine.reread import RereadConfig
from extract.engine.second import second_pass, second_pass_cells
from extract.errors import EngineLoadError
from extract.guard import readings
from extract.guard.checker import Guard
from extract.options import Options
from extract.outputs import jsondoc
from tests.fake_guard import FakeGuard
from tests.support import SPANISH_LINES, SetLexicon, scanned_pdf, text_page_image
from tests.test_engine_second import CONFIG, FAKE, READERS, FixedReaders, doubtful_line, page_of

PASSES = second_reading.thorough_passes()
TEXT = "Centro de Salud pública"


class OptionTest(unittest.TestCase):
    def test_the_option_takes_false_true_and_thorough_and_nothing_else(self):
        for good in (False, True, "thorough"):
            self.assertEqual(Options(second_reading=good).second_reading, good)
        for bad in (1, 0, "yes", "Thorough", "full", None, ["thorough"]):
            with self.assertRaises(ValueError, msg=repr(bad)):
                Options(second_reading=bad)

    def test_only_the_word_thorough_is_the_thorough_mode(self):
        self.assertTrue(Options(second_reading="thorough").thorough)
        self.assertFalse(Options(second_reading=True).thorough)
        self.assertFalse(Options().thorough)

    def test_the_file_says_what_the_mode_reads_and_a_bad_block_is_refused(self):
        self.assertEqual(PASSES, (("line", ("latin", "wide")), ("zone", ("deep",))))
        self.assertEqual(second_reading.thorough_readers(), ("latin", "wide", "deep"))
        block = second_reading.thorough()
        for key, value in (("lines", 17), ("zones", 0), ("readers_lines", ["nope"]), ("reader_zones", "x")):
            with self.assertRaises(ValueError, msg=key):
                second_reading.validate_thorough({**block, key: value})
        with self.assertRaises(ValueError):
            second_reading.validate_thorough({k: v for k, v in block.items() if k != "zones"})


class WiringTest(unittest.TestCase):
    def test_a_batch_reader_of_each_mode_gets_the_settings_of_that_mode(self):
        for mode, passes, unit in (("thorough", PASSES, THOROUGH_UNIT), (True, None, "line"), (False, None, "line")):
            with tempfile.TemporaryDirectory() as tmp:
                batch = BatchReader(Options(work_dir=Path(tmp), second_reading=mode), factory=FAKE)
                self.assertEqual((batch.reader_settings.passes, batch.reader_settings.unit), (passes, unit), mode)

    def test_a_word_that_is_not_thorough_is_not_the_mode(self):
        for other in ("Thorough", "thorough ", "full", "", True, None):
            self.assertFalse(second_reading.is_thorough(other), repr(other))
        self.assertEqual(settings_for("full"), ReaderSettings())

    def test_the_seconds_and_the_memory_of_the_thorough_mode_are_the_measured_ones(self):
        """The panel shows them: the mean and the p95 of the clean run (60 pilot pages, quiet machine, no cap) and the
        highest memory the reader process reached in any run of `bench/results/thorough-run*.json`."""
        results = Path(__file__).resolve().parents[1] / "bench" / "results"
        clean = json.loads((results / "thorough-run-nocap.json").read_text())
        peaks = [json.loads(f.read_text())["reader_rss_peak_mb"] for f in results.glob("thorough-run*.json")]
        block = second_reading.thorough()
        self.assertEqual(clean["parity"]["compared"], 60)
        self.assertEqual(block["seconds_per_page"], round(clean["seconds_per_page_mean"], 2))
        self.assertEqual(block["seconds_p95"], round(clean["seconds_per_page_p95"], 2))
        self.assertEqual(block["reader_ram_mb"], max(peaks))
        self.assertLessEqual(block["reader_ram_mb"], readerproc.THOROUGH_RSS_CAP_MB)

    def test_the_gain_the_panel_shows_is_the_sealed_test_and_the_dev_figure_is_the_paired_run(self):
        """`fixes_more_pct_by_language` is T5 (`read-test-v5.json`): the net per 100 wrong words of the thorough mode
        minus the default model on all the pages of the test, paired by document; the Spanish development figure (net fixes of
        L+Z over none, out of fold) is `cells-paired.json`'s and stays as `net_vs_none_es_dev`."""
        results = Path(__file__).resolve().parents[1] / "bench" / "results"
        block = second_reading.thorough()
        t5 = json.loads((results / "read-test-v5.json").read_text())["by_language"]
        for language in ("es", "en"):
            paired = t5[language]["all"]["paired_thorough_minus_default"]["more"]["net_fixed_per_100_wrong_difference"]
            self.assertEqual(block["fixes_more_pct_by_language"][language], round(paired["value"], 2), language)
        self.assertEqual(block["fixes_more_pct_vs_default"], block["fixes_more_pct_by_language"]["es"])
        dev = results / "cells-paired.json"
        if not dev.is_file():
            self.skipTest("cells-paired.json is not in this tree")
        row = json.loads(dev.read_text())["paired"]["lightgbm"]["latin+wide+deep - none"]
        self.assertEqual(block["net_vs_none_es_dev"], round(row["more_net"]["difference"]))

    def test_the_cost_the_panel_reads_has_the_measured_fields_and_a_block_without_them_is_refused(self):
        cost = second_reading.thorough_cost()
        block = second_reading.thorough()
        self.assertEqual(cost["reader_ram_mb"], block["reader_ram_mb"])
        self.assertEqual(cost["seconds_per_page"], block["seconds_per_page"])
        self.assertEqual(cost["seconds_p95"], block["seconds_p95"])
        self.assertEqual(cost["fixes_more_pct_by_language"], block["fixes_more_pct_by_language"])
        self.assertEqual(cost["fixes_more_pct"], block["fixes_more_pct_vs_default"])
        self.assertEqual(
            set(cost),
            {"seconds_per_page", "seconds_p95", "reader_ram_mb", "fixes_more_pct", "fixes_more_pct_by_language"},
        )
        block = second_reading.thorough()
        for key in ("seconds_p95", "reader_ram_mb", "fixes_more_pct_by_language"):
            with self.assertRaises(ValueError, msg=key):
                second_reading.validate_thorough({k: v for k, v in block.items() if k != key})
        for bad in ({}, {"es": -1}, {"es": "2"}):
            with self.assertRaises(ValueError, msg=repr(bad)):
                second_reading.validate_thorough({**block, "fixes_more_pct_by_language": bad})


class PassesTest(unittest.TestCase):
    def setUp(self):
        self.reading, self.crops, _ = page_of([doubtful_line(["Ccntro", "de", "Salud", "publica"])] * 8)
        self.texts = {n: [TEXT] * 8 for n in ("latin", "wide", "deep")}

    def run_cells(self):
        readers = FixedReaders(("latin", "wide", "deep"), self.texts)
        return second_pass_cells(self.reading, self.crops, readers, PASSES, CONFIG), readers

    def test_lines_are_read_by_the_cheap_readers_and_zones_by_the_deep_one_and_a_word_keeps_both(self):
        page, readers = self.run_cells()
        self.assertEqual([r[0] for r in readers.seen], ["latin", "wide", "deep"])
        self.assertEqual(
            [(r.source, r.text) for r in page.words[0].readings][:2], [("latin", "Centro"), ("wide", "Centro")]
        )
        self.assertEqual({r.source for r in page.words[0].readings}, {"latin", "wide", "deep"})
        self.assertEqual(page.rereads[0], ("latin", "wide", "deep"))

    def test_the_counts_add_up_and_the_first_reading_is_not_touched(self):
        page, _ = self.run_cells()
        lines = second_pass(self.reading, self.crops, FixedReaders(("latin", "wide"), self.texts), "line", CONFIG)
        zones = second_pass(self.reading, self.crops, FixedReaders(("deep",), self.texts), "zone", CONFIG)
        for key in ("units_read", "units_skipped", "reread_unaligned"):
            self.assertEqual(page.reread_report[key], lines.reread_report[key] + zones.reread_report[key], key)
        self.assertEqual(page.reread_report["reread_words"], sum(1 for w in page.words if w.readings))
        self.assertEqual(
            [(w.text, w.box, w.conf) for w in page.words], [(w.text, w.box, w.conf) for w in self.reading.words]
        )

    def test_the_result_equals_the_readings_of_the_two_cells_put_together(self):
        page, _ = self.run_cells()
        lines = second_pass(self.reading, self.crops, FixedReaders(("latin", "wide"), self.texts), "line", CONFIG)
        zones = second_pass(self.reading, self.crops, FixedReaders(("deep",), self.texts), "zone", CONFIG)
        for merged, a, b in zip(page.words, lines.words, zones.words, strict=True):
            self.assertEqual(merged.readings, ((a.readings or ()) + (b.readings or ())) or None)

    def test_the_words_with_readings_are_counted_over_both_passes(self):
        sure = [(t, c, b) for (t, c, b) in doubtful_line([f"w{i}" for i in range(10)], conf=0.99)]
        words = sure[:-1] + doubtful_line([f"w{i}" for i in range(10)])[-1:]
        reading, crops, _ = page_of([words] * 8)
        texts = {n: [" ".join(f"w{i}" for i in range(10))] * 8 for n in ("latin", "wide", "deep")}
        page = second_pass_cells(reading, crops, FixedReaders(("latin", "wide", "deep"), texts), PASSES, CONFIG)
        lines = second_pass(reading, crops, FixedReaders(("latin", "wide"), texts), "line", CONFIG)
        self.assertEqual(page.reread_report["reread_words"], 80)
        self.assertEqual(lines.reread_report["reread_words"], 80)
        self.assertLess(sum(1 for w in page.words if any(r.source == "deep" for r in w.readings)), 80)

    def test_one_pass_alone_still_replaces_as_before(self):
        once = second_pass(self.reading, self.crops, FixedReaders(("deep",), self.texts), "zone", CONFIG)
        twice = second_pass(once, self.crops, FixedReaders(("deep",), self.texts), "zone", CONFIG)
        self.assertEqual(once.words, twice.words)
        self.assertEqual(once.reread_report, twice.reread_report)


class SettingsTest(unittest.TestCase):
    def test_thorough_settings_carry_the_passes_the_readers_and_the_units_of_the_file(self):
        settings = settings_for("thorough", factory=READERS)
        block = second_reading.thorough()
        self.assertEqual(
            (settings.unit, settings.passes, settings.readers), (THOROUGH_UNIT, PASSES, ("latin", "wide", "deep"))
        )
        self.assertEqual(settings.config, RereadConfig(study_units=block["lines"], study_zones=block["zones"]))
        self.assertEqual(settings.factory, READERS)

    def test_the_other_modes_keep_the_packaged_cell(self):
        for mode in (False, True):
            self.assertEqual(settings_for(mode), ReaderSettings())
        self.assertIsNone(settings_for(True).passes)


class BatchThoroughTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        images = [text_page_image(SPANISH_LINES[:3], width=827, height=1169, size=14) for _ in range(2)]
        self.pdf = scanned_pdf(self.tmp / "scan.pdf", images)
        self.log = self.tmp / "readers.log"
        self.env = mock.patch.dict(
            os.environ,
            {"EXTRACT_FAKE_DOUBT": "1", "EXTRACT_FAKE_READER_LOG": str(self.log), "EXTRACT_FAKE_READ": TEXT},
        )
        self.env.start()
        self.readers = []

    def tearDown(self):
        for reader in self.readers:
            reader.close()
        self.env.stop()

    def reader(self, mode):
        options = Options(work_dir=self.tmp / "work", processes=1, threads=1, second_reading=mode)
        settings = settings_for(mode, factory=READERS, threads=1)
        reader = BatchReader(options, factory=FAKE, free_memory=lambda: 8000, cores=2, reader_settings=settings)
        self.readers.append(reader)
        return reader

    def test_a_document_read_in_the_thorough_mode_has_the_readings_of_the_three_readers(self):
        document = self.reader("thorough").read(self.pdf)
        sources = {r.source for p in document.pages for w in p.words for r in w.readings or ()}
        self.assertEqual(sources, {"latin", "wide", "deep"})
        names = [line.split()[0] for line in self.log.read_text().splitlines()]
        self.assertEqual(sorted(set(names)), ["deep", "latin", "wide"])
        self.assertTrue(all(p.reread_report["units_read"] > 0 for p in document.pages))

    def test_the_cache_key_of_pass_2_names_the_mode_and_a_cell_page_is_not_served_for_it(self):
        thorough, cell = self.reader("thorough"), self.reader(True)
        self.assertIn(THOROUGH_UNIT, thorough._rr2_key("k"))
        self.assertNotEqual(thorough._rr2_key("k"), cell._rr2_key("k"))
        cell.read(self.pdf)
        calls = self.log.read_text().count("\n")
        thorough.read(self.pdf)
        self.assertEqual(thorough.pool.pages_done, 0)  # pass 1 is shared: the units and their crops are the same
        self.assertGreater(self.log.read_text().count("\n"), calls)  # pass 2 ran again, with its own key


class GuardTest(unittest.TestCase):
    def stub(self, spec, thorough):
        model = SimpleNamespace(fix={"readings": spec})
        return Guard(model, thorough=thorough)

    def page(self):
        reading, crops, _ = page_of([doubtful_line(["Ccntro", "de", "Salud", "publica"])] * 8)
        texts = {n: [TEXT] * 8 for n in ("latin", "wide", "deep")}
        page = second_pass_cells(reading, crops, FixedReaders(("latin", "wide", "deep"), texts), PASSES, CONFIG)
        return reading, page

    def test_the_thorough_cell_is_the_lines_and_the_zones_of_all_three_readers_with_no_cap(self):
        spec = readings.product_spec(True)
        self.assertEqual(
            (spec.unit, spec.readers, spec.cap, spec.cell), (THOROUGH_UNIT, ("deep", "latin", "wide"), None, "LZ")
        )
        self.assertNotEqual(readings.product_spec(False).unit, THOROUGH_UNIT)

    def test_a_thorough_guard_with_a_model_trained_on_the_thorough_cell_sees_every_reading(self):
        _, page = self.page()
        guard = self.stub(readings.product_spec(True).as_meta(), True)
        words = guard._words_as_trained(page)
        self.assertEqual(guard.last_warnings, [])
        self.assertEqual({r.source for r in words[0].readings}, {"latin", "wide", "deep"})

    def test_a_thorough_page_judged_by_a_model_of_another_cell_drops_the_readings_and_says_so(self):
        _, page = self.page()
        other = readings.ReadingSpec("line", ("latin", "wide"), 16).as_meta()
        guard = self.stub(other, True)
        words = guard._words_as_trained(page)
        self.assertTrue(all(w.readings is None for w in words))
        self.assertEqual(guard.last_warnings[0]["key"], "guard.warning.readings_differ")
        self.assertEqual(guard.last_warnings[0]["product"], "line+zone deep+latin+wide")

    def test_the_cell_guard_still_judges_the_cell_readings_after_this_change(self):
        reading, crops, _ = page_of([doubtful_line(["Ccntro", "de", "Salud", "publica"])] * 8)
        cell = second_pass(
            reading, crops, FixedReaders(("latin", "wide"), {n: [TEXT] * 8 for n in ("latin", "wide")}), "line", CONFIG
        )
        guard = self.stub(readings.product_spec(False).as_meta(), False)
        self.assertEqual(guard.last_warnings, [])
        self.assertTrue(guard._words_as_trained(cell)[0].readings)
        self.assertEqual(guard.last_warnings, [])


class ThoroughModelFileTest(unittest.TestCase):
    def test_without_its_model_file_the_mode_stops_with_a_clear_error_and_does_not_use_the_other_model(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.dict(os.environ, {second_reading.THOROUGH_MODEL_ENV: str(Path(tmp) / "none.model")}),
        ):
            with self.assertRaises(EngineLoadError) as caught:
                api.Reader(Options(second_reading="thorough"))._prepare()
        self.assertIn("thorough", str(caught.exception))
        self.assertIn("none.model", str(caught.exception))

    def test_a_file_that_is_not_a_model_is_refused_with_the_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bad.model"
            path.write_text("not a model")
            with mock.patch.dict(os.environ, {second_reading.THOROUGH_MODEL_ENV: str(path)}):
                with self.assertRaises(EngineLoadError) as caught:
                    api._thorough_guard()
        self.assertIn("bad.model", str(caught.exception))

    def test_a_model_not_trained_on_the_thorough_readings_is_refused(self):
        for spec in (None, readings.ReadingSpec("line", ("latin", "wide"), 16).as_meta()):
            guard = Guard(SimpleNamespace(fix={"readings": spec}), thorough=True)
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "m.model"
                path.write_text("x")
                with (
                    mock.patch.dict(os.environ, {second_reading.THOROUGH_MODEL_ENV: str(path)}),
                    mock.patch.object(Guard, "load", return_value=guard),
                ):
                    with self.assertRaises(EngineLoadError) as caught:
                        api._thorough_guard()
            self.assertIn("not trained on the thorough readings", str(caught.exception))

    def test_a_model_trained_on_the_thorough_cell_is_accepted(self):
        guard = Guard(SimpleNamespace(fix={"readings": readings.product_spec(True).as_meta()}), thorough=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.model"
            path.write_text("x")
            with (
                mock.patch.dict(os.environ, {second_reading.THOROUGH_MODEL_ENV: str(path)}),
                mock.patch.object(Guard, "load", return_value=guard) as load,
            ):
                self.assertIs(api._thorough_guard(), guard)
        self.assertTrue(load.call_args.kwargs["thorough"])


class BudgetTest(unittest.TestCase):
    def test_the_thorough_process_does_not_start_where_its_cap_does_not_fit_and_the_default_one_does(self):
        events = []
        log = lambda code, **params: events.append(code)  # noqa: E731
        for mode, started in ((True, True), ("thorough", False)):
            settings = settings_for(mode, factory=READERS, threads=1)
            process = ReaderProcess(settings, free_memory=lambda: 1200, log=log)
            try:
                self.assertEqual(process.start(), started, mode)
            finally:
                process.stop()
        self.assertEqual(events.count("reader.skipped"), 1)
        self.assertEqual(ReaderProcess(settings_for("thorough")).budget_mb, readerproc.THOROUGH_BUDGET_MB)
        self.assertEqual(ReaderProcess(settings_for(True)).budget_mb, readerproc.BUDGET_MB["openvino"])
        self.assertEqual(readerproc.THOROUGH_RSS_CAP_MB, 1600)  # the memory limit of the thorough mode
        self.assertGreaterEqual(readerproc.THOROUGH_BUDGET_MB, readerproc.THOROUGH_RSS_CAP_MB)


class FailedPagesTest(unittest.TestCase):
    """The pages that did not get their second reading are counted in the document, the API and the JSON."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        scan = text_page_image(SPANISH_LINES, width=827, height=1169, size=14)
        self.pdf = scanned_pdf(self.tmp / "scan.pdf", [scan, scan])
        patch = mock.patch.object(api.Reader, "_engine_factory", FAKE)
        patch.start()
        self.addCleanup(patch.stop)

    def read(self, second):
        options = Options(work_dir=self.tmp / "work", processes=1, threads=1, second_reading=second)
        lexicon = SetLexicon({w.lower().strip(".,:") for line in SPANISH_LINES for w in line.split() if len(w) > 2})
        reader = api.Reader(options, guard=FakeGuard(), lexicon=lexicon)
        self.addCleanup(reader.close)
        return reader.read(self.pdf)

    def test_a_document_whose_reader_process_could_not_start_lists_its_pages_in_the_result_and_the_json(self):
        fake = functools.partial(BatchReader, reader_settings=settings_for(True, factory=READERS, threads=1))
        with mock.patch.dict(readerproc.BUDGET_MB, {"openvino": 10**7}), mock.patch.object(api, "BatchReader", fake):
            document = self.read(True)
        self.assertEqual(document.fixes["second_reading_failed_pages"], [1, 2])
        raw = jsondoc.to_dict(document._data)
        self.assertEqual(raw["options"]["second_reading_failed_pages"], [1, 2])
        self.assertEqual(raw["fixes"]["second_reading_failed_pages"], [1, 2])
        jsonschema.validate(
            raw, json.loads((Path(jsondoc.__file__).parent.parent / "schema" / "document.schema.json").read_text())
        )

    def test_without_the_second_reading_no_page_is_listed(self):
        document = self.read(False)
        self.assertEqual(document.fixes["second_reading_failed_pages"], [])
        self.assertNotIn("second_reading_failed_pages", jsondoc.to_dict(document._data)["options"])


if __name__ == "__main__":
    unittest.main()
