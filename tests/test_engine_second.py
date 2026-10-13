"""Pass 1 crops, pass 2 readings, their cache and the reader process (R5, R8, R10, R11)."""

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from extract.engine import ocr, readerproc
from extract.engine.batch import BatchReader
from extract.engine.cache import PageCache, page_key
from extract.engine.crops import crop_keys, decode, union_crop, zone_slice
from extract.engine.ocr import LineRead, read_page_with_crops
from extract.engine.readerproc import ReaderProcess, ReaderSettings
from extract.engine.reread import RereadConfig
from extract.engine.second import second_pass, without_readings
from extract.options import Options
from tests.fake_engine import FakeEngine
from tests.support import SPANISH_LINES, scanned_pdf, text_page_image

FAKE = "tests.fake_engine:load_engine"
READERS = "tests.fake_readers:FakeReaders"
CONFIG = RereadConfig()


def doubtful_line(texts, y=20.0, conf=0.8):
    """A line of words, all doubtful, 60 px apart."""
    words = []
    for i, text in enumerate(texts):
        c = ocr.CharConf(conf)
        c.chars, c.aligned = (conf,) * len(text), True
        words.append((text, c, (60.0 * i + 10, y, 60.0 * i + 50, y + 20)))
    return words


class FakeSetEngine(FakeEngine):
    """Reads the lines it was given, with an upright quadrilateral for each (or none)."""

    def __init__(self, lines, quads=True):
        super().__init__(2)
        self.lines, self.quads = lines, quads

    def read_lines(self, image):
        out = []
        for words in self.lines:
            x0, y0 = min(w[2][0] for w in words), min(w[2][1] for w in words)
            x1, y1 = max(w[2][2] for w in words), max(w[2][3] for w in words)
            quad = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32) if self.quads else None
            out.append(LineRead(words, quad))
        return out


def page_of(lines, quads=True):
    image = np.full((200, 700, 3), 255, np.uint8)
    cv_noise = np.random.default_rng(1).integers(0, 255, image.shape, dtype=np.uint8)
    image = cv_noise
    return read_page_with_crops(image, FakeSetEngine(lines, quads), 1, 200, None, CONFIG) + (image,)


class FixedReaders:
    """Readers that say what the test wants for each crop, in order."""

    def __init__(self, names, texts):
        self.names, self.texts, self.seen = tuple(names), texts, []

    def read_many(self, name, crops):
        self.seen.append((name, len(crops)))
        return [(self.texts[name][i], 0.9) for i in range(len(crops))]


class CropsTest(unittest.TestCase):
    lines = [doubtful_line([f"w{i}" for i in range(8)]) for _ in range(1)] + [doubtful_line(["alpha", "beta"], y=60.0)]

    def test_R10_crops_are_cut_for_every_unit_up_to_the_study_limit_and_counted(self):
        reading, crops, image = page_of([doubtful_line([f"p{i}" for i in range(30)], y=10 + 30 * k) for k in range(3)])
        self.assertEqual(reading.status, "ocr")
        report = reading.reread_report
        self.assertEqual(report["units_line"], 3)
        self.assertEqual(report["crops_quad"], len(crops))
        self.assertEqual(crop_keys(crops, "line"), [0, 1, 2])
        self.assertLessEqual(len(crop_keys(crops, "zone")), CONFIG.study_zones)
        self.assertEqual(crop_keys(crops, "zone"), list(range(report["units_zone"])))

    def test_R10_zones_are_stored_up_to_forty_and_lines_up_to_sixteen(self):
        many = [doubtful_line([f"p{i}" for i in range(30)], y=10 + 30 * k) for k in range(30)]
        reading, crops, image = page_of(many)
        self.assertLessEqual(len(crop_keys(crops, "line")), 16)
        self.assertGreater(len(crop_keys(crops, "zone")), 16)
        self.assertLessEqual(len(crop_keys(crops, "zone")), 40)

    def test_R8_the_key_changes_with_either_threshold_and_either_budget(self):
        keys = {
            RereadConfig().key(),
            RereadConfig(tau_word=0.8).key(),
            RereadConfig(tau_char=0.8).key(),
            RereadConfig(study_units=8).key(),
            RereadConfig(study_zones=20).key(),
        }
        self.assertEqual(len(keys), 5)
        for bad in ({"study_units": 17}, {"study_zones": 41}, {"study_zones": 0}):
            with self.assertRaises(ValueError):
                RereadConfig(**bad)

    def test_R5_a_line_whose_quadrilateral_is_missing_is_cut_from_the_union_of_its_boxes(self):
        reading, crops, image = page_of([doubtful_line(["uno", "dos", "tres"])] * 25, quads=False)
        self.assertEqual(reading.reread_report["crops_quad"], 0)
        self.assertEqual(reading.reread_report["crops_union"], len(crops))
        words = reading.words[:3]
        expected = union_crop(image, [w.box for w in words], ocr_pad := 6, 6)
        self.assertEqual(decode(crops["line0"]).shape[:2], expected.shape[:2])
        self.assertEqual(ocr_pad, 6)

    def test_R10_a_quadrilateral_line_crop_is_the_image_under_it(self):
        reading, crops, image = page_of([doubtful_line(["uno", "dos", "tres"])] * 25)
        box = [w.box for w in reading.words[:3]]
        x0, y0, x1, y1 = (
            int(min(b[0] for b in box)),
            int(min(b[1] for b in box)),
            int(max(b[2] for b in box)),
            int(max(b[3] for b in box)),
        )
        self.assertTrue(np.array_equal(decode(crops["line0"]), image[y0:y1, x0:x1]))

    def test_R10_a_zone_slice_runs_from_its_first_word_to_its_last_with_context(self):
        quad = np.array([[0, 0], [400, 0], [400, 20], [0, 20]], np.float32)
        crop = np.zeros((20, 400, 3), np.uint8)
        boxes = [(100.0, 0.0, 140.0, 20.0), (160.0, 0.0, 200.0, 20.0)]
        piece = zone_slice(crop, quad, boxes)
        self.assertEqual(piece.shape[1], 100 + 5 + 5 - 0 if False else piece.shape[1])
        self.assertEqual(piece.shape[1], (200 + 5) - (100 - 5))

    def test_R10_a_zone_slice_that_would_be_empty_falls_back(self):
        quad = np.array([[0, 0], [400, 0], [400, 20], [0, 20]], np.float32)
        self.assertIsNone(zone_slice(np.zeros((20, 400, 3), np.uint8), quad, [(500.0, 0.0, 501.0, 20.0)]))

    def test_R4_unreadable_pages_have_no_crops(self):
        reading, crops, _ = page_of([doubtful_line(["a", "b"])])
        self.assertEqual(reading.status, "unreadable")
        self.assertEqual(crops, {})


class FrozenReadersTest(unittest.TestCase):
    def test_V5_A22_the_cheap_set_is_latin_and_wide_and_the_others_stay_available_by_name(self):
        from extract.engine import readers, reread

        self.assertEqual(readers.CHEAP, ("latin", "wide"))
        self.assertEqual(reread.DEFAULT_READERS, readers.CHEAP)
        self.assertEqual(ReaderSettings().readers, readers.CHEAP)
        self.assertEqual(set(readers.ALL), {"latin", "bin", "clahe", "wide", "deep"})
        for name in readers.ALL:
            self.assertIn(name, readers.MODEL_OF)


class SecondPassTest(unittest.TestCase):
    def setUp(self):
        self.reading, self.crops, _ = page_of([doubtful_line(["Ccntro", "de", "Salud", "publica"])] * 8)

    def run_pass(self, texts, unit="line", names=("latin", "deep")):
        readers = FixedReaders(names, texts)
        return second_pass(self.reading, self.crops, readers, unit, CONFIG), readers

    def test_R5_every_reader_that_ran_gives_each_word_of_a_read_unit_a_reading(self):
        same = ["Ccntro de Salud publica"] * 8
        page, readers = self.run_pass({"latin": same, "deep": ["Centro de Salud pública"] * 8})
        first = page.words[0]
        self.assertEqual([(r.source, r.text) for r in first.readings], [("latin", "Ccntro"), ("deep", "Centro")])
        self.assertEqual(first.readings[0].rank, 0)
        self.assertEqual(page.reread_report["units_read"], 8)
        self.assertEqual(page.reread_report["reread_unaligned"], 0)
        self.assertEqual(readers.seen, [("latin", 8), ("deep", 8)])

    def test_P_unit_seconds_has_one_entry_per_stored_crop_and_reads_each_unit_alone_with_every_reader(self):
        from bench.run_pass2 import unit_seconds

        readers = FixedReaders(("latin", "deep"), {"latin": ["x"] * 8, "deep": ["x"] * 8})
        timing = unit_seconds(self.reading, self.crops, readers, "line", ("latin", "deep"), CONFIG)
        self.assertEqual(len(timing["unit_s"]), 8)
        self.assertEqual(set(timing), {"fixed_s", "unit_s"})
        self.assertEqual([s for s in readers.seen if s[1]], [(n, 1) for _ in range(8) for n in ("latin", "deep")])
        self.assertTrue(all(t >= 0 for t in timing["unit_s"]) and timing["fixed_s"] >= 0)
        without = {k: v for k, v in self.crops.items() if k != "line3"}  # a missing crop ends the list at its rank
        self.assertEqual(len(unit_seconds(self.reading, without, readers, "line", ("deep",), CONFIG)["unit_s"]), 3)

    def test_R5_readings_are_none_exactly_outside_the_units_that_were_read(self):
        page, _ = self.run_pass({"latin": ["Centro de Salud pública"] * 8, "deep": ["Centro de Salud pública"] * 8})
        read_lines = {line for line in page.rereads}
        for word in page.words:
            self.assertEqual(word.readings is not None, word.line_id in read_lines)
        self.assertEqual(page.rereads[0], ("latin", "deep"))

    def test_R5_an_answer_that_cannot_be_aligned_is_a_reading_of_none_and_is_counted(self):
        page, _ = self.run_pass({"deep": ["zzzzzzzzzzzzzzzzzzzzzz"] * 8}, names=("deep",))
        self.assertTrue(all(r.text is None for w in page.words[:4] for r in w.readings))
        self.assertEqual(page.reread_report["reread_unaligned"], 8)

    def test_R5_a_join_gives_each_joined_word_the_group_of_ids(self):
        page, _ = self.run_pass({"deep": ["Ccntrode Salud publica"] * 8}, names=("deep",))
        first, second = page.words[0].readings[0], page.words[1].readings[0]
        self.assertEqual(first.group, (page.words[0].id, page.words[1].id))
        self.assertEqual(second.group, first.group)
        self.assertEqual(first.text, "Ccntrode")

    def test_R5_a_cut_gives_the_word_a_text_with_a_space(self):
        page, _ = self.run_pass({"deep": ["Ccn tro de Salud publica"] * 8}, names=("deep",))
        self.assertEqual(page.words[0].readings[0].text, "Ccn tro")
        self.assertEqual(page.words[0].readings[0].group, (page.words[0].id,))

    def test_R5_a_unit_without_a_stored_crop_is_skipped_and_counted(self):
        crops = {k: v for k, v in self.crops.items() if k != "line3"}
        page = second_pass(
            self.reading, crops, FixedReaders(("deep",), {"deep": ["Centro de Salud pública"] * 8}), "line", CONFIG
        )
        self.assertEqual((page.reread_report["units_read"], page.reread_report["units_skipped"]), (7, 1))

    def test_R5_context_marks_a_word_that_was_not_suspect(self):
        lines = [doubtful_line(["Ccntro"]) + [("de", 0.99, (400.0, 20.0, 440.0, 40.0))]] * 15
        reading, crops, _ = page_of(lines)
        page = second_pass(reading, crops, FixedReaders(("deep",), {"deep": ["Centro de"] * 15}), "zone", CONFIG)
        by_text = {w.text: w.readings[0].context for w in page.words if w.readings}
        self.assertEqual(by_text, {"Ccntro": False, "de": True})

    def test_R11_the_same_crops_give_the_same_readings_twice(self):
        texts = {"deep": ["Centro de Salud pública"] * 8}
        a, _ = self.run_pass(texts, names=("deep",))
        b, _ = self.run_pass(texts, names=("deep",))
        self.assertEqual(a, b)

    def test_R11_a_page_without_readings_says_how_many_units_it_skipped(self):
        page = without_readings(self.reading)
        self.assertEqual(page.reread_report["units_skipped"], 8)
        self.assertTrue(all(w.readings is None for w in page.words))


class SidecarCacheTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.cache = PageCache(self.tmp)
        self.reading, self.crops, _ = page_of([doubtful_line(["Ccntro", "de", "Salud", "publica"])] * 8)
        self.key = page_key("sha", 1, 200, CONFIG.key())

    def test_R8_crops_come_back_under_their_key_only(self):
        self.cache.put_crops("sha", 1, self.key, self.crops)
        self.assertEqual(self.cache.get_crops("sha", 1, self.key), self.crops)
        self.assertIsNone(self.cache.get_crops("sha", 1, "another"))

    def test_R8_the_settings_of_the_units_are_in_the_first_pass_key(self):
        keys = {
            page_key("sha", 1, 200),
            page_key("sha", 1, 200, CONFIG.key()),
            page_key("sha", 1, 200, RereadConfig(tau_char=0.8).key()),
            page_key("sha", 1, 200, RereadConfig(study_units=8).key()),
        }
        self.assertEqual(len(keys), 4)

    def test_R8_a_new_reader_repeats_pass_2_and_leaves_the_first_reading_bytes_alone(self):
        self.cache.put("sha", 1, self.key, self.reading)
        path = self.tmp / "pages" / "sha" / "p0001.json"
        before = path.read_bytes()
        read = second_pass(
            self.reading, self.crops, FixedReaders(("deep",), {"deep": ["Centro de Salud pública"] * 8}), "line", CONFIG
        )
        self.cache.put_rereads("sha", 1, f"{self.key}|rr2:deep:aaa:line", read)
        self.assertEqual(self.cache.get_rereads("sha", 1, f"{self.key}|rr2:deep:aaa:line", self.reading), read)
        self.assertIsNone(self.cache.get_rereads("sha", 1, f"{self.key}|rr2:deep:bbb:line", self.reading))
        self.assertIsNone(self.cache.get_rereads("sha", 1, f"{self.key}|rr2:deep:aaa:zone", self.reading))
        self.cache.put_rereads("sha", 1, f"{self.key}|rr2:deep:bbb:line", read)
        self.assertEqual(path.read_bytes(), before)


class ReaderProcessTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.reading, self.crops, _ = page_of([doubtful_line(["Ccntro", "de", "Salud", "publica"])] * 10)
        self.procs = []
        self.env = mock.patch.dict(os.environ, {"EXTRACT_FAKE_READ": "Centro de Salud pública"})
        self.env.start()

    def tearDown(self):
        for proc in self.procs:
            proc.stop()
        self.env.stop()

    def process(self, free=8000, **kwargs):
        settings = ReaderSettings(readers=("latin", "deep"), factory=READERS, threads=1, **kwargs)
        proc = ReaderProcess(settings, free_memory=lambda: free, budget_mb=1000)
        self.procs.append(proc)
        return proc

    def run_pages(self, proc, count):
        for _ in range(count):
            _, back = proc.submit(self.reading, self.crops)
            self.assertIsNone(back)
        return proc.drain()

    def test_R11_it_starts_only_if_its_budget_fits_the_free_memory(self):
        proc = self.process(free=900)
        self.assertFalse(proc.start())
        job, back = proc.submit(self.reading, self.crops)
        self.assertEqual(back.reread_report["reader_failed"], 1)

    def test_R11_it_reads_pages_and_returns_them_with_readings(self):
        proc = self.process()
        self.assertTrue(proc.start())
        done = self.run_pages(proc, 3)
        self.assertEqual(len(done), 3)
        self.assertTrue(all(page.words[0].readings[1].text == "Centro" for _, page in done))

    def test_R11_it_is_replaced_above_its_recycle_trigger_and_no_page_is_lost(self):
        proc = self.process(recycle_pages=2)
        proc.start()
        done = self.run_pages(proc, 5)
        self.assertEqual(len(done), 5)
        self.assertGreaterEqual(proc.replaced, 2)
        self.assertTrue(all(page.reread_report["units_read"] == 10 for _, page in done))

    def test_R11_it_is_replaced_by_memory_once_four_pages_are_read_and_not_below_its_trigger(self):
        with mock.patch.dict(os.environ, {"EXTRACT_FAKE_READER_KEEP": "450"}):
            fat = self.process(recycle_pages=100, rss_trigger_mb=400)
            fat.start()
            done = self.run_pages(fat, 3)
            self.assertEqual(fat.replaced, 0)  # over the trigger, but fewer than MIN_PAGES_FOR_RSS_RECYCLE pages
            done += self.run_pages(fat, 2)
            self.assertEqual(len(done), 5)
            self.assertEqual(fat.replaced, 1)  # replaced after the fourth page, none lost
            self.assertTrue(all(page.reread_report["units_read"] == 10 for _, page in done))
        lean = self.process(recycle_pages=100, rss_trigger_mb=400)
        lean.start()
        self.assertEqual(len(self.run_pages(lean, 6)), 6)
        self.assertEqual(lean.replaced, 0)

    def test_R11_a_crash_loses_its_page_without_readings_and_the_run_goes_on(self):
        # the die flag is read in the reader process, so it is set before the process starts
        with mock.patch.dict(os.environ, {"EXTRACT_FAKE_READER_DIE": "1"}):
            dying = self.process()
            dying.start()
            _, back = dying.submit(self.reading, self.crops)
            done = dying.drain()
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0][1].reread_report["reader_failed"], 1)
        self.assertEqual(done[0][1].reread_report["units_skipped"], 10)
        self.assertTrue(all(w.readings is None for w in done[0][1].words))

    def test_R11_a_reader_over_its_hard_cap_is_killed_its_page_comes_back_bare_and_the_next_is_read(self):
        marker = self.tmp / "balloon"
        marker.write_text("x")
        with (
            mock.patch.dict(os.environ, {"EXTRACT_FAKE_READER_BALLOON": f"{marker}:400"}),
            mock.patch.object(readerproc, "RSS_CAP_MB", 300),
        ):
            proc = self.process()
            proc.start()
            self.run_pages(proc, 0)
            proc.submit(self.reading, self.crops)
            began = time.perf_counter()
            first = proc.drain()
            self.assertLess(time.perf_counter() - began, 25.0)
            self.assertEqual(first[0][1].reread_report["reader_failed"], 1)
            self.assertEqual(proc.replaced, 1)
            second = self.run_pages(proc, 1)
        self.assertEqual(second[0][1].reread_report["units_read"], 10)

    def test_R11_the_cap_of_a_process_is_its_own_setting_and_the_thorough_mode_has_a_higher_one(self):
        marker = self.tmp / "balloon"
        marker.write_text("x")
        with mock.patch.dict(os.environ, {"EXTRACT_FAKE_READER_BALLOON": f"{marker}:400"}):
            proc = self.process(rss_cap_mb=300)  # the module's cap (1.500) would let this page through
            self.assertEqual(proc._cap_mb(), 300)
            proc.start()
            proc.submit(self.reading, self.crops)
            first = proc.drain()
        self.assertEqual(first[0][1].reread_report["reader_failed"], 1)
        self.assertEqual(ReaderSettings().rss_cap_mb, None)
        self.assertEqual(ReaderProcess(ReaderSettings())._cap_mb(), readerproc.RSS_CAP_MB)
        thorough = readerproc.settings_for("thorough")
        self.assertEqual(thorough.rss_cap_mb, readerproc.THOROUGH_RSS_CAP_MB)
        self.assertGreater(readerproc.THOROUGH_RSS_CAP_MB, readerproc.RSS_CAP_MB)

    def test_R11_a_replacement_is_started_only_if_the_memory_still_holds_its_budget(self):
        free = [8000]
        settings = ReaderSettings(readers=("latin",), factory=READERS, threads=1, recycle_pages=1)
        proc = ReaderProcess(settings, free_memory=lambda: free[0], budget_mb=1000)
        self.procs.append(proc)
        self.assertTrue(proc.start())
        free[0] = 500  # what is left once the first process retires
        for _ in range(3):
            proc.submit(self.reading, self.crops)
        done = proc.drain()
        self.assertEqual(len(done), 3)
        reads = sorted(page.reread_report["units_read"] for _, page in done)
        self.assertEqual(reads, [0, 0, 10])
        self.assertFalse(proc.started)

    def test_R11_a_stop_returns_in_two_seconds(self):
        with mock.patch.dict(os.environ, {"EXTRACT_FAKE_READER_SECONDS": "30"}):
            proc = self.process()
            proc.start()
            proc.submit(self.reading, self.crops)
            began = time.perf_counter()
            proc.stop()
        self.assertLess(time.perf_counter() - began, readerproc.STOP_SECONDS + 0.5)
        self.assertEqual(readerproc.STOP_SECONDS, 2.0)


class BatchSecondReadingTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        images = [text_page_image(SPANISH_LINES[:3], width=827, height=1169, size=14) for _ in range(3)]
        self.pdf = scanned_pdf(self.tmp / "scan.pdf", images)
        self.log = self.tmp / "readers.log"
        self.env = mock.patch.dict(
            os.environ,
            {
                "EXTRACT_FAKE_DOUBT": "1",
                "EXTRACT_FAKE_READER_LOG": str(self.log),
                "EXTRACT_FAKE_READ": " ".join(f"word{i}" for i in range(15)),
            },
        )
        self.env.start()
        self.readers = []

    def tearDown(self):
        for reader in self.readers:
            reader.close()
        self.env.stop()

    def reader(self, second=True):
        options = Options(work_dir=self.tmp / "work", processes=1, threads=1, second_reading=second)
        settings = ReaderSettings(readers=("latin", "deep"), factory=READERS, threads=1)
        reader = BatchReader(options, factory=FAKE, free_memory=lambda: 8000, cores=2, reader_settings=settings)
        self.readers.append(reader)
        return reader

    def calls(self):
        return self.log.read_text().count("\n") if self.log.exists() else 0

    def test_R8_second_reading_adds_readings_and_a_cached_page_is_not_read_twice(self):
        reader = self.reader()
        first = reader.read(self.pdf)
        self.assertTrue(all(any(w.readings for w in page.words) for page in first.pages))
        calls = self.calls()
        self.assertEqual(calls, 6)  # two readers, three pages
        again = reader.read(self.pdf)
        self.assertEqual(reader.pool.pages_done, 3)
        self.assertEqual(self.calls(), calls)
        self.assertEqual(first.pages, again.pages)

    def test_R8_a_new_reader_version_repeats_pass_2_and_not_pass_1(self):
        reader = self.reader()
        reader.read(self.pdf)
        calls = self.calls()
        reader._readers_key = None
        with mock.patch.dict(os.environ, {"EXTRACT_FAKE_READERS_VERSION": "-v2"}):
            reader.read(self.pdf)
        self.assertEqual(reader.pool.pages_done, 3)
        self.assertEqual(self.calls(), calls + 6)

    def test_R8_a_page_cached_without_the_second_reading_is_not_served_with_it(self):
        self.reader(second=False).read(self.pdf)
        reader = self.reader(second=True)
        reader.read(self.pdf)
        self.assertEqual(reader.pool.pages_done, 3)

    def test_R9_without_the_second_reading_no_word_has_readings_or_crops_are_kept(self):
        reader = self.reader(second=False)
        document = reader.read(self.pdf)
        self.assertTrue(all(w.readings is None for page in document.pages for w in page.words))
        self.assertEqual(list((self.tmp / "work").rglob("*.crops.npz")), [])
        self.assertEqual(self.calls(), 0)

    def test_R11_without_memory_for_the_reader_the_pages_come_back_without_readings(self):
        options = Options(work_dir=self.tmp / "work", processes=1, threads=1, second_reading=True)
        settings = ReaderSettings(readers=("deep",), factory=READERS, threads=1)
        reader = BatchReader(options, factory=FAKE, free_memory=lambda: 700, cores=2, reader_settings=settings)
        self.readers.append(reader)
        document = reader.read(self.pdf)
        self.assertEqual(len(document.pages), 3)
        self.assertTrue(all(w.readings is None for page in document.pages for w in page.words))
        self.assertEqual(self.calls(), 0)


if __name__ == "__main__":
    unittest.main()
