import argparse
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from bench import run_corpus
from bench.corpus import invoice_pages
from extract.engine.batch import BatchReader
from extract.engine.worker import PageTask
from extract.options import Options
from extract.types import PageReading, Word


def record(page=1, ocr=None):
    return {
        "page": page,
        "image": {"width": 100, "height": 100, "dpi": 199.6},
        "truth": {
            "width_pt": 595.0,
            "height_pt": 842.0,
            "words": [{"id": "p1w0", "text": "keep", "box": [0, 0, 1, 1]}],
        },
        "ocr": ocr,
    }


class PagesTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.pages = []
        for n, ocr in enumerate(
            (
                None,
                {"engine": "old", "backend": "openvino", "reading": {}},
                {"engine": "now", "backend": "openvino", "reading": {}},
                {"engine": "now", "backend": "openvino"},
                {"engine": "now", "backend": "onnxruntime", "reading": {}},
            ),
            1,
        ):
            path = self.root / "pages" / "train" / "docA" / f"{n}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(record(n, ocr)), encoding="utf-8")
            self.pages.append(("train", path))

    def tearDown(self):
        self.dir.cleanup()

    def test_pages_without_a_reading_for_this_engine_are_pending(self):
        todo = run_corpus.pending(self.pages, "now")
        self.assertEqual([p.name for _, p in todo], ["1.json", "2.json", "4.json", "5.json"])

    def test_a_changed_engine_rereads_everything(self):
        self.assertEqual(len(run_corpus.pending(self.pages, "newer")), 5)

    def test_pages_of_follows_the_manifest_and_orders_by_document_and_page(self):
        manifest = self.root / "split.json"
        entries = [{"set": "train", "doc_id": "docA", "page": n} for n in (3, 1, 2)] + [
            {"set": "val", "doc_id": "docB", "page": 1}
        ]
        manifest.write_text(json.dumps({"pages": entries}), encoding="utf-8")
        found = run_corpus.pages_of(manifest, ["train"], self.root)
        self.assertEqual([p.name for _, p in found], ["1.json", "2.json", "3.json"])
        self.assertEqual({name for name, _ in found}, {"train"})

    def test_the_test_set_is_listed_once_with_the_unseen_slice_through_the_sealed_reader(self):
        for name, doc in (("test5", "docT"), ("unseen5", "docU")):
            path = self.root / "pages" / name / doc / "1.json"
            path.parent.mkdir(parents=True)
            cv2.imwrite(str(path.with_suffix(".png")), np.full((10, 10, 3), 255, np.uint8))
            path.write_text(json.dumps({**record(1), "set": name, "doc_id": doc}), encoding="utf-8")
        manifest = self.root / "split.json"
        manifest.write_text(
            json.dumps({"pages": [{"set": "unseen5", "doc_id": "docU", "page": 1, "role": "unseen5"}]}),
            encoding="utf-8",
        )
        before = Path.cwd()
        os.chdir(self.root)
        try:
            found = run_corpus.pages_of(manifest, ["test5"], self.root)
        finally:
            os.chdir(before)
        self.assertEqual(sorted((name, p.parent.name) for name, p in found), [("test5", "docT"), ("unseen5", "docU")])

    def test_documents_limits_the_catalogue_documents_of_t5_to_the_first_of_its_list_and_keeps_invoices_and_unseen(
        self,
    ):
        names = (
            ("test5", "docA", 0),
            ("test5", "docB", 1),
            ("test5", "docC", 2),
            ("test5", "inv-t5-0", None),
            ("unseen5", "docU", 130),
        )
        entries = []
        for name, doc, position in names:
            path = self.root / "pages" / name / doc / "1.json"
            path.parent.mkdir(parents=True)
            cv2.imwrite(str(path.with_suffix(".png")), np.full((10, 10, 3), 255, np.uint8))
            path.write_text(json.dumps({**record(1), "set": name, "doc_id": doc}), encoding="utf-8")
            entry = {"set": name, "doc_id": doc, "page": 1, "role": name}
            if position is not None:
                entry["position"] = position
            entries.append(entry)
        manifest = self.root / "split.json"
        manifest.write_text(json.dumps({"pages": entries}), encoding="utf-8")
        before = Path.cwd()
        os.chdir(self.root)
        try:
            found = run_corpus.pages_of(manifest, ["test5"], self.root, documents=2)
        finally:
            os.chdir(before)
        self.assertEqual(sorted(p.parent.name for _, p in found), ["docA", "docB", "docU", "inv-t5-0"])

    def test_writing_a_reading_keeps_the_rest_of_the_record_and_leaves_no_partial_file(self):
        path = self.pages[0][1]
        run_corpus.write_reading(path, {"engine": "now", "reading": {"words": []}})
        stored = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(stored["ocr"]["engine"], "now")
        self.assertEqual(stored["truth"]["words"][0]["text"], "keep")
        self.assertEqual([p.name for p in path.parent.glob("*.part")], [])

    def test_a_page_read_by_another_backend_is_read_again(self):
        todo = run_corpus.pending(self.pages, "now")
        self.assertIn("5.json", [p.name for _, p in todo])

    def test_the_task_carries_the_image_path_page_dpi_and_page_size(self):
        task = run_corpus.task_of(self.pages[2][1])
        self.assertTrue(task.image.endswith("3.png"))
        self.assertEqual((task.number, task.dpi, task.size_pt), (3, 200, (595.0, 842.0)))


def word(char_conf):
    return Word("w", 1, "x", (0, 0, 1, 1), 0.9, char_conf, 0, 0)


def reading(*char_confs):
    words = tuple(word(c) for c in char_confs)
    unaligned = sum(1 for c in char_confs if c is None)
    return PageReading(1, 10, 10, 595.0, 842.0, 200, words, "ok", None, None, 0.9, unaligned)


class ChunkingTest(unittest.TestCase):
    def test_a_document_goes_in_chunks_of_four_pages_in_page_order(self):
        tasks = [PageTask(n, 200, "", f"{n}.png", (1.0, 1.0)) for n in (6, 1, 3, 2, 5, 4, 7)]
        chunks = run_corpus.chunks_of_document(tasks)
        self.assertEqual([[t.number for t in c.pages] for c in chunks], [[1, 2, 3, 4], [5, 6, 7]])
        self.assertEqual([c.id for c in run_corpus.chunks_of_document(tasks, first_id=10)], [10, 11])

    def test_pages_are_grouped_by_the_folder_of_their_document(self):
        todo = [("train", Path("a/1.json")), ("train", Path("b/1.json")), ("train", Path("a/2.json"))]
        self.assertEqual(run_corpus.by_document(todo), [[Path("a/1.json"), Path("a/2.json")], [Path("b/1.json")]])


class MeasuresTest(unittest.TestCase):
    def test_the_unaligned_share_counts_words_without_character_confidences(self):
        readings = [reading((0.9,), None), reading((0.9,), (0.8,))]
        self.assertEqual(run_corpus.unaligned_share(readings), 0.25)
        self.assertEqual(run_corpus.unaligned_share([]), 0.0)

    def test_recycled_workers_are_counted_by_the_reason_they_left(self):
        entries = [{"code": "worker.recycled", "reason": r} for r in ("pages", "rss", "pages")] + [{"code": "x"}]
        self.assertEqual(run_corpus.recycled_by(entries), {"pages": 2, "rss": 1})

    def test_the_backend_of_each_worker_comes_from_the_log(self):
        entries = [{"code": "worker.ready", "pid": 7, "backend": "openvino"}, {"code": "pool.page", "pid": 7}]
        self.assertEqual(run_corpus.backends_of(entries), {7: "openvino"})

    def test_every_run_is_kept_and_o9_is_met_only_if_some_run_met_it(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "endurance.json"
            run_corpus.append_run(path, {"pages": 403, "recycle_pages": 120, "passed": False})
            self.assertFalse(json.loads(path.read_text(encoding="utf-8"))["o9_met"])
            run_corpus.append_run(path, {"pages": 420, "recycle_pages": 50, "passed": True})
            run_corpus.append_run(path, {"pages": 30, "recycle_pages": 50, "passed": False})
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual([r["pages"] for r in data["runs"]], [403, 420, 30])
            self.assertTrue(data["o9_met"])
            self.assertEqual(list(Path(folder).glob("*.part")), [])

    def test_a_file_written_as_a_single_run_is_kept_as_the_first_of_the_list(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "endurance.json"
            path.write_text(json.dumps({"pages": 403, "passed": False}), encoding="utf-8")
            run_corpus.append_run(path, {"pages": 10, "passed": False})
            self.assertEqual([r["pages"] for r in json.loads(path.read_text(encoding="utf-8"))["runs"]], [403, 10])


class InvoicePagesTest(unittest.TestCase):
    def test_every_invoice_page_is_built_once_and_listed_with_its_band(self):
        from reportlab.pdfgen import canvas

        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            for split, name, pages in (("train", "inv-a", 2), ("test", "inv-b", 1)):
                (data / "invoices" / split).mkdir(parents=True)
                sheet = canvas.Canvas(str(data / "invoices" / split / f"{name}.pdf"))
                for _ in range(pages):
                    sheet.drawString(50, 700, "Factura")
                    sheet.showPage()
                sheet.save()
            built = []

            def fake(pdf, number, doc_id, split, data_dir):
                built.append((doc_id, number, split))
                return "light"

            entries = invoice_pages.build(data, build_page=fake)
            self.assertEqual(built, [("inv-a", 1, "train"), ("inv-a", 2, "train"), ("inv-b", 1, "test")])
            self.assertEqual([e["set"] for e in entries], ["train", "train", "test"])
            manifest = invoice_pages.write_manifest(data, entries)
            self.assertEqual(len(json.loads(manifest.read_text(encoding="utf-8"))["pages"]), 3)
            self.assertEqual(list(manifest.parent.glob("*.part")), [])


class ReadPagesTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        self.todo = []
        for doc, count in (("docA", 5), ("docB", 2)):
            for n in range(1, count + 1):
                path = self.root / "pages" / "train" / doc / f"{n}.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(path.with_suffix(".png")), np.full((60, 80, 3), 255, np.uint8))
                path.write_text(json.dumps(record(n)), encoding="utf-8")
                self.todo.append(("train", path))

    def tearDown(self):
        self.dir.cleanup()

    def read(self, todo, write=True):
        reader = BatchReader(
            Options(work_dir=self.root / "run", processes=1, threads=1), factory="tests.fake_engine:load_engine"
        )
        try:
            return run_corpus.read_pages(todo, reader, "fp-1", write=write), reader
        finally:
            reader.close()

    def test_every_page_gets_its_reading_in_its_own_json_and_nothing_else_changes(self):
        (readings, failed), reader = self.read(self.todo)
        self.assertEqual((len(readings), failed), (7, []))
        stored = json.loads(self.todo[0][1].read_text(encoding="utf-8"))
        self.assertEqual(stored["ocr"]["engine"], "fp-1")
        self.assertEqual(stored["ocr"]["backend"], "fake")
        self.assertEqual(len(stored["ocr"]["reading"]["words"]), 30)
        self.assertEqual(stored["truth"]["words"][0]["text"], "keep")
        self.assertEqual([p.name for _, p in self.todo if list(p.parent.glob("*.part"))], [])

    def test_the_pages_of_two_documents_with_the_same_number_keep_their_own_readings(self):
        self.read(self.todo)
        for _, path in self.todo:
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["ocr"]["reading"]["number"], int(path.stem))

    def test_images_are_read_as_they_are_and_leave_no_page_cache(self):
        _, reader = self.read(self.todo)
        self.assertFalse((self.root / "run" / "pages").exists())
        self.assertEqual(sum(1 for e in reader.log.entries if e["code"] == "pool.page"), 7)

    def test_a_page_that_was_read_is_not_pending_for_the_same_engine_and_backend(self):
        self.read(self.todo)
        with mock.patch.object(run_corpus, "BACKEND", "fake"):
            self.assertEqual(run_corpus.pending(self.todo, "fp-1"), [])
            self.assertEqual(len(run_corpus.pending(self.todo, "fp-2")), 7)

    def test_a_run_that_keeps_nothing_reads_every_page_and_changes_no_file(self):
        before = {p: p.read_text(encoding="utf-8") for _, p in self.todo}
        (readings, failed), _ = self.read(self.todo, write=False)
        self.assertEqual((len(readings), failed), (7, []))
        self.assertEqual({p: p.read_text(encoding="utf-8") for _, p in self.todo}, before)

    def test_an_unreadable_image_is_reported_and_the_run_goes_on(self):
        self.todo[1][1].with_suffix(".png").write_bytes(b"not an image")
        (readings, failed), _ = self.read(self.todo)
        self.assertEqual(len(readings), 6)
        self.assertEqual(len(failed), 1)
        self.assertIn("2.json", failed[0])
        self.assertIsNone(json.loads(self.todo[1][1].read_text(encoding="utf-8"))["ocr"])


class ScopeTest(unittest.TestCase):
    def test_a_run_without_a_memory_cap_is_refused(self):
        original = run_corpus.scope_limit_mb
        run_corpus.scope_limit_mb = lambda: None
        try:
            args = argparse.Namespace(unscoped=False, set=["train"])
            with contextlib.redirect_stdout(io.StringIO()) as shown:
                self.assertEqual(run_corpus.run(args), 2)
                run_corpus.scope_limit_mb = lambda: 16000
                self.assertEqual(run_corpus.run(args), 2)
            self.assertIn("memory cap", shown.getvalue())
        finally:
            run_corpus.scope_limit_mb = original


if __name__ == "__main__":
    unittest.main()
