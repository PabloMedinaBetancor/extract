import json
import re
import tempfile
import unittest
from pathlib import Path

from bench.corpus import sealed


class SealedTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tmp = Path(folder.name)
        for doc in ("cA", "cB"):
            page = self.tmp / "pages" / "test5" / doc
            page.mkdir(parents=True)
            (page / "1.json").write_text(json.dumps({"doc_id": doc, "truth": {"words": []}}), encoding="utf-8")
        for doc in ("cA", "cB"):
            (self.tmp / "pages" / "test5" / doc / "1.png").write_bytes(b"png " + doc.encode())
        (self.tmp / "pages" / "unseen5" / "cU").mkdir(parents=True)
        (self.tmp / "pages" / "unseen5" / "cU" / "1.json").write_text(json.dumps({"doc_id": "cU"}), encoding="utf-8")
        (self.tmp / "pages" / "unseen5" / "cU" / "1.png").write_bytes(b"png cU")
        (self.tmp / "pages" / "val" / "cC").mkdir(parents=True)
        (self.tmp / "pages" / "val" / "cC" / "1.json").write_text("{}", encoding="utf-8")
        self.reads = self.tmp / "results" / "test_reads_v5.json"

    def test_G4_every_read_is_logged_and_only_test_pages_are_returned(self):
        pages = sealed.read_test_pages(self.tmp, "ocr", self.reads)
        self.assertEqual([record["doc_id"] for _, record in pages], ["cA", "cB", "cU"])
        sealed.read_test_pages(self.tmp, "ocr", self.reads)
        lines = sealed.load_reads(self.reads)
        self.assertEqual([line["purpose"] for line in lines], ["ocr", "ocr"])
        self.assertEqual(lines[0]["pages"], 3)
        self.assertEqual(sealed.metric_reads(self.reads), 0)

    def test_G17_the_pool_recall_loader_refuses_the_sealed_sets(self):
        from bench.guard import pool_recall

        with tempfile.TemporaryDirectory() as folder:
            for name in sealed.SEALED_SETS:
                with self.assertRaises(sealed.SealedTestError):
                    pool_recall.load_cases(Path(folder), [name])

    def test_G4_the_fingerprint_follows_the_images_and_the_truth_but_not_the_ocr_reading(self):
        before = sealed.fingerprint(sealed.read_test_pages(self.tmp, "ocr", self.reads))
        first = self.tmp / "pages" / "test5" / "cA" / "1.json"
        record = json.loads(first.read_text(encoding="utf-8"))
        first.write_text(json.dumps({**record, "ocr": {"words": []}}), encoding="utf-8")
        self.assertEqual(sealed.fingerprint(sealed.read_test_pages(self.tmp, "ocr", self.reads)), before)
        (self.tmp / "pages" / "test5" / "cA" / "1.png").write_bytes(b"changed")
        self.assertNotEqual(sealed.fingerprint(sealed.read_test_pages(self.tmp, "ocr", self.reads)), before)
        (self.tmp / "pages" / "test5" / "cA" / "1.png").write_bytes(b"png cA")
        first.write_text(json.dumps({**record, "truth": {"words": [{"text": "another"}]}}), encoding="utf-8")
        self.assertNotEqual(sealed.fingerprint(sealed.read_test_pages(self.tmp, "ocr", self.reads)), before)

    def test_G4_the_parts_follow_the_images_and_the_truth_and_report_the_patches_apart(self):
        before = sealed.fingerprint_parts(sealed.read_test_pages(self.tmp, "ocr", self.reads))
        first = self.tmp / "pages" / "test5" / "cA" / "1.json"
        record = json.loads(first.read_text(encoding="utf-8"))
        first.write_text(json.dumps({**record, "ocr": {"words": []}, "ungrounded": [[1, 2, 3, 4]]}), encoding="utf-8")
        after = sealed.fingerprint_parts(sealed.read_test_pages(self.tmp, "ocr", self.reads))
        self.assertEqual((after["images"], after["truth"]), (before["images"], before["truth"]))
        self.assertNotEqual(after["ungrounded"], before["ungrounded"])
        first.write_text(json.dumps({**record, "truth": {"words": [{"text": "another"}]}}), encoding="utf-8")
        changed = sealed.fingerprint_parts(sealed.read_test_pages(self.tmp, "ocr", self.reads))
        self.assertNotEqual(changed["truth"], before["truth"])

    def test_G4_a_freeze_record_keeps_the_hashes_and_the_note_and_counts_as_an_ocr_read_only(self):
        out = self.tmp / "results" / "freeze.json"
        sealed.freeze_record(self.tmp, out, "before the patches were redone", self.reads)
        sealed.freeze_record(self.tmp, out, "after", self.reads)
        records = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual([r["note"] for r in records], ["before the patches were redone", "after"])
        self.assertEqual(records[0]["pages"], 3)
        self.assertEqual(sealed.metric_reads(self.reads), 0)

    def test_G4_only_the_sealed_reader_and_the_corpus_runner_name_the_sealed_pages(self):
        root = Path(__file__).resolve().parents[1]
        allowed = {
            "bench/corpus/sealed.py",
            "bench/run_corpus.py",
            "tests/test_corpus_sealed.py",
            "tests/test_build_en.py",
            "bench/read_test_v5.py",
            "tests/test_read_test_v5.py",
        }
        pattern = re.compile(r"""pages["']?\s*[/,]\s*["']?(test5|unseen5)\b|SEALED_SETS""")
        for folder in ("bench", "src", "scripts", "tests"):
            for path in (root / folder).rglob("*.py"):
                name = str(path.relative_to(root))
                if name not in allowed and pattern.search(path.read_text(encoding="utf-8")):
                    self.fail(f"{name} names the sealed test pages")

    def test_G17_the_v4_test_is_open_to_development_and_the_v5_test_is_not(self):
        sealed.refuse_sealed_set("test")
        sealed.refuse_sealed_set("unseen")
        for name in ("test5", "unseen5"):
            with self.assertRaises(sealed.SealedTestError):
                sealed.refuse_sealed_set(name)

    def test_G4_the_second_read_for_metrics_needs_a_reason(self):
        sealed.read_test_pages(self.tmp, "metrics", self.reads)
        with self.assertRaises(sealed.SealedTestError):
            sealed.read_test_pages(self.tmp, "metrics", self.reads)
        self.assertEqual(sealed.metric_reads(self.reads), 1)
        sealed.read_test_pages(self.tmp, "metrics", self.reads, reason="a written reason")
        self.assertEqual(sealed.metric_reads(self.reads), 2)
        self.assertEqual(sealed.load_reads(self.reads)[-1]["reason"], "a written reason")

    def test_G4_an_unknown_purpose_is_refused(self):
        with self.assertRaises(ValueError):
            sealed.read_test_pages(self.tmp, "peek", self.reads)
        self.assertEqual(sealed.load_reads(self.reads), [])


if __name__ == "__main__":
    unittest.main()
