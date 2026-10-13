import tempfile
import unittest
from pathlib import Path

from bench import read_test

ENGINE = "engine|v1"


class FreezeTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        for name in ("models/guard.model", "src/extract/guard/data/dict_es.tsv.gz", "src/extract/guard/signals.py"):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_bytes(name.encode())

    def test_G5_a_freeze_lists_the_model_the_data_and_the_code_and_a_check_passes_until_one_changes(self):
        self.assertEqual(read_test.freeze(self.root, ENGINE), 3)
        self.assertEqual(read_test.verify(self.root), ENGINE)
        (self.root / "models" / "guard.model").write_bytes(b"retrained")
        with self.assertRaises(read_test.FrozenError):
            read_test.verify(self.root)

    def test_G5_a_new_file_under_a_frozen_folder_breaks_the_check(self):
        read_test.freeze(self.root, ENGINE)
        (self.root / "src" / "extract" / "guard" / "extra.py").write_text("x = 1\n", encoding="utf-8")
        with self.assertRaises(read_test.FrozenError):
            read_test.verify(self.root)

    def test_G5_the_frozen_set_is_written_once(self):
        read_test.freeze(self.root, ENGINE)
        with self.assertRaises(read_test.FrozenError):
            read_test.freeze(self.root, ENGINE)

    def test_G5_reading_without_a_freeze_is_refused(self):
        with self.assertRaises(read_test.FrozenError):
            read_test.verify(self.root)


class SplitTest(unittest.TestCase):
    def test_G12_the_sealed_pages_go_to_the_catalogue_the_invoice_and_the_unseen_slice(self):
        pages = [
            (Path("a"), {"set": "test", "source": "catalog"}),
            (Path("b"), {"set": "test", "source": "invoice"}),
            (Path("c"), {"set": "unseen", "source": "catalog"}),
        ]
        got = read_test.split_records(pages)
        self.assertEqual(
            {name: [p.name for p, _ in records] for name, records in got.items()},
            {"pages": ["a"], "invoices": ["b"], "unseen": ["c"]},
        )


if __name__ == "__main__":
    unittest.main()
