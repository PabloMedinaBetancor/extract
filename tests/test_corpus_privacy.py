import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OWN_RESULTS = (
    "pilot-bands.json",
    "font-families*.json",
    "test_reads*.json",
    "test-freeze-v5.json",
    "split-v5-checks.json",
    "injection-fixes.json",
    "guard-*.json",
    "labels-*.json",
)


class PrivacyTest(unittest.TestCase):
    def test_G4_page_images_selections_and_work_folders_are_outside_git(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
        self.assertIn("bench/data/", ignored)
        self.assertIn("work/", ignored)

    def test_G4_no_result_file_carries_document_text(self):
        folder = ROOT / "bench" / "results"
        results = [p for pattern in OWN_RESULTS for p in sorted(folder.glob(pattern))]
        self.assertGreater(len(results), 0)
        for path in results:
            self.assertEqual(self.long_strings(json.loads(path.read_text(encoding="utf-8"))), [], path.name)

    def long_strings(self, value, limit=64):
        if isinstance(value, str):
            return [value] if len(value) > limit else []
        if isinstance(value, dict):
            return [s for k, v in value.items() for s in self.long_strings(k, limit) + self.long_strings(v, limit)]
        if isinstance(value, list):
            return [s for v in value for s in self.long_strings(v, limit)]
        return []


if __name__ == "__main__":
    unittest.main()
