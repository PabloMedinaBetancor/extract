import hashlib
import os
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "extract" / "guard" / "data"
LINE = re.compile(r"([0-9a-f]{64}) [ *](.+)")


class DataFilesTest(unittest.TestCase):
    def listed(self):
        text = (ROOT / "models" / "SHA256SUMS").read_text(encoding="utf-8")
        return {m.group(2): m.group(1) for m in map(LINE.fullmatch, text.splitlines()) if m}

    def test_G4_every_data_file_is_listed_and_matches_its_checksum(self):
        listed = self.listed()
        files = sorted(p for pattern in ("*.npy", "*.tsv.gz") for p in DATA.glob(pattern))
        self.assertEqual(len(files), 6)
        for path in files:
            name = f"../src/extract/guard/data/{path.name}"
            with self.subTest(file=path.name):
                self.assertIn(name, listed)
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), listed[name])

    def test_G4_every_listed_data_file_has_a_row_in_the_models_document(self):
        document = (ROOT / "MODELS.md").read_text(encoding="utf-8")
        for name in self.listed():
            self.assertIn(f"`{Path(name).name}`", document)

    def test_G4_the_data_files_are_kept_out_of_git(self):
        ignored = (ROOT / ".gitignore").read_text(encoding="utf-8").split()
        self.assertIn("src/extract/guard/data/*.npy", ignored)
        self.assertIn("src/extract/guard/data/*.tsv.gz", ignored)

    def test_G4_the_copy_of_the_model_inside_the_package_is_the_model(self):
        model, packaged = ROOT / "models" / "guard.model", ROOT / "src" / "extract" / "data" / "guard.model"
        self.assertEqual(model.read_bytes(), packaged.read_bytes())
        self.assertEqual(self.listed()["../src/extract/data/guard.model"], self.listed()["guard.model"])

    def test_G4_the_copy_of_the_thorough_model_inside_the_package_is_the_model(self):
        model = ROOT / "models" / "guard-thorough.model"
        packaged = ROOT / "src" / "extract" / "data" / "guard-thorough.model"
        self.assertEqual(model.read_bytes(), packaged.read_bytes())
        self.assertEqual(
            self.listed()["../src/extract/data/guard-thorough.model"], self.listed()["guard-thorough.model"]
        )

    def test_G4_the_thorough_model_is_tracked_by_git_and_not_ignored(self):
        git = os.environ.get("GIT") or shutil.which("git")
        if not git or not (ROOT / ".git").exists():
            self.skipTest("no git, or not a checkout")
        for name in ("models/guard.model", "models/guard-thorough.model", "src/extract/data/guard-thorough.model"):
            with self.subTest(file=name):
                ignored = subprocess.run([git, "-C", str(ROOT), "check-ignore", "-q", name]).returncode
                self.assertEqual(ignored, 1, f"{name} is ignored by git")

    def test_G4_no_word_list_is_tracked_by_git(self):
        git = os.environ.get("GIT") or shutil.which("git")
        if not git or not (ROOT / ".git").exists():
            self.skipTest("no git, or not a checkout")
        done = subprocess.run(
            [git, "-C", str(ROOT), "ls-files", "src/extract/guard/data"], capture_output=True, text=True, check=True
        )
        tracked = [Path(line).name for line in done.stdout.split()]
        self.assertEqual([n for n in tracked if n.endswith((".npy", ".tsv.gz"))], [])


if __name__ == "__main__":
    unittest.main()
