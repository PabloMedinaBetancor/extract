"""Every generated documentation file equals what its script writes now, so the docs cannot drift from the code.

`llms-full.txt` (the pages of the site as one file) comes from `scripts/docs.py`; the API reference, the JSON format
page and the measured block of `docs/MEASURED.md` come from `scripts/api_docs.py`. Fix a failure by running
`python scripts/docs.py build`, then read the diff.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import docs_support as ds


def load_docs_script():
    spec = importlib.util.spec_from_file_location("docs_script", ds.ROOT / "scripts" / "docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class GeneratedFilesTest(unittest.TestCase):
    def test_the_generated_files_are_the_committed_ones(self):
        stale = load_docs_script().stale_files()
        ds.expect_none(self, stale, "stale generated files (run `python scripts/docs.py build`)")

    def test_llms_full_is_every_page_in_navigation_order_and_regenerates_identically(self):
        script = load_docs_script()
        first = script.render_llms_full()
        self.assertEqual(first, script.render_llms_full())
        headers = [line[2:] for line in first.splitlines() if line.startswith("# docs/") and line.endswith(".md")]
        excluded = {f"docs/{p}" for p in script.excluded_pages()}
        pages = {f"docs/{p.relative_to(ds.DOCS).as_posix()}" for p in ds.DOCS.rglob("*.md")} - excluded
        self.assertEqual(set(headers), pages)
        self.assertTrue(excluded)
        self.assertEqual(excluded & set(headers), set())
        listed = [f"docs/{p}" for p in script.nav_pages() if (ds.DOCS / p).is_file()]
        self.assertEqual([h for h in headers if h in listed], listed)

    def test_the_help_of_the_docs_script_changes_nothing(self):
        before = {p: p.stat().st_mtime_ns for p in [ds.ROOT / "llms-full.txt"] if p.exists()}
        done = subprocess.run(
            [sys.executable, str(ds.ROOT / "scripts" / "docs.py"), "--help"], capture_output=True, text=True, timeout=60
        )
        self.assertEqual(done.returncode, 0, done.stderr[-300:])
        self.assertIn("python scripts/docs.py", done.stdout)
        self.assertEqual(before, {p: p.stat().st_mtime_ns for p in before})


class StaleFileIsCaughtTest(unittest.TestCase):
    def test_a_changed_generated_file_is_reported_stale_and_a_current_one_is_not(self):
        script = load_docs_script()
        folder = Path(tempfile.mkdtemp())
        self.addCleanup(__import__("shutil").rmtree, folder, True)
        target = folder / "llms-full.txt"
        original = script.LLMS_FULL
        script.LLMS_FULL = target
        self.addCleanup(lambda: setattr(script, "LLMS_FULL", original))

        def stale():
            return [
                p.name
                for p, text in script.generated_files().items()
                if not p.is_file() or p.read_text(encoding="utf-8") != text
            ]

        self.assertEqual(stale(), ["llms-full.txt"])  # not written yet
        target.write_text(script.render_llms_full(), encoding="utf-8")
        self.assertEqual(stale(), [])
        target.write_text(target.read_text(encoding="utf-8") + "an edit by hand\n", encoding="utf-8")
        self.assertEqual(stale(), ["llms-full.txt"])


if __name__ == "__main__":
    unittest.main()
