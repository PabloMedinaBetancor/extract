"""Extract 5.0.4: bugs found by the fourth client QA of 05/10/2026 (CHANGELOG.md, one entry each).
Every test here fails on 5.0.3."""

import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tests.test_ui import CHROME, Running, chrome
from tests.test_ui_v501 import read_result

FIXTURES = Path(__file__).parent / "fixtures"


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)


@unittest.skipIf(CHROME is None, "google-chrome is needed")
class WarningsOfManyPages(Scratch):
    """B34: the Read screen showed one box per page for a note that holds for the whole document."""

    def test_B34_a_note_for_every_page_is_one_box_and_the_progress_stays_in_view(self):
        run = Running(self.tmp / "work", [FIXTURES / "doubtful.extract.json"], root=self.tmp)
        self.addCleanup(run.close)
        url = run.url("/?selftest=ui&section=warninggroup&lang=en&nowait=1")
        result = read_result(chrome(["--virtual-time-budget=60000", "--window-size=1280,900", "--dump-dom", url]))
        self.assertEqual([c for c in result["checks"] if not c["ok"]], [])
        self.assertGreaterEqual(len(result["checks"]), 6)


class FieldStatusOfAMissingField(unittest.TestCase):
    """B35: the guide's example `doc.field_status(doc.invoice.total)` crashed whenever the invoice had no total."""

    def test_B35_a_field_the_invoice_does_not_have_is_unverified_not_an_attribute_error(self):
        import extract

        doc = extract.load(FIXTURES / "doubtful.extract.json")
        self.assertIsNone(doc.invoice.total)
        self.assertEqual(doc.field_status(doc.invoice.total), "unverified")
        self.assertEqual(doc.field_status(doc.invoice.number), "unverified")


class ReadmeExamplesAreComplete(unittest.TestCase):
    """B36: two of the three Python examples of the README used `extract` without importing it."""

    def test_B36_every_python_example_of_the_docs_that_uses_extract_imports_it(self):
        import ast
        import re

        root = Path(__file__).resolve().parents[1]
        bare = []
        names = ["README.md"] + [
            p.relative_to(root).as_posix() for p in sorted((root / "docs" / "reference").glob("*.md"))
        ]
        for name in names:
            if not (root / name).exists():  # a document that is not there is skipped
                continue
            text = (root / name).read_text(encoding="utf-8")
            for number, block in enumerate(re.findall(r"```python\n(.*?)```", text, re.S), 1):
                tree = ast.parse(block)
                uses = any(isinstance(n, ast.Name) and n.id == "extract" for n in ast.walk(tree))
                imports = any(
                    isinstance(n, ast.Import) and any(a.name == "extract" for a in n.names) for n in ast.walk(tree)
                )
                if uses and not imports:
                    bare.append(f"{name} example {number}")
        self.assertEqual(bare, [])

    def test_B37_the_docs_show_the_second_reading_as_an_option_of_options(self):
        # The README is a front page since the documentation site: the example lives on the concept page.
        root = Path(__file__).resolve().parents[1]
        text = (root / "docs" / "concepts" / "corrections.md").read_text(encoding="utf-8")
        self.assertIn('extract.Options(second_reading="thorough"', text)
        self.assertIn("extract.read(", text)


class MaintenanceScriptsAndHelp(unittest.TestCase):
    """B39: `--help` of the maintenance scripts ran them (THIRD-PARTY.md came out with no package table), and the help of
    the installer stopped in the middle of its first sentence."""

    ROOT = Path(__file__).resolve().parents[1]
    GUARDED = (
        "scripts/third_party.py",
        "scripts/write_sums.py",
        "scripts/api_docs.py",
        "scripts/make_fixtures.py",
        "scripts/docs.py",
    )

    def test_B39_help_of_a_maintenance_script_shows_its_text_and_changes_nothing(self):
        import hashlib
        import subprocess
        import sys

        watched = [self.ROOT / n for n in ("THIRD-PARTY.md", "models/SHA256SUMS", "docs/MEASURED.md")]
        watched += sorted((self.ROOT / "docs" / "reference").glob("*.md"))
        watched += sorted((self.ROOT / "tests/fixtures").glob("*"))
        watched = [p for p in watched if p.exists()]

        def digest():
            return {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in watched}

        before = digest()
        for script in self.GUARDED:
            done = subprocess.run(
                [sys.executable, str(self.ROOT / script), "--help"], capture_output=True, text=True, timeout=120
            )
            self.assertEqual(done.returncode, 0, script + done.stderr[-300:])
            self.assertIn("python", done.stdout, script)
            self.assertEqual(digest(), before, f"{script} --help changed a file")

    def test_B39_the_third_party_list_is_not_rewritten_when_there_is_nothing_to_list(self):
        import importlib.util
        import sys

        scripts = str(self.ROOT / "scripts")
        sys.path.insert(0, scripts)
        self.addCleanup(sys.path.remove, scripts)
        spec = importlib.util.spec_from_file_location("third_party_under_test", self.ROOT / "scripts/third_party.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        target = Path(tempfile.mkdtemp()) / "THIRD-PARTY.md"
        target.write_text("the list\n", encoding="utf-8")
        self.addCleanup(shutil.rmtree, target.parent, True)
        with mock.patch.object(module, "OUTPUT", target), mock.patch.object(module, "collect", return_value={}):
            with self.assertRaises(SystemExit) as stop:
                module.main()
        self.assertIn("lock.py", str(stop.exception))
        self.assertEqual(target.read_text(encoding="utf-8"), "the list\n")

    def test_B39_the_installer_help_has_its_whole_first_sentence_and_its_examples(self):
        import subprocess
        import sys

        done = subprocess.run(
            [sys.executable, str(self.ROOT / "scripts/install.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertIn("a reading of a sample PDF to prove it works", done.stdout)
        self.assertIn("--offline DIR", done.stdout)


class PythonVersionsAreSaidAsTested(unittest.TestCase):
    """B40: the docs called Python 3.12 untested although it installs, runs offline and reads like 3.10."""

    def test_B40_no_document_says_python_3_12_is_untested(self):
        root = Path(__file__).resolve().parents[1]
        stale = (
            "Python 3.12 and Windows have lock files and are not tested",
            "Tested on Linux only, with Python 3.10.",
            "Tested on Linux with Python 3.10 only.",
            "(the version Extract is tested on)",
            "The lock files for Python 3.12",
        )
        found = []
        for name in ("README.md", "QUICKSTART.md"):
            path = root / name
            if path.exists():
                found += [f"{name}: {phrase}" for phrase in stale if phrase in path.read_text(encoding="utf-8")]
        self.assertEqual(found, [])


class FilesOfOneExportShareTheirNumber(Scratch):
    """B41: after an export of some formats, an export of all of them numbered each file on its own."""

    def test_B41_the_four_files_of_an_export_get_the_same_copy_number(self):
        import extract

        doc = extract.load(FIXTURES / "doubtful.extract.json")
        out = self.tmp / "out"
        doc.export(out, formats=("txt", "json"))
        doc.export(out, formats=("txt",))
        names = sorted(p.name for p in doc.export(out, formats=("pdf", "xlsx", "json", "txt")))
        self.assertEqual(
            names, ["doubtful-3.extract.json", "doubtful-3.searchable.pdf", "doubtful-3.txt", "doubtful-3.xlsx"]
        )
        again = sorted(p.name for p in doc.export(out, formats=("xlsx", "txt")))
        self.assertEqual(again, ["doubtful-4.txt", "doubtful-4.xlsx"])

    def test_B41_a_file_that_appears_meanwhile_is_still_never_overwritten(self):
        from extract.outputs.export import claim

        folder = self.tmp / "f"
        folder.mkdir()
        (folder / "x.txt").write_text("mine", encoding="utf-8")
        partial = folder / ".x.partial.txt"
        partial.write_text("new", encoding="utf-8")
        self.assertEqual(claim(partial, folder, "x", ".txt", 1).name, "x-2.txt")
        self.assertEqual((folder / "x.txt").read_text(encoding="utf-8"), "mine")


class OneValueInvolved(unittest.TestCase):
    """B42: a join of two pieces said "1 values involved are outlined"."""

    def test_B42_the_count_of_values_involved_has_a_singular_in_both_languages(self):
        import json

        ui = Path(__file__).resolve().parents[1] / "src/extract/ui"
        for lang in ("en", "es"):
            strings = json.loads((ui / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            self.assertIn("ui.review.involved_one", strings, lang)
            self.assertNotIn("{n}", strings["ui.review.involved_one"])
            self.assertTrue(strings["ui.review.involved_one"].startswith("1 "), lang)
        self.assertIn('tn("ui.review.involved", S.involved.size)', (ui / "static/app.js").read_text(encoding="utf-8"))


class StalledReaderProcess(Scratch):
    """B43: a reader process that stopped answering (asleep, not over its memory cap) was waited for for ever."""

    def test_B43_a_reader_that_says_nothing_for_too_long_is_killed_and_its_page_comes_back_bare(self):
        import os
        import time

        from extract.engine import readerproc
        from extract.engine.readerproc import ReaderProcess, ReaderSettings
        from tests.test_engine_second import READERS, doubtful_line, page_of

        reading, crops, _ = page_of([doubtful_line(["Ccntro", "de", "Salud", "publica"])] * 10)
        env = mock.patch.dict(
            os.environ, {"EXTRACT_FAKE_READ": "Centro de Salud pública", "EXTRACT_FAKE_READER_SECONDS": "600"}
        )
        stall = mock.patch.object(readerproc, "STALL_SECONDS", 3.0)
        with env, stall:
            settings = ReaderSettings(readers=("latin", "deep"), factory=READERS, threads=1)
            proc = ReaderProcess(settings, free_memory=lambda: 8000, budget_mb=1000)
            self.addCleanup(proc.stop)
            self.assertTrue(proc.start())
            proc.submit(reading, crops)
            began = time.perf_counter()
            done = proc.drain()
        self.assertLess(time.perf_counter() - began, 40.0)
        self.assertEqual(len(done), 1)
        self.assertEqual(done[0][1].reread_report["reader_failed"], 1)
        self.assertGreaterEqual(proc.replaced, 1)
