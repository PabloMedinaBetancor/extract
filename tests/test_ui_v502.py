"""Extract 5.0.2: bugs found by the second client QA of 04/10/2026 (CHANGELOG.md, one entry each).
Every test here fails on 5.0.1."""

import json
import os
import shutil
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

import extract
from extract.api import Reader
from extract.ui import server as ui_server
from tests.fake_guard import FakeGuard
from tests.support import SPANISH_LINES, digital_pdf
from tests.test_ui import CHROME, FAKE, LEXICON, Running, chrome
from tests.test_ui_v501 import read_result

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
NOT_ROOT = unittest.skipIf(hasattr(os, "geteuid") and os.geteuid() == 0, "root can read everything")


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def running(self, work="work"):
        run = Running(self.tmp / work, root=self.tmp)
        self.addCleanup(run.close)
        return run

    def forbid(self, path):
        path.chmod(0)
        self.addCleanup(path.chmod, 0o755)


class FileSystemTrouble(Scratch):
    """B1: a folder or a file the file system does not let the interface look at must not break the screens."""

    def test_B1_a_link_to_nothing_named_like_a_pdf_does_not_hide_the_whole_folder(self):
        folder = self.tmp / "in"
        folder.mkdir()
        shutil.copy(FIXTURES / "doubtful.pdf", folder / "good.pdf")
        os.symlink(self.tmp / "gone" / "x.pdf", folder / "dangling.pdf")
        status, body = self.running().call(f"/api/browse?path={folder}")
        self.assertEqual(status, 200, body)
        self.assertEqual([p["name"] for p in body["pdfs"]], ["good.pdf"])

    @NOT_ROOT
    def test_B1_a_folder_without_permission_is_one_clear_error_not_an_internal_one(self):
        folder = self.tmp / "closed"
        folder.mkdir()
        self.forbid(folder)
        status, body = self.running().call(f"/api/browse?path={folder}")
        self.assertEqual((status, body["error"]["code"]), (400, "error.cannot_read_folder"))
        self.assertNotIn("detail", body["error"]["params"])

    @NOT_ROOT
    def test_B1_a_file_without_permission_is_skipped_with_its_reason_and_the_others_are_added(self):
        folder = self.tmp / "in"
        folder.mkdir()
        shutil.copy(FIXTURES / "doubtful.pdf", folder / "good.pdf")
        shutil.copy(FIXTURES / "doubtful.pdf", folder / "private.pdf")
        (folder / "private.pdf").chmod(0)
        self.addCleanup((folder / "private.pdf").chmod, 0o644)
        run = self.running()
        status, out = run.call("/api/inspect_many", {"paths": [str(folder / "private.pdf"), str(folder / "good.pdf")]})
        self.assertEqual(status, 200, out)
        self.assertEqual([f["name"] for f in out["files"]], ["good.pdf"])
        self.assertEqual([(s["name"], s["code"]) for s in out["skipped"]], [("private.pdf", "error.file_not_readable")])
        status, body = run.call(f"/api/inspect?path={folder / 'private.pdf'}")
        self.assertEqual((status, body["error"]["code"]), (400, "error.file_not_readable"))

    def test_B1_a_read_whose_file_went_away_ends_with_a_finished_event_not_in_silence(self):
        (self.tmp / "now-a-folder.pdf").mkdir()
        run = self.running()
        status, body = run.call("/api/read", {"paths": [str(self.tmp / "now-a-folder.pdf")]})
        self.assertEqual(status, 200, body)
        events = run.events("job.finished", timeout=20)
        errors = [e for e in events if e["event"] == "job.error"]
        self.assertEqual([e["code"] for e in errors], ["error.file_not_readable"])
        self.assertEqual(events[-1]["status"], "done")

    def test_B1_the_two_new_messages_exist_in_both_languages(self):
        for lang in ("en", "es"):
            texts = json.loads((ROOT / "src/extract/ui/i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            for code in ("error.cannot_read_folder", "error.file_not_readable"):
                self.assertIn(code, texts)
        self.assertTrue(issubclass(ui_server.CannotReadFolder, ui_server.UiError))


class SameContent(Scratch):
    """B2: two files with the same bytes are one document to the interface; the second must not vanish without a word."""

    def setUp(self):
        super().setUp()
        patch = mock.patch.object(Reader, "_engine_factory", FAKE)
        patch.start()
        self.addCleanup(patch.stop)
        self.a = digital_pdf(self.tmp / "invoice.pdf", [SPANISH_LINES])
        shutil.copy(self.a, self.tmp / "invoice (copy).pdf")
        self.other = digital_pdf(self.tmp / "other.pdf", [SPANISH_LINES[:3]])
        self.run_ = Running(
            self.tmp / "work", root=self.tmp, reader_factory=lambda o: Reader(o, guard=FakeGuard(), lexicon=LEXICON)
        )
        self.addCleanup(self.run_.close)

    def read(self, *names):
        """Start a read and return the events of that read only (the log of the server keeps every earlier read)."""
        before = len(self.run_.call("/api/events?after=0&wait=0")[1]["events"])
        self.run_.call("/api/read", {"paths": [str(self.tmp / n) for n in names], "options": {"languages": ["es"]}})
        seen = []
        while not any(e["event"] == "job.finished" for e in seen):
            seen += self.run_.call(f"/api/events?after={before + len(seen)}")[1]["events"]
        return seen

    def test_B2_a_copy_of_a_file_in_the_same_read_is_named_in_a_warning_and_shown_once(self):
        events = self.read("invoice.pdf", "invoice (copy).pdf", "other.pdf")
        warnings = [e for e in events if e["event"] == "job.warning"]
        self.assertEqual(len(warnings), 1, events)
        self.assertEqual(warnings[0]["code"], "job.warning.same_content")
        self.assertEqual(warnings[0]["params"], {"name": "invoice (copy).pdf", "same_as": "invoice.pdf"})
        reads = [e for e in events if e["event"] == "doc.read"]
        self.assertEqual([e["name"] for e in reads], ["invoice.pdf", "invoice.pdf", "other.pdf"])
        self.assertEqual(reads[0]["doc_id"], reads[1]["doc_id"])
        _, docs = self.run_.call("/api/docs")
        self.assertEqual([d["name"] for d in docs["docs"]], ["invoice.pdf", "other.pdf"])

    def test_B2_the_first_name_is_kept_when_the_copy_comes_in_a_later_read(self):
        self.read("invoice.pdf")
        events = self.read("invoice (copy).pdf")
        self.assertEqual([e["params"]["same_as"] for e in events if e["event"] == "job.warning"], ["invoice.pdf"])
        _, docs = self.run_.call("/api/docs")
        self.assertEqual([d["name"] for d in docs["docs"]], ["invoice.pdf"])

    def test_B2_reading_the_same_file_again_is_not_a_duplicate(self):
        self.read("invoice.pdf")
        events = self.read("invoice.pdf")
        self.assertEqual([e for e in events if e["event"] == "job.warning"], [])

    def test_B2_the_warning_has_both_languages_and_the_same_placeholders(self):
        for lang in ("en", "es"):
            texts = json.loads((ROOT / "src/extract/ui/i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            self.assertIn("{name}", texts["job.warning.same_content"])
            self.assertIn("{same_as}", texts["job.warning.same_content"])


class SearchableTextLayer(Scratch):
    """B3: words set close together must stay separate words for the PDF viewers that guess breaks from the gaps."""

    def test_B3_neighbouring_words_set_close_together_are_found_as_a_phrase_in_pdfium_and_poppler(self):
        import subprocess
        from dataclasses import replace

        import pypdfium2 as pdfium

        from extract.outputs.pdf import add_text_layer
        from tests.support import SAMPLE, SANS, SPANISH_LINES, grid_reading, scanned_pdf, text_page_image

        source = scanned_pdf(self.tmp / "scan.pdf", [text_page_image(SPANISH_LINES, SANS, 24, 1240, 1754)])
        reading = grid_reading(1, SAMPLE["pdf_words"] * 3)
        tight = []
        for (
            word
        ) in reading.words:  # a gap of 0.09 line heights between words of a row: the measured case that pdfium joins
            left, top, right, bottom = word.box
            tight.append(replace(word, box=(left, top, right + (right - left) * 0.2 - 0.09 * (bottom - top), bottom)))
        reading = replace(reading, words=tuple(tight))
        out = self.tmp / "out.pdf"
        add_text_layer(source, [reading], out)
        words = [w.text for w in reading.words]
        page = pdfium.PdfDocument(str(out))[0].get_textpage()
        self.assertEqual(len(page.get_text_range().split()), len(words))
        lost = []
        for before, after, a, b in zip(reading.words, reading.words[1:], words, words[1:], strict=False):
            if before.line_id == after.line_id and page.search(f"{a} {b}").get_next() is None:
                lost.append(f"{a} {b}")
        self.assertEqual(lost, [])
        poppler = subprocess.run(
            ["pdftotext", "-raw", str(out), "-"], capture_output=True, text=True, check=True
        ).stdout
        self.assertEqual(poppler.split(), words)


class Blocking:
    """A reader that holds the read until the test lets it go or the read is stopped."""

    def __init__(self, options, gate):
        self.gate = gate

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, path, on_event=None, cancel=None, **kw):
        while not self.gate.wait(0.05):
            if cancel is not None and cancel.is_set():
                raise extract.Cancelled("stopped")
        raise extract.ExtractError("released")


@unittest.skipIf(CHROME is None, "google-chrome is needed")
class ReadInProgress(Scratch):
    """B4 and B6: the screens when a read is going on that this page did not start, and a double click on Read."""

    def setUp(self):
        super().setUp()
        self.pdf = digital_pdf(self.tmp / "a.pdf", [SPANISH_LINES])
        self.gate = threading.Event()
        self.run_ = Running(self.tmp / "work", root=self.tmp, reader_factory=lambda o: Blocking(o, self.gate))
        self.addCleanup(self.run_.close)
        self.addCleanup(self.gate.set)

    def section(self, name, query=""):
        url = self.run_.url(f"/?selftest=ui&section={name}&lang=en&nowait=1&pdf={self.pdf}&{query}")
        result = read_result(chrome(["--virtual-time-budget=60000", "--window-size=1280,900", "--dump-dom", url]))
        self.assertEqual([c for c in result["checks"] if not c["ok"]], [])
        self.assertGreaterEqual(len(result["checks"]), 2)

    def start_read(self):
        self.assertEqual(self.run_.call("/api/read", {"paths": [str(self.pdf)]})[0], 200)
        self.run_.events("doc.begin")

    def finished(self):
        return [e for e in self.run_.events("job.finished") if e["event"] == "job.finished"]

    def test_B4_a_page_opened_during_a_read_shows_it_with_its_stop_button(self):
        self.start_read()
        self.section("reloadread")
        self.assertEqual([e["status"] for e in self.finished()], ["stopped"])

    def test_B4_a_page_that_stays_on_import_knows_about_the_read_and_opens_it_when_read_is_pressed(self):
        self.start_read()
        self.section("readelsewhere", "tab=import")
        self.assertEqual(
            len(self.run_.state.events and [e for e in self.run_.state.events if e["event"] == "job.started"]), 1
        )

    def test_B6_a_double_click_on_read_is_one_read_and_no_error(self):
        self.section("doubleclick", "tab=import")
        self.gate.set()
        events = self.run_.events("job.finished")
        self.assertEqual(len([e for e in events if e["event"] == "job.started"]), 1)


@unittest.skipIf(CHROME is None, "google-chrome is needed")
class KeyboardOnly(Scratch):
    """B5: the Tab key of the Review screen must not take the keyboard away."""

    def setUp(self):
        super().setUp()
        preload = [FIXTURES / f"{name}.extract.json" for name in ("doubtful", "mismatch", "unreadable")]
        self.run_ = Running(self.tmp / "work", preload, root=self.tmp)
        self.addCleanup(self.run_.close)

    def test_B5_tab_moves_the_focus_unless_it_has_a_suggestion_to_take(self):
        url = self.run_.url("/?selftest=ui&section=tabkey&lang=en&nowait=1")
        result = read_result(chrome(["--virtual-time-budget=60000", "--window-size=1440,900", "--dump-dom", url]))
        self.assertEqual([c for c in result["checks"] if not c["ok"]], [])
        self.assertGreaterEqual(len(result["checks"]), 9)


@unittest.skipIf(CHROME is None, "google-chrome is needed")
class ReadAgain(Scratch):
    """B10: a document read a second time (another Corrections choice) is a new reading on every screen."""

    def setUp(self):
        super().setUp()
        self.preload = [FIXTURES / f"{name}.extract.json" for name in ("doubtful", "mismatch")]
        self.run_ = Running(self.tmp / "work", self.preload, root=self.tmp)
        self.addCleanup(self.run_.close)

    def test_B10_the_page_cache_of_the_review_screen_is_dropped_when_the_document_is_read_again(self):
        url = self.run_.url("/?selftest=ui&section=reread&lang=en&nowait=1")
        result = read_result(chrome(["--virtual-time-budget=60000", "--window-size=1440,900", "--dump-dom", url]))
        self.assertEqual([c for c in result["checks"] if not c["ok"]], [])

    def test_B10_what_was_accepted_in_the_old_reading_is_not_carried_into_the_new_one(self):
        meta = self.run_.call("/api/docs")[1]["docs"][1]
        self.assertEqual(meta["name"], "mismatch.pdf")
        first = meta["doubtful"][0]["id"]
        _, out = self.run_.call("/api/accept", {"doc": meta["id"], "word": first})
        self.assertEqual(len(out["doubtful"]), len(meta["doubtful"]) - 1)
        again = extract.load(self.preload[1], options=extract.Options(work_dir=self.tmp / "work"))
        self.run_.state.add(again, FIXTURES / "mismatch.pdf")
        after = self.run_.call("/api/docs")[1]["docs"][1]
        self.assertEqual(len(after["doubtful"]), len(meta["doubtful"]))
        self.assertIn(first, [d["id"] for d in after["doubtful"]])


class ExportErrors(Scratch):
    """B9: an export that cannot write says why in words, and never as an internal error with a partial file's name."""

    def setUp(self):
        super().setUp()
        self.run_ = Running(self.tmp / "work", [FIXTURES / "doubtful.extract.json"], root=self.tmp)
        self.addCleanup(self.run_.close)
        self.doc = self.run_.call("/api/docs")[1]["docs"][0]["id"]

    def export(self, **changes):
        body = {"docs": [self.doc], "formats": ["txt"], "destination": str(self.tmp / "out"), "pattern": "{name}"}
        return self.run_.call("/api/export", {**body, **changes})

    @NOT_ROOT
    def test_B9_a_folder_without_permission_is_a_failed_document_with_the_folder_named(self):
        (self.tmp / "closed").mkdir()
        self.forbid(self.tmp / "closed")
        status, out = self.export(destination=str(self.tmp / "closed"))
        self.assertEqual(status, 200, out)
        self.assertEqual(out["written"], {})
        self.assertEqual(
            out["failed"][self.doc], {"code": "error.cannot_write", "params": {"path": str(self.tmp / "closed")}}
        )

    def test_B9_a_name_the_file_system_cannot_hold_says_it_is_too_long(self):
        status, out = self.export(pattern="x" * 300)
        self.assertEqual(status, 200, out)
        self.assertEqual(out["failed"][self.doc]["code"], "error.name_too_long")
        self.assertEqual(list((self.tmp / "out").glob(".*")), [])

    def test_B9_an_empty_pattern_or_one_with_a_slash_is_refused_with_its_own_message(self):
        for pattern in ("", "   ", "a/b", ".hidden"):
            with self.subTest(pattern=pattern):
                status, out = self.export(pattern=pattern)
                self.assertEqual((status, out["error"]["code"]), (400, "error.bad_pattern"))
        self.assertFalse((self.tmp / "out").exists())

    def test_B9_the_new_messages_exist_in_both_languages(self):
        for lang in ("en", "es"):
            texts = json.loads((ROOT / "src/extract/ui/i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            for code in ("error.cannot_write", "error.name_too_long", "error.bad_pattern"):
                self.assertIn(code, texts)


class EngineLoadMessage(Scratch):
    """B8: a missing word list or model is named on the screen, and the command that is offered is the documented one."""

    def test_B8_the_screen_gets_the_reason_the_engine_could_not_load(self):
        pdf = digital_pdf(self.tmp / "d.pdf", [SPANISH_LINES])

        def factory(options):
            raise extract.EngineLoadError("the word lists for es are missing; build them with: x")

        run = Running(self.tmp / "work", root=self.tmp, reader_factory=factory)
        self.addCleanup(run.close)
        run.call("/api/read", {"paths": [str(pdf)]})
        errors = [e for e in run.events("job.finished") if e["event"] == "job.error"]
        self.assertEqual([e["code"] for e in errors], ["error.engine_load"])
        self.assertIn("word lists for es", errors[0]["params"]["detail"])

    def test_B8_the_missing_lists_message_names_the_command_of_the_guide(self):
        from extract import api

        with self.assertRaises(extract.EngineLoadError) as caught:
            api._default_lexicon(("zz",))
        self.assertIn(".venv/bin/python scripts/fetch_models.py --lists", str(caught.exception))
        guide = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(".venv/bin/python scripts/fetch_models.py --lists", guide)


class WorkFolder(Scratch):
    """B11: a work folder that cannot be written is named, not reported as an internal error."""

    def test_B11_a_read_that_cannot_write_its_work_folder_says_which_folder(self):
        pdf = digital_pdf(self.tmp / "d.pdf", [SPANISH_LINES])

        def factory(options):
            raise PermissionError(13, "Permission denied", str(self.tmp / "work"))

        run = Running(self.tmp / "work", root=self.tmp, reader_factory=factory)
        self.addCleanup(run.close)
        run.call("/api/read", {"paths": [str(pdf)]})
        errors = [e for e in run.events("job.finished") if e["event"] == "job.error"]
        self.assertEqual(
            [(e["code"], e["params"]) for e in errors], [("error.cannot_write_work", {"path": str(self.tmp / "work")})]
        )
        for lang in ("en", "es"):
            texts = json.loads((ROOT / "src/extract/ui/i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            self.assertIn("{path}", texts["error.cannot_write_work"])
            self.assertIn("--work-dir", texts["error.cannot_write_work"])

    def test_B11_an_error_about_another_file_is_not_blamed_on_the_work_folder(self):
        pdf = digital_pdf(self.tmp / "d.pdf", [SPANISH_LINES])

        def factory(options):
            raise PermissionError(13, "Permission denied", str(self.tmp / "elsewhere" / "x"))

        run = Running(self.tmp / "work", root=self.tmp, reader_factory=factory)
        self.addCleanup(run.close)
        run.call("/api/read", {"paths": [str(pdf)]})
        errors = [e for e in run.events("job.finished") if e["event"] == "job.error"]
        self.assertEqual([e["code"] for e in errors], ["error.internal"])


class CommandLine(Scratch):
    """B12: `--open` with a file that is not a saved document is one line of explanation, not a Python trace."""

    def run_ui(self, *args):
        import subprocess
        import sys

        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        return subprocess.run(
            [sys.executable, "-m", "extract.ui", "--no-browser", "--work-dir", str(self.tmp / "w"), *args],
            capture_output=True, text=True, timeout=60, env=env, cwd=self.tmp,
        )  # fmt: skip

    def test_B12_a_file_that_does_not_exist_is_named(self):
        done = self.run_ui("--open", "typo.extract.json")
        self.assertEqual(done.returncode, 2, done.stderr)
        self.assertIn("typo.extract.json", done.stderr)
        self.assertNotIn("Traceback", done.stderr)

    def test_B12_a_file_that_is_not_a_saved_document_is_one_line(self):
        (self.tmp / "other.json").write_text('{"x": 1}')
        (self.tmp / "text.json").write_text("not json at all")
        for name in ("other.json", "text.json"):
            with self.subTest(name=name):
                done = self.run_ui("--open", name)
                self.assertEqual(done.returncode, 2, done.stderr)
                self.assertNotIn("Traceback", done.stderr)
                self.assertTrue(done.stderr.strip().startswith("python -m extract.ui:"), done.stderr)


@unittest.skipIf(CHROME is None, "google-chrome is needed")
class WithheldForLanguage(Scratch):
    """B13: fixes withheld because of the page's language are explained as such, with no share of changed words."""

    def test_B13_the_two_causes_have_their_own_texts_in_the_review_panel_and_on_the_read_screen(self):
        preload = [FIXTURES / f"{name}.extract.json" for name in ("doubtful", "mismatch", "unreadable")]
        run = Running(self.tmp / "work", preload, root=self.tmp)
        self.addCleanup(run.close)
        url = run.url("/?selftest=ui&section=withheld&lang=en&nowait=1")
        result = read_result(chrome(["--virtual-time-budget=60000", "--window-size=1440,900", "--dump-dom", url]))
        self.assertEqual([c for c in result["checks"] if not c["ok"]], [])
        self.assertGreaterEqual(len(result["checks"]), 5)

    def test_B13_the_text_exists_in_both_languages_with_the_page_only(self):
        for lang in ("en", "es"):
            texts = json.loads((ROOT / "src/extract/ui/i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            self.assertIn("{page}", texts["guard.warning.fixes_withheld_language"])
            self.assertNotIn("{share}", texts["guard.warning.fixes_withheld_language"])


@unittest.skipIf(CHROME is None, "google-chrome is needed")
class KeyboardLists(Scratch):
    """B14: folders and table rows are reachable and pressable with the keyboard."""

    def test_B14_folders_of_the_import_list_and_rows_of_the_fields_table_work_with_the_keyboard(self):
        (self.tmp / "inbox").mkdir()
        (self.tmp / "archive").mkdir()
        preload = [FIXTURES / f"{name}.extract.json" for name in ("doubtful", "mismatch", "unreadable")]
        run = Running(self.tmp / "work", preload, root=self.tmp)
        self.addCleanup(run.close)
        url = run.url(f"/?selftest=ui&section=keyboardlist&lang=en&nowait=1&path={self.tmp}")
        result = read_result(chrome(["--virtual-time-budget=60000", "--window-size=1440,900", "--dump-dom", url]))
        self.assertEqual([c for c in result["checks"] if not c["ok"]], [])
        self.assertGreaterEqual(len(result["checks"]), 9)


class PageNumbers(Scratch):
    """B15: a page that does not exist is a 404, as for the words of a page, not an internal error."""

    def test_B15_the_image_of_a_page_out_of_range_is_not_found(self):
        run = Running(self.tmp / "work", [FIXTURES / "doubtful.extract.json"], root=self.tmp)
        self.addCleanup(run.close)
        doc = run.call("/api/docs")[1]["docs"][0]
        for number in (0, len(doc["pages"]) + 1, 99999):
            with self.subTest(page=number):
                status, body = run.call(f"/api/page/{doc['id']}/{number}.png")
                self.assertEqual((status, body["error"]["code"]), (404, "error.not_found"))
        status, body = run.call(f"/api/page/{doc['id']}/1.png")
        self.assertEqual((status, body[:4]), (200, b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
