"""Extract 5.0.1: what a client meets on screen (QA report of 04/10/2026). Each test fails on 5.0.0."""

import html
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import extract
from extract.outputs.excel import build_workbook
from extract.ui import server as ui_server
from tests.test_fixes import document
from tests.test_ui import CHROME, Running, chrome

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def load(self, name="doubtful"):
        return extract.load(FIXTURES / f"{name}.extract.json", options=extract.Options(work_dir=self.tmp / "work"))


class CorrectionsSheet(Scratch):
    def test_SF1_a_correction_typed_by_the_user_is_in_the_corrections_sheet(self):
        doc = self.load()
        word = doc.doubtful()[0]
        doc.correct(word.id, "1234")
        rows = list(build_workbook(doc._data)["Corrections"].iter_rows(min_row=2, values_only=True))
        mine = [r for r in rows if r[4] == "1234"]
        self.assertEqual(len(mine), 1, rows)
        page, operation, kind, original, final, source, _, _, state = mine[0]
        self.assertEqual((page, operation, kind, source, state), (word.page, "edit", "user", "user", "applied"))
        self.assertEqual(original, word.text)

    def test_SF1_accepting_or_retyping_the_same_text_is_not_a_correction(self):
        doc = self.load()
        first, second = doc.doubtful()[:2]
        doc.accept(first.id)
        doc.correct(second.id, second.text)
        rows = list(build_workbook(doc._data)["Corrections"].iter_rows(min_row=2, values_only=True))
        self.assertEqual(rows, [])

    def test_SF1_the_exported_workbook_lists_the_user_and_the_guard_together(self):
        doc = self.load("unreadable")
        applied = doc.fixes["applied"]
        word = doc.doubtful()[0]
        doc.correct(word.id, "XYZ")
        rows = list(build_workbook(doc._data)["Corrections"].iter_rows(min_row=2, values_only=True))
        self.assertEqual(len(rows), applied + 1)
        self.assertEqual(sorted(r[5] for r in rows).count("user"), 1)


class OpenFolder(Scratch):
    def setUp(self):
        super().setUp()
        self.opened = []
        patch = mock.patch.object(ui_server, "open_folder", self.opened.append)
        patch.start()
        self.addCleanup(patch.stop)
        (self.tmp / "out").mkdir()
        self.run_ = Running(self.tmp / "work", root=self.tmp)
        self.addCleanup(self.run_.close)

    def test_SF2_the_open_folder_action_opens_a_folder_of_the_user(self):
        status, body = self.run_.call("/api/open_folder", {"path": str(self.tmp / "out")})
        self.assertEqual(status, 200, body)
        self.assertEqual(self.opened, [self.tmp / "out"])

    def test_SF2_it_refuses_a_folder_outside_the_root_and_a_path_that_is_not_a_folder(self):
        status, body = self.run_.call("/api/open_folder", {"path": "/etc"})
        self.assertEqual((status, body["error"]["code"]), (403, "error.path_outside_home"))
        (self.tmp / "file.txt").write_text("x")
        status, body = self.run_.call("/api/open_folder", {"path": str(self.tmp / "file.txt")})
        self.assertEqual((status, body["error"]["code"]), (404, "error.not_found"))
        status, _ = self.run_.call("/api/open_folder", {"path": str(self.tmp / "out")}, token=False)
        self.assertEqual(status, 403)
        self.assertEqual(self.opened, [])


class Reopen(Scratch):
    def test_P_an_accepted_word_can_be_taken_back_and_is_doubtful_again(self):
        doc = self.load()
        word = doc.doubtful()[0]
        before = [w.id for w in doc.doubtful()]
        doc.accept(word.id)
        self.assertNotIn(word.id, [w.id for w in doc.doubtful()])
        doc.reopen(word.id)
        self.assertIsNone(doc.word(word.id).review)
        self.assertEqual([w.id for w in doc.doubtful()], before)

    def test_P_a_retyped_word_gets_the_text_it_was_read_with(self):
        doc = self.load()
        word = doc.doubtful()[0]
        doc.correct(word.id, "1234")
        doc.reopen(word.id)
        back = doc.word(word.id)
        self.assertEqual((back.text, back.review, back.original, back.corrected_by), (word.text, None, None, None))
        self.assertIn(word.id, [w.id for w in doc.doubtful()])
        rows = list(build_workbook(doc._data)["Corrections"].iter_rows(min_row=2, values_only=True))
        self.assertEqual([r for r in rows if r[5] == "user"], [])

    def test_P_a_word_nobody_decided_on_and_a_word_with_a_fix_of_the_guard_are_refused(self):
        doc = document(mode="more")
        with self.assertRaises(ValueError):
            doc.reopen(next(doc.words()).id)
        fixed = doc.fixed()[0]
        doc.correct(fixed.id, "zzz")
        with self.assertRaises(ValueError):
            doc.reopen(fixed.id)

    def test_P_the_reopening_is_logged_and_a_number_that_did_not_add_up_is_to_check_again(self):
        doc = self.load("mismatch")
        failing = ui_server.pending(doc)[0]
        doc.accept(failing["id"])
        doc.reopen(failing["id"])
        lines = (self.tmp / "work" / "corrections.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual([json.loads(line)["action"] for line in lines], ["accept", "reopen"])

    def test_P_the_server_reopens_a_word_and_puts_its_number_back_in_the_list(self):
        run = Running(self.tmp / "work2", [FIXTURES / "mismatch.extract.json"], root=self.tmp)
        self.addCleanup(run.close)
        meta = run.call("/api/docs")[1]["docs"][0]
        first = meta["doubtful"][0]["id"]
        _, out = run.call("/api/accept", {"doc": meta["id"], "word": first})
        self.assertEqual(len(out["doubtful"]), len(meta["doubtful"]) - 1)
        status, out = run.call("/api/reopen", {"doc": meta["id"], "word": first})
        self.assertEqual(status, 200, out)
        self.assertEqual(len(out["doubtful"]), len(meta["doubtful"]))
        self.assertIn(first, out["flagged"])
        self.assertEqual(run.call("/api/reopen", {"doc": meta["id"], "word": first})[0], 400)


def make_inbox(folder):
    """Two readable PDFs and one of each kind a client may meet: empty, text, truncated and password protected."""
    import pikepdf

    folder.mkdir()
    shutil.copy(FIXTURES / "doubtful.pdf", folder / "good1.pdf")
    shutil.copy(FIXTURES / "mismatch.pdf", folder / "good2.pdf")
    (folder / "empty.pdf").write_bytes(b"")
    (folder / "text.pdf").write_text("this is a text file named like a PDF")
    (folder / "corrupt.pdf").write_bytes((FIXTURES / "doubtful.pdf").read_bytes()[:400])
    with pikepdf.open(FIXTURES / "doubtful.pdf") as pdf:
        pdf.save(folder / "locked.pdf", encryption=pikepdf.Encryption(owner="o", user="u"))
    return ["empty.pdf", "text.pdf", "corrupt.pdf", "locked.pdf"]


class InspectMany(Scratch):
    def test_SF3_every_file_that_cannot_be_added_comes_back_with_its_name_and_reason(self):
        bad = make_inbox(self.tmp / "in")
        run = Running(self.tmp / "work", root=self.tmp)
        self.addCleanup(run.close)
        paths = sorted(str(p) for p in (self.tmp / "in").glob("*.pdf"))
        status, out = run.call("/api/inspect_many", {"paths": paths})
        self.assertEqual(status, 200, out)
        self.assertEqual(sorted(f["name"] for f in out["files"]), ["good1.pdf", "good2.pdf"])
        self.assertEqual(sorted(s["name"] for s in out["skipped"]), sorted(bad))
        reasons = {s["name"]: s["code"] for s in out["skipped"]}
        self.assertEqual(reasons["locked.pdf"], "error.password_protected")
        self.assertEqual(reasons["empty.pdf"], "error.unsupported_file")
        self.assertTrue(all(f["pages"] > 0 for f in out["files"]))

    def test_SF3_a_path_outside_the_root_is_one_more_skipped_file_not_a_failed_request(self):
        make_inbox(self.tmp / "in")
        run = Running(self.tmp / "work", root=self.tmp)
        self.addCleanup(run.close)
        status, out = run.call("/api/inspect_many", {"paths": ["/etc/passwd", str(self.tmp / "in" / "good1.pdf")]})
        self.assertEqual(status, 200)
        self.assertEqual([s["code"] for s in out["skipped"]], ["error.path_outside_home"])
        self.assertEqual([f["name"] for f in out["files"]], ["good1.pdf"])

    def test_SF3_the_single_inspect_still_answers_with_the_same_fields(self):
        make_inbox(self.tmp / "in")
        run = Running(self.tmp / "work", root=self.tmp)
        self.addCleanup(run.close)
        _, one = run.call(f"/api/inspect?path={self.tmp / 'in' / 'good1.pdf'}")
        _, many = run.call("/api/inspect_many", {"paths": [str(self.tmp / "in" / "good1.pdf")]})
        self.assertEqual(one, {"v": 1, **many["files"][0]})


def read_result(done):
    match = re.search(r'<pre id="selftest-result"[^>]*>(.*?)</pre>', done.stdout, re.S)
    if match is None:
        raise AssertionError(done.stdout[-600:] + done.stderr[-600:])
    return json.loads(html.unescape(match.group(1)))


@unittest.skipIf(CHROME is None, "google-chrome is needed")
class InBrowser(Scratch):
    """The page drives itself (static/selftest-ui.js) in a headless Chrome; each test reads one section."""

    PRELOAD = ("doubtful", "mismatch", "unreadable")

    def setUp(self):
        super().setUp()
        self.opened = []
        patch = mock.patch.object(ui_server, "open_folder", self.opened.append)
        patch.start()
        self.addCleanup(patch.stop)
        (self.tmp / "out").mkdir()
        preload = [FIXTURES / f"{name}.extract.json" for name in self.PRELOAD]
        self.run_ = Running(self.tmp / "work", preload, root=self.tmp)
        self.addCleanup(self.run_.close)

    def section(self, name, query="", width=1440, height=900, language="en"):
        url = self.run_.url(f"/?selftest=ui&section={name}&lang={language}&nowait=1&{query}")
        done = chrome(["--virtual-time-budget=60000", f"--window-size={width},{height}", "--dump-dom", url])
        result = read_result(done)
        failed = [c for c in result["checks"] if not c["ok"]]
        self.assertEqual(failed, [])
        self.assertGreaterEqual(len(result["checks"]), 1)
        return result

    def test_SF2_export_ends_with_a_summary_the_folder_and_a_new_button_state(self):
        self.section("export", f"dest={self.tmp / 'out'}")
        self.assertEqual(self.opened, [self.tmp / "out"])
        self.assertEqual(len(list((self.tmp / "out").glob("*-2.*"))), 6)

    def test_SF2_a_link_can_open_the_export_screen_already_exported(self):
        self.section("exportlink", "tab=export&export=1")

    def test_SF3_add_all_names_every_skipped_file_with_its_own_reason(self):
        make_inbox(self.tmp / "in")
        run = Running(self.tmp / "work2", root=self.tmp)
        self.addCleanup(run.close)
        self.run_ = run
        self.section("addall", f"tab=import&path={self.tmp / 'in'}")

    def test_SF4_the_layout_fits_480_800_tablet_portrait_and_landscape_and_wide_screens(self):
        shutil.copy(FIXTURES / "doubtful.pdf", self.tmp / "doubtful.pdf")
        self.section("layout", f"pdf={self.tmp / 'doubtful.pdf'}")

    def test_P_the_read_screen_does_not_jump_away_shows_a_review_button_and_stops_honestly(self):
        self.section("read")

    def test_P_ready_and_partial_are_never_shown_together_and_the_badges_wrap(self):
        self.section("badges")

    def test_P_the_fields_view_has_none_of_the_page_view_controls(self):
        self.section("fields")

    def test_P_touch_buttons_for_the_next_and_the_previous_word(self):
        self.section("reviewtouch")

    def test_P_an_accepted_or_retyped_word_can_be_reopened(self):
        self.section("reopen")

    def test_P_touch_targets_of_44_px_the_keyboard_the_language_of_the_page(self):
        self.section("touch")

    def test_P_plurals_in_english_and_spanish(self):
        self.section("plural", language="en")
        self.section("plural", language="es")

    def test_P_the_spanish_screens_do_not_overflow_their_side_panel(self):
        shutil.copy(FIXTURES / "doubtful.pdf", self.tmp / "doubtful.pdf")
        self.section("spanish", f"pdf={self.tmp / 'doubtful.pdf'}", language="es")

    def test_P_the_time_estimate_is_a_range_and_says_one_page_in_the_singular(self):
        self.section("estimate")

    def test_P_the_typed_path_the_error_banner_long_paths_and_the_thorough_button(self):
        deep = self.tmp / "a" / "b" / "c" / "d" / "e"
        deep.mkdir(parents=True)
        self.section("import", f"deep={deep}")

    def test_P_after_a_reload_the_import_list_and_the_folder_are_still_there(self):
        deep = self.tmp / "a" / "b"
        deep.mkdir(parents=True)
        shutil.copy(FIXTURES / "doubtful.pdf", self.tmp / "doubtful.pdf")
        self.section("reload", f"pdf={self.tmp / 'doubtful.pdf'}&deep={deep}")


class Texts(unittest.TestCase):
    def test_SF4_the_page_view_is_told_when_its_panel_changes_size(self):
        source = (ROOT / "src" / "extract" / "ui" / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("new ResizeObserver(() => this.onResize()).observe(c)", source)

    def test_P_every_text_with_a_plural_has_its_singular_in_both_languages(self):
        source = (ROOT / "src" / "extract" / "ui" / "static" / "app.js").read_text(encoding="utf-8")
        keys = set(re.findall(r'\btn\("([\w.]+)"', source))
        self.assertGreaterEqual(len(keys), 5)
        for lang in ("en", "es"):
            texts = json.loads((ROOT / "src/extract/ui/i18n" / f"{lang}.json").read_text(encoding="utf-8"))
            for key in keys:
                with self.subTest(lang=lang, key=key):
                    self.assertIn(key, texts)
                    self.assertIn(f"{key}_one", texts)
                    self.assertNotIn("{n}", texts[f"{key}_one"])

    def test_P_the_spanish_singulars_agree_in_number(self):
        es = json.loads((ROOT / "src/extract/ui/i18n/es.json").read_text(encoding="utf-8"))
        self.assertEqual(es["ui.review.fixed_count_one"], "1 arreglada")
        self.assertEqual(es["ui.review.fixed_count"].format(n=2), "2 arregladas")
        self.assertTrue(es["ui.import.read_one"].endswith("1 documento"))
        self.assertEqual(es["ui.export.go_one"], "Exportar 1 documento")

    def test_P_the_short_estimate_follows_the_measured_ratio_of_a_page_with_little_text(self):
        measured = json.loads((ROOT / "bench" / "results" / "ui-estimate-seconds.json").read_text(encoding="utf-8"))
        wanted = ui_server.SECONDS_PER_SCANNED_PAGE * measured["ratio_sparse_to_dense"]
        self.assertAlmostEqual(ui_server.SECONDS_PER_SCANNED_PAGE_LOW, wanted, delta=0.1)
        self.assertLess(measured["sparse_mean"], measured["dense_mean"])

    def test_P_the_defaults_carry_a_short_and_a_long_time_per_scanned_page(self):
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, True)
        run = Running(tmp / "work", root=tmp)
        self.addCleanup(run.close)
        _, body = run.call("/api/defaults")
        self.assertLess(0, body["seconds_per_scanned_page_low"])
        self.assertLess(body["seconds_per_scanned_page_low"], body["seconds_per_scanned_page"])


if __name__ == "__main__":
    unittest.main()
