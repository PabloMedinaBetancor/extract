"""Extract 5.0.6: a screen never shows the name, the page, the fields or the notes of one document next to the data of
another (found on 05/10/2026 by a real read of three documents: the Read screen named one document over the page of
another). One test per case; each fails on 5.0.5. The reading engine did not move."""

import html
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

from extract.engine.cache import file_sha256
from tests.support import SANS, SPANISH_LINES, scanned_pdf, text_page_image
from tests.test_fixes import document
from tests.test_ui import CHROME, Running, chrome

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"


def read_result(done):
    match = re.search(r'<pre id="selftest-result"[^>]*>(.*?)</pre>', done.stdout, re.S)
    if match is None:
        raise AssertionError(done.stdout[-600:] + done.stderr[-600:])
    return json.loads(html.unescape(match.group(1)))


class Scratch(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)


@unittest.skipIf(CHROME is None, "google-chrome is needed")
class InBrowser(Scratch):
    """The page drives itself (static/selftest-ui.js) in a headless Chrome; each test reads one section."""

    PRELOAD = ("doubtful", "mismatch", "unreadable")

    def preload(self):
        return [FIXTURES / f"{name}.extract.json" for name in self.PRELOAD]

    def setUp(self):
        super().setUp()
        self.run_ = Running(self.tmp / "work", self.preload(), root=self.tmp)
        self.addCleanup(self.run_.close)

    def section(self, name, minimum=3):
        url = self.run_.url(f"/?selftest=ui&section={name}&lang=en&nowait=1")
        result = read_result(chrome(["--virtual-time-budget=60000", "--window-size=1440,900", "--dump-dom", url]))
        self.assertEqual([c for c in result["checks"] if not c["ok"]], [])
        self.assertGreaterEqual(len(result["checks"]), minimum)


class ReadScreen(InBrowser):
    def test_B46_the_page_drawn_on_the_read_screen_is_that_of_the_document_it_names(self):
        self.section("readpreview", 4)

    def test_B46_when_the_last_document_could_not_be_read_the_screen_follows_the_one_before(self):
        self.section("readfailed", 6)

    def test_B46_a_file_with_the_content_of_an_open_one_says_which_document_it_opens(self):
        self.section("readsame", 2)


class ReviewScreen(InBrowser):
    def test_B47_choosing_a_document_that_has_no_such_page_opens_its_first_page(self):
        self.section("reviewswitch", 5)

    def test_B48_open_in_review_never_draws_the_page_of_the_document_that_was_open_before(self):
        self.section("openfromread", 5)

    def test_B49_a_document_read_again_is_shown_as_the_new_reading_and_new_documents_are_listed(self):
        self.section("rereadreview", 5)


class FixedFilter(InBrowser):
    """B50: the filter of fixed words and the rule picked belong to the document they were set on."""

    def preload(self):
        pdf = scanned_pdf(self.tmp / "fixed.pdf", [text_page_image(SPANISH_LINES, SANS, 24, 1240, 1754)])
        doc = document(pdf, None, mode="off")
        doc.set_mode("more")
        path = self.tmp / "fixed.extract.json"
        doc.save(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        data["source"]["sha256"] = file_sha256(pdf)
        path.write_text(json.dumps(data), encoding="utf-8")
        return [FIXTURES / "doubtful.extract.json", path]

    def test_B50_the_fixed_filter_and_the_rule_focus_are_not_carried_to_another_document(self):
        self.section("fixedfilter", 5)


class ExportList(InBrowser):
    def test_B51_a_document_read_after_the_export_screen_was_opened_is_ticked_there(self):
        self.section("exportnew", 2)

    def test_B49_a_document_read_while_the_export_screen_is_open_is_listed_there(self):
        self.section("exportevent", 3)


class ErrorsNameTheirDocument(Scratch):
    """B52: the error of a file that could not be opened carries the position of that file in the read."""

    def test_B52_each_unreadable_file_is_reported_with_its_own_index(self):
        for name in ("a.pdf", "b.pdf", "c.pdf"):
            (self.tmp / name).mkdir()
        run = Running(self.tmp / "work", root=self.tmp)
        self.addCleanup(run.close)
        paths = [str(self.tmp / n) for n in ("a.pdf", "b.pdf", "c.pdf")]
        status, body = run.call("/api/read", {"paths": paths})
        self.assertEqual(status, 200, body)
        events = run.events("job.finished", timeout=20)
        errors = [e for e in events if e["event"] == "job.error"]
        self.assertEqual(
            [(e["index"], Path(e["path"]).name) for e in errors], [(0, "a.pdf"), (1, "b.pdf"), (2, "c.pdf")]
        )


if __name__ == "__main__":
    unittest.main()
