"""The interface protocol for corrections: undo, undo a page, the mode switch, the defaults and the words
that carry their fix, all acting on the same Document as the API."""

import html
import json
import os
import re
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import extract
from extract import api, second_reading
from extract.engine.cache import file_sha256
from tests.fake_fix_guard import fix
from tests.support import SANS, SPANISH_LINES, digital_pdf, scanned_pdf, text_page_image
from tests.test_fixes import document, ids
from tests.test_ui import CHROME, Running, chrome


class FixesProtocolTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.doc = document(mode="more")
        self.path = self.tmp / "doc.extract.json"
        self.doc.save(self.path)
        self.run_ = Running(self.tmp / "work", preload=[self.path], root=self.tmp)
        self.addCleanup(self.run_.close)
        self.id = self.doc.doc_id

    def test_C7_defaults_carry_the_two_options_and_the_cost_text(self):
        _, body = self.run_.call("/api/defaults")
        self.assertEqual(
            (body["correct"], body["second_reading"], body["modes"]), ("off", False, ["off", "safe", "more"])
        )
        self.assertEqual(set(body["second_reading_thorough_cost"]), set(second_reading.thorough_cost()))

    def test_A26_a_missing_thorough_model_gets_its_own_error_text(self):
        from extract.errors import EngineLoadError
        from extract.ui.server import wire_error

        marked = EngineLoadError("x", path="/anywhere/other-name.model", cause="thorough_model_missing")
        self.assertEqual(wire_error(marked)["code"], "error.thorough_model_missing")
        self.assertEqual(
            wire_error(EngineLoadError("x", path="/anywhere/guard-thorough.model"))["code"], "error.engine_load"
        )

    def test_A26_defaults_say_whether_the_thorough_model_is_installed(self):
        _, body = self.run_.call("/api/defaults")
        self.assertIsInstance(body["thorough_available"], bool)

    def test_C7_the_doc_and_the_words_carry_the_fixes(self):
        _, body = self.run_.call(f"/api/doc/{self.id}")
        doc = body["doc"]
        self.assertEqual((doc["mode"], doc["fixes"]["applied"], len(doc["fixed"])), ("more", 5, 5))
        self.assertEqual(doc["pages"][0]["fixed"], 5)
        _, words = self.run_.call(f"/api/words/{self.id}/1")
        by_id = {w["id"]: w for w in words["words"]}
        self.assertEqual(
            (by_id["p1w40"]["status"], by_id["p1w40"]["original"], by_id["p1w40"]["fix"]["applied"]),
            ("fixed", "telefono", True),
        )
        self.assertEqual(by_id["p1w4"]["absorbed_by"], "p1w3")

    def test_C7_undo_acts_on_the_document_and_answers_with_what_changed(self):
        status, body = self.run_.call("/api/undo", {"doc": self.id, "word": "p1w40"})
        self.assertEqual(status, 200)
        self.assertEqual((body["word"]["text"], body["fixes"]["undone"], len(body["fixed"])), ("telefono", 1, 4))
        self.assertEqual(self.run_.state.doc(self.id).word("p1w40").text, "telefono")
        status, body = self.run_.call("/api/undo", {"doc": self.id, "word": "p1w40"})
        self.assertEqual(status, 400)

    def test_C7_undo_page_and_mode(self):
        status, body = self.run_.call("/api/undo_page", {"doc": self.id, "page": 1})
        self.assertEqual((status, body["undone"], len(body["fixed"])), (200, 5, 0))
        self.assertEqual(len(body["words"]), 120)
        status, body = self.run_.call("/api/mode", {"doc": self.id, "mode": "off"})
        self.assertEqual((status, body["mode"]), (200, "off"))
        status, body = self.run_.call("/api/mode", {"doc": self.id, "mode": "safe"})
        self.assertEqual(body["fixes"]["applied"], 0)  # the undone fixes are not applied again
        self.assertEqual(self.run_.call("/api/mode", {"doc": self.id, "mode": "maybe"})[0], 400)

    def test_C7_the_mode_switch_re_applies_without_reading(self):
        self.run_.call("/api/mode", {"doc": self.id, "mode": "off"})
        _, body = self.run_.call("/api/mode", {"doc": self.id, "mode": "more"})
        self.assertEqual((body["fixes"]["applied"], len(body["fixed"])), (5, 5))


def braked_proposals():
    """Twelve text fixes on a 120-word page: 10 % of the words, over the 8 % brake."""
    return {f"p1w{n}": fix("text", "accent", f"arreglada{n}", ids(n)) for n in range(50, 62)}


@unittest.skipIf(CHROME is None, "google-chrome is needed")
class FixesInBrowser(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        paths = []
        for name, lines, proposals in (
            ("fixed", SPANISH_LINES, None),
            ("braked", SPANISH_LINES[:5], braked_proposals()),
        ):
            pdf = scanned_pdf(self.tmp / f"{name}.pdf", [text_page_image(lines, SANS, 24, 1240, 1754)])
            doc = document(pdf, proposals, mode="off")
            doc.set_mode("more")
            path = self.tmp / f"{name}.extract.json"
            doc.save(path)
            data = json.loads(path.read_text(encoding="utf-8"))
            data["source"]["sha256"] = file_sha256(pdf)
            path.write_text(json.dumps(data), encoding="utf-8")
            paths.append(path)
        self.run_ = Running(self.tmp / "work", paths, root=self.tmp)
        self.addCleanup(self.run_.close)

    def selftest(self, lang):
        done = chrome(["--virtual-time-budget=40000", "--dump-dom", self.run_.url(f"/?selftest=fixes&lang={lang}")])
        match = re.search(r'<pre id="selftest-result"[^>]*>(.*?)</pre>', done.stdout, re.S)
        self.assertIsNotNone(match, done.stdout[-600:] + done.stderr[-600:])
        result = json.loads(html.unescape(match.group(1)))
        self.assertEqual([c for c in result["checks"] if not c["ok"]], [])
        self.assertGreaterEqual(len(result["checks"]), 20)

    def test_C7_undo_keep_the_filter_the_mode_switch_and_the_page_brake_work_in_a_browser(self):
        self.selftest("en")

    def test_C7_the_same_flow_works_in_spanish(self):
        self.selftest("es")

    def second_selftest(self, lang):
        done = chrome(["--virtual-time-budget=20000", "--dump-dom", self.run_.url(f"/?selftest=second&lang={lang}")])
        match = re.search(r'<pre id="selftest-result"[^>]*>(.*?)</pre>', done.stdout, re.S)
        self.assertIsNotNone(match, done.stdout[-600:] + done.stderr[-600:])
        result = json.loads(html.unescape(match.group(1)))
        self.assertEqual([c for c in result["checks"] if not c["ok"]], [])
        self.assertGreaterEqual(len(result["checks"]), 14)

    def test_A26_the_read_order_carries_the_mode_and_the_hints_follow_the_choices_in_en_and_es(self):
        self.second_selftest("en")
        self.second_selftest("es")

    def import_dom(self, query):
        done = chrome(["--virtual-time-budget=15000", "--dump-dom", self.run_.url(f"/?tab=import&nowait=1&{query}")])
        return html.unescape(done.stdout)

    def test_A26_import_offers_off_and_thorough_with_the_measured_cost_in_both_languages(self):
        en = self.import_dom("lang=en&correct=safe&second=thorough")
        self.assertIn(
            f"about {round(second_reading.thorough_cost()['seconds_per_page'])} s more per page (measured in development)",
            en,
        )
        self.assertNotIn("of every 100", en)
        self.assertIn("Safe only fixes control digits (DNI/NIE, IBAN)", en)
        self.assertIn("Its gain is applied with More", en)
        self.assertRegex(
            en, r'data-second="thorough"[^>]*aria-pressed="true"|aria-pressed="true"[^>]*data-second="thorough"'
        )
        self.assertEqual(en.count("data-second="), 2)
        es = self.import_dom("lang=es&correct=more")
        self.assertIn("Cada página se lee una sola vez.", es)
        self.assertNotIn("Safe solo arregla", es)
        self.assertRegex(es, r'data-second="off"[^>]*aria-pressed="true"|aria-pressed="true"[^>]*data-second="off"')
        thorough_es = self.import_dom("lang=es&second=thorough")
        self.assertIn(f"unos {round(second_reading.thorough_cost()['seconds_per_page'])} s más por página", thorough_es)


class ThoroughOrder(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.pdf = digital_pdf(self.tmp / "d.pdf", [SPANISH_LINES])
        self.options = []
        gone = str(self.tmp / "no-such-guard.model")
        patch = mock.patch.dict(os.environ, {second_reading.THOROUGH_MODEL_ENV: gone})
        patch.start()
        self.addCleanup(patch.stop)

        def factory(options):
            self.options.append(options)
            if options.second_reading == "thorough":
                api._thorough_guard()
            raise extract.EngineLoadError("stop here")

        self.run_ = Running(self.tmp / "work", root=self.tmp, reader_factory=factory)
        self.addCleanup(self.run_.close)

    def test_A26_the_word_thorough_reaches_the_options_and_a_missing_model_has_its_own_error(self):
        status, _ = self.run_.call("/api/read", {"paths": [str(self.pdf)], "options": {"second_reading": "thorough"}})
        self.assertEqual(status, 200)
        events = self.run_.events("job.finished")
        self.assertEqual(self.options[0].second_reading, "thorough")
        errors = [e for e in events if e["event"] == "job.error"]
        self.assertEqual([e["code"] for e in errors], ["error.thorough_model_missing"])
        self.assertIn("no-such-guard.model", errors[0]["params"]["path"])

    def test_A26_defaults_say_whether_the_thorough_model_file_exists(self):
        self.assertIs(self.run_.call("/api/defaults")[1]["thorough_available"], False)
        present = self.tmp / "guard-thorough.model"
        present.write_bytes(b"x")
        with mock.patch.dict(os.environ, {second_reading.THOROUGH_MODEL_ENV: str(present)}):
            with mock.patch("extract.engine.modelfiles.has_model", return_value=True):
                self.assertIs(self.run_.call("/api/defaults")[1]["thorough_available"], True)
            # F1: the guard model travels in the wheel, the reader models do not (fetch_models.py --all)
            with mock.patch("extract.engine.modelfiles.has_model", return_value=False):
                self.assertIs(self.run_.call("/api/defaults")[1]["thorough_available"], False)

    def test_A26_off_reaches_the_options_as_false(self):
        self.run_.call("/api/read", {"paths": [str(self.pdf)], "options": {"second_reading": False}})
        events = self.run_.events("job.finished")
        self.assertIs(self.options[0].second_reading, False)
        self.assertEqual([e["code"] for e in events if e["event"] == "job.error"], ["error.engine_load"])


if __name__ == "__main__":
    unittest.main()
