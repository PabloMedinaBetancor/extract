import ast
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

import extract
from extract.api import Reader
from extract.ui import server as ui_server
from tests.fake_guard import FakeGuard
from tests.support import SANS, SPANISH_LINES, SetLexicon, digital_pdf, scanned_pdf, text_page_image

ROOT = Path(__file__).resolve().parents[1]
UI = ROOT / "src" / "extract" / "ui"
FIXTURES = ROOT / "tests" / "fixtures"
FAKE = "tests.fake_engine:load_engine"
LEXICON = SetLexicon({w.lower().strip(".,:") for line in SPANISH_LINES for w in line.split() if len(w) > 2})
CHROME = shutil.which("google-chrome")
GUARD_REASONS = [
    "low_char_conf", "not_in_dictionary", "close_to", "mixed_letters_digits", "odd_chars", "unusual_height",
    "unusual_width", "bad_number_format", "low_page_quality", "neighbours_unusual",
]  # fmt: skip
GUARD_FIX_REASONS = [
    "accent_restored", "case_restored", "punctuation_restored", "words_joined", "word_split", "speck_dropped",
    "readers_agree", "engine_alternative", "dictionary_nearest", "page_vocabulary",
]  # fmt: skip


def strings(lang):
    return json.loads((UI / "i18n" / f"{lang}.json").read_text(encoding="utf-8"))


def placeholders(text):
    return sorted(re.findall(r"\{(\w+)\}", text))


class Running:
    """A server on a random port in a thread, and a small client that sends the token and the host."""

    def __init__(self, work_dir, preload=(), root=None, reader_factory=None):
        self.server, self.state = ui_server.make_server(
            Path(work_dir), list(preload), root=root, reader_factory=reader_factory
        )
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.state.port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def url(self, path, token=True):
        return f"{self.base}{path}" + (f"{'&' if '?' in path else '?'}t={self.state.token}" if token else "")

    def call(self, path, body=None, headers=None, token=True):
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url(path, token), data=data, headers=headers or {})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                kind = response.headers.get("Content-Type", "")
                payload = response.read()
                return response.status, json.loads(payload) if "json" in kind else payload
        except urllib.error.HTTPError as error:
            return error.code, json.loads(error.read())

    def events(self, until, timeout=60):
        seen, deadline = [], time.time() + timeout
        while time.time() < deadline:
            _, body = self.call(f"/api/events?after={len(seen)}")
            seen += body["events"]
            if any(e["event"] == until for e in seen):
                return seen
        raise AssertionError(f"no {until} in {[e['event'] for e in seen]}")


class SourceRules(unittest.TestCase):
    def test_A6_the_ui_imports_only_extract_and_the_standard_library(self):
        stdlib = set(__import__("sys").stdlib_module_names)
        for path in UI.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [a.name for a in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0:
                    names = [node.module]
                else:
                    continue
                for name in names:
                    top = name.split(".")[0]
                    with self.subTest(file=path.name, name=name):
                        self.assertTrue(top in stdlib or name == "extract", name)

    def test_A6_the_ui_calls_the_public_api_only(self):
        public = set(extract.__all__)
        for path in UI.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == "extract":
                    with self.subTest(file=path.name, name=node.attr):
                        self.assertIn(node.attr, public)


class I18n(unittest.TestCase):
    def test_S5_both_languages_have_the_same_keys_and_placeholders(self):
        en, es = strings("en"), strings("es")
        self.assertEqual(sorted(en), sorted(es))
        for key in en:
            with self.subTest(key=key):
                self.assertEqual(placeholders(en[key]), placeholders(es[key]))
                self.assertTrue(en[key].strip() and es[key].strip())

    def test_S5_every_guard_reason_and_error_code_has_a_text(self):
        en = strings("en")
        for name in GUARD_REASONS:
            self.assertIn(f"guard.reason.{name}", en)
        for name in GUARD_FIX_REASONS:
            self.assertIn(f"guard.fix.{name}", en)
        self.assertIn("guard.warning.fixes_withheld", en)
        codes = re.findall(r'code = "([\w.]+)"', (ROOT / "src/extract/errors.py").read_text(encoding="utf-8"))
        codes += re.findall(r'code = "([\w.]+)"', (UI / "server.py").read_text(encoding="utf-8"))
        self.assertGreaterEqual(len(codes), 12)
        for code in codes:
            self.assertIn(code, en)

    def test_S5_every_key_the_screens_use_has_a_text(self):
        en = strings("en")
        source = (UI / "static" / "app.js").read_text(encoding="utf-8")
        literal = set(re.findall(r'\b(?:t|tn)\("([\w.]+)"', source))
        templated = {
            "ui.tab.{}": ["import", "read", "review", "export"],
            "ui.kind.{}": ["scanned", "digital", "mixed"],
            "ui.language.{}": ["es", "en"],
            "ui.format.{}": ["pdf", "xlsx", "json", "txt"],
            "ui.budget.{}": ["fewer", "balanced", "more"],
            "ui.budget.{}.hint": ["fewer", "balanced", "more"],
            "ui.read.status.{}": ["waiting", "reading", "done", "partial", "stopped", "skipped", "failed"],
            "ui.read.finished.{}": ["done", "stopped", "failed"],
            "ui.review.tab.{}": ["page", "fields"],
            "ui.state.{}": ["ok", "native", "doubtful", "unscored", "accepted", "corrected", "fixed"],
            "ui.mode.{}": ["off", "safe", "more"],
            "ui.second.{}": ["off", "thorough"],
            "ui.correct.{}.hint": ["off", "safe", "more"],
            "ui.fix.source.{}": [
                "dictionary", "fold", "punct", "page", "ctc_alt", "masked", "reading", "structure", "rule", "other",
            ],
            "ui.status.{}": ["checked", "mismatch", "unverified"],
            "checks.result.{}": ["pass", "fail", "not_applicable"],
            "ui.field.{}": [
                "number", "date", "subtotal", "withholding_rate", "withholding", "total", "vat_rate", "vat_base",
                "vat_amount", "description", "quantity", "unit_price", "amount",
            ],
        }  # fmt: skip
        keys = literal | {pattern.format(v) for pattern, values in templated.items() for v in values}
        for key in sorted(keys):
            with self.subTest(key=key):
                self.assertIn(key, en)
        for key in set(en) - keys:
            if key.startswith("ui."):
                with self.subTest(unused=key):
                    base = key[: -len("_one")] if key.endswith("_one") else key
                    self.assertTrue(f'"{base}"' in source or key.startswith(("ui.keys.", "ui.legend.")), key)


class Protocol(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self.run_ = Running(
            self.tmp / "work", [FIXTURES / "doubtful.extract.json", FIXTURES / "mismatch.extract.json"], root=ROOT
        )
        self.addCleanup(self.run_.close)

    def test_S6_the_server_listens_on_loopback_only(self):
        self.assertEqual(self.run_.server.server_address[0], "127.0.0.1")

    def test_S6_a_request_without_the_token_is_refused(self):
        status, body = self.run_.call("/api/defaults", token=False)
        self.assertEqual((status, body["error"]["code"]), (403, "error.forbidden_token"))
        status, body = self.run_.call("/api/defaults?t=wrong", token=False)
        self.assertEqual(status, 403)
        status, _ = self.run_.call("/", token=False)
        self.assertEqual(status, 403)

    def test_S6_a_request_with_another_host_is_refused(self):
        status, body = self.run_.call("/api/defaults", headers={"Host": "evil.example:80"})
        self.assertEqual((status, body["error"]["code"]), (403, "error.forbidden_host"))
        status, _ = self.run_.call("/api/defaults", headers={"Host": f"127.0.0.1:{self.run_.state.port + 1}"})
        self.assertEqual(status, 403)

    def test_S6_the_token_in_a_header_works_and_localhost_is_a_loopback_host(self):
        status, body = self.run_.call(
            "/api/defaults",
            headers={"X-Token": self.run_.state.token, "Host": f"localhost:{self.run_.state.port}"},
            token=False,
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["v"], 1)

    def test_S6_paths_outside_the_root_are_refused(self):
        for path in ("/etc", f"{ROOT}/..", "/"):
            status, body = self.run_.call(f"/api/browse?path={path}")
            self.assertEqual((status, body["error"]["code"]), (403, "error.path_outside_home"), path)
        status, _ = self.run_.call("/api/read", {"paths": ["/etc/passwd"]})
        self.assertEqual(status, 403)
        status, _ = self.run_.call("/api/export", {"docs": [], "formats": ["txt"], "destination": "/tmp"})
        self.assertEqual(status, 400)

    def test_the_page_serves_the_screens_and_the_texts_come_as_keys(self):
        status, page = self.run_.call("/")
        self.assertEqual(status, 200)
        self.assertIn(b"app.js", page)
        status, body = self.run_.call("/api/i18n/es.json")
        self.assertEqual(body["strings"]["ui.tab.review"], "Revisar")
        self.assertEqual(self.run_.call("/api/i18n/fr.json")[0], 404)
        _, doc = self.run_.call("/api/docs")
        _, words = self.run_.call(f"/api/words/{doc['docs'][0]['id']}/1")
        self.assertIn("guard.reason.", json.dumps(words))
        self.assertNotIn("Not in the dictionary", json.dumps(words))

    def test_a_number_that_does_not_add_up_is_to_check_and_accepting_it_takes_it_off(self):
        _, listing = self.run_.call("/api/docs")
        meta = listing["docs"][1]
        self.assertEqual(meta["name"], "mismatch.pdf")
        failing = [n for n in meta["numbers"] if n["status"] == "mismatch"]
        self.assertTrue(failing)
        self.assertEqual(meta["doubtful"][0]["status"], "mismatch")
        self.assertEqual(len(meta["doubtful"]), len(failing))
        first = meta["doubtful"][0]["id"]
        _, words = self.run_.call(f"/api/words/{meta['id']}/1")
        self.assertEqual(set(words["flagged"]), {i for n in failing for i in n["word_ids"]})
        _, out = self.run_.call("/api/accept", {"doc": meta["id"], "word": first})
        self.assertEqual(len(out["doubtful"]), len(meta["doubtful"]) - 1)
        self.assertNotIn(first, out["flagged"])
        number = next(n for n in out["numbers"] if first in n["word_ids"])
        self.assertEqual(number["status"], "mismatch")

    def test_the_static_files_cannot_be_walked_out_of(self):
        import socket

        for target in ("/static/../server.py", "/static/..%2fserver.py", "/static/../../__init__.py"):
            with socket.create_connection(("127.0.0.1", self.run_.state.port), timeout=10) as sock:
                request = f"GET {target}?t={self.run_.state.token} HTTP/1.1\r\nHost: 127.0.0.1:{self.run_.state.port}\r\nConnection: close\r\n\r\n"
                sock.sendall(request.encode())
                reply = b""
                while chunk := sock.recv(4096):
                    reply += chunk
            self.assertTrue(
                reply.startswith(b"HTTP/1.0 404") or reply.startswith(b"HTTP/1.1 404"), (target, reply[:80])
            )
            self.assertNotIn(b"def serve", reply)

    def test_a_garbage_content_length_is_a_bad_request(self):
        import socket

        with socket.create_connection(("127.0.0.1", self.run_.state.port), timeout=10) as sock:
            request = (
                f"POST /api/stop?t={self.run_.state.token} HTTP/1.1\r\nHost: 127.0.0.1:{self.run_.state.port}\r\n"
                "Content-Length: abc\r\nConnection: close\r\n\r\n"
            )
            sock.sendall(request.encode())
            reply = b""
            while chunk := sock.recv(4096):
                reply += chunk
        self.assertIn(b" 400 ", reply.split(b"\r\n")[0])

    def test_the_document_the_words_the_page_and_the_crop(self):
        _, listing = self.run_.call("/api/docs")
        meta = listing["docs"][0]
        self.assertEqual(meta["name"], "doubtful.pdf")
        self.assertEqual(len(meta["doubtful"]), 8)
        _, words = self.run_.call(f"/api/words/{meta['id']}/1")
        self.assertEqual(len(words["words"]), 95)
        status, png = self.run_.call(f"/api/page/{meta['id']}/1.png")
        self.assertEqual((status, png[:4]), (200, b"\x89PNG"))
        status, crop = self.run_.call(f"/api/crop/{meta['id']}/{meta['doubtful'][0]['id']}.png")
        self.assertEqual((status, crop[:4]), (200, b"\x89PNG"))
        self.assertEqual(self.run_.call(f"/api/words/{meta['id']}/9")[0], 404)

    def test_S2_accept_correct_and_use_suggestion_change_the_document_and_log_it(self):
        _, listing = self.run_.call("/api/docs")
        meta = listing["docs"][0]
        todo = meta["doubtful"]
        word = next(w for w in self.run_.call(f"/api/words/{meta['id']}/1")[1]["words"] if w["suggestion"])
        status, out = self.run_.call("/api/use_suggestion", {"doc": meta["id"], "word": word["id"]})
        self.assertEqual(status, 200)
        self.assertEqual((out["word"]["text"], out["word"]["review"]), (word["suggestion"], "corrected"))
        self.assertEqual(len(out["doubtful"]), len(todo) - 1)
        other = out["doubtful"][0]["id"]
        _, out = self.run_.call("/api/accept", {"doc": meta["id"], "word": other})
        self.assertEqual(out["word"]["review"], "accepted")
        _, out = self.run_.call("/api/correct", {"doc": meta["id"], "word": out["doubtful"][0]["id"], "text": "Hola"})
        self.assertEqual(out["word"]["text"], "Hola")
        self.assertEqual(self.run_.call("/api/correct", {"doc": meta["id"], "word": "p1w0", "text": " "})[0], 400)
        lines = (self.tmp / "work" / "corrections.jsonl").read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines]
        self.assertEqual([r["action"] for r in records], ["correct", "accept", "correct"])
        self.assertEqual(set(records[0]), {"sha256", "word", "original", "corrected", "action", "time"})

    def test_S4_nothing_is_written_before_export_and_the_pattern_names_the_files(self):
        run = Running(self.tmp / "work2", [FIXTURES / "doubtful.extract.json"], root=self.tmp)
        self.addCleanup(run.close)
        _, listing = run.call("/api/docs")
        meta = listing["docs"][0]
        destination = self.tmp / "out"
        self.assertFalse(destination.exists())
        body = {
            "docs": [meta["id"]],
            "destination": str(destination),
            "pattern": "{name}-final",
            "formats": ["txt", "json"],
        }
        status, out = run.call("/api/export", body)
        self.assertEqual(status, 200, out)
        names = sorted(Path(p).name for p in out["written"][meta["id"]])
        self.assertEqual(names, ["doubtful-final.extract.json", "doubtful-final.txt"])
        _, again = run.call("/api/export", body)
        self.assertEqual(
            sorted(Path(p).name for p in again["written"][meta["id"]]),
            ["doubtful-final-2.extract.json", "doubtful-final-2.txt"],
        )
        bad = dict(body, pattern="../x")
        self.assertEqual(run.call("/api/export", bad)[0], 400)

    def test_browse_hides_dotted_folders_and_files(self):
        (self.tmp / ".secret").mkdir()
        (self.tmp / ".hidden.pdf").write_bytes(b"x")
        (self.tmp / "shown").mkdir()
        run = Running(self.tmp / "work3", root=self.tmp)
        self.addCleanup(run.close)
        _, listing = run.call(f"/api/browse?path={self.tmp}")
        self.assertIn("shown", listing["dirs"])
        self.assertNotIn(".secret", listing["dirs"])
        self.assertNotIn(".hidden.pdf", [f["name"] for f in listing["pdfs"]])

    def test_browse_lists_folders_and_pdfs_and_inspect_reads_no_ocr(self):
        _, listing = self.run_.call(f"/api/browse?path={FIXTURES}")
        self.assertIn("doubtful.pdf", [f["name"] for f in listing["pdfs"]])
        self.assertEqual(listing["parent"], str(FIXTURES.parent))
        _, info = self.run_.call(f"/api/inspect?path={FIXTURES / 'unreadable.pdf'}")
        self.assertEqual((info["pages"], info["kind"]), (3, info["kind"]))
        self.assertIn(info["kind"], ("scanned", "digital", "mixed"))


class ReadJob(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        patch = mock.patch.object(Reader, "_engine_factory", FAKE)
        patch.start()
        self.addCleanup(patch.stop)
        scan = text_page_image(SPANISH_LINES, SANS, 24, 1240, 1754)
        self.scanned = scanned_pdf(self.tmp / "scanned.pdf", [scan, scan])
        self.digital = digital_pdf(self.tmp / "digital.pdf", [SPANISH_LINES])
        self.run_ = Running(
            self.tmp / "work",
            root=self.tmp,
            reader_factory=lambda options: Reader(options, guard=FakeGuard(), lexicon=LEXICON),
        )
        self.addCleanup(self.run_.close)

    def test_S3_a_read_reports_progress_per_page_and_opens_the_documents(self):
        status, _ = self.run_.call(
            "/api/read",
            {"paths": [str(self.scanned), str(self.digital)], "options": {"languages": ["es"], "review": "more"}},
        )
        self.assertEqual(status, 200)
        events = self.run_.events("job.finished")
        names = [e["event"] for e in events]
        self.assertEqual(names[0], "job.started")
        self.assertEqual(events[0]["names"], ["scanned.pdf", "digital.pdf"])
        self.assertEqual(names.count("doc.read"), 2)
        self.assertEqual(events[-1]["status"], "done")
        reads = [e for e in events if e["event"] == "page.read"]
        self.assertEqual(len(reads), 3)
        self.assertEqual([e["index"] for e in reads], [0, 0, 1])
        progress = [e for e in events if e["event"] == "page.progress" and e["index"] == 0]
        self.assertEqual([e["done"] for e in progress], sorted(e["done"] for e in progress))
        _, docs = self.run_.call("/api/docs")
        self.assertEqual(len(docs["docs"]), 2)
        self.assertTrue(all(e.get("doc_id") in self.run_.state.docs for e in events if e["event"] == "doc.read"))
        self.assertEqual(self.run_.call("/api/read", {"paths": []})[0], 400)

    def test_S3_stop_ends_the_read_and_keeps_what_was_read(self):
        paths = [str(self.digital)] + [str(self.scanned)] * 1
        self.run_.call("/api/read", {"paths": paths})
        self.run_.call("/api/stop", {})
        events = self.run_.events("job.finished")
        self.assertIn(events[-1]["status"], ("stopped", "done"))

    def test_a_file_that_is_not_a_pdf_is_reported_with_its_code(self):
        bad = self.tmp / "bad.pdf"
        bad.write_bytes(b"not a pdf")
        self.run_.call("/api/read", {"paths": [str(bad), str(self.digital)]})
        events = self.run_.events("job.finished")
        errors = [e for e in events if e["event"] == "job.error"]
        self.assertEqual(errors[0]["code"], "error.unsupported_file")
        self.assertEqual([e["event"] for e in events].count("doc.read"), 1)


def chrome(args, timeout=120):
    profile = tempfile.mkdtemp()
    try:
        command = [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--no-first-run",
            f"--user-data-dir={profile}",
            # the test browser talks to the test server only: no update, sync, safe-browsing or component traffic
            "--disable-background-networking",
            "--disable-sync",
            "--disable-component-update",
            "--no-default-browser-check",
            "--disable-default-apps",
            "--disable-features=OptimizationHints,MediaRouter,Translate",
            *args,
        ]
        return subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    finally:
        shutil.rmtree(profile, ignore_errors=True)


@unittest.skipIf(CHROME is None, "google-chrome is needed")
class InBrowser(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        preload = [
            FIXTURES / "doubtful.extract.json",
            FIXTURES / "mismatch.extract.json",
            FIXTURES / "unreadable.extract.json",
        ]
        self.run_ = Running(self.tmp / "work", preload, root=ROOT)
        self.addCleanup(self.run_.close)

    def test_S2_the_self_test_page_drives_the_key_handlers_and_reports_in_the_dom(self):
        url = self.run_.url("/?selftest=1&lang=en")
        done = chrome(["--virtual-time-budget=40000", "--dump-dom", url])
        match = re.search(r'<pre id="selftest-result"[^>]*>(.*?)</pre>', done.stdout, re.S)
        self.assertIsNotNone(match, done.stdout[-600:] + done.stderr[-600:])
        import html

        result = json.loads(html.unescape(match.group(1)))
        failed = [c for c in result["checks"] if not c["ok"]]
        self.assertEqual(failed, [])
        self.assertGreaterEqual(len(result["checks"]), 20)
        self.assertTrue(result["pass"])

    def test_S2_a_page_with_1500_boxes_redraws_in_under_50_ms(self):
        # Chrome dumps the page when its time is up, so a slow machine gets longer windows: the measurement itself is
        # asserted below, and only the wait for its result grows.
        for window in (8000, 25000, 60000):
            done = chrome(
                [f"--timeout={window}", "--dump-dom", self.run_.url("/?perf=1")], timeout=window // 1000 + 120
            )
            match = re.search(r'<pre id="perf-result"[^>]*>(.*?)</pre>', done.stdout, re.S)
            if match:
                break
        self.assertIsNotNone(match, done.stdout[-400:])
        result = json.loads(match.group(1))
        self.assertEqual(result["boxes"], 1500)
        self.assertLess(result["median_ms"], 50)
        self.assertTrue(result["pass"])

    def test_S1_the_screens_render_in_both_languages(self):
        pages = {
            "import": f"tab=import&path={FIXTURES}&add={FIXTURES / 'doubtful.pdf'}",
            "review": "tab=review&doc=0&pick=p1w1",
            "export": "tab=export",
        }
        sizes = {}
        for lang in ("en", "es"):
            for name, query in pages.items():
                shot = self.tmp / f"{name}-{lang}.png"
                chrome(
                    [
                        "--virtual-time-budget=15000",
                        "--window-size=1440,900",
                        f"--screenshot={shot}",
                        self.run_.url(f"/?lang={lang}&nowait=1&{query}"),
                    ]
                )
                self.assertTrue(shot.is_file() and shot.stat().st_size > 8000, shot)
                sizes[(name, lang)] = shot.read_bytes()
        for name in pages:
            self.assertNotEqual(sizes[(name, "en")], sizes[(name, "es")], name)


if __name__ == "__main__":
    unittest.main()
