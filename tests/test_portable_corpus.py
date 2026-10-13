import hashlib
import http.server
import random
import tempfile
import threading
import unittest
from pathlib import Path

from bench import fetch_corpus


class FakeClock:
    def __init__(self):
        self.time = 1_000_000.0
        self.slept = []

    def now(self):
        return self.time

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.time += seconds


class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.server.hits.append(self.path)
        if self.path.startswith("/blocked"):
            self.send_response(429)
            self.end_headers()
            return
        body = self.server.documents.get(self.path, b"")
        self.send_response(200 if body else 404)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def start_server(documents: dict[str, bytes]) -> http.server.ThreadingHTTPServer:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.documents, server.hits = documents, []
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def row(server, path: str, body: bytes, doc_id: str, claimed: bytes | None = None) -> dict:
    digest = hashlib.sha256(body if claimed is None else claimed).hexdigest()
    return {"doc_id": doc_id, "source_url": f"http://127.0.0.1:{server.server_address[1]}{path}", "sha256": digest}


class FetchCorpusTest(unittest.TestCase):
    def setUp(self):
        self.out = Path(tempfile.mkdtemp())
        self.clock = FakeClock()
        self.addCleanup(lambda: __import__("shutil").rmtree(self.out, ignore_errors=True))

    def server(self, documents: dict[str, bytes]):
        server = start_server(documents)
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        return server

    def fetcher(self) -> fetch_corpus.Fetcher:
        return fetch_corpus.Fetcher(self.out, self.clock, random.Random(7))

    def test_P7_requests_to_one_site_are_8_to_12_seconds_apart(self):
        docs = {f"/d{i}.pdf": f"doc {i}".encode() for i in range(6)}
        server = self.server(docs)
        rows = [row(server, path, body, f"d{i}") for i, (path, body) in enumerate(docs.items())]
        report = self.fetcher().run(rows)
        self.assertEqual(report["fetched"], 6)
        self.assertEqual(len(self.clock.slept), 5)
        self.assertTrue(all(8 <= gap <= 12 for gap in self.clock.slept), self.clock.slept)
        self.assertGreater(len({round(gap, 3) for gap in self.clock.slept}), 1)

    def test_P7_the_hour_never_has_more_than_150_requests_to_a_site(self):
        docs = {f"/d{i}.pdf": f"doc {i}".encode() for i in range(160)}
        server = self.server(docs)
        rows = [row(server, path, body, f"d{i}") for i, (path, body) in enumerate(docs.items())]
        fetcher = self.fetcher()
        stamps = []
        original = fetcher.fetch_one
        fetcher.fetch_one = lambda r: (stamps.append(self.clock.now()), original(r))[1]
        fetcher.run(rows)
        self.assertEqual(len(stamps), 160)
        for i, moment in enumerate(stamps):
            in_hour = [t for t in stamps[: i + 1] if t > moment - 3600]
            self.assertLessEqual(len(in_hour), fetch_corpus.PER_HOUR)

    def test_P7_the_daily_cap_is_800_requests_to_a_site(self):
        site = "127.0.0.1:1"
        fetcher = self.fetcher()
        fetcher.requests[site] = [self.clock.now() - 86400 + i * 100 for i in range(fetch_corpus.PER_DAY)]
        self.assertGreaterEqual(fetcher.ready_at(site), fetcher.requests[site][0] + 86400)

    def test_P7_two_sites_are_paced_each_on_its_own(self):
        first, second = self.server({"/a.pdf": b"a"}), self.server({"/b.pdf": b"b"})
        rows = [row(first, "/a.pdf", b"a", "a"), row(second, "/b.pdf", b"b", "b")]
        self.fetcher().run(rows)
        self.assertEqual(self.clock.slept, [])

    def test_P7_a_changed_document_is_counted_and_not_kept(self):
        server = self.server({"/x.pdf": b"new text"})
        report = self.fetcher().run([row(server, "/x.pdf", b"new text", "x", claimed=b"old text")])
        self.assertEqual(report["changed"], ["x"])
        self.assertEqual(report["fetched"], 0)
        self.assertFalse((self.out / fetch_corpus.ORIGINALS / "x.pdf").exists())

    def test_P7_a_second_run_downloads_nothing_and_a_limit_is_respected(self):
        docs = {f"/d{i}.pdf": f"doc {i}".encode() for i in range(4)}
        server = self.server(docs)
        rows = [row(server, path, body, f"d{i}") for i, (path, body) in enumerate(docs.items())]
        self.assertEqual(self.fetcher().run(rows, limit=2)["fetched"], 2)
        self.assertEqual(len(server.hits), 2)
        self.assertEqual(self.fetcher().run(rows)["fetched"], 2)
        self.assertEqual(self.fetcher().run(rows)["fetched"], 0)
        self.assertEqual(len(server.hits), 4)

    def test_P7_the_pace_survives_a_restart(self):
        server = self.server({"/a.pdf": b"a", "/b.pdf": b"b"})
        rows = [row(server, "/a.pdf", b"a", "a"), row(server, "/b.pdf", b"b", "b")]
        self.fetcher().run(rows, limit=1)
        self.clock.slept.clear()
        self.fetcher().run(rows)
        self.assertEqual(len(self.clock.slept), 1)
        self.assertGreaterEqual(self.clock.slept[0], 8 - 1e-9)

    def test_P7_a_site_that_answers_429_is_left_alone(self):
        server = self.server({})
        rows = [row(server, f"/blocked{i}.pdf", b"x", f"b{i}") for i in range(5)]
        report = self.fetcher().run(rows)
        self.assertEqual(len(server.hits), 1)
        self.assertEqual(len(report["failed"]), 5)
        self.assertEqual(len(report["skipped_sites"]), 1)

    def test_P7_a_dry_run_asks_nobody(self):
        server = self.server({"/a.pdf": b"a"})
        text = fetch_corpus.plan([row(server, "/a.pdf", b"a", "a")], self.fetcher())
        self.assertIn("1 of 1 documents", text)
        self.assertEqual(server.hits, [])

    def test_P7_the_catalogue_written_beside_the_documents_is_the_one_the_corpus_builder_reads(self):
        from bench.corpus import catalog

        body = b"%PDF-1.4 a public document"
        digest = hashlib.sha256(body).hexdigest()
        rows = [
            {
                "doc_id": f"c{digest[:10]}",
                "source_url": "https://www.boe.es/a.pdf",
                "sha256": digest,
                "catalog_id": "7",
            }
        ]
        folder = self.out / fetch_corpus.ORIGINALS
        folder.mkdir(parents=True)
        (folder / f"{rows[0]['doc_id']}.pdf").write_bytes(body)
        self.assertEqual(fetch_corpus.write_catalogue(self.out, rows), 1)
        documents, skipped = catalog.list_documents(self.out / "catalog")
        self.assertEqual([(d.doc_id, d.catalog_id, d.sha256) for d in documents], [(rows[0]["doc_id"], 7, digest)])
        self.assertEqual(sum(skipped.values()), 0)

    def test_P7_the_real_sources_file_has_a_hash_and_a_licence_for_every_document(self):
        rows = fetch_corpus.read_sources(Path(__file__).resolve().parents[1] / "bench" / "SOURCES.csv")
        self.assertEqual(len(rows), 420)  # 275 documents of v4 + the 145 of the v5 test T5
        for r in rows:
            self.assertRegex(r["sha256"], r"^[0-9a-f]{64}$", r["doc_id"])
            self.assertTrue(r["source_url"].startswith(("http://", "https://")), r["doc_id"])
            self.assertTrue(r["licence"], r["doc_id"])
        self.assertEqual(len({r["doc_id"] for r in rows}), 420)
