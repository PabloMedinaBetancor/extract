import hashlib
import http.server
import importlib.util
import json
import os
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest import mock

from extract.engine import modelfiles
from extract.errors import EngineLoadError

ROOT = Path(__file__).resolve().parents[1]
CONTENT = bytes(range(256)) * 4000


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class RangeServer(http.server.ThreadingHTTPServer):
    """Serves CONTENT and remembers the Range header of each request."""

    def __init__(self):
        self.ranges = []
        super().__init__(("127.0.0.1", 0), RangeHandler)


class RangeHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        header = self.headers.get("Range")
        self.server.ranges.append(header)
        start = int(header.split("=")[1].rstrip("-")) if header else 0
        body = CONTENT[start:]
        self.send_response(206 if header else 200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


def entry(url: str, sha256: str = hashlib.sha256(CONTENT).hexdigest()) -> dict:
    return {"name": "m", "file": "m.onnx", "url": url, "sha256": sha256, "size": len(CONTENT), "default": True}


class FetchModelsTest(unittest.TestCase):
    def setUp(self):
        self.fetch = load_script("fetch_models")
        self.folder = Path(tempfile.mkdtemp())
        self.addCleanup(lambda: __import__("shutil").rmtree(self.folder, ignore_errors=True))
        self.source = self.folder / "source.bin"
        self.source.write_bytes(CONTENT)

    def test_P4_a_model_with_the_right_hash_is_kept(self):
        target = self.fetch.fetch(entry(self.source.as_uri()), self.folder / "out")
        self.assertEqual(target.read_bytes(), CONTENT)
        self.assertEqual(list((self.folder / "out").glob("*.part")), [])

    def test_P4_a_model_with_a_wrong_hash_fails_aloud_and_leaves_nothing(self):
        bad = entry(self.source.as_uri(), sha256="0" * 64)
        with self.assertRaisesRegex(self.fetch.HashMismatch, "sha256 is"):
            self.fetch.fetch(bad, self.folder / "out")
        self.assertEqual(list((self.folder / "out").iterdir()), [])

    def test_P4_a_corrupt_file_already_in_place_is_not_trusted(self):
        out = self.folder / "out"
        out.mkdir()
        (out / "m.onnx").write_bytes(CONTENT[:-1] + b"x")
        self.fetch.fetch(entry(self.source.as_uri()), out)
        self.assertEqual((out / "m.onnx").read_bytes(), CONTENT)

    def test_P4_the_command_line_stops_with_an_error_on_a_wrong_hash(self):
        catalogue = self.folder / "models.json"
        catalogue.write_text(json.dumps({"models": [entry(self.source.as_uri(), sha256="1" * 64)]}), encoding="utf-8")
        argv = ["fetch_models.py", "--catalogue", str(catalogue), "--dir", str(self.folder / "out")]
        with mock.patch.object(sys, "argv", argv), self.assertRaises(SystemExit) as stop:
            self.fetch.main()
        self.assertIn("ERROR", str(stop.exception))

    def test_P4_a_file_in_place_is_not_downloaded_again(self):
        out = self.folder / "out"
        self.fetch.fetch(entry(self.source.as_uri()), out)
        with mock.patch.object(self.fetch, "download", side_effect=AssertionError("downloaded again")):
            self.fetch.fetch(entry("http://127.0.0.1:1/never"), out)

    def test_P4_a_stopped_download_is_resumed_with_a_range_request(self):
        server = RangeServer()
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        out = self.folder / "out"
        out.mkdir()
        (out / "m.onnx.part").write_bytes(CONTENT[:5000])
        url = f"http://127.0.0.1:{server.server_address[1]}/m.onnx"
        self.fetch.fetch(entry(url), out)
        self.assertEqual((out / "m.onnx").read_bytes(), CONTENT)
        self.assertEqual(server.ranges, ["bytes=5000-"])

    def test_P4_the_catalogue_has_a_hash_a_size_and_a_licence_for_every_model(self):
        models = self.fetch.read_catalogue(ROOT / "models" / "models.json")
        self.assertGreaterEqual(len(models), 2)
        for model in models:
            self.assertRegex(model["sha256"], r"^[0-9a-f]{64}$", model["name"])
            self.assertGreater(model["size"], 1_000_000, model["name"])
            self.assertTrue(model["url"].startswith("https://"), model["name"])
            self.assertTrue(model["licence"], model["name"])


class ModelFilesTest(unittest.TestCase):
    def make(self, folder: Path, name: str = modelfiles.DETECTION) -> Path:
        folder.mkdir(parents=True, exist_ok=True)
        (folder / name).write_bytes(b"model")
        return folder / name

    def test_P4_the_environment_folder_wins_over_the_user_folder(self):
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            wanted = self.make(scratch / "env")
            self.make(scratch / "user")
            with (
                mock.patch.dict(
                    os.environ, {modelfiles.ENV: str(scratch / "env"), "XDG_DATA_HOME": str(scratch / "user")}
                ),
                mock.patch.object(modelfiles, "user_models_dir", return_value=scratch / "user"),
            ):
                self.assertEqual(modelfiles.find_model(modelfiles.DETECTION), wanted)

    def test_P4_the_user_folder_comes_before_the_rapidocr_package(self):
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            wanted = self.make(scratch / "user")
            self.make(scratch / "package")
            with (
                mock.patch.dict(os.environ, {}, clear=False),
                mock.patch.object(modelfiles, "user_models_dir", return_value=scratch / "user"),
                mock.patch.object(modelfiles, "search_folders", return_value=[scratch / "user", scratch / "package"]),
            ):
                self.assertEqual(modelfiles.find_model(modelfiles.DETECTION), wanted)

    def test_P4_a_model_nobody_has_stops_the_load_and_says_where_it_looked(self):
        with tempfile.TemporaryDirectory() as scratch:
            with mock.patch.dict(os.environ, {modelfiles.ENV: scratch}):
                with self.assertRaises(EngineLoadError) as caught:
                    modelfiles.find_model("nothing-like-this.onnx")
        self.assertIn(scratch, str(caught.exception))
        self.assertIn("fetch_models.py", str(caught.exception))

    def test_P4_the_fetch_script_and_the_engine_use_the_same_default_folder(self):
        fetch = load_script("fetch_models")
        with mock.patch.dict(os.environ):
            os.environ.pop(modelfiles.ENV, None)
            self.assertEqual(fetch.target_dir(None), modelfiles.user_models_dir())
        with mock.patch.dict(os.environ, {modelfiles.ENV: "/somewhere"}):
            self.assertEqual(fetch.target_dir(None), Path("/somewhere"))
