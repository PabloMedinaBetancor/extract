"""Read PDFs for a web service: one thread owns the Reader, the request threads wait on a queue.

This is the script of docs/how-to/web-service.md. Run it from the repository folder:

    .venv/bin/python examples/web_service.py

It starts a small HTTP server on a free local port, posts a sample invoice and a file that is not a PDF to it, prints the
answers and stops. The standard library is all it uses besides Extract.
"""

import json
import queue
import tempfile
import threading
from concurrent.futures import Future
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import extract


class ReadingService:
    """Reads one PDF at a time, on one thread that owns the Reader. Any thread may call `submit`."""

    def __init__(self, options=None):
        self._jobs = queue.Queue()
        self._thread = threading.Thread(target=self._run, args=(options,), name="extract-reader")
        self._thread.start()

    def submit(self, path):
        """Queue `path` and return a Future whose result is the Document, or whose exception is what the read raised."""
        future = Future()
        self._jobs.put((path, future))
        return future

    def close(self):
        """Finish the jobs already queued, then stop the Reader and its workers."""
        self._jobs.put(None)
        self._thread.join()

    def _run(self, options):
        with extract.Reader(options) as reader:
            while (job := self._jobs.get()) is not None:
                path, future = job
                try:
                    future.set_result(reader.read(path))
                except BaseException as error:  # the waiting request thread re-raises it
                    future.set_exception(error)


def summary(doc):
    """What the service answers for a document."""
    return {
        "status": doc.status,
        "unreadable_pages": list(doc.unreadable_pages),
        "to_review": [{"id": word.id, "text": word.text, "suggestion": word.suggestion} for word in doc.doubtful()],
        "numbers": {
            status: sum(1 for number in doc.numbers if number.status == status)
            for status in ("checked", "mismatch", "unverified")
        },
    }


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = self.rfile.read(int(self.headers["Content-Length"]))
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "upload.pdf"
                path.write_bytes(body)
                try:
                    answer, code = summary(service.submit(path).result(timeout=300)), 200
                except (extract.UnsupportedFile, extract.PasswordProtected) as error:
                    answer, code = {"error": error.code}, 422
            data = json.dumps(answer).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def log_message(self, *args):
            pass

    return Handler


def post(url, body):
    try:
        with urlopen(Request(url, data=body, method="POST")) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def main():
    service = ReadingService()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{server.server_port}/"
    try:
        code, answer = post(url, Path("tests/fixtures/mismatch.pdf").read_bytes())
        print(code, answer["status"], [word["id"] for word in answer["to_review"]], answer["numbers"])
        code, answer = post(url, b"this is not a PDF")
        print(code, answer)
    finally:
        server.shutdown()
        service.close()


if __name__ == "__main__":
    main()
