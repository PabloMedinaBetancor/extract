# Use Extract in a web service

This page puts Extract behind a web service: files arrive in requests, one thread reads them, and each request gets its
answer. It uses the standard library only, and the pattern carries over to any framework. It is for a backend developer
who has to read PDFs on a request.

## The shape

Three facts decide the design:

- A read takes seconds a page, and the first one also loads the models. Do not read in a request thread: the request
  would wait for the models, and the next would start another read.
- A `Reader` reads one document at a time, and calling `Reader.read` from two threads at once is not supported. One
  `Reader` for the whole service keeps the OCR workers and the models loaded between requests.
- Once the reading thread has returned a `Document` and nobody changes it, any number of threads may read it at the same
  time. The methods that change a document hold a lock, but the methods that read it take none, so a thread must not read
  a document while another changes it. This service never changes a document after it returns it.

So the service has one thread that owns the `Reader` and takes jobs from a queue, and a `Future` that hands each result
back to the request that asked for it.

## The reading service

```python
import queue
import threading
from concurrent.futures import Future

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
```

`submit` returns at once. `future.result(timeout=...)` blocks the caller until the read is done and returns the
`Document`, or raises what the read raised. Jobs run in the order they arrive. `close` puts a marker at the end of the
queue, so the jobs already queued are finished first, and then stops the `Reader` and its worker processes: call it when
the service shuts down. Creating the service starts a thread and loads nothing, but create it under
`if __name__ == "__main__":`, as below, because the worker processes import your main script.

## A request handler

This handler receives a PDF in the body of a POST, writes it to a temporary folder, asks the service to read it, and
answers with JSON. A file that is not a PDF, or that needs a password, is the caller's mistake, and the answer says so with
the error's `code`.

```python
import json
import tempfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path


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
```

The temporary folder lives until the answer is built, because a `Document` needs its PDF for the picture of a word and
for the searchable PDF. `summary` only reads the `Document`, which the reading thread never touches again. Any error other
than a bad upload, an `EngineLoadError` or a full disk, is the service's fault and should reach your error logging as a
500.

## Try it

The whole script starts the server on a free local port, posts the sample invoice and a file that is not a PDF, and
stops:

```python
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen


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
```

```text
# output
200 done ['p1w11', 'p1w13', 'p1w4'] {'checked': 13, 'mismatch': 3, 'unverified': 11}
422 {'error': 'error.unsupported_file'}
```

The three blocks of this page, with their imports in one place, are `examples/web_service.py`. `ThreadingHTTPServer`
gives each request its own thread, and all of them wait on the one reading thread.

## Before production

This is the shape, not a finished service. A real one also limits the size of an upload and the length of the queue,
answers before the read is done when a read may take minutes, and chooses the memory settings for its machine: each read
starts up to two workers, each budgeted 1,500 MB, and `Options(processes=1, backend="onnxruntime")` suits a small machine.
`result(timeout=300)` raises `concurrent.futures.TimeoutError` when a read takes longer, and nothing here catches it:
decide what the caller is told, and whether the read should be cancelled ([Progress and cancel](progress-and-cancel.md)).
