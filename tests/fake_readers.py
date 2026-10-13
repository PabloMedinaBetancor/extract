"""Stand-in recognisers for the reader-process tests: no model, no network. What they return for every crop comes from
the environment, so a test decides what the readers "see"."""

import os
import socket
import time


def refuse_network(*args, **kwargs):
    raise AssertionError("network access attempted")


class FakeReaders:
    def __init__(self, names, threads=2, backend="auto"):
        self.names = tuple(names)
        self.calls = 0

    def key(self):
        return "+".join(f"{name}:fake" for name in self.names) + os.environ.get("EXTRACT_FAKE_READERS_VERSION", "")

    def read_many(self, name, crops):
        socket.socket.connect = refuse_network  # only ever called in the reader process, never in the test's
        if os.environ.get("EXTRACT_FAKE_READER_DIE"):
            os._exit(4)
        if os.environ.get("EXTRACT_FAKE_READER_SECONDS"):
            time.sleep(float(os.environ["EXTRACT_FAKE_READER_SECONDS"]))
        marker = os.environ.get("EXTRACT_FAKE_READER_BALLOON")  # "<file>:<MB>": once, while the file exists
        if marker:
            path, _, megabytes = marker.rpartition(":")
            if os.path.exists(path):
                os.remove(path)
                self.ballast = bytearray(int(megabytes) * 1024 * 1024)
                for i in range(0, len(self.ballast), 4096):
                    self.ballast[i] = 1
                time.sleep(30)
        keep = os.environ.get("EXTRACT_FAKE_READER_KEEP")  # "<MB>": hold that much memory for the rest of the process
        if keep and not hasattr(self, "kept"):
            self.kept = bytearray(int(keep) * 1024 * 1024)
            for i in range(0, len(self.kept), 4096):
                self.kept[i] = 1
        log = os.environ.get("EXTRACT_FAKE_READER_LOG")
        if log:
            with open(log, "a", encoding="utf-8") as handle:
                handle.write(f"{name} {len(crops)}\n")
        text = os.environ.get("EXTRACT_FAKE_READ", " ".join(f"word{i}" for i in range(15)))
        return [(text, 0.9)] * len(crops)
