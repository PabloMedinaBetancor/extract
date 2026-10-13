"""A stand-in for the OCR engine that the pool tests start in worker processes: instant, deterministic, and unable to
touch the network."""

import json
import os
import signal
import socket
import time

SLOW_SECONDS = float(os.environ.get("EXTRACT_FAKE_SECONDS", "0"))


def refuse_network(*args, **kwargs):
    raise AssertionError("network access attempted")


def doubt(conf):
    """With EXTRACT_FAKE_DOUBT the words carry character confidences below the selection threshold, so the second
    reading has units to read."""
    if not os.environ.get("EXTRACT_FAKE_DOUBT"):
        return conf
    from extract.engine.ocr import CharConf

    low = CharConf(0.8)
    low.chars, low.aligned = (0.8,) * 5, True
    return low


class FakeEngine:
    backend = "fake"
    fallback_reason = None

    def __init__(self, threads):
        self.threads = threads

    def read_words(self, image):
        if os.environ.get("EXTRACT_FAKE_DIE"):
            os._exit(3)
        if SLOW_SECONDS:
            time.sleep(SLOW_SECONDS)
        page = os.environ.get("EXTRACT_FAKE_PAGE")
        if page:
            with open(page, encoding="utf-8") as handle:
                return [[(t, c, tuple(b)) for t, c, b in line] for line in json.load(handle)]
        words = [(f"word{i}", doubt(0.9), (10.0 * i, 5.0, 10.0 * i + 8, 15.0)) for i in range(30)]
        return [words[:15], words[15:]]


def load_engine(threads, backend, tuning=None):
    socket.socket.connect = refuse_network
    if os.environ.get("EXTRACT_FAKE_IGNORE_TERM"):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
    return FakeEngine(threads)
