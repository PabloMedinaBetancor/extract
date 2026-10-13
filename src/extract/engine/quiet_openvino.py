"""Import OpenVINO without letting it report anything.

Each time OpenVINO is imported it sends an anonymous usage event (a random client id, the OpenVINO version, the
operating system and a count of uses) to Google Analytics, unless a consent file in the user's home folder says no or the
process runs under a CI service (the `CI=true` variable). Extract reads documents on the user's own computer and says
that nothing leaves it, so OpenVINO is imported here with `CI=true`, which it honours, and the previous value of the
variable is put back at once. Nothing is written to the user's home folder.

OpenVINO reads that variable only while it is imported, so the first import of a process has to go through
`import_openvino`: `engine.ocr._patch_rapidocr` does it before any engine is loaded, in the main process, in every OCR
worker and in the reader process of the second reading.
"""

from __future__ import annotations

import contextlib
import importlib
import os
from collections.abc import Iterator

SWITCH = "CI"


@contextlib.contextmanager
def openvino_quiet() -> Iterator[None]:
    """Within the block `CI` is `true`, which OpenVINO's usage reporting honours; on exit it has its old value again."""
    missing = object()
    previous = os.environ.get(SWITCH, missing)
    os.environ[SWITCH] = "true"
    try:
        yield
    finally:
        if previous is missing:
            os.environ.pop(SWITCH, None)
        else:
            os.environ[SWITCH] = str(previous)


def import_openvino() -> None:
    """Import OpenVINO with its reporting off. A package that is not installed is left for the caller to report."""
    with openvino_quiet():
        try:
            importlib.import_module("openvino")
        except ImportError:
            pass
