"""A worker whose parent process has gone must go too.

The parent (the interface, or a script that reads) can be killed, or closed with its terminal, in the middle of a read.
The workers are children of a `spawn` context: nothing tells them, they wait for a task that never comes and keep their
800 MB for ever. They look for their parent between two tasks and between two pages."""

from __future__ import annotations

import multiprocessing
import queue
from typing import Any

POLL_SECONDS = 2.0


def parent_gone() -> bool:
    parent = multiprocessing.parent_process()
    return parent is not None and not parent.is_alive()


def next_task(task_queue: Any) -> Any:
    """The next item of the queue. `None`, the stop sentinel, when the parent is gone: nobody is left to send one."""
    while True:
        try:
            return task_queue.get(timeout=POLL_SECONDS)
        except queue.Empty:
            if parent_gone():
                return None
