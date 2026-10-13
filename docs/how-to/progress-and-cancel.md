# Progress and cancel

This page follows a read page by page and stops one that is under way. It is for a tool with a progress bar or a Cancel
button, and for a service whose caller can go away.

## Follow a read

Pass a function as `on_event`. Extract calls it, in the thread that called `read`, with an event object at each step:

| Event | When | Fields worth knowing |
|---|---|---|
| `Started` | Once, when the read begins. | `pages`: how many pages the file has. |
| `PageRead` | After each page. | `page`, `status`, `words`, `doubtful` (the `doubtful` and `unscored` words of the page). |
| `Progress` | Right after each `PageRead`. | `done` and `total`: pages read so far, and pages in all. |
| `DocumentRead` | Once, after the last page. | `status` (`done` or `partial`), `unreadable_pages`, `doubtful`, `numbers_checked`. |
| `Warning` | Any time. | `code` and `params`: something the read noticed, such as `ram.low` when memory is short. |
| `Finished` | Last. | `status`: `done`, `stopped` after a cancel, or `failed`. |

Pages come back in the order the workers finish them, not always 1, 2, 3, and pages that are cached or digital come first.
Drive a progress bar from `Progress.done`, and use `PageRead.page` to know which page arrived. The
[events reference](../reference/events.md) lists every field and every warning code.

```python
import tempfile

import extract

SAMPLE = "tests/fixtures/unreadable.pdf"


def show(event):
    """Runs in the thread that called read, once for each event."""
    if isinstance(event, extract.Started):
        print(f"started: {event.pages} pages")
    elif isinstance(event, extract.Progress):
        print(f"progress: {event.done}/{event.total}")
    elif isinstance(event, extract.DocumentRead):
        print(f"read: {event.status}, unreadable pages {event.unreadable_pages}, {event.doubtful} to review")


def follow():
    with tempfile.TemporaryDirectory() as folder:
        extract.read(SAMPLE, options=extract.Options(work_dir=folder), on_event=show)


if __name__ == "__main__":
    follow()
```

```text
# output
started: 3 pages
progress: 1/3
progress: 2/3
progress: 3/3
read: partial, unreadable pages (2,), 11 to review
```

The read takes a new work folder on purpose. A page that was read before comes from the page cache, so a second read of
the same file in the same work folder sends its events at once and shows nothing worth following. For a test of progress,
timing or cancel, always give the read a new `work_dir`.

## Cancel a read

A `CancelToken` stops a read. Pass one as `cancel`, and call `token.cancel()` from any thread, or from the callback
itself, as here:

```python
def cancel_after_the_first_page():
    token = extract.CancelToken()
    seen = []

    def stop_at_the_first_page(event):
        seen.append(event)
        if isinstance(event, extract.PageRead):
            token.cancel()

    with tempfile.TemporaryDirectory() as folder:
        try:
            extract.read(SAMPLE, options=extract.Options(work_dir=folder), on_event=stop_at_the_first_page, cancel=token)
        except extract.Cancelled as error:
            print("stopped:", error.code, "-", seen[-1].status)


if __name__ == "__main__":
    cancel_after_the_first_page()
```

```text
# output
stopped: error.cancelled - stopped
```

The read raises `extract.Cancelled`, and the last event is `Finished` with the status `stopped`. A cancel is not instant:
the workers are told to stop, and the read returns within about two seconds. The token is looked at while the pages are
read, so a cancel that comes after the last page is too late: the read returns its `Document` as usual, and its last
event is `Finished("done")`. From another thread it is the same call:
`threading.Timer(0.5, token.cancel).start()` stops a read half a second after it starts. A token works once and stays set,
so make a new `CancelToken` for each read. `Reader.read` takes the same `on_event` and `cancel`, and the two scripts above
are `examples/progress_and_cancel.py`.

## What a cancel leaves behind

The pages that were finished stay in the page cache, in the work folder: read the file again with the same `work_dir`,
and only the rest is read. A cancelled read returns no `Document`, so nothing else is left, and a temporary work folder, as
in the examples, takes the cache with it. The interface keeps neither the stopped document nor its pages.

`Document.export` also takes a `cancel` token, but it looks at it only between formats: a format that has begun is
finished and checked, and the formats already written keep their names.

## Next

[Batch a folder](batch-a-folder.md) lists what else a read can raise. [Use Extract in a web service](web-service.md) puts
a read behind a request whose caller may go away.
