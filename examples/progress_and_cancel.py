"""Follow a read with on_event, then stop one from inside the callback.

This is the script of docs/how-to/progress-and-cancel.md. Run it from the repository folder:

    .venv/bin/python examples/progress_and_cancel.py

It reads tests/fixtures/unreadable.pdf twice. Each read gets a new work folder, because a page that was read before
comes from the page cache and is not read again, which would leave nothing to follow or to cancel.
"""

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


def main():
    follow()
    cancel_after_the_first_page()


if __name__ == "__main__":
    main()
