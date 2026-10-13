from __future__ import annotations

import argparse
from pathlib import Path

import extract

from .server import serve


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m extract.ui", description="Open the Extract interface in a browser."
    )
    parser.add_argument("--work-dir", type=Path, help="where the cache, the job log and corrections.jsonl go")
    parser.add_argument("--open", nargs="*", default=[], type=Path, metavar="JSON", help="saved documents to open")
    parser.add_argument("--no-browser", action="store_true", help="print the address and do not open a browser")
    parser.add_argument("--lang", choices=("en", "es"), help="interface language")
    args = parser.parse_args()
    for saved in args.open:
        if not saved.is_file():
            parser.error(f"--open: {saved} is not a file")
    try:
        serve(work_dir=args.work_dir, preload=args.open, browser=not args.no_browser, lang=args.lang)
    except (
        extract.ExtractError,
        OSError,
        ValueError,
    ) as error:  # a saved document that cannot be read: one line, not a trace
        parser.exit(2, f"python -m extract.ui: {error}\n")


main()
