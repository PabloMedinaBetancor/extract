"""Take the screenshots of the four screens, in English and Spanish, into docs/screenshots/.

The Read screen needs a finished job, so one is run here with the test engine over two small PDFs; the other screens
open the saved fixture documents. Needs google-chrome.

The screens show the folder they browse, so the script refuses to run unless the home folder is the neutral
`/home/user`. On a machine with another user name, give it that home in a throw-away mount namespace, for example

    bwrap --dev-bind / / --bind "$HOME" /home/user --tmpfs /home/user/Documents --dir /home/user/Documents/samples \\
          --bind "$PWD" /home/user/extract --setenv HOME /home/user --setenv EXTRACT_SHOTS_NO_SANDBOX 1 \\
          --chdir /home/user/extract python3 scripts/ui_screenshots.py

(`EXTRACT_SHOTS_NO_SANDBOX=1` passes `--no-sandbox` to Chrome, which cannot create its own sandbox inside that
namespace.) The sample documents then sit in `~/Documents/samples`.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import replace
from pathlib import Path
from unittest import mock

import extract

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from extract.api import Document, Reader  # noqa: E402
from extract.types import Reason  # noqa: E402
from extract.ui import server as ui_server  # noqa: E402
from tests.fake_fix_guard import FixGuard, fix  # noqa: E402
from tests.support import SANS, SPANISH_LINES, SetLexicon, digital_pdf, scanned_pdf, text_page_image  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
OUT = ROOT / "docs" / "screenshots"
LEXICON = SetLexicon({w.lower().strip(".,:") for line in SPANISH_LINES for w in line.split() if len(w) > 2})
SHOTS = {
    "import": "tab=import&correct=more&second=thorough&path={fixtures}&add={fixtures}/doubtful.pdf;{fixtures}/mismatch.pdf;{fixtures}/unreadable.pdf",
    "read": "tab=read",
    "review": "tab=review&doc=0&pick=p1w9&filter=fixed",
    "export": "tab=export&export=1",
    "review-tablet": "tab=review&doc=0&pick=p1w9&filter=fixed",
    "import-phone": "tab=import&correct=more&second=thorough&path={fixtures}&add={fixtures}/doubtful.pdf;{fixtures}/mismatch.pdf",
}
# The window of each capture; the ones not listed are 1440x900. The last two show the stacked layout of a portrait
# tablet (1200x1920) and of a phone (480x900).
SIZES = {"review-tablet": "1200,1920", "import-phone": "480,900"}


def chrome(args: list[str]) -> None:
    if os.environ.get("EXTRACT_SHOTS_NO_SANDBOX"):
        args = ["--no-sandbox", *args]
    profile = tempfile.mkdtemp()
    try:
        subprocess.run(
            [
                "google-chrome",
                "--headless=new",
                "--disable-gpu",
                "--no-first-run",
                "--disable-background-networking",
                "--disable-sync",
                f"--user-data-dir={profile}",
                *args,
            ],
            capture_output=True,
            timeout=120,
            check=True,
        )
    finally:
        shutil.rmtree(profile, ignore_errors=True)


def accent_fix(word_id: str, text: str):
    proposal = fix("text", "accent", text, (word_id,))
    return replace(proposal, reasons=(Reason("guard.fix.accent_restored"),))


def fixed_demo(work: Path) -> Path:
    """The doubtful fixture with the guard's fixes applied to three words, saved next to its PDF."""
    doc = extract.load(FIXTURES / "doubtful.extract.json", FIXTURES / "doubtful.pdf")
    data = doc._data
    proposals = {"p1w1": "n\u00famero", "p1w9": "consultor\u00eda", "p1w32": "documentaci\u00f3n"}
    words = tuple(
        replace(w, fix=accent_fix(w.id, proposals[w.id])) if w.id in proposals else w for w in data.pages[0].words
    )
    page = replace(data.pages[0], words=words)
    demo = Document(replace(data, pages=(page,), options=data.options | {"correct": "off"}), FIXTURES / "doubtful.pdf")
    demo.set_mode("more")
    folder = work / "demo"
    folder.mkdir()
    shutil.copy(FIXTURES / "doubtful.pdf", folder / "doubtful.pdf")
    demo.save(folder / "doubtful.extract.json")
    return folder / "doubtful.extract.json"


def main() -> None:
    """Take the screenshots; the home folder must be the neutral one (see the module docstring)."""
    if Path.home() != Path("/home/user"):
        sys.exit("the screens show the home folder: run this with HOME=/home/user (see the docstring)")
    work = Path.home() / "Documents" / "scans"  # the server may only browse under the home folder
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    samples = Path.home() / "Documents" / "samples"
    shutil.copytree(FIXTURES, samples, dirs_exist_ok=True)
    scan = text_page_image(SPANISH_LINES, SANS, 24, 1240, 1754)
    pdfs = [
        scanned_pdf(work / "contract.pdf", [scan, scan, scan]),
        digital_pdf(work / "report.pdf", [SPANISH_LINES, SPANISH_LINES[:4]]),
    ]
    mock.patch.object(Reader, "_engine_factory", "tests.fake_engine:load_engine").start()
    mock.patch.object(ui_server, "thorough_available", lambda: True).start()
    page = work / "page.json"
    lines = [
        [[w, 0.97, [10 + 12 * n, 10 * k, 20 + 12 * n, 10 * k + 8]] for n, w in enumerate(line.split())]
        for k, line in enumerate(SPANISH_LINES)
    ]
    page.write_text(json.dumps(lines), encoding="utf-8")
    os.environ["EXTRACT_FAKE_PAGE"] = str(page)
    guard = FixGuard(
        {
            "p1w1": accent_fix("p1w1", "n\u00famero"),
            "p1w9": accent_fix("p1w9", "consultor\u00eda"),
            "p1w32": accent_fix("p1w32", "documentaci\u00f3n"),
        }
    )
    preload = [fixed_demo(work), FIXTURES / "mismatch.extract.json"]
    server, state = ui_server.make_server(
        work / "work",
        preload,
        root=Path.home(),
        reader_factory=lambda options: Reader(options, guard=guard, lexicon=LEXICON),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    state.start([str(p) for p in pdfs], {"options": {"languages": ["es"], "review": "balanced", "correct": "more"}})
    while state.job is not None and state.job.is_alive():
        state.job.join(timeout=1)
    OUT.mkdir(parents=True, exist_ok=True)
    for lang in ("en", "es"):
        for name, query in SHOTS.items():
            # The import capture ticks one language, the one of the interface, so the panel shows the measured T5 gain
            # of that language instead of "gain not measured" (it is never a mixed figure).
            only = f"&languages={lang}" if name.startswith("import") else ""
            url = f"http://127.0.0.1:{state.port}/?t={state.token}&lang={lang}&nowait=1&{query.format(fixtures=samples)}{only}"
            target = OUT / f"{name}-{lang}.png"
            chrome(
                [
                    "--virtual-time-budget=20000",
                    f"--window-size={SIZES.get(name, '1440,900')}",
                    f"--screenshot={target}",
                    url,
                ]
            )
            print(target)
    server.shutdown()
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
