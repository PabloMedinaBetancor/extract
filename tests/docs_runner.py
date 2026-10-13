"""Runs documentation examples in a fresh process, so worker processes and the `__main__` guard behave as they do for a reader.

python tests/docs_runner.py samples FOLDER          # copy the sample files into FOLDER and save a real reading of each
python tests/docs_runner.py page SPEC.json REPORT.json   # run the blocks named in SPEC.json, write one result per block

Not meant to be run by hand: `tests/docs_support.py`'s callers (`test_docs_examples.py`, `test_examples.py`) start it.
Needs `extract` importable (`PYTHONPATH=src`).
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import shutil
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The names the docs use for the three sample documents (DOCS-PLAN section 4) and the fixtures behind them.
SAMPLES = {"scan.pdf": "doubtful.pdf", "invoice.pdf": "mismatch.pdf", "damaged.pdf": "unreadable.pdf"}


def make_samples(folder: Path, saved_readings: Path | None = None) -> None:
    """Fill `folder` with the sample PDFs, `scans/`, a `tests/fixtures/` copy and a saved reading of each PDF.

    The saved readings come from a real read, never from the scripted fixtures, so a block that loads one sees what a
    reader of the docs would get. With `saved_readings`, copy readings made earlier (same names) instead of reading.
    """
    import extract

    folder.mkdir(parents=True, exist_ok=True)
    fixtures = ROOT / "tests" / "fixtures"
    (folder / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    for source in fixtures.iterdir():
        shutil.copy2(source, folder / "tests" / "fixtures" / source.name)
    (folder / "scans").mkdir(exist_ok=True)
    for name, fixture in SAMPLES.items():
        shutil.copy2(fixtures / fixture, folder / name)
        shutil.copy2(fixtures / fixture, folder / "scans" / name)
    if saved_readings is not None:
        for saved in saved_readings.glob("*.extract.json"):
            shutil.copy2(saved, folder / saved.name)
        return
    work = Path(os.environ.get("EXTRACT_DOCS_WORK", folder.parent / "docs-work"))
    options = extract.Options(work_dir=work)
    with extract.Reader(options) as reader:
        for name in SAMPLES:
            doc = reader.read(folder / name)
            doc.save(folder / (Path(name).stem + ".extract.json"))


def run_page(spec_path: Path, report_path: Path) -> int:
    """Run the blocks of one page in order, in one namespace, in the current folder. Stops at the first failure."""
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "__main__"}
    results = []
    failed = False
    for block in spec["blocks"]:
        entry = {
            "number": block["number"],
            "line": block["line"],
            "status": "ok",
            "error": None,
            "output_mismatch": None,
        }
        if failed:
            entry["status"] = "not-run"
        elif block["skip_reason"]:
            entry["status"] = "skipped"
            entry["error"] = block["skip_reason"]
        else:
            captured = io.StringIO()
            try:
                with contextlib.redirect_stdout(captured):
                    exec(compile(block["code"], f"{spec['page']} block {block['number']}", "exec"), namespace)  # noqa: S102
            except BaseException:  # noqa: BLE001 - an example may fail in any way, and the report says how
                failed = True
                entry["status"] = "fail"
                entry["error"] = "".join(traceback.format_exception_only(*sys.exc_info()[:2])).strip()
                entry["traceback"] = traceback.format_exc()[-1500:]
            else:
                expected = block["expected_output"]
                if expected is not None and captured.getvalue().strip() != expected.strip():
                    failed = True
                    entry["status"] = "fail"
                    entry["error"] = "output differs from the `# output` block that follows"
                    entry["output_mismatch"] = {"expected": expected.strip(), "actual": captured.getvalue().strip()}
        results.append(entry)
    report_path.write_text(json.dumps({"page": spec["page"], "results": results}, indent=1), encoding="utf-8")
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    if len(argv) >= 2 and argv[0] == "samples":
        make_samples(Path(argv[1]), Path(argv[2]) if len(argv) > 2 else None)
        return 0
    if len(argv) == 3 and argv[0] == "page":
        return run_page(Path(argv[1]), Path(argv[2]))
    print(__doc__)
    return 0 if any(a in ("-h", "--help") for a in argv) else 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
