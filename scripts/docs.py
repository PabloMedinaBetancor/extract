"""Build, serve and check the documentation site, and print the PDF manuals.

python scripts/docs.py build   # regenerate every generated file, then build site/ with mkdocs --strict
python scripts/docs.py serve   # regenerate, then serve the site at http://127.0.0.1:8000 for editing
python scripts/docs.py check   # change nothing: generated files are current, the site builds, every example runs
python scripts/docs.py pdf [OUT.pdf] [--manual user|developer]   # print the documentation to a PDF manual with headless Chrome

`check` is the gate: it exits 0 when the generated files equal the committed ones, `mkdocs build --strict` has no
warning, the built site asks no other host for anything, every ```python block of the pages and every `examples/*.py`
runs on the real models, every internal link and anchor resolves, and no banned phrase or personal detail is in a page.
It prints a short summary and writes the full output to `work/docs-check/`. The examples read real PDFs: expect
several minutes. Run it niced (it lowers its own priority) and one at a time.

MkDocs comes from `requirements/docs.txt`; install it in a separate environment and name that interpreter in
`DOCS_PYTHON` (default: the interpreter running this script; it also assembles the PDF pages, so it needs the
Markdown, Pygments and PyYAML that requirements/docs.txt brings). The examples run with the interpreter running this script,
so that one needs Extract installed (or `PYTHONPATH=src`) and `pytest`.

`--help` prints this text and does nothing else.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
LLMS_FULL = ROOT / "llms-full.txt"
MKDOCS_YML = ROOT / "mkdocs.yml"
OUT = ROOT / "work" / "docs-check"

CHECK_TESTS = [
    "tests/test_docs_generated.py",
    "tests/test_docs_style.py",
    "tests/test_docs_links.py",
    "tests/test_docs_pdf.py",
    "tests/test_docs_examples.py",
    "tests/test_examples.py",
]


# --- generated files ----------------------------------------------------------------------------------------------


def _name(path: Path) -> str:
    return path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else path.name


def nav_pages() -> list[str]:
    """The page paths of `mkdocs.yml`'s `nav`, in order (every `- Title: path.md` line)."""
    if not MKDOCS_YML.is_file():
        return []
    text = MKDOCS_YML.read_text(encoding="utf-8")
    nav = text.split("\nnav:", 1)[1] if "\nnav:" in text else ""
    return re.findall(r":\s+([\w./-]+\.md)\s*$", nav, re.M)


def excluded_pages() -> list[str]:
    """The pages that `exclude_docs` in `mkdocs.yml` keeps out of the site, as paths relative to `docs/`, one per line.
    They stay in the repository; the site does not build them, so no page of the site may link to them."""
    if not MKDOCS_YML.is_file():
        return []
    found = re.search(r"^exclude_docs:\s*\|\n((?:[ \t]+\S.*\n?)+)", MKDOCS_YML.read_text(encoding="utf-8"), re.M)
    if not found:
        return []
    lines = [line.strip() for line in found.group(1).splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def render_llms_full() -> str:
    """`llms-full.txt`: every page of the site as one plain file, in navigation order and then any other page the site
    builds; the pages `exclude_docs` keeps out of the site are left out. Each page follows a line `# docs/path.md`; the
    page text is written as it is."""
    excluded = set(excluded_pages())
    pages = []
    for name in nav_pages():
        if (DOCS / name).is_file() and name not in pages and name not in excluded:
            pages.append(name)
    for path in sorted(DOCS.rglob("*.md")) if DOCS.is_dir() else []:
        name = path.relative_to(DOCS).as_posix()
        if name not in pages and name not in excluded:
            pages.append(name)
    parts = []
    for name in pages:
        body = (DOCS / name).read_text(encoding="utf-8")
        parts.append(f"# docs/{name}\n\n{body.rstrip()}\n")
    return "\n".join(parts)


def generated_files() -> dict[Path, str]:
    """The files this script writes itself, with what they must contain: `llms-full.txt`. (The API reference and the
    measured block come from `scripts/api_docs.py`, which has its own `--check`.)"""
    return {LLMS_FULL: render_llms_full()}


def regenerate() -> list[str]:
    """Rewrite every generated file; return the names of those that changed."""
    changed = []
    api = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "api_docs.py")], cwd=ROOT, capture_output=True, text=True
    )
    if api.returncode != 0:
        raise SystemExit("scripts/api_docs.py failed:\n" + (api.stderr or api.stdout)[-1500:])
    for path, text in generated_files().items():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
            changed.append(_name(path))
    return changed


def stale_files() -> list[str]:
    """Names of generated files that differ from what their script writes now. Changes nothing."""
    stale = []
    for path, text in generated_files().items():
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            stale.append(_name(path))
    api = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "api_docs.py"), "--check"], cwd=ROOT, capture_output=True, text=True
    )
    if api.returncode != 0:
        stale.append("docs/API.md or docs/MEASURED.md or the reference pages (scripts/api_docs.py --check)")
    return stale


# --- the site -----------------------------------------------------------------------------------------------------


def docs_python() -> str:
    return os.environ.get("DOCS_PYTHON", sys.executable)


def mkdocs_command(*args: str) -> list[str]:
    return [docs_python(), "-m", "mkdocs", *args]


def mkdocs_env() -> dict:
    env = dict(os.environ)
    env["NO_MKDOCS_2_WARNING"] = "true"  # the theme's notice about the next MkDocs version is not about this site
    return env


def build_site(destination: Path) -> tuple[bool, str]:
    done = subprocess.run(
        mkdocs_command("build", "--strict", "--clean", "-d", str(destination)),
        cwd=ROOT,
        env=mkdocs_env(),
        capture_output=True,
        text=True,
    )
    log = done.stdout + done.stderr
    if done.returncode != 0 and "No module named mkdocs" in log:
        log += "\nInstall the toolchain: python -m pip install --require-hashes -r requirements/docs.txt, and set DOCS_PYTHON."
    return done.returncode == 0, log


def outside_requests(site: Path) -> list[str]:
    sys.path.insert(0, str(ROOT))
    try:
        from tests import docs_support
    finally:
        sys.path.remove(str(ROOT))
    return docs_support.outside_requests(site)


def cmd_build(_args) -> int:
    changed = regenerate()
    print("regenerated:", ", ".join(changed) if changed else "nothing to change")
    ok, log = build_site(ROOT / "site")
    print(log.rstrip())
    if not ok:
        return 1
    problems = outside_requests(ROOT / "site")
    for problem in problems:
        print("outside request:", problem)
    return 1 if problems else 0


def cmd_serve(_args) -> int:
    regenerate()
    return subprocess.call(mkdocs_command("serve"), cwd=ROOT, env=mkdocs_env())


# --- check --------------------------------------------------------------------------------------------------------


PRIVATE_NAMES_FILE = "docs-private-names.txt"


def private_names_path() -> str | None:
    """The private internal-names list: EXTRACT_DOCS_PRIVATE_NAMES, else the file beside the docs venv's folder."""
    named = os.environ.get("EXTRACT_DOCS_PRIVATE_NAMES")
    if named:
        return named if Path(named).is_file() else None
    python = os.environ.get("DOCS_PYTHON")
    if python:
        beside = Path(python).absolute().parents[2] / PRIVATE_NAMES_FILE
        if beside.is_file():
            return str(beside)
    return None


def run_tests(report: Path, junit: Path, log: Path) -> int:
    env = dict(os.environ)
    env.update(EXTRACT_DOCS="1", EXTRACT_DOCS_REPORT=str(report))
    private = private_names_path()
    if private:
        env["EXTRACT_DOCS_PRIVATE_NAMES"] = private
    else:
        env.pop("EXTRACT_DOCS_PRIVATE_NAMES", None)
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", f"--junitxml={junit}", *CHECK_TESTS]
    with open(log, "w", encoding="utf-8") as out:
        return subprocess.run(command, cwd=ROOT, env=env, stdout=out, stderr=subprocess.STDOUT).returncode


def example_summary(report: Path) -> tuple[int, list[str], list[str]]:
    """(blocks run, failures as `page block N: error`, skips as `page block N: reason`) from the runner's report."""
    run, failures, skips = 0, [], []
    if report.is_file():
        for line in report.read_text(encoding="utf-8").splitlines():
            entry = json.loads(line)
            for r in entry["results"]:
                name = (
                    f"{entry['page']} block {r['number']}"
                    if not entry["page"].startswith("examples/")
                    else entry["page"]
                )
                if r["status"] in ("ok", "fail"):
                    run += 1
                if r["status"] == "fail":
                    failures.append(f"{name}: {str(r['error']).splitlines()[-1] if r['error'] else 'failed'}")
                if r["status"] == "skipped":
                    skips.append(f"{name}: {r['error']}")
    return run, failures, skips


def junit_failures(junit: Path) -> dict[str, list[str]]:
    """Failed test cases by test file, each as `test name: first line of the message`."""
    found: dict[str, list[str]] = {}
    if not junit.is_file():
        return found
    for case in ET.parse(junit).getroot().iter("testcase"):
        bad = case.find("failure")
        bad = bad if bad is not None else case.find("error")
        if bad is not None:
            module = (
                case.get("classname", "").split(".")[1]
                if "." in case.get("classname", "")
                else case.get("classname", "")
            )
            message = (bad.get("message") or "").strip().splitlines()
            found.setdefault(module, []).append(f"{case.get('name')}: {message[0] if message else 'failed'}")
    return found


def cmd_check(args) -> int:
    if not args.no_nice:
        try:
            os.nice(19)
        except OSError:
            pass
    OUT.mkdir(parents=True, exist_ok=True)
    failed = False

    stale = stale_files()
    print(f"generated files: {'all current' if not stale else 'STALE: ' + ', '.join(stale)}")
    failed |= bool(stale)

    site = Path(tempfile.mkdtemp(prefix="extract-site-"))
    try:
        ok, log = build_site(site)
        (OUT / "mkdocs.log").write_text(log, encoding="utf-8")
        warnings = [ln for ln in log.splitlines() if ln.startswith(("WARNING", "ERROR", "Aborted"))]
        print(
            f"mkdocs build --strict: {'ok' if ok else 'FAILED'}"
            + ("" if ok else f" ({len(warnings)} lines; see {OUT / 'mkdocs.log'})")
        )
        for line in warnings[:8]:
            print("   ", line)
        failed |= not ok
        if ok:
            problems = outside_requests(site)
            print(f"requests outside the site: {'none' if not problems else len(problems)}")
            for problem in problems[:8]:
                print("   ", problem)
            failed |= bool(problems)
    finally:
        shutil.rmtree(site, True)

    report, junit, log = OUT / "examples.jsonl", OUT / "junit.xml", OUT / "pytest.log"
    report.unlink(missing_ok=True)
    junit.unlink(missing_ok=True)
    code = run_tests(report, junit, log)
    print(f"private-names scan: {'on' if private_names_path() else 'off'}")
    run, failures, skips = example_summary(report)
    print(
        f"examples: {run} blocks and scripts run, {len(failures)} fail"
        + (": " + "; ".join(failures[:6]) if failures else "")
    )
    for skip in skips:
        print("   skipped", skip)
    by_file = junit_failures(junit)
    for label, module in (
        ("generated files", "test_docs_generated"),
        ("style and privacy", "test_docs_style"),
        ("links and anchors", "test_docs_links"),
    ):
        bad = by_file.get(module, [])
        print(f"{label}: {'ok' if not bad else str(len(bad)) + ' fail'}")
        for line in bad[:6]:
            print("   ", line)
    failed |= code != 0 or bool(failures)
    if code != 0 and not (failures or any(by_file.values())):
        print(f"pytest exit {code} with no failing test reported: see {log}")
    print(f"full output: {OUT}")
    return 1 if failed else 0


# --- the PDF manual -------------------------------------------------------------------------------------------------

OFFLINE_FLAGS = [
    # The same flags the Studio's own PDF report uses: a fresh Chrome profile otherwise talks to its makers' servers.
    "--proxy-server=socks5://127.0.0.1:1",
    "--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE localhost",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-sync",
    "--disable-default-apps",
    "--disable-domain-reliability",
    "--disable-client-side-phishing-detection",
    "--no-pings",
    "--disable-breakpad",
]
MIN_FREE_MB = 900
CHROME_TIMEOUT = 300


def free_mb() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) // 1024
    return 10**6  # not Linux: no check


def find_chrome() -> str | None:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser", "microsoft-edge", "msedge"):
        found = shutil.which(name)
        if found:
            return found
    return None


def page_count(pdf: Path) -> int | None:
    if shutil.which("pdfinfo"):
        done = subprocess.run(["pdfinfo", str(pdf)], capture_output=True, text=True)
        found = re.search(r"^Pages:\s+(\d+)", done.stdout, re.M)
        return int(found[1]) if found else None
    return len(re.findall(rb"/Type\s*/Page[^s]", pdf.read_bytes())) or None


def chrome_print(chrome: str, work: Path, page: Path, target: Path) -> bool:
    """Print `page` to `target` with a headless browser at low priority; False when it fails or times out."""
    command = [
        "nice",
        "-n",
        "19",
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-first-run",
        "--hide-scrollbars",
        f"--user-data-dir={work / 'profile'}",
        f"--print-to-pdf={target}",
        "--no-pdf-header-footer",
        # the PDF's Creator field is the browser's user agent: say what the file is, not which machine made it
        "--user-agent=Extract manual",
        *OFFLINE_FLAGS,
        page.as_uri(),
    ]
    proc = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    try:
        proc.wait(timeout=CHROME_TIMEOUT)
    except subprocess.TimeoutExpired:
        print(f"Chrome did not finish in {CHROME_TIMEOUT} s; stopping it")
        return False
    finally:
        # Chrome closes by itself after printing; whatever it left (helper processes) belongs to the group this script started.
        try:
            os.killpg(proc.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
    return True


def sheet_numbers(pdf: Path, titles_file: Path) -> dict[str, int]:
    """The sheet each page of the manual starts on, read from the printed PDF: every page begins on a new sheet, right
    under the running header, with its first heading. Pages are matched in navigation order; a page not found is left out."""
    try:
        titles = json.loads(titles_file.read_text(encoding="utf-8"))
        text = subprocess.run(["pdftotext", str(pdf), "-"], capture_output=True, text=True, timeout=120).stdout
    except (OSError, ValueError, subprocess.SubprocessError):
        return {}
    sheets = [
        re.sub(r"^Extract \S+, (?:user manual|developer reference|manual)\s+", "", " ".join(chunk.split()))
        for chunk in text.split("\f")
    ]
    numbers: dict[str, int] = {}
    at = 1  # the cover is sheet 1 and carries no running header
    for path, title in titles.items():
        wanted = " ".join(title.split())[:50]
        for index in range(at, len(sheets)):
            if wanted and sheets[index].startswith(wanted):
                numbers[path] = index + 1
                at = index + 1
                break
    return numbers


def cmd_pdf(args) -> int:
    chrome = find_chrome()
    if not chrome:
        print("no Chrome, Chromium or Edge found on this computer (needed to print the PDF)")
        return 1
    if free_mb() < MIN_FREE_MB and not args.force:
        print(
            f"only {free_mb()} MB of memory is free; the printing needs about {MIN_FREE_MB} MB. Close something, or use --force."
        )
        return 1
    regenerate()
    work = Path(tempfile.mkdtemp(prefix="extract-pdf-"))
    manual = getattr(args, "manual", None)
    extra = ["--manual", manual] if manual else []
    default_name = {
        None: "extract-manual.pdf",
        "user": "extract-user-manual.pdf",
        "developer": "extract-developer-reference.pdf",
    }[manual]
    target = Path(args.out).resolve() if args.out else ROOT / "work" / default_name
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        page = work / "manual.html"
        done = subprocess.run(
            [docs_python(), str(ROOT / "scripts" / "docs_pdf.py"), str(page), *extra],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        if done.returncode != 0:
            print("could not assemble the print page:\n" + (done.stderr or done.stdout)[-1500:])
            return 1
        started = time.time()
        if not chrome_print(chrome, work, page, target):
            return 1
        # A second printing puts the page numbers into the contents list: the first one finds where each page starts.
        numbers = sheet_numbers(target, work / "manual.titles.json")
        if numbers:
            numbered = work / "manual-numbered.json"
            numbered.write_text(json.dumps(numbers), encoding="utf-8")
            done = subprocess.run(
                [docs_python(), str(ROOT / "scripts" / "docs_pdf.py"), str(page), str(numbered), *extra],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if done.returncode == 0 and not chrome_print(chrome, work, page, target):
                return 1
        if not target.is_file() or target.stat().st_size < 1000:
            print("Chrome wrote no PDF")
            return 1
        pages = page_count(target)
        print(
            f"{target}: {pages if pages else '?'} pages, {target.stat().st_size // 1024} kB, {time.time() - started:.0f} s"
        )
        return 0
    finally:
        shutil.rmtree(work, True)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="python scripts/docs.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="regenerate, then build site/ with mkdocs --strict").set_defaults(run=cmd_build)
    sub.add_parser("serve", help="regenerate, then serve the site for editing").set_defaults(run=cmd_serve)
    check = sub.add_parser(
        "check", help="verify generated files, the build, the examples, links and style; changes nothing"
    )
    check.add_argument("--no-nice", action="store_true", help="keep the normal priority (the default lowers it to 19)")
    check.set_defaults(run=cmd_check)
    pdf = sub.add_parser("pdf", help="print the documentation to one PDF manual with headless Chrome")
    pdf.add_argument("out", nargs="?", help="where to write it (default: work/extract-manual.pdf)")
    pdf.add_argument(
        "--manual",
        choices=("user", "developer"),
        help="print only the user manual (no code) or the developer reference, each a shorter PDF",
    )
    pdf.add_argument("--force", action="store_true", help="print even when little memory is free")
    pdf.set_defaults(run=cmd_pdf)
    args = parser.parse_args(argv)
    return args.run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
