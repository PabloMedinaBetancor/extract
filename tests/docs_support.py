"""Shared helpers of the documentation tests (standard library only).

What lives here: finding the documentation files, reading their fenced code blocks, the banned-phrase scan, the link and
anchor check, the privacy scan, and the scan of a built site for requests that leave it. The tests
(`test_docs_*.py`, `test_examples.py`) and `scripts/docs.py` call these; none of them changes a file.
"""

from __future__ import annotations

import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"

# Pages written by hand or generated, as the reader meets them. A generated copy of another file would be listed in
# GENERATED_COPIES, which the style scan leaves out (the freshness test checks it against its source); there is none.
ROOT_PAGES = ("README.md", "QUICKSTART.md", "AGENTS.md", "llms.txt")
GENERATED_COPIES: set[str] = set()


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:  # outside the repository (a test's temporary folder)
        return path.as_posix()


def doc_pages() -> list[Path]:
    """Every Markdown page under `docs/`, then the root pages that exist, in a stable order."""
    pages = sorted(DOCS.rglob("*.md")) if DOCS.is_dir() else []
    pages += [ROOT / name for name in ROOT_PAGES if name.endswith(".md") and (ROOT / name).is_file()]
    return pages


def prose_files() -> list[Path]:
    """The files the style and privacy scans read: pages, `llms.txt`, the changelog's source."""
    files = [p for p in doc_pages() if rel(p) not in GENERATED_COPIES]
    for name in ("llms.txt", "CHANGELOG.md", "mkdocs.yml"):
        if (ROOT / name).is_file() and ROOT / name not in files:
            files.append(ROOT / name)
    return files


# --- fenced blocks ------------------------------------------------------------------------------------------------

FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<mark>`{3,}|~{3,})[ \t]*(?P<lang>[\w+.-]*)[ \t]*(?P<rest>.*)$")
SKIP_COMMENT = re.compile(r"^\s*<!--\s*docs-test:\s*skip\s*(?:-\s*|:\s*)(?P<reason>.+?)\s*-->\s*$")


@dataclass(frozen=True)
class Block:
    page: str  # path relative to the repository root, as written in a failure message
    number: int  # 1-based among the ```python blocks of the page
    line: int  # 1-based line of the opening fence
    code: str
    expected_output: str | None  # text of a following ```text block that starts with `# output`
    skip_reason: str | None  # from a `<!-- docs-test: skip - reason -->` line right before the fence

    @property
    def name(self) -> str:
        return f"{self.page} block {self.number}"


@dataclass(frozen=True)
class Fence:
    lang: str
    start: int  # 1-based line of the opening fence
    end: int  # 1-based line of the closing fence (or the last line when it never closes)
    body: tuple[str, ...]
    indent: str


def fences(text: str) -> list[Fence]:
    """Every fenced block of a Markdown text, in order. A fence closes with the same character, at least as long."""
    lines = text.splitlines()
    found, i = [], 0
    while i < len(lines):
        m = FENCE.match(lines[i])
        if not m or (m["mark"][0] == "`" and "`" in m["rest"]):
            i += 1
            continue
        mark, indent = m["mark"], m["indent"]
        j = i + 1
        while j < len(lines):
            close = FENCE.match(lines[j])
            if (
                close
                and close["mark"][0] == mark[0]
                and len(close["mark"]) >= len(mark)
                and not close["lang"]
                and not close["rest"]
            ):
                break
            j += 1
        body = [ln[len(indent) :] if ln.startswith(indent) else ln.lstrip() for ln in lines[i + 1 : j]]
        found.append(Fence(m["lang"].lower(), i + 1, min(j + 1, len(lines)), tuple(body), indent))
        i = j + 1
    return found


def python_blocks(path: Path) -> list[Block]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    all_fences = fences(text)
    blocks = []
    for k, fence in enumerate(all_fences):
        if fence.lang not in {"python", "py", "python3"}:
            continue
        output = None
        if k + 1 < len(all_fences):
            nxt = all_fences[k + 1]
            between = lines[fence.end : nxt.start - 1]
            if (
                nxt.lang == "text"
                and not any(b.strip() for b in between)
                and nxt.body
                and nxt.body[0].strip() == "# output"
            ):
                output = "\n".join(nxt.body[1:]).strip("\n")
        skip = None
        before = fence.start - 2
        while before >= 0 and not lines[before].strip():
            before -= 1
        if before >= 0 and (m := SKIP_COMMENT.match(lines[before])):
            skip = m["reason"]
        blocks.append(Block(rel(path), len(blocks) + 1, fence.start, "\n".join(fence.body) + "\n", output, skip))
    return blocks


def strip_code(text: str) -> list[tuple[int, str]]:
    """The prose lines of a Markdown text as (line number, text): fenced blocks and HTML comments blanked, inline code
    spans replaced by a placeholder, so a scan reads only what a person reads."""
    lines = text.splitlines()
    for f in fences(text):
        for n in range(f.start - 1, f.end):
            lines[n] = ""
    prose = "\n".join(lines)
    prose = re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"), prose, flags=re.S)
    out = []
    for number, line in enumerate(prose.split("\n"), 1):
        out.append((number, re.sub(r"(`+)(?:(?!\1).)+\1", "CODE", line)))
    return out


# --- the banned words and forms: marketing words, filler openings, hedges and recap endings -------------------------

BANNED_WORDS = [
    r"let'?s dive in",
    r"dive into",
    r"delve",
    r"in today'?s fast-paced world",
    r"whether you'?re .{1,40}? or ",
    r"this guide will walk you through",
    r"look no further",
    r"seamless(?:ly)?",
    r"robust(?:ness)?",
    r"powerful",
    r"comprehensive",
    r"leverag(?:e|es|ed|ing)",
    r"cutting-edge",
    r"state-of-the-art",
    r"blazing(?:ly)?[ -]fast",
    r"effortless(?:ly)?",
    r"elegant(?:ly)?",
    r"unlock(?:s|ed|ing)?",
    r"empower(?:s|ed|ing)?",
    r"elevate[sd]?",
    r"harness(?:es|ed|ing)?",
    r"streamlin(?:e|es|ed|ing)",
    r"game-changer",
    r"revolutionis(?:e|es|ed|ing)",
    r"revolutioniz(?:e|es|ed|ing)",
    r"supercharge[sd]?",
    r"best-in-class",
    r"world-class",
    r"next-level",
    r"simply",
    r"just (?:run|use|add|install|call|type|click|copy|open|pass|set|import|follow|do|write|pip|press|launch|start)",
    r"easy",
    r"easily",
    r"it'?s important to note",
    r"it'?s worth noting",
    r"in conclusion",
    r"key takeaways",
    r"utilis(?:e|es|ed|ing)",
    r"utiliz(?:e|es|ed|ing)",
    r"ensur(?:e|es|ed|ing)",
    r"a wide range of",
    r"plethora",
    r"myriad",
    r"crucial",
    r"vital",
    r"navigate the complexities",
    r"at the end of the day",
    r"happy coding",
]
BANNED_WORD_RE = re.compile(r"(?<![\w-])(?:" + "|".join(BANNED_WORDS) + r")(?![\w-])", re.I)
# Connectors and openers that are banned only in that position.
SENTENCE_START = re.compile(r"(?:^|[.!?:]\s+)(?P<w>note that|moreover|furthermore|additionally)\b", re.I)
NOT_ONLY = re.compile(r"\bnot only\b.{1,160}?\bbut also\b", re.I)
EXCLAMATION = re.compile(r"[\w)\"'»\]]!(?=\s|$|[\"'»)\]])")
EMOJI = re.compile("[←-⇿•‣▪-◿☀-➿⬀-⯿✓✔✅❌\U0001f000-\U0001faff️‍]")
EM_DASH = "—"
EMPTY_HEADINGS = {"introduction", "overview", "conclusion", "summary"}


def style_findings(text: str, *, faq: bool = False) -> list[tuple[int, str]]:
    """(line, what was found) for every banned word or form in the prose of one Markdown text."""
    found = []
    for number, line in strip_code(text):
        for m in BANNED_WORD_RE.finditer(line):
            found.append((number, f"banned word: {m.group(0).strip()}"))
        for m in SENTENCE_START.finditer(line):
            found.append((number, f"banned opener: {m['w']}"))
        if NOT_ONLY.search(line):
            found.append((number, "banned form: not only ... but also"))
        if EM_DASH in line:
            found.append((number, "em dash"))
        if EXCLAMATION.search(line):
            found.append((number, "exclamation mark"))
        m = EMOJI.search(line)
        if m:
            found.append((number, f"emoji or decorative symbol: {m.group(0)!r}"))
        heading = re.match(r"#{1,6}\s+(.*?)\s*#*\s*$", line)
        if heading:
            title = heading.group(1).strip("*_ ").lower()
            if title in EMPTY_HEADINGS:
                found.append((number, f"empty heading: {heading.group(1)}"))
            if title.endswith("?") and not faq:
                found.append((number, "question as a heading"))
    return found


def docstring_style_findings(text: str) -> list[tuple[int, str]]:
    """The same scan over plain text (docstrings): no Markdown stripping beyond backticks."""
    found = []
    for number, line in enumerate(text.splitlines(), 1):
        line = re.sub(r"(`+)(?:(?!\1).)+\1", "CODE", line)
        for m in BANNED_WORD_RE.finditer(line):
            found.append((number, f"banned word: {m.group(0).strip()}"))
        if EM_DASH in line:
            found.append((number, "em dash"))
        if EXCLAMATION.search(line):
            found.append((number, "exclamation mark"))
        if EMOJI.search(line):
            found.append((number, "emoji or decorative symbol"))
    return found


# --- nothing personal ---------------------------------------------------------------------------------------------

EMAIL = re.compile(r"[\w.+-]+@(?!example\.(?:com|org|net|invalid)\b)[\w-]+\.[\w.-]+")
PHONE = re.compile(r"(?<![\w/-])(?<!\d[.,])(?:\+\d{1,3}[ .-]?)?(?:\d[ .-]?){8,12}\d(?![\w/-])(?![.,]\d)")
HOME_PATH = re.compile(r"/home/[a-z_][\w-]*|/Users/\w+|[A-Za-z]:\\Users\\\w+")
PROFILE_LINK = re.compile(
    r"https?://(?:www\.)?(?:upwork\.com/(?:freelancers|fl)|linkedin\.com/in|github\.com/[\w-]+/?(?=[\s)\"'>]|$))", re.I
)
# Internal names (team roles, tools, private folders) are NOT spelled in this public file. They live in a private file,
# one regex per line, named by the environment variable below; without it that part of the scan is skipped.
PRIVATE_NAMES_ENV = "EXTRACT_DOCS_PRIVATE_NAMES"


def load_internal_names(path: str | None = None) -> re.Pattern | None:
    """Compile the private name list (blank lines and lines starting with # ignored); None when there is none."""
    path = path or os.environ.get(PRIVATE_NAMES_ENV)
    if not path or not Path(path).is_file():
        return None
    parts = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines()]
    parts = [ln for ln in parts if ln and not ln.startswith("#")]
    return re.compile("|".join(f"(?:{ln})" for ln in parts)) if parts else None


# Phone-like runs that are figures: years, byte counts and the like are filtered by the shape of a phone number.


def privacy_findings(text: str, internal: re.Pattern | None = None) -> list[tuple[int, str]]:
    """(line, what) for a home path, e-mail, phone number, profile link or internal name. Code blocks are read too:
    an example may not hold them either."""
    found = []
    internal = internal or load_internal_names()
    checks = [(HOME_PATH, "home path"), (EMAIL, "e-mail address"), (PROFILE_LINK, "profile link")]
    if internal is not None:
        checks.append((internal, "internal name"))
    for number, line in enumerate(text.splitlines(), 1):
        for regex, label in checks:
            for m in regex.finditer(line):
                found.append((number, f"{label}: {m.group(0)}"))
        for m in PHONE.finditer(line):
            digits = re.sub(r"\D", "", m.group(0))
            if 9 <= len(digits) <= 13 and (m.group(0).startswith("+") or re.match(r"[6-9]\d{8}$", digits)):
                found.append((number, f"phone number: {m.group(0)}"))
    return found


# --- links and anchors --------------------------------------------------------------------------------------------

LINK = re.compile(
    r"(?<!\\)!?\[(?P<text>(?:[^\[\]]|\[[^\]]*\])*)\]\((?P<target><[^>]+>|[^()\s]*(?:\([^()]*\)[^()\s]*)*)(?:\s+\"[^\"]*\")?\)"
)
REF_DEF = re.compile(r"^\s{0,3}\[[^\]]+\]:\s+(?P<target>\S+)")
HTML_LINK = re.compile(r"""<(?:a|img)\b[^>]*?\b(?:href|src)=["'](?P<target>[^"']+)["']""", re.I)
ATTR_ID = re.compile(r"\{[^}]*#(?P<id>[\w-]+)[^}]*\}")
HTML_ID = re.compile(r"""\b(?:id|name)=["'](?P<id>[^"']+)["']""")


def slugify(text: str) -> str:
    """The slug Python-Markdown's `toc` extension (MkDocs) gives a heading."""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def heading_text(raw: str) -> str:
    raw = ATTR_ID.sub("", raw).strip()
    raw = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", raw)
    raw = re.sub(r"<[^>]+>", "", raw)
    raw = re.sub(r"[`*]", "", raw)
    return re.sub(r"(?<!\w)_|_(?!\w)", "", raw).strip()


_anchor_cache: dict[Path, set[str]] = {}


def anchors(path: Path) -> set[str]:
    """Every anchor a Markdown page offers: heading slugs (duplicates get `_1`, `_2`), `{#id}` and HTML ids."""
    path = path.resolve()
    if path in _anchor_cache:
        return _anchor_cache[path]
    text = path.read_text(encoding="utf-8")
    found, seen = set(), {}
    for number, line in strip_code(text):
        m = re.match(r"\s{0,3}#{1,6}\s+(.*?)\s*#*\s*$", line)
        if m:
            raw = text.splitlines()[number - 1]
            raw = re.sub(r"^\s{0,3}#{1,6}\s+", "", raw).rstrip("# ")
            explicit = ATTR_ID.search(raw)
            slug = explicit["id"] if explicit else slugify(heading_text(raw))
            if slug in seen:
                seen[slug] += 1
                slug = f"{slug}_{seen[slug] - 1}"
            else:
                seen[slug] = 1
            found.add(slug)
    for m in HTML_ID.finditer(text):
        found.add(m["id"])
    for m in ATTR_ID.finditer(text):
        found.add(m["id"])
    _anchor_cache[path] = found
    return found


@dataclass(frozen=True)
class LinkProblem:
    page: str
    line: int
    target: str
    problem: str

    def __str__(self) -> str:
        return f"{self.page}:{self.line}: {self.target}: {self.problem}"


def is_external(target: str) -> bool:
    return bool(re.match(r"^(?:[a-z][a-z0-9+.-]*:|//)", target, re.I))


def links_of(path: Path) -> list[tuple[int, str]]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    for f in fences(text):
        for n in range(f.start - 1, f.end):
            lines[n] = ""
    prose = "\n".join(lines)
    prose = re.sub(r"<!--.*?-->", lambda m: "\n" * m.group(0).count("\n"), prose, flags=re.S)
    out = []
    for number, line in enumerate(prose.split("\n"), 1):
        line = re.sub(r"(`+)(?:(?!\1).)+\1", "CODE", line)
        targets = [m["target"].strip("<>") for m in LINK.finditer(line)]
        targets += [m["target"] for m in HTML_LINK.finditer(line)]
        if m := REF_DEF.match(line):
            targets.append(m["target"].strip("<>"))
        out += [(number, t) for t in targets]
    return out


def site_excluded(docs: Path = DOCS) -> set[Path]:
    """The pages that `exclude_docs` in `mkdocs.yml` keeps out of the site (one path relative to `docs/` per line)."""
    config = docs.parent / "mkdocs.yml"
    if not config.is_file():
        return set()
    found = re.search(r"^exclude_docs:\s*\|\n((?:[ \t]+\S.*\n?)+)", config.read_text(encoding="utf-8"), re.M)
    if not found:
        return set()
    lines = [line.strip() for line in found.group(1).splitlines()]
    return {(docs / line).resolve() for line in lines if line and not line.startswith("#")}


def check_links(
    pages: list[Path], excluded: set[Path] | None = None, site_root: Path = DOCS
) -> tuple[list[LinkProblem], list[tuple[str, int, str]]]:
    """(problems, external links). A relative link must name a file or folder that exists, a `#anchor` a heading or id
    of the page it points at; a page under `docs/` may not link outside `docs/` (the site cannot follow it), and a page
    of the site may not link to a page that `exclude_docs` keeps out of it (the link would be dead on the site)."""
    excluded = site_excluded() if excluded is None else excluded
    problems, external = [], []
    for page in pages:
        in_site = site_root.resolve() in page.resolve().parents and page.resolve() not in excluded
        for number, target in links_of(page):
            if is_external(target):
                if not target.lower().startswith(("mailto:", "tel:")):
                    external.append((rel(page), number, target))
                else:
                    problems.append(LinkProblem(rel(page), number, target, "a mail or phone link in a public page"))
                continue
            path_part, _, fragment = target.partition("#")
            path_part = unquote(path_part.split("?")[0])
            dest = page if not path_part else (page.parent / path_part).resolve()
            if path_part and not dest.exists():
                problems.append(LinkProblem(rel(page), number, target, "no such file"))
                continue
            if path_part and DOCS in page.resolve().parents and DOCS not in (dest, *dest.parents):
                problems.append(LinkProblem(rel(page), number, target, "leaves docs/, so the site cannot follow it"))
            if path_part and in_site and dest in excluded:
                problems.append(
                    LinkProblem(
                        rel(page), number, target, "the site leaves that page out: name it as a path in code font"
                    )
                )
            if fragment and dest.suffix == ".md" and dest.is_file() and unquote(fragment) not in anchors(dest):
                problems.append(LinkProblem(rel(page), number, target, "no such anchor"))
    return problems, external


# --- the built site makes no request outside itself -----------------------------------------------------------------

URL_ATTR = re.compile(
    r"""<(?P<tag>link|script|img|source|iframe|video|audio|embed|object|form|a)\b(?P<attrs>[^>]*)>""", re.I
)
ATTR = re.compile(r"""\b(?P<name>href|src|srcset|poster|data|action)\s*=\s*(?:"(?P<d>[^"]*)"|'(?P<s>[^']*)')""", re.I)
CSS_URL = re.compile(r"""(?:url\(\s*["']?|@import\s+["']?)(?P<u>(?:https?:)?//[^)"'\s]+)""", re.I)
JS_LITERAL = re.compile(
    r"""(?:fetch|import|importScripts|XMLHttpRequest\(\)\.open|sendBeacon|new Worker|new EventSource|new WebSocket)\(\s*[\w.,\s"']{0,40}?["'](?P<u>(?:https?:)?//[^"']+)["']"""
)
# Strings in Material's script that could become a request only with a setting this site does not use.
DORMANT_JS = (
    ("api.github.com", "repo_url", "GitHub repository facts load only when `repo_url` is set in mkdocs.yml"),
    ("unpkg.com/mermaid", 'class="mermaid"', "Mermaid loads from unpkg.com when a page has a diagram: vendor it first"),
)


def outside_requests(site: Path) -> list[str]:
    """Every place in a built site that would make the browser ask another host: a script, style, font, image, frame or
    form target on another origin, `url()` and `@import` in CSS, a literal URL in a `fetch`-like call, and Material's
    own dormant requests that this site's settings would switch on. Plain `<a href>` links in the text are not requests."""
    problems = []
    html_pages = list(site.rglob("*.html"))
    for page in html_pages:
        text = page.read_text(encoding="utf-8", errors="replace")
        for m in URL_ATTR.finditer(text):
            tag = m["tag"].lower()
            for a in ATTR.finditer(m["attrs"]):
                value = a["d"] if a["d"] is not None else a["s"]
                name = a["name"].lower()
                values = [v.strip().split()[0] for v in value.split(",") if v.strip()] if name == "srcset" else [value]
                for v in values:
                    if not re.match(r"^(?:https?:)?//", v):
                        continue
                    is_nav_link = tag == "a" or (
                        tag == "link"
                        and re.search(r'rel=["\'](?:canonical|alternate|next|prev|author|help)["\']', m["attrs"])
                    )
                    if not is_nav_link or name == "src":
                        problems.append(f"{page.relative_to(site).as_posix()}: <{tag} {name}> {v}")
        for _needle, setting, why in DORMANT_JS:
            if (
                setting == 'class="mermaid"'
                and 'class="mermaid"' in text
                and not (site / "assets" / "javascripts" / "mermaid.min.js").exists()
            ):
                problems.append(f"{page.relative_to(site).as_posix()}: {why}")
    for css in site.rglob("*.css"):
        for m in CSS_URL.finditer(css.read_text(encoding="utf-8", errors="replace")):
            problems.append(f"{css.relative_to(site).as_posix()}: css {m['u']}")
    for js in site.rglob("*.js"):
        for m in JS_LITERAL.finditer(js.read_text(encoding="utf-8", errors="replace")):
            problems.append(f"{js.relative_to(site).as_posix()}: script request {m['u']}")
    config = (ROOT / "mkdocs.yml").read_text(encoding="utf-8") if (ROOT / "mkdocs.yml").is_file() else ""
    if re.search(r"^repo_url\s*:", config, re.M):
        problems.append("mkdocs.yml: repo_url is set, so Material asks api.github.com for the repository facts")
    if re.search(r"^\s*(?:analytics|provider\s*:\s*google)", config, re.M):
        problems.append("mkdocs.yml: analytics are configured")
    if re.search(r"^\s*font\s*:\s*(?!false\b)\S", config, re.M) or "font:" not in config:
        problems.append("mkdocs.yml: theme.font is not false, so fonts come from Google")
    return problems


def expect_none(test, found: list, what: str) -> None:
    """Fail `test` when `found` is not empty; the first line of the message names the count and the first findings, so a
    one-line summary (scripts/docs.py check) is useful, and the full list follows."""
    if found:
        lines = [str(f) for f in found]
        test.fail(
            f"{len(lines)} {what}: "
            + "; ".join(lines[:2])
            + ("; ..." if len(lines) > 2 else "")
            + "\n"
            + "\n".join(lines)
        )


# --- running examples -----------------------------------------------------------------------------------------------

import atexit  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402

RUNNER = ROOT / "tests" / "docs_runner.py"
REAL_MODEL_TIMEOUT = 1500  # seconds for one page of examples, workers start-up and reads included
_state: dict = {}


def enabled() -> bool:
    """The real-model example tests run when `EXTRACT_DOCS=1` (set by `scripts/docs.py check`)."""
    return os.environ.get("EXTRACT_DOCS") == "1"


def child_env() -> dict:
    env = dict(os.environ)
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else src
    env.setdefault("PYTHONUNBUFFERED", "1")
    return env


def samples() -> Path:
    """A folder with the samples of DOCS-PLAN section 4, made once per process by a real read (or the folder named by
    `EXTRACT_DOCS_SAMPLES`, made by an earlier run, whose saved readings are copied instead of read again)."""
    if "samples" in _state:
        return _state["samples"]
    base = Path(tempfile.mkdtemp(prefix="extract-docs-"))
    atexit.register(shutil.rmtree, base, True)
    folder = base / "samples"
    command = [sys.executable, str(RUNNER), "samples", str(folder)]
    earlier = os.environ.get("EXTRACT_DOCS_SAMPLES")
    if earlier:
        command.append(earlier)
    done = subprocess.run(
        command, cwd=base, env=child_env(), capture_output=True, text=True, timeout=REAL_MODEL_TIMEOUT
    )
    if done.returncode != 0:
        raise RuntimeError("could not prepare the sample files:\n" + done.stderr[-1500:])
    _state["samples"] = folder
    return folder


def run_blocks(page: str, blocks: list[Block]) -> list[dict]:
    """Run the blocks of one page in a fresh copy of the samples; return one result dict per block (status `ok`, `fail`,
    `skipped` or `not-run`). The result is also appended to the file named by `EXTRACT_DOCS_REPORT`, if any."""
    work = Path(tempfile.mkdtemp(prefix="extract-docs-run-"))
    try:
        shutil.copytree(samples(), work / "cwd")
        spec = {
            "page": page,
            "blocks": [
                {
                    "number": b.number,
                    "line": b.line,
                    "code": b.code,
                    "expected_output": b.expected_output,
                    "skip_reason": b.skip_reason,
                }
                for b in blocks
            ],
        }
        (work / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
        done = subprocess.run(
            [sys.executable, str(RUNNER), "page", str(work / "spec.json"), str(work / "report.json")],
            cwd=work / "cwd",
            env=child_env(),
            capture_output=True,
            text=True,
            timeout=REAL_MODEL_TIMEOUT,
        )
        report = work / "report.json"
        if report.is_file():
            results = json.loads(report.read_text(encoding="utf-8"))["results"]
        else:
            results = [
                {
                    "number": b.number,
                    "line": b.line,
                    "status": "fail",
                    "error": "the runner died: " + done.stderr[-600:],
                }
                for b in blocks[:1]
            ]
            results += [{"number": b.number, "line": b.line, "status": "not-run", "error": None} for b in blocks[1:]]
        path = os.environ.get("EXTRACT_DOCS_REPORT")
        if path:
            with open(path, "a", encoding="utf-8") as out:
                out.write(json.dumps({"page": page, "results": results}) + "\n")
        return results
    finally:
        shutil.rmtree(work, True)


def failure_lines(page: str, results: list[dict]) -> list[str]:
    return [f"{page} block {r['number']} (line {r['line']}): {r['error']}" for r in results if r["status"] == "fail"]
