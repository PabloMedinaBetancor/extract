"""Assemble the documentation into one print page (the PDF manual's source). Run by `scripts/docs.py pdf`.

python scripts/docs_pdf.py OUT.html [PAGES.json] [--manual user|developer]
    # write the page; needs markdown, pymdownx, pygments and yaml (the docs venv)

PAGES.json maps a page path of the navigation to the sheet it starts on: the contents list then shows page numbers
(`scripts/docs.py pdf` prints once to find them, then prints again).

The page holds a cover (title and version only), a contents list, then every page of `mkdocs.yml`'s navigation in order,
each starting on a new sheet, with the same Markdown extensions as the site. Links between pages become links inside the
page; a link to a file that is not in the manual becomes plain text; images are read from `docs/` by absolute file path.
With `--manual user` or `--manual developer` the page holds only that manual's pages (see manuals.yml): the user manual
(no Python examples) or the developer reference (tutorial, how-to guides, API reference). A link to a page that lives in
the other manual stays readable and names that manual and page, for example "(in the Developer reference, Options)"; a
link to a page left out of both (`left_out` in manuals.yml) names the full manual: "(in the full manual, Number checks)".
Nothing is fetched: no script, no web font. `docs.py pdf` prints this page with a headless browser.
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
NOT_IN_THE_MANUAL: set[str] = set()  # every page of the navigation is in the manual
FULL_MANUAL = "full manual"  # what a link to a page of neither short manual says: the page is in the full manual

CSS = """
@page { size: A4; margin: 22mm 18mm 20mm 18mm;
  @top-left { content: "Extract VERSION, LABEL"; font: 8.5pt system-ui, 'DejaVu Sans', sans-serif; color: #666; }
  @bottom-right { content: counter(page); font: 8.5pt system-ui, 'DejaVu Sans', sans-serif; color: #666; } }
@page cover { margin: 0; @top-left { content: none; } @bottom-right { content: none; } }
html { font: 10.5pt/1.5 system-ui, 'DejaVu Sans', 'Segoe UI', Helvetica, Arial, sans-serif; color: #1b1f24; }
body { margin: 0; }
.cover { page: cover; height: 297mm; box-sizing: border-box; padding: 95mm 28mm 0 28mm; break-after: page; }
.cover h1 { font-size: 40pt; line-height: 1.1; margin: 0 0 6mm 0; border: 0; }
.cover p { font-size: 15pt; color: #444; margin: 0; }
.cover .rule { width: 40mm; border-top: 3px solid #1b1f24; margin: 8mm 0; }
.contents { break-after: page; }
.contents h2 { font-size: 20pt; border: 0; margin-top: 0; }
.contents ol { list-style: none; padding: 0; margin: 0 0 2.5mm 0; font-size: 9.5pt; }
.contents li { margin: 0.3mm 0; display: flex; align-items: baseline; }
.contents li .dots { flex: 1; border-bottom: 1px dotted #9aa1ab; margin: 0 2mm 0.8mm 2mm; }
.contents li .num { color: #444; font-variant-numeric: tabular-nums; }
.contents .part { font-weight: 600; margin-top: 3mm; break-after: avoid; }
.contents a { color: inherit; text-decoration: none; }
section.doc { break-before: page; }
h1 { font-size: 22pt; line-height: 1.2; margin: 0 0 6mm 0; padding-bottom: 2mm; border-bottom: 1px solid #c9ced6; }
h2 { font-size: 15pt; margin: 6mm 0 2.5mm 0; break-after: avoid; }
h3 { font-size: 12pt; margin: 4.5mm 0 1.5mm 0; break-after: avoid; }
h4 { font-size: 10.5pt; margin: 5mm 0 2mm 0; break-after: avoid; }
p, ul, ol, dl { margin: 0 0 2.5mm 0; orphans: 3; widows: 3; }
li { margin: 0.5mm 0; }
a { color: #0b4f9c; text-decoration: none; }
code { font: 9pt 'DejaVu Sans Mono', Consolas, Menlo, monospace; background: #f1f3f5; padding: 0.2mm 1mm; border-radius: 1mm; }
pre { font: 7.9pt/1.4 'DejaVu Sans Mono', Consolas, Menlo, monospace; background: #f6f7f9; border: 1px solid #dfe3e8;
  border-radius: 1.5mm; padding: 1.8mm 3mm; margin: 0 0 3mm 0; white-space: pre-wrap; overflow-wrap: anywhere; break-inside: auto; orphans: 5; widows: 5; }
pre code { background: none; padding: 0; font: inherit; display: block; padding-left: 5ch; text-indent: -5ch each-line; }
pre.keep { break-inside: avoid-page; }
div.highlight:has(+ div.highlight) { break-after: avoid; }  /* a code block and the output block under it stay together */
p:has(> strong:only-child) { break-after: avoid; }  /* a bold label such as "Fields" stays with the table under it */
h3 code, h4 code { font-size: 1em; font-weight: 700; background: none; padding: 0; }
.highlight { margin: 0; }
table { border-collapse: collapse; width: 100%; margin: 0 0 4mm 0; font-size: 9pt; }
th, td { border: 1px solid #cfd4db; padding: 1.4mm 2mm; text-align: left; vertical-align: top; overflow-wrap: break-word; }
th { background: #eef0f3; }
td code, th code { overflow-wrap: break-word; word-break: normal; }
tr { break-inside: avoid; }
thead { display: table-header-group; }
img, svg { display: block; margin: 0 auto 3mm auto; max-width: 100%; max-height: 80mm; width: auto; height: auto; break-inside: avoid; }
blockquote { margin: 0 0 3mm 0; padding: 1mm 4mm; border-left: 3px solid #c9ced6; color: #444; }
.admonition, details { border: 1px solid #cfd4db; border-left: 4px solid #6b7785; border-radius: 1.5mm; padding: 1.5mm 3.5mm; margin: 0 0 4mm 0; break-inside: avoid; background: #fafbfc; }
.admonition-title, summary { font-weight: 600; margin: 0 0 1mm 0; }
.admonition.warning, .admonition.danger { border-left-color: #b4540a; }
.admonition.tip, .admonition.success { border-left-color: #2a7a3b; }
.elsewhere { color: #555; font-size: 0.92em; }
hr { border: 0; border-top: 1px solid #c9ced6; }
"""


# The two shorter manuals are listed in manuals.yml, next to mkdocs.yml: edit the lists there, not here. Pages are
# printed in the order of the navigation; `appendices` come last, under their own heading; `part` renames the navigation
# section a page is listed under.
def manuals() -> dict[str, dict]:
    import yaml

    whole = yaml.safe_load((ROOT / "manuals.yml").read_text(encoding="utf-8"))
    found = whole["manuals"]
    for name, spec in found.items():
        for key, default in (
            ("appendices", []),
            ("part", {}),
            ("strip_code", False),
        ):
            spec.setdefault(key, default)
        if not all(spec.get(key) for key in ("label", "cover", "other", "pages")):
            raise SystemExit(f"manuals.yml: manual {name!r} needs label, cover, other and pages")
    return found


def left_out() -> list[str]:
    """The pages of the navigation that belong to neither manual (manuals.yml: left_out)."""
    import yaml

    return yaml.safe_load((ROOT / "manuals.yml").read_text(encoding="utf-8")).get("left_out", [])


def drop_python_sections(text: str) -> str:
    """A section that is only about Python ("## With Python") goes whole, down to the next heading of its level or higher:
    its prose explains code the user manual leaves out. One line says where it is."""
    out: list[str] = []
    level = 0  # the level of the section being dropped, 0 when none
    fenced = False  # inside a ``` block, where "# output" is not a heading
    for line in text.splitlines(keepends=True):
        if line.lstrip().startswith("```"):
            fenced = not fenced
        heading = None if fenced or line.lstrip().startswith("```") else re.match(r"(#{1,6})[ \t]+(.*?)[ \t]*$", line)
        if level and heading and len(heading[1]) <= level:
            level = 0
        if level:
            continue
        if heading and heading[2] == "With Python" and len(heading[1]) >= 2:
            level = len(heading[1])
            out.append(line + "\n*The same steps in Python are in the full manual and on the documentation site.*\n\n")
            continue
        out.append(line)
    return "".join(out)


def strip_python_examples(text: str, where: str = "the full manual") -> str:
    """The user manual has no code: a Python example becomes one line, and what it printed stays (it is not code).
    `where` names the manual that has the code: the other short manual only when the page is in it, else the full one."""
    shown = f"*The Python code for this step is in {where}. This is what it printed.*"
    alone = f"*The Python example for this step is in {where}.*"

    def swap(match: re.Match) -> str:
        return match[1] + (shown if match[2] else alone) + "\n\n" + (match[2] or "")

    text = drop_python_sections(text)
    text = re.sub(
        r"^([ \t]*)```python[^\n]*\n.*?^\1```[ \t]*\n(?:\s*(^\1```text\n)# output\n)?",
        swap,
        text,
        flags=re.S | re.M,
    )
    # The sentences that talk about the removed code are reworded or go with it.
    plain = r"(?:[^.\n`]|`[^`\n]*`|\.(?=\w))*\."
    text = re.sub(r"^The block uses .*\n", "", text, flags=re.M)
    for sentence in (rf"Run the block {plain}", r"Run it from the repository root\."):
        text = re.sub(rf" ?{sentence}", "", text)
    text = re.sub(r"\bThis block\b", "The online example", text)
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"^[ \t]+\n", "\n", text, flags=re.M))


def version() -> str:
    text = (ROOT / "src" / "extract" / "__init__.py").read_text(encoding="utf-8")
    found = re.search(r'__version__\s*=\s*"([^"]+)"', text)
    return found[1] if found else ""


def load_config() -> dict:
    import yaml

    class Loader(yaml.SafeLoader):
        pass

    # mkdocs.yml may use python-name tags for extensions; the page only needs names and settings.
    Loader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: None)
    return yaml.load((ROOT / "mkdocs.yml").read_text(encoding="utf-8"), Loader=Loader)


def flatten(nav, part: str = "") -> list[tuple[str, str, str]]:
    """(part, title, path) for every page of the navigation, in order. `part` is the top-level section's title."""
    found = []
    for entry in nav:
        if isinstance(entry, str):
            found.append((part, Path(entry).stem.replace("-", " ").capitalize(), entry))
            continue
        for title, value in entry.items():
            if isinstance(value, str):
                found.append((part or title, title, value))
            else:
                found += flatten(value, part or title)
    return found


def extensions(config: dict):
    names, settings = [], {}
    for item in config.get("markdown_extensions", []):
        if isinstance(item, str):
            names.append(item)
        else:
            ((name, options),) = item.items()
            names.append(name)
            settings[name] = options or {}
    settings.setdefault("toc", {})["permalink"] = False
    return names, settings


def key_of(path: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", path.lower()).strip("-")


def convert(text: str, md_names: list, md_settings: dict) -> str:
    import markdown

    return markdown.Markdown(extensions=md_names, extension_configs=md_settings).convert(text)


def where_to_look(label: str, link_text: str) -> str:
    """'User manual, Quickstart' for a link whose text is not the page's title; just 'User manual' when the text says it."""
    manual, _, title = label.partition(", ")
    plain = html.unescape(re.sub(r"<[^>]+>", "", link_text)).strip().lower()
    return f"in the {manual}" if plain == html.unescape(title).strip().lower() else f"in the {manual}, {title}"


def relink(body: str, page: str, pages: dict[str, str], others: dict[str, str] | None = None) -> str:
    """Make one page's ids unique in the whole document, point links at other pages' ids, drop links that leave the manual."""
    key = pages[page]
    body = re.sub(r'\bid="([^"]+)"', lambda m: f'id="{key}--{m[1]}"', body)
    here = (DOCS / page).parent

    def fix_href(match: re.Match) -> str:
        whole, target = match.group(0), match.group(1)
        if re.match(r"^[a-z][a-z0-9+.-]*:|^//", target, re.I):
            return whole  # an outside link stays as text; the PDF makes no request
        path_part, _, fragment = target.partition("#")
        if not path_part:
            return f'href="#{key}--{fragment}"' if fragment else whole
        dest = (here / unquote(path_part)).resolve()
        try:
            name = dest.relative_to(DOCS).as_posix()
        except ValueError:
            return 'href="#" data-dead="1"'
        if name in pages:
            return f'href="#{pages[name]}--{fragment}"' if fragment else f'href="#{pages[name]}"'
        if others and name in others:
            return f'href="#" data-dead="1" data-other="{html.escape(others[name])}"'
        return 'href="#" data-dead="1"'

    body = re.sub(r'href="([^"]*)"', fix_href, body)
    body = re.sub(
        r'<a href="#" data-dead="1"(?: data-other="([^"]*)")?[^>]*>(.*?)</a>',
        lambda m: m[2] + (f' <span class="elsewhere">({where_to_look(m[1], m[2])})</span>' if m[1] else ""),
        body,
        flags=re.S,
    )

    def fix_src(match: re.Match) -> str:
        target = match.group(1)
        if re.match(r"^[a-z][a-z0-9+.-]*:|^//", target, re.I):
            return match.group(0)
        dest = (here / unquote(target)).resolve()
        return f'src="{dest.as_uri()}"' if dest.is_file() else match.group(0)

    return re.sub(r'src="([^"]*)"', fix_src, body)


SHORT_CODE_LINES = 12


def keep_short_code_whole(body: str) -> str:
    """Mark the code blocks of at most SHORT_CODE_LINES lines, so that the print CSS never splits one across two sheets
    (a split leaves an empty grey slice at the foot of a sheet); a longer block may break between lines."""

    def mark(match: re.Match) -> str:
        if match.group(2).count("\n") >= SHORT_CODE_LINES:
            return match.group(0)
        return f'<pre class="keep"{match.group(1)}>{match.group(2)}</pre>'

    return re.sub(r"<pre\b([^>]*)>(.*?)</pre>", mark, body, flags=re.S)


def h1_of(body: str) -> str:
    """The text of the first level-one heading of a converted page."""
    found = re.search(r"<h1[^>]*>(.*?)</h1>", body, re.S)
    return html.unescape(re.sub(r"<[^>]+>", "", found.group(1))).strip() if found else ""


def manual_entries(
    entries: list[tuple[str, str, str]], manual: str
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    """The entries of one manual in reading order, and the label of every page that lives in the other manual
    (so that a link to it can name where to find it)."""
    every = manuals()
    spec = every[manual]
    listed = [path for each in every.values() for path in each["pages"] + each["appendices"]] + left_out()
    unplaced = [path for _, _, path in entries if path not in listed]
    missing = [path for path in listed if path not in {p for _, _, p in entries}]
    twice = sorted({path for path in listed if listed.count(path) > 1})
    if unplaced or missing or twice:
        raise SystemExit(
            "manuals.yml and the navigation disagree: "
            + "; ".join(
                f"{what}: {', '.join(paths)}"
                for what, paths in (
                    ("in no manual (add to a manual or to left_out)", unplaced),
                    ("not in the navigation", missing),
                    ("listed twice", twice),
                )
                if paths
            )
        )
    by_path = {path: (part, title, path) for part, title, path in entries}
    chosen: list[tuple[str, str, str]] = []
    for path in spec["pages"]:
        if path in by_path:
            part, title, _ = by_path[path]
            chosen.append((spec["part"].get(path, part), title, path))
    for path in spec["appendices"]:
        if path in by_path:
            chosen.append(("Appendices", by_path[path][1], path))
    other_paths = {
        path for name, other in every.items() if name != manual for path in other["pages"] + other["appendices"]
    }
    others = {path: f"{spec['other']}, {by_path[path][1]}" for path in other_paths if path in by_path}
    for path in left_out():  # in no short manual: the full manual has it
        if path in by_path:
            others[path] = f"{FULL_MANUAL}, {by_path[path][1]}"
    return chosen, others


def assemble(
    numbers: dict[str, int] | None = None, titles: dict[str, str] | None = None, manual: str | None = None
) -> str:
    config = load_config()
    md_names, md_settings = extensions(config)
    entries = [e for e in flatten(config["nav"]) if (DOCS / e[2]).is_file() and e[2] not in NOT_IN_THE_MANUAL]
    others: dict[str, str] = {}
    every = manuals() if manual else {}
    spec = every.get(manual) if manual else None
    if manual and not spec:
        raise SystemExit(f"unknown manual {manual!r}: use one of {', '.join(every)}")
    if spec:
        entries, others = manual_entries(entries, manual)
    pages = {path: key_of(path) for _, _, path in entries}
    from pygments.formatters import HtmlFormatter

    ver = version()
    label = spec["label"] if spec else "manual"
    css = CSS.replace("VERSION", ver).replace("LABEL", label) + HtmlFormatter(style="default").get_style_defs(
        ".highlight"
    )
    parts = [
        '<!doctype html><html lang="en"><head><meta charset="utf-8">',
        f"<title>Extract {html.escape(ver)}: {label}</title><style>{css}</style></head><body>",
        '<div class="cover"><h1>Extract</h1><div class="rule"></div>'
        f"<p>{spec['cover'] if spec else 'Manual'}</p><p>Version {html.escape(ver)}</p></div>",
        '<div class="contents"><h2>Contents</h2>',
    ]
    last = None
    for part, title, path in entries:
        if part != last:
            if last is not None:
                parts.append("</ol>")
            parts.append(f'<div class="part">{html.escape(part)}</div><ol>')
            last = part
        number = (
            f'<span class="dots"></span><span class="num">{numbers[path]}</span>' if numbers and path in numbers else ""
        )
        parts.append(f'<li><a href="#{pages[path]}">{html.escape(title)}</a>{number}</li>')
    parts.append("</ol>")
    parts.append("</div>")
    for _, _, path in entries:
        text = (DOCS / path).read_text(encoding="utf-8")
        if spec and spec["strip_code"]:
            holders = [o["other"] for n, o in every.items() if n != manual and path in o["pages"] + o["appendices"]]
            text = strip_python_examples(text, f"the {holders[0]}" if holders else "the full manual")
        body = convert(text, md_names, md_settings)
        if titles is not None:
            titles[path] = h1_of(body)
        parts.append(
            f'<section class="doc" id="{pages[path]}">{keep_short_code_whole(relink(body, path, pages, others))}</section>'
        )
    parts.append("</body></html>")
    return "\n".join(parts)


def main(argv: list[str]) -> int:
    manual = None
    if "--manual" in argv:
        at = argv.index("--manual")
        manual = argv[at + 1] if at + 1 < len(argv) else None
        argv = argv[:at] + argv[at + 2 :]
    if len(argv) not in (1, 2) or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if argv and argv[0] in ("-h", "--help") else 2
    import json

    numbers = json.loads(Path(argv[1]).read_text(encoding="utf-8")) if len(argv) == 2 else None
    titles: dict[str, str] = {}
    Path(argv[0]).write_text(assemble(numbers, titles, manual), encoding="utf-8")
    Path(argv[0]).with_suffix(".titles.json").write_text(json.dumps(titles), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
