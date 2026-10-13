"""The PDF manual's print page (`scripts/docs_pdf.py`): what it assembles from a small invented documentation tree.

The page is built by the docs environment's Python (`DOCS_PYTHON`, which has Markdown, Pygments and PyYAML); the tests
skip when that interpreter is not named. Printing it to PDF needs Chrome and is tried by `scripts/docs.py pdf`.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests import docs_support as ds

DOCS_PYTHON = os.environ.get("DOCS_PYTHON")

MKDOCS = """site_name: T
nav:
  - Home: index.md
  - Guide:
      - Other: other.md
      - Changes: changelog.md
markdown_extensions:
  - admonition
  - tables
  - toc:
      permalink: true
  - pymdownx.highlight:
      anchor_linenums: false
  - pymdownx.superfences
"""
INDEX = """# Home

Go to [other](other.md#second-part), [missing](nope.md), [dev](../README.md), [self](#home) and [web](https://example.org/).

```python
print("hello")
```

| Name | Value |
|---|---|
| `a` | 1 |

![diagram](img/d.svg)
"""
USER_INDEX = """# Home

See [the second part](other.md#second-part), [Other](other.md) and [changes](changelog.md).

```python
print("hello")
```

```text
# output
hello
```

This block prints a greeting.
"""
SPLIT = """
manuals:
  user:
    label: user manual
    cover: User manual
    other: Developer reference
    strip_code: true
    pages: [index.md]
  developer:
    label: developer reference
    cover: Developer reference
    other: User manual
    pages: [other.md]
    appendices: [changelog.md]
"""
OTHER = "# Other\n\n## Second part\n\nText.\n"
SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="40" height="20"><rect width="40" height="20"/></svg>'


@unittest.skipUnless(
    DOCS_PYTHON and Path(DOCS_PYTHON).is_file(), "set DOCS_PYTHON to the docs environment's interpreter"
)
class PrintPageTest(unittest.TestCase):
    def assemble(self, numbers=None, manual=None, manuals=None, fail=False) -> str:
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, True)
        (root / "docs" / "img").mkdir(parents=True)
        (root / "src" / "extract").mkdir(parents=True)
        (root / "src" / "extract" / "__init__.py").write_text('__version__ = "9.9.9"\n', encoding="utf-8")
        (root / "mkdocs.yml").write_text(MKDOCS, encoding="utf-8")
        (root / "docs" / "index.md").write_text(USER_INDEX if manuals else INDEX, encoding="utf-8")
        (root / "docs" / "other.md").write_text(OTHER, encoding="utf-8")
        (root / "docs" / "changelog.md").write_text("# Changelog\n\nLong.\n", encoding="utf-8")
        if manuals:
            (root / "manuals.yml").write_text(manuals, encoding="utf-8")
        (root / "docs" / "img" / "d.svg").write_text(SVG, encoding="utf-8")
        code = (
            "import sys, pathlib\n"
            f"sys.path.insert(0, {str(ds.ROOT / 'scripts')!r})\n"
            "import docs_pdf\n"
            f"docs_pdf.ROOT = pathlib.Path({str(root)!r})\n"
            "docs_pdf.DOCS = docs_pdf.ROOT / 'docs'\n" + f"print(docs_pdf.assemble({numbers!r}, None, {manual!r}))\n"
        )
        done = subprocess.run([DOCS_PYTHON, "-c", code], capture_output=True, text=True, timeout=120)
        if fail:
            self.assertNotEqual(done.returncode, 0)
            return done.stderr
        self.assertEqual(done.returncode, 0, done.stderr[-800:])
        return done.stdout

    def test_cover_contents_and_pages_in_navigation_order(self):
        page = self.assemble()
        self.assertIn("Version 9.9.9", page)
        self.assertIn("Extract 9.9.9, manual", page)
        sections = re.findall(r'<section class="doc" id="([^"]+)"', page)
        self.assertEqual(sections, ["index-md", "other-md", "changelog-md"])

    def test_the_contents_list_shows_the_sheet_each_page_starts_on_when_it_is_given(self):
        plain = self.assemble()
        self.assertNotIn('class="num"', plain)
        page = self.assemble({"index.md": 3, "other.md": 4})
        self.assertEqual(re.findall(r'<span class="num">(\d+)</span>', page), ["3", "4"])

    def test_links_between_pages_stay_inside_the_page_and_a_dead_link_becomes_text(self):
        page = self.assemble()
        self.assertIn('href="#other-md--second-part"', page)
        self.assertIn('href="#index-md--home"', page)
        self.assertIn('id="other-md--second-part"', page)
        self.assertNotIn("nope.md", page)
        self.assertNotIn("README.md", page)
        self.assertIn("missing, dev, ", page)  # the words stay

    def test_every_id_is_unique_and_every_inner_link_has_a_target(self):
        page = self.assemble()
        ids = re.findall(r'\bid="([^"]+)"', page)
        self.assertEqual(len(ids), len(set(ids)))
        for target in re.findall(r'href="#([^"]+)"', page):
            self.assertIn(target, ids, target)

    def test_images_are_read_from_docs_by_file_path_and_nothing_is_fetched(self):
        page = self.assemble()
        self.assertTrue(re.search(r'src="file://[^"]+/docs/img/d\.svg"', page))
        for tag in re.findall(r"<(?:script|link|img|iframe|source)\b[^>]*>", page):
            self.assertNotRegex(tag, r'(?:src|href)="(?:https?:)?//', tag)
        self.assertNotIn("<script", page)
        self.assertNotIn("@import", page)

    def test_the_user_manual_has_no_code_a_summary_page_and_names_the_other_manual(self):
        page = self.assemble(manual="user", manuals=SPLIT)
        self.assertEqual(re.findall(r'<section class="doc" id="([^"]+)"', page), ["index-md"])
        self.assertIn("User manual", page)
        self.assertIn("Extract 9.9.9, user manual", page)
        self.assertNotIn('print("hello")', page)
        self.assertIn("Python code for this step is in the full manual", page)  # index.md is in no other manual
        self.assertIn("hello</pre>", page.replace("\n</code>", "").replace("</code>", ""))
        self.assertNotIn("# output", page)
        self.assertIn("The online example prints a greeting", page)
        self.assertIn("(in the Developer reference)", page)  # [Other](other.md): the text is the page's title
        self.assertIn("(in the Developer reference, Other)", page)  # [other](other.md#...): it names the page
        self.assertEqual(page.count("(in the Developer reference)"), 2)  # [Other] and [changes]
        self.assertNotIn("changelog.md", page)

    def test_a_link_to_a_page_left_out_of_both_manuals_names_the_full_manual(self):
        left = SPLIT.replace("    appendices: [changelog.md]\n", "") + "left_out: [changelog.md]\n"
        page = self.assemble(manual="user", manuals=left)
        self.assertEqual(re.findall(r'<section class="doc" id="([^"]+)"', page), ["index-md"])
        self.assertIn("(in the full manual)", page)  # [changes](changelog.md): the text is the page's title
        self.assertIn("(in the Developer reference, Other)", page)
        self.assertNotIn("changelog.md", page)

    def test_a_page_in_no_list_stops_the_print_and_is_named(self):
        forgotten = SPLIT.replace("    appendices: [changelog.md]\n", "")
        said = self.assemble(manual="user", manuals=forgotten, fail=True)
        self.assertIn("in no manual", said)
        self.assertIn("changelog.md", said)
        gone = SPLIT.replace("[other.md]", "[other.md, vanished.md]")
        self.assertIn("vanished.md", self.assemble(manual="user", manuals=gone, fail=True))

    def test_the_developer_reference_keeps_its_code_and_puts_the_appendices_last(self):
        page = self.assemble(manual="developer", manuals=SPLIT)
        self.assertEqual(re.findall(r'<section class="doc" id="([^"]+)"', page), ["other-md", "changelog-md"])
        self.assertIn('<div class="part">Appendices</div>', page)
        self.assertIn("Extract 9.9.9, developer reference", page)

    def test_code_and_tables_are_there(self):
        page = self.assemble()
        self.assertIn('<div class="highlight">', page)
        self.assertIn("<table>", page)


class PythonSectionTest(unittest.TestCase):
    SAMPLE = (
        "# Quickstart\n\n## With the interface\n\nClick Read.\n\n## With Python\n\nThe same work in code.\n\n"
        "```python\nprint(1)\n```\n\n```text\n# output\n1\n```\n\n### A detail\n\nMore about the code.\n\n"
        "## The samples\n\nThree PDFs.\n"
    )

    def drop(self, text: str) -> str:
        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(ds.ROOT / 'scripts')!r})\n"
            "import docs_pdf\n"
            f"print(docs_pdf.drop_python_sections({text!r}), end='')\n"
        )
        return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout

    def test_a_with_python_section_goes_whole_down_to_the_next_heading_of_its_level(self):
        out = self.drop(self.SAMPLE)
        self.assertIn("## With Python\n\n*The same steps in Python are in the full manual", out)
        for gone in ("The same work in code.", "print(1)", "# output", "### A detail", "More about the code."):
            self.assertNotIn(gone, out)
        self.assertIn("## With the interface\n\nClick Read.", out)
        self.assertIn("## The samples\n\nThree PDFs.", out)

    def strip(self, text: str, *where: str) -> str:
        script = (
            "import sys\n"
            f"sys.path.insert(0, {str(ds.ROOT / 'scripts')!r})\n"
            "import docs_pdf\n"
            f"print(docs_pdf.strip_python_examples({text!r}, *{where!r}), end='')\n"
        )
        return subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout

    def test_the_line_that_replaces_a_code_block_names_the_manual_that_has_it(self):
        text = "```python\nprint(1)\n```\n\n```text\n# output\n1\n```\n\n```python\nprint(2)\n```\n"
        full = self.strip(text)
        self.assertIn("The Python code for this step is in the full manual. This is what it printed.", full)
        self.assertIn("The Python example for this step is in the full manual.", full)
        self.assertNotIn("Developer reference", full)
        dev = self.strip(text, "the Developer reference")
        self.assertIn("is in the Developer reference. This is what it printed.", dev)
        self.assertIn("example for this step is in the Developer reference.", dev)

    def test_text_without_such_a_section_is_unchanged(self):
        text = "# Page\n\n## Steps\n\n```text\n# output\n```\n"
        self.assertEqual(self.drop(text), text)


class ChromeFlagsTest(unittest.TestCase):
    def test_the_print_command_cuts_the_browser_off_the_network(self):
        spec_text = (ds.ROOT / "scripts" / "docs.py").read_text(encoding="utf-8")
        self.assertIn("--proxy-server=socks5://127.0.0.1:1", spec_text)
        self.assertIn("--host-resolver-rules=MAP * ~NOTFOUND , EXCLUDE localhost", spec_text)
        self.assertIn("--no-pdf-header-footer", spec_text)
        self.assertTrue(sys.version_info >= (3, 10))


if __name__ == "__main__":
    unittest.main()
