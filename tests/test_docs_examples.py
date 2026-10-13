"""Every ```python block of every documentation page runs, on the real models, in a folder that holds the sample files.

A page's blocks share one namespace and run in order, as a reader would paste them. A failure names the block as
`docs/how-to/x.md block 3`. Fenced `text` and `console` blocks are not run; a block with
`<!-- docs-test: skip - the reason -->` on the line before it is listed as skipped, with its reason.

Runs only with `EXTRACT_DOCS=1` (`scripts/docs.py check` sets it): reading takes real time.
"""

from __future__ import annotations

import unittest

import pytest

from tests import docs_support as ds

pytestmark = [
    pytest.mark.docs,
    pytest.mark.skipif(
        not ds.enabled(), reason="real-model documentation examples: set EXTRACT_DOCS=1 or run scripts/docs.py check"
    ),
]

PAGES = [(ds.rel(p), p) for p in ds.doc_pages() if ds.python_blocks(p)]


@pytest.mark.parametrize("name,path", PAGES, ids=[n for n, _ in PAGES])
def test_every_python_block_of_a_page_runs(name, path):
    results = ds.run_blocks(name, ds.python_blocks(path))
    failures = ds.failure_lines(name, results)
    assert not failures, "\n".join(failures)


class BlockReaderTest(unittest.TestCase):
    """The block reader and the failure report, on invented pages (fast, no models, always run)."""

    def blocks(self, text):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "page.md"
            path.write_text(text, encoding="utf-8")
            return ds.python_blocks(path)

    def test_blocks_are_numbered_by_page_and_text_blocks_are_ignored(self):
        text = "# T\n\n```python\nprint(1)\n```\n\n```text\nnot code\n```\n\n```python\nprint(2)\n```\n"
        blocks = self.blocks(text)
        self.assertEqual([(b.number, b.line) for b in blocks], [(1, 3), (2, 11)])

    def test_a_text_block_that_starts_with_output_is_the_expected_output(self):
        text = "```python\nprint(1)\n```\n\n```text\n# output\n1\n```\n\n```python\nprint(2)\n```\n"
        blocks = self.blocks(text)
        self.assertEqual([b.expected_output for b in blocks], ["1", None])

    def test_a_skip_comment_needs_its_reason_and_is_carried(self):
        text = (
            "<!-- docs-test: skip - needs a Windows machine -->\n```python\nimport msvcrt\n```\n```python\nx = 1\n```\n"
        )
        blocks = self.blocks(text)
        self.assertEqual([b.skip_reason for b in blocks], ["needs a Windows machine", None])

    def test_a_fence_inside_a_longer_fence_is_not_a_block(self):
        text = "````markdown\n```python\nprint(1)\n```\n````\n"
        self.assertEqual(self.blocks(text), [])

    def test_a_broken_block_is_named_by_page_and_number(self):
        import json
        import subprocess
        import sys
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            blocks = [
                ds.Block("docs/how-to/x.md", 1, 3, "value = 2\n", None, None),
                ds.Block("docs/how-to/x.md", 2, 9, "print(value + undefined_name)\n", None, None),
                ds.Block("docs/how-to/x.md", 3, 15, "print('after')\n", None, None),
            ]
            spec = {
                "page": "docs/how-to/x.md",
                "blocks": [
                    dict(
                        number=b.number,
                        line=b.line,
                        code=b.code,
                        expected_output=b.expected_output,
                        skip_reason=b.skip_reason,
                    )
                    for b in blocks
                ],
            }
            (folder / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
            done = subprocess.run(
                [sys.executable, str(ds.RUNNER), "page", str(folder / "spec.json"), str(folder / "r.json")], cwd=folder
            )
            self.assertEqual(done.returncode, 1)
            results = json.loads((folder / "r.json").read_text())["results"]
        self.assertEqual([r["status"] for r in results], ["ok", "fail", "not-run"])
        self.assertEqual(
            ds.failure_lines("docs/how-to/x.md", results)[0][:48], "docs/how-to/x.md block 2 (line 9): NameError: na"
        )
