"""The documentation reads as written by a person: no banned word or form (lists in docs_support.py), nothing personal.

Scans every page, `README.md`, `AGENTS.md`, `llms.txt`, the changelog's source and the docstrings of the public API.
Code blocks and inline code are left out of the style scan; the privacy scan reads them too. A failure prints
`file:line: what was found`. The helpers are in `tests/docs_support.py`.
"""

from __future__ import annotations

import inspect
import re
import unittest

import extract
from tests import docs_support as ds


def public_docstrings():
    """(dotted name, docstring) for every public name of `extract.__all__` and its public methods."""
    for name in extract.__all__:
        obj = getattr(extract, name, None)
        if obj is None:
            continue
        if obj.__doc__ and not isinstance(obj, (str, int, float, tuple, dict, list, set, frozenset)):
            yield name, inspect.getdoc(obj) or ""
        if inspect.isclass(obj):
            for member, value in vars(obj).items():
                if member.startswith("_"):
                    continue
                target = value.fget if isinstance(value, property) else value
                if callable(target) and getattr(target, "__doc__", None):
                    yield f"{name}.{member}", inspect.getdoc(target) or ""


SYNTHETIC_INTERNAL = re.compile(r"team-lead-zeta|ModelOmega")


class StyleScanTest(unittest.TestCase):
    def test_no_page_uses_a_banned_word_or_form(self):
        found = []
        for path in ds.prose_files():
            if path.suffix not in {".md", ".txt"}:
                continue
            faq = path.name.lower() == "faq.md"
            found += [
                f"{ds.rel(path)}:{n}: {what}"
                for n, what in ds.style_findings(path.read_text(encoding="utf-8"), faq=faq)
            ]
        ds.expect_none(self, found, "banned phrases in pages")

    def test_no_docstring_of_the_public_api_uses_a_banned_word_or_form(self):
        found = []
        for name, doc in public_docstrings():
            found += [f"{name}:{n}: {what}" for n, what in ds.docstring_style_findings(doc)]
        ds.expect_none(self, found, "banned phrases in docstrings")

    def test_nothing_personal_is_in_a_public_page(self):
        found = []
        for path in ds.prose_files():
            found += [
                f"{ds.rel(path)}:{n}: {what}" for n, what in ds.privacy_findings(path.read_text(encoding="utf-8"))
            ]
        ds.expect_none(self, found, "personal details in public pages")


class ScannerCatchesSeededFaultsTest(unittest.TestCase):
    """Each rule fires on an invented fault and stays quiet on the allowed form (the scan catches that and nothing else)."""

    def kinds(self, text, **kw):
        return [what for _, what in ds.style_findings(text, **kw)]

    def test_banned_words_and_forms_are_found(self):
        for text, expect in [
            ("Extract is a robust reader.", "banned word: robust"),
            ("You can simply run it.", "banned word: simply"),
            ("Then just run the script.", "banned word: just run"),
            ("This is easy to install.", "banned word: easy"),
            ("Moreover, it reads Spanish.", "banned opener: Moreover"),
            ("It reads. Note that it is slow.", "banned opener: Note that"),
            ("It works — mostly.", "em dash"),
            ("It works!", "exclamation mark"),
            ("It works ✓", "emoji or decorative symbol: '✓'"),
            ("It does not only read but also corrects.", "banned form: not only ... but also"),
            ("# Overview\n", "empty heading: Overview"),
            ("## Why use it?\n", "question as a heading"),
            ("We ensure the result.", "banned word: ensure"),
        ]:
            with self.subTest(text=text):
                self.assertIn(expect, self.kinds(text))

    def test_allowed_forms_are_not_flagged(self):
        quiet = [
            "```python\nprint('simply robust!')  # —\n```\n",
            "Use `just run` or `robust` in code.\n",
            "<!-- simply an easy comment -->\n",
            "It is not equal (`a != b`) and `![x](y)` is an image.\n",
            "He was just there.\n",  # \"just\" is banned only before a verb of instruction
            "Range 49.7–86.5 uses an en dash.\n",
        ]
        for text in quiet:
            with self.subTest(text=text):
                self.assertEqual(self.kinds(text), [])
        self.assertEqual(self.kinds("## Why Extract?\n", faq=True), [])

    def test_personal_details_are_found(self):
        for text, expect in [
            ("Open /ho" + "me/someone/work.", "home path"),
            ("Write to someone@" + "gmail.com.", "e-mail address"),
            ("Call +34 600" + " 123 456.", "phone number"),
            ("Ask the team-lead-zeta.", "internal name"),
            ("Made with ModelOmega.", "internal name"),
        ]:
            with self.subTest(text=text):
                found = ds.privacy_findings(text, internal=SYNTHETIC_INTERNAL)
                self.assertTrue(any(expect in what for _, what in found), text)
        self.assertEqual(ds.privacy_findings("Mail a@example.com about 1,330 MB in 553 pages, 3.616 s, 2026-0042."), [])
