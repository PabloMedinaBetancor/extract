"""The hand-written and trimmed parts of the reference stay true to the code.

* `docs/reference/example-reading.extract.json` follows the schema and loads (the JSON page quotes it).
* Every `--flag` that `docs/reference/command-line.md` names for a command is one its own `--help` prints.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

import extract

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "docs" / "reference"


class ExampleReadingTest(unittest.TestCase):
    def test_the_trimmed_reading_follows_the_schema_and_loads(self):
        import jsonschema

        raw = json.loads((REFERENCE / "example-reading.extract.json").read_text(encoding="utf-8"))
        schema = json.loads((ROOT / "src" / "extract" / "schema" / "document.schema.json").read_text(encoding="utf-8"))
        jsonschema.validate(raw, schema)
        doc = extract.load(REFERENCE / "example-reading.extract.json")
        self.assertEqual(doc.status, "done")
        self.assertEqual([w.id for w in doc.doubtful()], ["p1w11", "p1w13", "p1w4"])

    def test_the_json_page_names_every_key_of_the_schema(self):
        page = (REFERENCE / "json-format.md").read_text(encoding="utf-8")
        schema = json.loads((ROOT / "src" / "extract" / "schema" / "document.schema.json").read_text(encoding="utf-8"))
        keys = set(schema["properties"])
        for definition in schema["$defs"].values():
            for option in definition.get("anyOf", [definition]):
                keys |= set(option.get("properties", {}))
        for key in sorted(keys):
            self.assertIn(f"| `{key}` |", page, key)


class CommandLineFlagsTest(unittest.TestCase):
    def help_flags(self, command: list[str]) -> set[str]:
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
        done = subprocess.run(command + ["--help"], capture_output=True, text=True, cwd=ROOT, env=env, timeout=120)
        self.assertEqual(done.returncode, 0, done.stderr[-300:])
        return set(re.findall(r"--[a-z][a-z-]*", done.stdout))

    def sections(self) -> dict[str, list[str]]:
        """The lines of the page that belong to each command, by the name of its script (or `extract.ui`)."""
        found: dict[str, list[str]] = {}
        current = None
        for line in (REFERENCE / "command-line.md").read_text(encoding="utf-8").splitlines():
            heading = re.match(r"#{2,3} (.*)", line)
            if heading:
                title = heading.group(1).strip("` ")
                if title == "The interface":
                    current = "extract.ui"
                elif title.startswith("scripts/") and title.endswith(".py"):
                    current = title[len("scripts/") : -len(".py")]
                else:
                    current = None
            row = re.match(r"\| `(?:python3? |\.venv/bin/python )?scripts/([a-z_]+)\.py", line)
            owner = row.group(1) if row else current
            if owner:
                found.setdefault(owner, []).append(line)
        return found

    def test_every_flag_on_the_page_is_one_the_command_prints_in_its_help(self):
        checked = 0
        for name, lines in self.sections().items():
            flags = set(re.findall(r"`(--[a-z][a-z-]*)", "\n".join(lines)))
            if not flags:
                continue
            command = (
                [sys.executable, "-m", "extract.ui"] if name == "extract.ui" else [sys.executable, f"scripts/{name}.py"]
            )
            known = self.help_flags(command) | {"--help"}
            self.assertEqual(flags - known, set(), f"{name}: flags on the page that its --help does not print")
            checked += len(flags)
        self.assertGreater(checked, 12)


if __name__ == "__main__":
    unittest.main()
