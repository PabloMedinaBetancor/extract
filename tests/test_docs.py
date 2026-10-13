import importlib.util
import unittest
from pathlib import Path
from unittest import mock

import extract

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    spec = importlib.util.spec_from_file_location("api_docs", ROOT / "scripts" / "api_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class DocsTest(unittest.TestCase):
    def test_A7_the_api_reference_is_the_one_the_docstrings_give(self):
        script = load_script()
        pages = script.render_reference()
        self.assertTrue(pages)
        for path, text in pages.items():
            self.assertEqual(path.read_text(encoding="utf-8"), text, path.name)
        self.assertFalse((ROOT / "docs" / "API.md").exists(), "docs/API.md was replaced by docs/reference/")

    def test_A7_the_measured_numbers_come_from_the_results_files(self):
        script = load_script()
        measured = (ROOT / "docs" / "MEASURED.md").read_text(encoding="utf-8")
        self.assertEqual(script.refresh_measured(measured), measured)
        self.assertIn("s a page", measured)

    def test_A7_the_headline_figures_of_the_readme_are_in_the_measured_file(self):
        measured = (ROOT / "docs" / "MEASURED.md").read_text(encoding="utf-8")
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        summary = readme[readme.index("## Measured") : readme.index("## Limits")]
        for figure in ("2.78 %", "2.85 %", "68.6 %", "77.1 %", "7.6 %", "19.5", "21.2", "10.9", "13.0", "3.6", "3.2"):
            self.assertIn(figure, summary)
        for figure in (
            "2.78 %",
            "2.85 %",
            "68.6 %",
            "77.1 %",
            "7.62 %",
            "19.5 (",
            "21.2 (",
            "10.9 (",
            "13.0 (",
            "3.616 s",
            "3.245 s",
        ):
            self.assertIn(figure, measured)
        self.assertIn("232 numbers", measured)
        self.assertIn("287 numbers", measured)

    def test_A7_the_check_mode_of_the_script_finds_a_stale_reference(self):
        script = load_script()
        self.assertEqual(script.main(check=True), 0)
        with mock.patch.object(script, "render_reference", lambda: {script.REFERENCE / "index.md": "stale"}):
            self.assertEqual(script.main(check=True), 1)

    def test_A7_every_public_name_is_in_the_reference(self):
        reference = "\n".join(
            path.read_text(encoding="utf-8") for path in sorted((ROOT / "docs" / "reference").glob("*.md"))
        )
        for name in extract.__all__:
            self.assertIn(f"`{name}", reference, name)

    def test_A7_the_documents_say_what_they_must(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("test suite has been run on Linux with Python 3.10 only", readme)


if __name__ == "__main__":
    unittest.main()
