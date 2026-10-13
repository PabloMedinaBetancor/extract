import os
import subprocess
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from tests.support import SANS, SPANISH_LINES, scanned_pdf, text_page_image

SCRIPT = """\
from extract.api import Reader

Reader._engine_factory = "tests.fake_engine:load_engine"
{body}
"""
READ = """\
with Reader(guard=False) as reader:
    doc = reader.read({pdf!r})
print(doc.status)
"""


class SpawnTest(unittest.TestCase):
    """The workers are started with `spawn`, so they import the script again; a script that reads at its top level
    would start workers from inside a worker."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        scanned_pdf(self.tmp / "scan.pdf", [text_page_image(SPANISH_LINES, SANS, 24, 1240, 1754)])

    def run_script(self, body: str) -> subprocess.CompletedProcess:
        script = self.tmp / "script.py"
        script.write_text(SCRIPT.format(body=body), encoding="utf-8")
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(p for p in sys.path if p),
            "XDG_CACHE_HOME": str(self.tmp / "cache"),
        }
        return subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=120)

    def test_X1_a_script_that_reads_under_the_main_guard_works(self):
        read = textwrap.indent(READ.format(pdf=str(self.tmp / "scan.pdf")), "    ")
        done = self.run_script(f'if __name__ == "__main__":\n{read}')
        self.assertEqual(done.stdout.strip(), "done", done.stderr)

    def test_X1_a_script_that_reads_at_its_top_level_fails_loudly_and_is_told_what_to_do(self):
        done = self.run_script(READ.format(pdf=str(self.tmp / "scan.pdf")))
        self.assertNotEqual(done.returncode, 0, "a script without the guard must not exit 0 with a partial document")
        self.assertEqual(done.stdout.strip(), "")
        self.assertIn("extract.errors.EngineLoadError", done.stderr)
        self.assertIn('under `if __name__ == "__main__":`', done.stderr)
        self.assertNotIn("bootstrapping", done.stderr)

    def test_X2_the_error_names_its_cause_so_that_a_program_can_react(self):
        body = (
            "from extract import EngineLoadError\n"
            "try:\n"
            + textwrap.indent(READ.format(pdf=str(self.tmp / "scan.pdf")), "    ")
            + "except EngineLoadError as error:\n"
            "    print(error.params['cause'])\n"
        )
        done = self.run_script(body)
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertEqual(done.stdout.strip(), "no_main_guard")

    def test_X3_a_worker_that_is_importing_the_script_leaves_with_the_documented_code(self):
        from extract.engine.pool import EXIT_NO_MAIN_GUARD

        code = (
            "import multiprocessing, sys\n"
            "multiprocessing.current_process()._inheriting = True\n"
            "from extract.engine.pool import leave_if_started_by_an_unguarded_script\n"
            "leave_if_started_by_an_unguarded_script()\n"
            "print('not reached')\n"
        )
        done = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)},
            timeout=60,
        )
        self.assertEqual(done.returncode, EXIT_NO_MAIN_GUARD)
        self.assertEqual(done.stdout, "")
        self.assertIn("reads a document at its top level", done.stderr)


if __name__ == "__main__":
    unittest.main()
