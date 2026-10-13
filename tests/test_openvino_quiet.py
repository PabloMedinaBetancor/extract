"""Extract 5.0.5 (B45): OpenVINO reports an anonymous usage event to Google Analytics each time it is imported, unless a
consent file in the home folder says no or the process runs under a CI service. Extract imports it with that off."""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from extract.engine.quiet_openvino import openvino_quiet

ROOT = Path(__file__).resolve().parents[1]
INSTALLED = all(importlib.util.find_spec(name) for name in ("openvino", "openvino_telemetry", "rapidocr"))

# The sender of OpenVINO's usage reporting is replaced by a function that counts what it is given and sends nothing.
PROBE = """
import os, sys
import openvino_telemetry.utils.sender as sender
sent = []
sender.TelemetrySender.send = lambda self, backend, message, *args, **kwargs: sent.append(message)
{statement}
print("sent", len(sent))
print("openvino", "openvino" in sys.modules)
print("CI", os.environ.get("CI"))
"""
ENGINE = "from extract.engine import ocr; ocr._patch_rapidocr()"


class Switch(unittest.TestCase):
    def test_B45_ci_is_true_inside_the_block_and_has_its_old_value_after(self):
        for before in (None, "false", "true"):
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("CI", None)
                if before is not None:
                    os.environ["CI"] = before
                with openvino_quiet():
                    self.assertEqual(os.environ["CI"], "true")
                self.assertEqual(os.environ.get("CI"), before)

    def test_B45_the_old_value_comes_back_when_the_import_fails(self):
        with mock.patch.dict(os.environ, {"CI": "no"}):
            with self.assertRaises(RuntimeError):
                with openvino_quiet():
                    raise RuntimeError("the import failed")
            self.assertEqual(os.environ["CI"], "no")


@unittest.skipUnless(INSTALLED, "OpenVINO, its usage reporting and RapidOCR are needed")
class ImportReportsNothing(unittest.TestCase):
    def run_probe(self, statement: str, ci: str | None = None) -> dict[str, str]:
        home = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, home, True)
        env = {k: v for k, v in os.environ.items() if k not in ("CI", "TF_BUILD", "JENKINS_URL")}
        env |= {"HOME": home, "USERPROFILE": home, "PYTHONPATH": os.pathsep.join([str(ROOT / "src"), str(ROOT)])}
        if ci is not None:
            env["CI"] = ci
        done = subprocess.run(
            [sys.executable, "-c", PROBE.format(statement=statement)],
            capture_output=True,
            text=True,
            env=env,
            timeout=300,
        )
        self.assertEqual(done.returncode, 0, done.stderr[-600:])
        result = dict(
            line.split(" ", 1) for line in done.stdout.splitlines() if line.startswith(("sent ", "openvino ", "CI "))
        )
        result["intel_folder"] = str(Path(home, "intel").exists())
        return result

    def test_B45_loading_the_engine_code_reports_nothing_and_writes_nothing_in_the_home(self):
        result = self.run_probe(ENGINE)
        self.assertEqual(result["openvino"], "True")
        self.assertEqual(result["sent"], "0")
        self.assertEqual(result["intel_folder"], "False")
        self.assertEqual(result["CI"], "None")

    def test_B45_a_value_of_ci_that_was_there_is_kept(self):
        result = self.run_probe(ENGINE, ci="false")
        self.assertEqual(result["sent"], "0")
        self.assertEqual(result["CI"], "false")

    def test_B45_the_control_without_the_guard_does_reach_the_sender(self):
        result = self.run_probe("import openvino")
        if result["sent"] == "0":
            self.skipTest("this OpenVINO no longer reports on import; the guard is not needed for it")
        self.assertEqual(result["sent"], "1")


if __name__ == "__main__":
    unittest.main()
