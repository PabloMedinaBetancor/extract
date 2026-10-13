import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import install  # noqa: E402
import make_offline_bundle  # noqa: E402


class OfflineBundleTest(unittest.TestCase):
    def test_P5_the_bundle_holds_the_tree_the_wheels_the_models_and_the_lists_and_not_the_junk(self):
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            for folder, name in (("wheels", "pip-1-py3-none-any.whl"), ("models", "m.onnx")):
                (scratch / folder).mkdir()
                (scratch / folder / name).write_bytes(b"x")
            (scratch / "lists.tar.gz").write_bytes(b"x")
            bundle = scratch / "bundle"
            make_offline_bundle.assemble(
                "linux-x86_64-py310", bundle, scratch / "wheels", scratch / "models", scratch / "lists.tar.gz"
            )
            archive = scratch / "bundle.zip"
            make_offline_bundle.zip_folder(bundle, archive)
            with zipfile.ZipFile(archive) as zipped:
                names = zipped.namelist()
        for wanted in (
            "bundle/scripts/install.py",
            "bundle/requirements/linux-x86_64-py310.txt",
            "bundle/pyproject.toml",
            "bundle/src/extract/__init__.py",
            "bundle/offline/wheelhouse/pip-1-py3-none-any.whl",
            "bundle/offline/models/m.onnx",
            "bundle/offline/lists.tar.gz",
            "bundle/README.txt",
        ):
            self.assertIn(wanted, names)
        junk = [
            n for n in names if "/.venv/" in n or "/dist/" in n or "__pycache__" in n or n.endswith((".npy", ".tsv.gz"))
        ]
        self.assertEqual(junk, [])

    def test_P5_the_install_script_names_a_lock_file_that_exists_for_this_machine(self):
        self.assertTrue(install.lock_file().is_file())
        version, hashes = install.pip_pin(install.lock_file())
        self.assertRegex(version, r"^\d+(\.\d+)+$")
        self.assertTrue(hashes)

    def test_P5_the_bundle_version_is_the_package_version(self):
        import extract

        self.assertEqual(make_offline_bundle.version(), extract.__version__)
