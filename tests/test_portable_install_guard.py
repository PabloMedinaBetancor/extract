import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_install():
    spec = importlib.util.spec_from_file_location("install_script", ROOT / "scripts" / "install.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ClearVenvTest(unittest.TestCase):
    def test_refuses_a_folder_that_is_not_a_venv(self):
        install = load_install()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / "mine"
            folder.mkdir()
            (folder / "file.txt").write_text("keep me")
            with self.assertRaises(SystemExit):
                install.clear_venv(folder)
            self.assertTrue((folder / "file.txt").exists())

    def test_removes_a_venv_and_an_empty_folder(self):
        install = load_install()
        with tempfile.TemporaryDirectory() as tmp:
            venv = Path(tmp) / "v"
            venv.mkdir()
            (venv / "pyvenv.cfg").write_text("home = /usr/bin")
            install.clear_venv(venv)
            self.assertFalse(venv.exists())
            empty = Path(tmp) / "e"
            empty.mkdir()
            install.clear_venv(empty)
            self.assertFalse(empty.exists())
            install.clear_venv(Path(tmp) / "missing")


if __name__ == "__main__":
    unittest.main()
