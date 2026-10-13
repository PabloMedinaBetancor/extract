import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest import mock

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "extract"


class PackageTest(unittest.TestCase):
    def test_X1_package_imports(self):
        import extract

        self.assertRegex(extract.__version__, r"^\d+\.\d+\.\d+$")

    def test_X1_importing_the_package_loads_none_of_the_heavy_libraries(self):
        script = (
            "import sys, extract\n"
            "heavy = ['numpy', 'cv2', 'PIL', 'pypdfium2', 'pikepdf', 'openpyxl', 'rapidocr', 'onnxruntime']\n"
            "print([m for m in heavy if m in sys.modules])\n"
        )
        done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)
        self.assertEqual(done.stdout.strip(), "[]")

    def test_X1_a_name_of_the_api_loads_it_on_first_use(self):
        import extract

        self.assertEqual(extract.Options().recycle_pages, 120)
        self.assertEqual(extract.Invoice.__name__, "InvoiceFields")
        self.assertFalse(hasattr(extract, "nothing_like_this"))

    def test_X1_every_file_that_is_not_code_is_in_the_package_data(self):
        config = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
        patterns = config["tool"]["setuptools"]["package-data"]["extract"]
        shipped = {path for pattern in patterns for path in PACKAGE.glob(pattern)}
        data = [
            path
            for path in PACKAGE.rglob("*")
            if path.is_file() and path.suffix not in (".py", ".pyc") and "__pycache__" not in path.parts
        ]
        self.assertGreater(len(data), 10)
        self.assertEqual(sorted(str(p.relative_to(PACKAGE)) for p in data if p not in shipped), [])

    def test_X1_an_installed_package_finds_its_model_inside_itself(self):
        import extract.api as api

        packaged = api.PACKAGED_GUARD_MODEL
        self.assertTrue(packaged.is_file())
        self.assertTrue(packaged.resolve().is_relative_to(Path(api.__file__).resolve().parent))
        with (
            mock.patch.dict(os.environ, {}, clear=False),
            mock.patch("extract.guard.checker.Guard.load", return_value="guard") as load,
        ):
            os.environ.pop("EXTRACT_GUARD_MODEL", None)
            self.assertEqual(api._default_guard(), "guard")
            load.assert_called_once_with(packaged)

    def test_X1_the_environment_variable_still_names_the_model(self):
        import extract.api as api

        with (
            mock.patch.dict(os.environ, {"EXTRACT_GUARD_MODEL": "/elsewhere/guard.model"}),
            mock.patch("extract.guard.checker.Guard.load", return_value="guard") as load,
        ):
            api._default_guard()
            load.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
