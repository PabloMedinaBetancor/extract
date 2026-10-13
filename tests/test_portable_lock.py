import hashlib
import os
import re
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = ["linux-x86_64-py310", "linux-x86_64-py312", "win-amd64-py310", "win-amd64-py312"]
HOME_PATH = re.compile(r"/home/(?!user\b)[A-Za-z0-9_.-]+|/Users/[A-Za-z0-9_.-]+|[A-Za-z]:\\Users\\[A-Za-z0-9_.-]+")
NOT_VERSIONED = {".git", ".venv", "dist", "__pycache__", ".pytest_cache", ".ruff_cache", "work", "site"}


def read_lock(platform: str) -> dict[str, tuple[str, list[str]]]:
    text = (ROOT / "requirements" / f"{platform}.txt").read_text(encoding="utf-8").replace("\\\n", " ")
    lock = {}
    for line in text.splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = re.match(r"(\S+)==(\S+)", line).groups()
            lock[canonicalize_name(name)] = (version, re.findall(r"--hash=sha256:([0-9a-f]{64})", line))
    return lock


def project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def tiny_wheel(folder: Path) -> Path:
    """A wheel with no code, built here, to try pip's hash check without a network."""
    path = folder / "tinything-1.0-py3-none-any.whl"
    with zipfile.ZipFile(path, "w") as wheel:
        wheel.writestr("tinything/__init__.py", "")
        wheel.writestr("tinything-1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: tinything\nVersion: 1.0\n")
        wheel.writestr(
            "tinything-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
        wheel.writestr("tinything-1.0.dist-info/RECORD", "")
    return path


class LockTest(unittest.TestCase):
    def test_P2_every_platform_has_a_lock_and_every_line_of_it_has_a_hash(self):
        for platform in PLATFORMS:
            lock = read_lock(platform)
            self.assertGreater(len(lock), 40, platform)
            for name, (_, hashes) in lock.items():
                self.assertTrue(hashes, f"{platform}: {name} has no hash")

    def test_P2_the_lock_has_every_dependency_of_the_project_within_its_bounds(self):
        wanted = (
            project()["dependencies"]
            + project()["optional-dependencies"]["openvino"]
            + project()["optional-dependencies"]["dev"]
        )
        for platform in PLATFORMS:
            lock = read_lock(platform)
            for text in wanted:
                requirement = Requirement(text)
                if requirement.marker and not requirement.marker.evaluate({"python_version": f"3.{platform[-2:]}"}):
                    continue
                name = canonicalize_name(requirement.name)
                self.assertIn(name, lock, f"{platform} lacks {name}")
                self.assertTrue(
                    requirement.specifier.contains(lock[name][0]),
                    f"{platform}: {name} {lock[name][0]} is out of bounds",
                )

    def test_P2_the_lock_pins_what_building_the_package_needs(self):
        for platform in PLATFORMS:
            self.assertTrue({"pip", "setuptools", "wheel"} <= set(read_lock(platform)), platform)

    def test_P2_a_wheel_with_an_altered_hash_does_not_install(self):
        with tempfile.TemporaryDirectory() as scratch:
            scratch = Path(scratch)
            wheel = tiny_wheel(scratch)
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            command = [
                sys.executable,
                "-m",
                "pip",
                "install",
                "--no-index",
                "--find-links",
                str(scratch),
                "--require-hashes",
                "--target",
                str(scratch / "site"),
                "--disable-pip-version-check",
                "-r",
            ]
            good = scratch / "good.txt"
            good.write_text(f"tinything==1.0 --hash=sha256:{digest}\n", encoding="utf-8")
            done = subprocess.run(command + [str(good)], capture_output=True, text=True)
            self.assertEqual(done.returncode, 0, done.stdout + done.stderr)
            self.assertTrue((scratch / "site" / "tinything").is_dir())
            bad = scratch / "bad.txt"
            bad.write_text(f"tinything==1.0 --hash=sha256:{'0' * 64}\n", encoding="utf-8")
            done = subprocess.run(command + [str(bad)], capture_output=True, text=True)
            self.assertNotEqual(done.returncode, 0)
            self.assertIn("HASHES", done.stdout + done.stderr)

    def test_P2_the_hashes_of_the_local_wheel_in_the_lock_are_those_of_the_file(self):
        folder = ROOT / "requirements" / "wheels"
        for wheel in folder.glob("*.whl"):
            name = canonicalize_name(wheel.name.split("-")[0])
            digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
            for platform in PLATFORMS:
                self.assertIn(digest, read_lock(platform)[name][1], f"{platform}: {wheel.name}")


class ThirdPartyTest(unittest.TestCase):
    def test_P8_every_package_of_every_lock_has_its_row_with_a_licence(self):
        rows = {}
        for line in (ROOT / "THIRD-PARTY.md").read_text(encoding="utf-8").splitlines():
            cells = [c.strip() for c in line.strip("|").split("|")]
            if line.startswith("| ") and len(cells) == 4:
                rows[canonicalize_name(cells[0])] = cells
        for platform in PLATFORMS:
            for name, (version, _) in read_lock(platform).items():
                self.assertIn(name, rows, f"{platform}: {name} has no row in THIRD-PARTY.md")
                self.assertIn(version, rows[name][1], f"{platform}: {name} {version}")
                self.assertNotIn("see the project", rows[name][2], name)

    def test_P8_the_wordfreq_data_licence_and_what_it_asks_are_stated(self):
        text = (ROOT / "THIRD-PARTY.md").read_text(encoding="utf-8") + (ROOT / "MODELS.md").read_text(encoding="utf-8")
        self.assertIn("CC BY-SA 4.0", text)
        self.assertIn("Robyn Speer", text)


class MachineNamesTest(unittest.TestCase):
    def versioned_files(self):
        for folder, names, files in os.walk(ROOT):
            names[:] = [n for n in names if n not in NOT_VERSIONED and not n.endswith(".egg-info")]
            relative = Path(folder).relative_to(ROOT).as_posix()
            if relative == "bench/data":
                names[:] = []
                continue
            for file in (f for f in files if f not in NOT_VERSIONED):
                yield Path(folder, file), (Path(relative) / file).as_posix()

    def test_P3_no_versioned_file_names_a_personal_home_folder(self):
        found = []
        for path, relative in self.versioned_files():
            if path.suffix in {
                ".png",
                ".jpg",
                ".avif",
                ".npy",
                ".gz",
                ".model",
                ".whl",
                ".pdf",
            }:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            found += [f"{relative}: {match}" for match in HOME_PATH.findall(text)]
        self.assertEqual(found, [])


class ContinuousIntegrationTest(unittest.TestCase):
    def workflow(self) -> dict:
        import yaml

        return yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8"))

    def steps(self) -> list[dict]:
        return self.workflow()["jobs"]["test"]["steps"]

    def test_P6_the_workflow_is_valid_yaml_and_runs_the_tested_combination(self):
        # Only Linux with Python 3.10 is run in CI: the lock files of Python 3.12 and Windows exist but have not been run.
        job = self.workflow()["jobs"]["test"]
        self.assertEqual(job["runs-on"], "ubuntu-latest")
        setup = next(step for step in self.steps() if str(step.get("uses", "")).startswith("actions/setup-python"))
        self.assertEqual(str(setup["with"]["python-version"]), "3.10")

    def test_P6_it_installs_lints_tests_and_repeats_verify_install(self):
        commands = "\n".join(step.get("run", "") for step in self.steps())
        for wanted in ("scripts/install.py --ci", "ruff check", "pytest", "scripts/verify_install.py"):
            self.assertIn(wanted, commands)

    def test_P6_every_script_a_step_runs_exists(self):
        commands = "\n".join(step.get("run", "") for step in self.steps())
        named = re.findall(r"scripts/[\w.-]+\.py", commands)
        self.assertGreaterEqual(len(named), 2)
        for script in named:
            self.assertTrue((ROOT / script).is_file(), script)

    def test_P6_the_model_cache_is_keyed_by_the_catalogue_that_holds_the_hashes(self):
        cache = next(step for step in self.steps() if str(step.get("uses", "")).startswith("actions/cache"))
        self.assertIn("hashFiles('models/models.json')", cache["with"]["key"])
