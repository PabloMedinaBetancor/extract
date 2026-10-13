"""Writes requirements/<platform>.txt: every package the product and its tests need, pinned with sha256 hashes.

For each platform the wheels are resolved and downloaded with `pip download --only-binary=:all:` for that platform
(the dependencies come from pyproject.toml, the pins from requirements/pins.in), hashed, and written as a file that
`pip install --require-hashes` accepts. The wheels stay in `dist/wheelhouse/<platform>/`, which the offline bundle reuses.

    python3 scripts/lock.py                      # every platform
    python3 scripts/lock.py linux-x86_64-py310   # one platform
"""

import argparse
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements"
WHEELHOUSE = ROOT / "dist" / "wheelhouse"

LINUX_TAGS = ["manylinux_2_35_x86_64", "manylinux_2_28_x86_64", "manylinux_2_27_x86_64", "manylinux2014_x86_64"]
PLATFORMS = {
    "linux-x86_64-py310": (LINUX_TAGS, "3.10"),
    "linux-x86_64-py312": (LINUX_TAGS, "3.12"),
    "win-amd64-py310": (["win_amd64"], "3.10"),
    "win-amd64-py312": (["win_amd64"], "3.12"),
}
# Needed to build and install the package itself; they are not dependencies of it.
# Published only as a source archive, and pip cannot resolve for another platform from one. It is pure Python: the wheel
# built from that archive is kept in requirements/wheels/ and offered to pip as a local source.
SOURCE_ONLY = {"antlr4-python3-runtime": "4.9.3"}
LOCAL_WHEELS = REQUIREMENTS / "wheels"
INSTALL_TOOLS = ["pip", "setuptools>=64", "wheel"]
EXTRAS = ["openvino", "dev"]


def requirements_in() -> list[str]:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    names = list(project["dependencies"])
    for extra in EXTRAS:
        names += project["optional-dependencies"][extra]
    return INSTALL_TOOLS + names


def read_pins() -> dict[str, str]:
    pins = {}
    for line in (REQUIREMENTS / "pins.in").read_text(encoding="utf-8").splitlines():
        if line.strip() and not line.startswith("#"):
            name, version = line.split("==")
            pins[name] = version
    return pins


def canonical(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def current_pip() -> dict[str, str]:
    """Environment that runs the latest pip from its wheel: the resolver of the pip that ships with Ubuntu 22.04
    (22.0.2) picks versions that newer pips reject."""
    tools = ROOT / "dist" / "tools"
    if not list(tools.glob("pip-*.whl")):
        subprocess.run(
            [sys.executable, "-m", "pip", "download", "--no-deps", "--dest", str(tools), "pip"],
            check=True,
            capture_output=True,
        )
    wheel = sorted(tools.glob("pip-*.whl"))[-1]
    return {**os.environ, "PYTHONPATH": str(wheel)}


def build_local_wheels() -> None:
    """Builds the wheels of the source-only packages once; the result is committed, so its hash never changes."""
    for name, version in SOURCE_ONLY.items():
        if list(LOCAL_WHEELS.glob(f"{name.replace('-', '_')}-{version}-*.whl")):
            continue
        LOCAL_WHEELS.mkdir(exist_ok=True)
        subprocess.run(
            [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(LOCAL_WHEELS), f"{name}=={version}"],
            check=True,
            env=current_pip(),
        )


def download(platform: str, pins: dict[str, str], target: Path) -> list[str]:
    """Downloads the wheels of one platform into `target`. Returns the pins dropped because they have no wheel there."""
    tags, python = PLATFORMS[platform]
    dropped: list[str] = []
    with tempfile.TemporaryDirectory() as scratch:
        inputs = Path(scratch) / "in.txt"
        inputs.write_text("\n".join(requirements_in()) + "\n", encoding="utf-8")
        while True:
            constraints = Path(scratch) / "constraints.txt"
            constraints.write_text("".join(f"{n}=={v}\n" for n, v in pins.items()), encoding="utf-8")
            command = [sys.executable, "-m", "pip", "download", "--only-binary=:all:"]
            command += ["--find-links", str(LOCAL_WHEELS), "--python-version", python]
            for tag in tags:
                command += ["--platform", tag]
            command += ["--dest", str(target), "--constraint", str(constraints), "--requirement", str(inputs)]
            result = subprocess.run(command, capture_output=True, text=True, env=current_pip())
            if result.returncode == 0:
                return dropped
            culprit = _unsatisfied_pin(result.stdout + result.stderr, pins)
            if culprit is None:
                sys.exit(f"pip download failed for {platform}:\n{(result.stdout + result.stderr)[-2000:]}")
            print(f"{platform}: no wheel for {culprit}=={pins[culprit]}, pin dropped", file=sys.stderr)
            dropped.append(f"{culprit}=={pins.pop(culprit)}")


def _unsatisfied_pin(stderr: str, pins: dict[str, str]) -> str | None:
    for name, version in pins.items():
        if f"(constraint) {name}=={version}" in stderr or f"{name}=={version} " in stderr:
            return name
    return None


def name_version(path: Path) -> tuple[str, str]:
    if path.suffix == ".whl":
        name, version = path.name.split("-")[:2]
    else:
        name, version = path.name.removesuffix(".tar.gz").rsplit("-", 1)
    return canonical(name), version


def write_lock(platform: str, dropped: list[str]) -> Path:
    folder = WHEELHOUSE / platform
    wheels = sorted([*folder.glob("*.whl"), *folder.glob("*.tar.gz")], key=lambda p: name_version(p)[0])
    lines = [f"# {platform}: written by scripts/lock.py, do not edit by hand"]
    if dropped:
        lines.append("# no wheel for this platform, so not pinned to the tested version: " + ", ".join(dropped))
    for wheel in wheels:
        name, version = name_version(wheel)
        digest = hashlib.sha256(wheel.read_bytes()).hexdigest()
        lines.append(f"{name}=={version} \\\n    --hash=sha256:{digest}")
    path = REQUIREMENTS / f"{platform}.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("platforms", nargs="*", help=f"one or more of: {', '.join(PLATFORMS)} (default: all)")
    args = parser.parse_args()
    unknown = set(args.platforms) - set(PLATFORMS)
    if unknown:
        parser.error(f"unknown platform: {', '.join(sorted(unknown))}")
    build_local_wheels()
    for platform in args.platforms or list(PLATFORMS):
        target = WHEELHOUSE / platform
        target.mkdir(parents=True, exist_ok=True)
        dropped = download(platform, read_pins(), target)
        print(write_lock(platform, dropped))


if __name__ == "__main__":
    main()
