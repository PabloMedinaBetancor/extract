"""Installs Extract from a clone with one command: a virtual environment, every dependency pinned by hash, the package,
the OCR models, the word lists, and a reading of a sample PDF to prove it works. Standard library only.

    python3 scripts/install.py                  # creates ./.venv
    python3 scripts/install.py --venv DIR       # somewhere else
    python3 scripts/install.py --offline DIR    # from an unpacked offline bundle, with no network

Then: `.venv/bin/python -m extract.ui` (`.venv\\Scripts\\python -m extract.ui` on Windows).
"""

import argparse
import hashlib
import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIREMENTS = ROOT / "requirements"


def platform_name() -> str:
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        sys.exit(f"no lock file for {machine}; the lock files are for 64-bit x86 Linux and Windows")
    system = {"linux": "linux-x86_64", "win32": "win-amd64"}.get(sys.platform)
    if not system:
        sys.exit(f"no lock file for {sys.platform}; the lock files are for Linux and Windows")
    return f"{system}-py{sys.version_info.major}{sys.version_info.minor}"


def lock_file() -> Path:
    path = REQUIREMENTS / f"{platform_name()}.txt"
    if not path.is_file():
        available = ", ".join(p.stem for p in sorted(REQUIREMENTS.glob("*.txt")))
        sys.exit(
            f"{path.name} does not exist; this machine needs Python 3.10 or 3.12 on 64-bit Linux or Windows ({available})"
        )
    return path


def pip_pin(lock: Path) -> tuple[str, set[str]]:
    text = lock.read_text(encoding="utf-8").replace("\\\n", " ")
    match = re.search(r"^pip==(\S+)((?: +--hash=sha256:[0-9a-f]+)+)", text, re.M)
    if not match:
        sys.exit(f"{lock.name} has no pinned pip")
    return match.group(1), set(re.findall(r"sha256:([0-9a-f]+)", match.group(2)))


def pip_wheel(lock: Path, offline: Path | None, scratch: Path) -> Path:
    """The pinned pip as a wheel, which runs from its zip file with no pip around."""
    version, hashes = pip_pin(lock)
    if offline:
        candidates = list((offline / "wheelhouse").glob(f"pip-{version}-*.whl"))
        if not candidates:
            sys.exit(f"the bundle has no pip-{version} wheel")
        return candidates[0]
    with urllib.request.urlopen(f"https://pypi.org/pypi/pip/{version}/json", timeout=60) as response:
        files = json.load(response)["urls"]
    match = next((f for f in files if f["filename"].endswith(".whl") and f["digests"]["sha256"] in hashes), None)
    if not match:
        sys.exit(f"PyPI has no pip {version} wheel with the hash of {lock.name}")
    target = scratch / match["filename"]
    with urllib.request.urlopen(match["url"], timeout=120) as response:
        target.write_bytes(response.read())
    if hashlib.sha256(target.read_bytes()).hexdigest() not in hashes:
        sys.exit("the pip wheel does not match its hash")
    return target


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def shown(command: list) -> str:
    """The command as a person can paste it back into a terminal: a path with a space or an accent in it is quoted."""
    parts = [str(part) for part in command]
    return subprocess.list2cmdline(parts) if sys.platform == "win32" else shlex.join(parts)


def run(command: list, env: dict | None = None) -> None:
    command = [str(part) for part in command]
    print("+", shown(command), flush=True)
    result = subprocess.run(command, env={**os.environ, **(env or {})}, cwd=ROOT)
    if result.returncode:
        sys.exit(f"failed ({result.returncode}): {shown(command)}")


def clear_venv(venv: Path) -> None:
    """Removes an earlier environment, and only that: a folder that is not a venv (no pyvenv.cfg) and not empty is
    somebody's files, so the installer stops instead of deleting it."""
    if not venv.exists():
        return
    if venv.is_dir() and ((venv / "pyvenv.cfg").exists() or not any(venv.iterdir())):
        shutil.rmtree(venv, ignore_errors=True)
        return
    sys.exit(
        f"{venv} exists and is not a virtual environment (no pyvenv.cfg): choose another --venv or remove it yourself"
    )


def create_venv(venv: Path, lock: Path, offline: Path | None) -> dict:
    """Makes the environment and returns the extra environment pip needs. Ubuntu without python3-venv cannot run
    ensurepip, so the environment is made without pip and the pinned pip wheel installs everything, itself included."""
    clear_venv(venv)
    probe = subprocess.run([sys.executable, "-m", "venv", str(venv)], capture_output=True)
    if probe.returncode == 0:
        return {}
    print("python3 -m venv could not install pip (python3-venv is missing?); bootstrapping pip from its pinned wheel")
    clear_venv(venv)
    run([sys.executable, "-m", "venv", "--without-pip", venv])
    scratch = Path(tempfile.mkdtemp(prefix="extract-pip-"))
    return {"PYTHONPATH": str(pip_wheel(lock, offline, scratch))}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--venv", type=Path, default=ROOT / ".venv", help="where to create the environment (default: .venv here)"
    )
    parser.add_argument("--offline", type=Path, help="an unpacked offline bundle: no network is used")
    parser.add_argument("--ci", action="store_true", help="also run pip check and list what was installed")
    args = parser.parse_args()
    args.venv = args.venv.resolve()
    args.offline = args.offline.resolve() if args.offline else None

    lock = lock_file()
    env = create_venv(args.venv, lock, args.offline)
    python = venv_python(args.venv)
    sources = ["--find-links", REQUIREMENTS / "wheels"]
    if args.offline:
        sources += ["--no-index", "--find-links", args.offline / "wheelhouse"]
    run(
        [
            python,
            "-m",
            "pip",
            "install",
            "--require-hashes",
            "--only-binary=:all:",
            "--disable-pip-version-check",
        ]
        + sources
        + ["-r", lock],
        env,
    )
    run([python, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-index", "-e", "."])
    models = ["--source", args.offline / "models"] if args.offline else []
    run([python, ROOT / "scripts" / "fetch_models.py"] + models)
    lists = ["--from-release", (args.offline / "lists.tar.gz").as_uri()] if args.offline else []
    run([python, ROOT / "scripts" / "fetch_models.py", "--lists"] + lists)
    if args.ci:
        run([python, "-m", "pip", "check"])
        run([python, "-m", "pip", "list"])
    run([python, ROOT / "scripts" / "verify_install.py", "--here"])
    print(f"\nInstalled. Start the interface with: {shown([python, '-m', 'extract.ui'])}")


if __name__ == "__main__":
    main()
