"""Builds dist/extract-offline-<version>-<platform>.zip: the source tree, every wheel of the lock file, the OCR models
and the word lists, so a machine with no network (or no wish to use it) can install with one command.

    python3 scripts/make_offline_bundle.py                           # for this machine
    python3 scripts/make_offline_bundle.py --platform win-amd64-py312

The Windows bundles are built here but nobody has installed from them on Windows yet. To use a bundle: unzip it, then

    python3 scripts/install.py --offline offline
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path

from fetch_models import LISTS_DIR, fetch, fetch_lists, list_sums, read_catalogue
from install import platform_name
from lock import PLATFORMS, current_pip
from verify_install import skip

ROOT = Path(__file__).resolve().parents[1]
README = """Extract, offline bundle for {platform}

Unzip this folder, open a terminal in it and run

    python3 scripts/install.py --offline offline

It creates .venv here from the wheels in offline/wheelhouse (every one checked against requirements/{platform}.txt),
installs the package, copies the OCR models and the word lists, and reads a sample PDF. No network is used.
Then start the interface with

    .venv/bin/python -m extract.ui        (.venv\\Scripts\\python -m extract.ui on Windows)

Python {python} on 64-bit {system} is needed. Licences of everything inside: THIRD-PARTY.md and MODELS.md.
"""


def version() -> str:
    text = (ROOT / "src" / "extract" / "__init__.py").read_text(encoding="utf-8")
    return re.search(r'__version__ = "([^"]+)"', text).group(1)


def download_wheels(platform: str, target: Path) -> None:
    """Every wheel of the lock file, checked against its hash while it is downloaded."""
    tags, python = PLATFORMS[platform]
    command = [sys.executable, "-m", "pip", "download", "--no-deps", "--require-hashes", "--only-binary=:all:"]
    for tag in tags:
        command += ["--platform", tag]
    command += ["--python-version", python, "--find-links", str(ROOT / "requirements" / "wheels")]
    command += ["--dest", str(target), "--requirement", str(ROOT / "requirements" / f"{platform}.txt")]
    subprocess.run(command, check=True, env=current_pip())


def pack_lists(target: Path) -> None:
    fetch_lists(None)
    with tarfile.open(target, "w:gz") as archive:
        for name in list_sums():
            archive.add(LISTS_DIR / name, arcname=name)


def assemble(platform: str, folder: Path, wheelhouse: Path, models: Path, lists: Path) -> None:
    """Lays out the bundle in `folder`: the tree of the repository and `offline/` with what pip and the scripts need."""
    shutil.copytree(ROOT, folder, ignore=skip)
    offline = folder / "offline"
    shutil.copytree(wheelhouse, offline / "wheelhouse")
    shutil.copytree(models, offline / "models")
    shutil.copyfile(lists, offline / "lists.tar.gz")
    major, minor = platform[-3], platform[-2:]
    system = "Linux" if platform.startswith("linux") else "Windows"
    (folder / "README.txt").write_text(
        README.format(platform=platform, python=f"{major}.{minor}", system=system), encoding="utf-8"
    )


def zip_folder(folder: Path, archive: Path) -> None:
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(folder.rglob("*")):
            if path.is_file():
                bundle.write(path, Path(folder.name) / path.relative_to(folder))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--platform", default=platform_name(), choices=list(PLATFORMS))
    args = parser.parse_args()
    name = f"extract-offline-{version()}-{args.platform}"
    with tempfile.TemporaryDirectory() as scratch:
        scratch = Path(scratch)
        wheelhouse, models = scratch / "wheelhouse", scratch / "models"
        download_wheels(args.platform, wheelhouse)
        for entry in read_catalogue(ROOT / "models" / "models.json"):
            if entry.get("default"):
                fetch(entry, models)
        pack_lists(scratch / "lists.tar.gz")
        assemble(args.platform, scratch / name, wheelhouse, models, scratch / "lists.tar.gz")
        (ROOT / "dist").mkdir(exist_ok=True)
        archive = ROOT / "dist" / f"{name}.zip"
        zip_folder(scratch / name, archive)
    print(archive, f"{os.path.getsize(archive) / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
