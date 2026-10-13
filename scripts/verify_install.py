"""Checks that Extract installs and reads a PDF on a machine that has nothing of ours.

    python3 scripts/verify_install.py          # copies the tree to a temporary folder, installs it there from
                                               # scratch (new environment, cold pip cache, models downloaded again)
                                               # and reads a sample PDF
    python3 scripts/verify_install.py --here   # only the reading, with the Python that runs this script

The sample is a page of text drawn to an image and saved as a PDF without a text layer, so the OCR has to read it.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_LINES = [
    "Invoice number 2026 0042",
    "Date 2 October 2026",
    "Customer Zorvanta and Sons Limited",
    "Description of the work done in September",
    "Subtotal 1000 euros",
    "Total payable 1210 euros",
]
MIN_RECALL = 0.85
IGNORED = (".git", ".venv", "dist", "__pycache__", "*.egg-info", ".pytest_cache", ".ruff_cache", "*.npy", "*.tsv.gz")
UNTRACKED_FOLDERS = ("bench/data", "work")


def skip(folder: str, names: list[str]) -> set[str]:
    """What not to copy: caches, build output, the generated word lists and the folders git does not track."""
    ignored = set(shutil.ignore_patterns(*IGNORED)(folder, names))
    relative = Path(folder).resolve().relative_to(ROOT).as_posix()
    return ignored | {n for n in names if f"{relative}/{n}".lstrip("./") in UNTRACKED_FOLDERS}


def make_sample(path: Path) -> list[str]:
    """Writes the sample PDF and returns the words that are printed on it."""
    from PIL import Image, ImageDraw, ImageFont

    image = Image.new("RGB", (1654, 2339), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=46)
    for row, line in enumerate(SAMPLE_LINES):
        draw.text((140, 200 + row * 110), line, fill="black", font=font)
    image.save(path, "PDF", resolution=200)
    return [word.lower() for line in SAMPLE_LINES for word in line.split()]


def read_sample(folder: Path) -> float:
    import extract

    truth = make_sample(folder / "sample.pdf")
    document = extract.read(folder / "sample.pdf")
    found = {word.text.lower().strip(".,;:") for word in document.words()}
    hits = [word for word in truth if word in found]
    missing = sorted(set(truth) - found)
    print(f"read {len(document.pages)} page, {len(hits)} of {len(truth)} printed words found, status {document.status}")
    if missing:
        print("not found:", ", ".join(missing))
    return len(hits) / len(truth)


def check_here() -> None:
    with tempfile.TemporaryDirectory() as folder:
        recall = read_sample(Path(folder))
    if recall < MIN_RECALL:
        sys.exit(f"FAILED: only {recall:.0%} of the printed words were read (needs {MIN_RECALL:.0%})")
    print(f"OK: {recall:.0%} of the printed words read")


def clean_room() -> None:
    with tempfile.TemporaryDirectory(prefix="extract-verify-") as scratch:
        scratch = Path(scratch)
        tree = scratch / "extract"
        shutil.copytree(ROOT, tree, ignore=skip)
        env = {
            **os.environ,
            "EXTRACT_MODELS_DIR": str(scratch / "models"),
            "XDG_CACHE_HOME": str(scratch / "cache"),
            "PIP_NO_CACHE_DIR": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        for name in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV"):
            env.pop(name, None)
        result = subprocess.run(
            [sys.executable, "scripts/install.py", "--venv", str(scratch / "venv")], cwd=tree, env=env
        )
        if result.returncode:
            sys.exit(f"FAILED: the installation stopped with {result.returncode}")
    print("OK: clean installation and reading")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--here", action="store_true", help="only read the sample with this Python")
    args = parser.parse_args()
    if args.here:
        check_here()
    else:
        clean_room()


if __name__ == "__main__":
    main()
