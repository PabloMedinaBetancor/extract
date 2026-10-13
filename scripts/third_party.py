"""Writes THIRD-PARTY.md: the licence and home page of every package in the lock files, read from the metadata of
the wheels in dist/wheelhouse (scripts/lock.py leaves them there), plus the models and data files.

    python3 scripts/third_party.py
"""

import email
import email.message
import re
import sys
import zipfile
from pathlib import Path

from lock import PLATFORMS, WHEELHOUSE, name_version

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "THIRD-PARTY.md"
FILES = [
    (
        "PP-OCRv6 small, detection and recognition (ONNX)",
        "Apache-2.0",
        "https://github.com/RapidAI/RapidOCR (converted from PaddleOCR, https://github.com/PaddlePaddle/PaddleOCR)",
        "downloaded by scripts/fetch_models.py, not stored in this repository",
    ),
    (
        "Word lists, spelling index and character models built from wordfreq data",
        "CC BY-SA 4.0",
        "https://github.com/rspeer/wordfreq",
        "built on install by scripts/build_dictionaries.py; derived files keep the licence, see MODELS.md",
    ),
]
USAGE_REPORTING = (
    "OpenVINO sends an anonymous usage event to Google Analytics each time it is imported, unless a consent file in the "
    "home folder says no or the process runs with `CI=true`. Extract imports it with `CI=true` "
    "(`src/extract/engine/quiet_openvino.py`, tested); see B45 in docs/release-notes.md."
)


def metadata(wheel: Path) -> email.message.Message:
    with zipfile.ZipFile(wheel) as archive:
        name = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
        meta = email.message_from_bytes(archive.read(name))
        if licence_of(meta).startswith("Other"):
            files = [n for n in archive.namelist() if "/LICENSE" in n.upper() or "/COPYING" in n.upper()]
            head = archive.read(files[0]).decode("utf-8", "replace")[:200] if files else ""
            found = re.search(r"MIT License|Apache License|BSD \d?-?Clause|BSD License", head)
            if found:
                del meta["License"]
                meta["License"] = found.group(0)
        return meta


def licence_of(meta: email.message.Message) -> str:
    if meta["License-Expression"]:
        return meta["License-Expression"]
    declared = (meta["License"] or "").strip()
    if declared and "\n" not in declared and len(declared) <= 60 and declared.upper() != "UNKNOWN":
        return declared
    classifiers = [c.split(" :: ")[-1] for c in meta.get_all("Classifier", []) if c.startswith("License ::")]
    return " / ".join(classifiers) or "see the project page"


def url_of(meta: email.message.Message) -> str:
    if meta["Home-page"]:
        return meta["Home-page"]
    for entry in meta.get_all("Project-URL", []):
        label, _, url = entry.partition(",")
        if label.strip().lower() in {"homepage", "home", "source", "repository", "source code", "github"}:
            return url.strip()
    return f"https://pypi.org/project/{meta['Name']}/"


def collect() -> dict[str, dict]:
    packages: dict[str, dict] = {}
    for platform in PLATFORMS:
        for wheel in sorted((WHEELHOUSE / platform).glob("*.whl")):
            name, version = name_version(wheel)
            meta = metadata(wheel)
            row = packages.setdefault(name, {"versions": set(), "licence": licence_of(meta), "url": url_of(meta)})
            row["versions"].add(version)
    return packages


def render(packages: dict[str, dict]) -> str:
    lines = [
        "# Third-party software and data",
        "",
        "Written by `scripts/third_party.py` from the metadata of the packages in `requirements/*.txt`; do not edit by hand.",
        "A package is listed with the licence its own metadata declares.",
        "",
        "## Models and data",
        "",
        "| What | Licence | Source | How it gets here |",
        "|---|---|---|---|",
    ]
    lines += [f"| {what} | {licence} | {source} | {how} |" for what, licence, source, how in FILES]
    lines += ["", "## Python packages", "", "| Package | Versions | Licence | Home page |", "|---|---|---|---|"]
    for name in sorted(packages):
        row = packages[name]
        versions = ", ".join(sorted(row["versions"]))
        licence = re.sub(r"\s+", " ", row["licence"]).replace("|", "/")
        lines.append(f"| {name} | {versions} | {licence} | {row['url']} |")
    lines += ["", "## Usage reporting of a dependency", "", USAGE_REPORTING]
    return "\n".join(lines) + "\n"


def main() -> None:
    packages = collect()
    if not packages:
        sys.exit("no wheels in dist/wheelhouse: run `python3 scripts/lock.py` first. THIRD-PARTY.md was not touched.")
    OUTPUT.write_text(render(packages), encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):  # `--help` must show this text, not run the script
        print(__doc__)
        sys.exit(0)
    main()
