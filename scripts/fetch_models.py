"""Downloads the OCR models listed in models/models.json and checks each one against its sha256.

Files go to $EXTRACT_MODELS_DIR, or the user's data folder when it is not set (the same place the engine looks first).
A download that stops is resumed. A file whose hash is wrong is deleted and the script stops with an error: nothing
unverified is left behind. A file already in place with the right hash is not downloaded again.

    python3 scripts/fetch_models.py                   # the default models
    python3 scripts/fetch_models.py --all             # every model in the catalogue
    python3 scripts/fetch_models.py --source DIR      # copy from a folder instead of downloading (offline bundle)
    .venv/bin/python scripts/fetch_models.py --lists  # build the word lists the checker needs
    python3 scripts/fetch_models.py --lists --from-release URL.tar.gz
"""

import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from extract.engine.modelfiles import ENV, user_models_dir  # noqa: E402

CATALOGUE = ROOT / "models" / "models.json"
LISTS_DIR = ROOT / "src" / "extract" / "guard" / "data"
CHUNK = 1 << 20


class HashMismatch(Exception):
    pass


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def target_dir(given: Path | None) -> Path:
    if given:
        return given
    import os

    return Path(os.environ[ENV]) if os.environ.get(ENV) else user_models_dir()


def is_in_place(path: Path, entry: dict) -> bool:
    return path.is_file() and path.stat().st_size == entry["size"] and sha256_of(path) == entry["sha256"]


def download(url: str, part: Path) -> None:
    """Appends to `part` what is still missing. A server that ignores the range makes it start over."""
    request = urllib.request.Request(url, headers={"User-Agent": "extract-fetch-models"})
    have = part.stat().st_size if part.exists() else 0
    if have:
        request.add_header("Range", f"bytes={have}-")
    try:
        response = urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code == 416:
            return
        raise
    with response:
        resumed = have and response.status == 206
        with part.open("ab" if resumed else "wb") as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)


def fetch(entry: dict, folder: Path, source: Path | None = None) -> Path:
    """Puts the model in `folder` with the right hash or raises. Returns the path of the file."""
    folder.mkdir(parents=True, exist_ok=True)
    final = folder / entry["file"]
    if is_in_place(final, entry):
        print(f"{entry['file']}: already in place")
        return final
    part = final.with_name(final.name + ".part")
    if source:
        shutil.copyfile(source / entry["file"], part)
    else:
        print(f"{entry['file']}: downloading {entry['size'] / 1e6:.1f} MB from {entry['url']}")
        download(entry["url"], part)
    found = sha256_of(part)
    if found != entry["sha256"]:
        part.unlink()
        final.unlink(missing_ok=True)
        raise HashMismatch(f"{entry['file']}: sha256 is {found}, the catalogue says {entry['sha256']}; file removed")
    part.replace(final)
    print(f"{entry['file']}: ok")
    return final


def read_catalogue(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["models"]


def list_sums() -> dict[str, str]:
    """File name -> sha256 of the word lists, from models/SHA256SUMS."""
    sums = {}
    for line in (ROOT / "models" / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        digest, name = line.split(maxsplit=1)
        if name.startswith("../src/extract/guard/data/"):
            sums[Path(name).name] = digest
    return sums


def lists_ready() -> bool:
    return all(
        (LISTS_DIR / name).is_file() and sha256_of(LISTS_DIR / name) == digest for name, digest in list_sums().items()
    )


def build_lists() -> None:
    if importlib.util.find_spec("wordfreq") is None:
        sys.exit(
            "wordfreq is not installed in this Python. It is installed in the environment that scripts/install.py made: "
            "run `.venv/bin/python scripts/fetch_models.py --lists` instead"
        )
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_dictionaries.py")], check=True)


def unpack_lists(url: str) -> None:
    sums = list_sums()
    with tempfile.TemporaryDirectory() as scratch:
        archive = Path(scratch) / "lists.tar.gz"
        print(f"word lists: downloading {url}")
        download(url, archive)
        with tarfile.open(archive) as tar:
            members = {Path(m.name).name: m for m in tar.getmembers() if m.isfile()}
            missing = sorted(set(sums) - set(members))
            if missing:
                sys.exit(f"the release lacks {', '.join(missing)}")
            for name in sums:
                tar.makefile(members[name], str(Path(scratch) / name))
        for name, digest in sums.items():
            if sha256_of(Path(scratch) / name) != digest:
                sys.exit(f"{name}: sha256 differs from models/SHA256SUMS; nothing was installed")
        LISTS_DIR.mkdir(parents=True, exist_ok=True)
        for name in sums:
            shutil.copyfile(Path(scratch) / name, LISTS_DIR / name)


def fetch_lists(release: str | None) -> None:
    if lists_ready():
        print("word lists: already in place")
        return
    if release:
        unpack_lists(release)
    else:
        build_lists()
    if not lists_ready():
        sys.exit("the word lists do not match models/SHA256SUMS after the build")
    print("word lists: ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dir", type=Path, help="where the models go (default: $EXTRACT_MODELS_DIR or the user folder)"
    )
    parser.add_argument("--catalogue", type=Path, default=CATALOGUE)
    parser.add_argument("--all", action="store_true", help="every model of the catalogue, not only the default ones")
    parser.add_argument("--only", action="append", default=[], help="a model name; can be repeated")
    parser.add_argument("--source", type=Path, help="copy the files from this folder instead of downloading")
    parser.add_argument("--lists", action="store_true", help="build the word lists instead of fetching models")
    parser.add_argument("--from-release", metavar="URL", help="with --lists: download the lists as a .tar.gz")
    args = parser.parse_args()
    if args.lists:
        fetch_lists(args.from_release)
        return
    entries = [
        e
        for e in read_catalogue(args.catalogue)
        if e["name"] in args.only or args.all or (not args.only and e.get("default"))
    ]
    folder = target_dir(args.dir)
    try:
        for entry in entries:
            fetch(entry, folder, args.source)
    except HashMismatch as error:
        sys.exit(f"ERROR: {error}")
    print(f"models in {folder}")


if __name__ == "__main__":
    main()
