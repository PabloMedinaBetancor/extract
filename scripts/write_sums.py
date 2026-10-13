"""Writes models/SHA256SUMS: the checksum of the guard model (if built), of its copy inside the package and of the
guard's data files.

The data files are not in git: `scripts/build_dictionaries.py` writes them byte for byte from wordfreq, and
this list says which bytes are the right ones.

    python scripts/write_sums.py
"""

import hashlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "src" / "extract" / "guard" / "data"
PACKAGED = ROOT / "src" / "extract" / "data" / "guard.model"
PACKAGED_THOROUGH = ROOT / "src" / "extract" / "data" / "guard-thorough.model"
PATTERNS = ("*.npy", "*.tsv.gz")


def sums() -> list[str]:
    files = (
        [ROOT / "models" / name for name in ("guard.model", "guard-thorough.model", "guard-v4.model", "models.json")]
        + [p for p in (PACKAGED, PACKAGED_THOROUGH) if p.is_file()]
        + [p for pattern in PATTERNS for p in sorted(DATA.glob(pattern))]
    )
    return [f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {_relative(p)}" for p in files]


def _relative(path: Path) -> str:
    parts = path.relative_to(ROOT).parts
    return "/".join(parts[1:]) if parts[0] == "models" else "../" + "/".join(parts)


def main() -> None:
    (ROOT / "models" / "SHA256SUMS").write_text("\n".join(sums()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):  # `--help` must show this text, not run the script
        print(__doc__)
        sys.exit(0)
    main()
