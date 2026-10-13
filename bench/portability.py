"""Portability of the reading (V5, README): the install timed from scratch on another machine and the first reading of
five pages on that machine compared, byte for byte, with the one the development machine stored.

    python3 -m bench.portability --install-log verify-install.log --other-machine-reading other-machine-reading.json \\
        --data <bench data> --out bench/results/portability.json

The install log is what `time python3 scripts/verify_install.py` printed (a `real 2m17,824s` line, with the comma or the
point of the machine's locale); the reading file is what the page reader of the other machine wrote: the engine, the
backend and, per page `<doc id>-<page>`, the seconds and the serialised first reading. The development machine's reading
of the same page is the `ocr.reading` of its page file in the corpus data; both are compared as canonical JSON (numbers exactly, an integer and the same float being one value), without the
`reread_report` (what pass 1 counted about its crops), and each side's sha256 is recorded."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

REAL = re.compile(r"^real\s+(\d+)m(\d+)[,.](\d+)s", re.M)


def install_seconds(log: str) -> float | None:
    """The `real` time of a `time` output, in seconds (None when the log has none)."""
    found = REAL.findall(log)
    if not found:
        return None
    minutes, seconds, fraction = found[-1]
    return int(minutes) * 60 + float(f"{seconds}.{fraction}")


# what pass 1 of the second reading counts about its own crops: bookkeeping of the study, not part of the reading, and
# only the development machine ran the study
BOOKKEEPING = ("reread_report",)


def _floats(node):
    """The same tree with every number a float: `102` and `102.0` are the same value and one file may write either."""
    if isinstance(node, dict):
        return {key: _floats(value) for key, value in node.items()}
    if isinstance(node, list):
        return [_floats(value) for value in node]
    if isinstance(node, int) and not isinstance(node, bool):
        return float(node)
    return node


def canonical(reading: dict) -> str:
    """The reading as canonical JSON, without the bookkeeping of the second reading's study; the values are compared
    exactly (a confidence differing in the last digit is a difference)."""
    kept = {key: value for key, value in reading.items() if key not in BOOKKEEPING}
    return json.dumps(_floats(kept), sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stored_reading(data: Path, page_id: str) -> tuple[str, dict] | None:
    """(set, reading) of the page `<doc id>-<page>` in the corpus data: the first reading its page file stores."""
    doc, _, page = page_id.rpartition("-")
    for path in sorted((data / "pages").glob(f"*/{doc}/{page}.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        ocr = record.get("ocr")
        if ocr and "reading" in ocr:
            return path.parts[-3], ocr["reading"]
    return None


def compare(other: dict, data: Path) -> dict:
    pages = []
    for page_id, entry in sorted(other["pages"].items()):
        mine = canonical(entry["reading"])
        found = stored_reading(data, page_id)
        record = {
            "id": page_id,
            "words": len(entry["reading"]["words"]),
            "seconds_other_machine": round(entry["seconds"], 2),
            "sha256_other_machine": sha256(mine),
        }
        if found is None:
            record.update(set=None, identical=False, sha256_development=None, note="no stored reading for this page")
        else:
            theirs = canonical(found[1])
            record.update(set=found[0], identical=mine == theirs, sha256_development=sha256(theirs))
        pages.append(record)
    return {
        "engine_other_machine": other["engine"],
        "backend": other["backend"],
        "pages": pages,
        "all_identical": bool(pages) and all(p["identical"] for p in pages),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--install-log", type=Path, required=True)
    parser.add_argument("--other-machine-reading", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--machine", default="a second, smaller laptop (i3, 2 cores, 3.3 GB)")
    parser.add_argument("--out", type=Path, default=Path("bench/results/portability.json"))
    args = parser.parse_args(argv)
    seconds = install_seconds(args.install_log.read_text(encoding="utf-8", errors="replace"))
    result = {
        "machine": args.machine,
        "install": {
            "command": "python3 scripts/verify_install.py (a new environment, a cold pip cache, the models downloaded again)",
            "seconds": seconds,
        },
        "method": "the pages are read on the other machine with this tree's engine, the page's dpi rounded as the corpus run "
        "passes it; the readings are compared with the `ocr.reading` stored by the development machine, without the "
        "`reread_report` (the study's bookkeeping, which only the development machine ran)",
        "parity": compare(json.loads(args.other_machine_reading.read_text(encoding="utf-8")), args.data),
    }
    args.out.write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"install_seconds": seconds, "all_identical": result["parity"]["all_identical"]}))
    return 0 if seconds is not None and result["parity"]["all_identical"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
