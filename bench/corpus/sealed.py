"""The only reader of the sealed v5 test T5 (its standard pages, its invoices and its unseen-damage slice).

The test pages are built with the others, read by the engine without anyone looking at metrics
(purpose "ocr"), and read for numbers after the model is frozen (purpose "metrics"). Every
read is a line of `bench/results/test_reads_v5.json`. A second read for metrics needs a written
reason, which is stored in the line too.

The v4 test (sets `test` and `unseen`, archived in `bench/results/test_reads.json`) was retired as a test
when v5 was built: it is development data now, not sealed, and only `bench/read_test.py` still names it.
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

PURPOSES = ("ocr", "metrics")
SEALED_SETS = ("test5", "unseen5")
EN_TEST_SETS = ("test5en",)
"""The sealed English test (T5-EN): sealed exactly as T5, with its own freeze record and its own list; it is read through
`read_test_pages(..., sets=EN_TEST_SETS)`. It is not in `SEALED_SETS` so that the hashes of the Spanish T5 are the ones
recorded when it was frozen."""
READS_PATH = Path("bench/results/test_reads_v5.json")
EN_READS_PATH = Path("bench/results/test_reads_v5en.json")
V4_SETS = ("test", "unseen")
V4_READS_PATH = Path("bench/results/test_reads.json")


class SealedTestError(RuntimeError):
    pass


def refuse_sealed_set(set_name: str) -> None:
    """Raise for a sealed set: any reader of pages other than `read_test_pages` must call this first."""
    if set_name in SEALED_SETS or set_name in EN_TEST_SETS:
        raise SealedTestError(f"{set_name} is sealed: only read_test_pages opens it")


def load_reads(reads_path: Path = READS_PATH) -> list[dict]:
    if not reads_path.is_file():
        return []
    return json.loads(reads_path.read_text(encoding="utf-8"))


def read_test_pages(
    data_dir: Path,
    purpose: str,
    reads_path: Path = READS_PATH,
    reason: str | None = None,
    sets: tuple[str, ...] = SEALED_SETS,
) -> list[tuple[Path, dict]]:
    """Return (json path, record) for every test page, and log the read."""
    if purpose not in PURPOSES:
        raise ValueError(f"purpose must be one of {PURPOSES}, got {purpose!r}")
    reads = load_reads(reads_path)
    if purpose == "metrics" and any(r["purpose"] == "metrics" for r in reads) and not reason:
        raise SealedTestError("the test was already read for metrics; a second read needs a written reason")
    paths = sorted(
        path
        for name in sets
        for path in (data_dir / "pages" / name).glob("*/*.json")
        if path.stem.isdigit()  # not the `<n>.rr2-<cell>.json` caches of the second reading
    )
    pages = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    reads.append(
        {
            "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "purpose": purpose,
            "pages": len(pages),
            "fingerprint": fingerprint(pages),
            "reason": reason,
        }
    )
    _write(reads_path, reads)
    return pages


def fingerprint(pages: list[tuple[Path, dict]]) -> str:
    """SHA-256 over every page image and over its record without the OCR reading, which is written later."""
    digest = hashlib.sha256()
    for path, record in pages:
        digest.update(path.with_suffix(".png").read_bytes())
        digest.update(json.dumps({k: v for k, v in record.items() if k != "ocr"}, sort_keys=True).encode())
    return digest.hexdigest()


def fingerprint_parts(pages: list[tuple[Path, dict]]) -> dict[str, str]:
    """The pieces `fingerprint` mixes, apart: the page images, the truth (the record without its OCR reading and its
    `ungrounded` patches) and the patches. The patches are derived from the clean render and the truth by a rule that
    may be redone before any metric of the test is read, so a later check compares images and truth and reports
    the patches' hash beside the recorded one."""
    images, truth, patches = hashlib.sha256(), hashlib.sha256(), hashlib.sha256()
    for path, record in pages:
        images.update(path.with_suffix(".png").read_bytes())
        truth.update(
            json.dumps({k: v for k, v in record.items() if k not in ("ocr", "ungrounded")}, sort_keys=True).encode()
        )
        patches.update(json.dumps(record.get("ungrounded"), sort_keys=True).encode())
    return {"images": images.hexdigest(), "truth": truth.hexdigest(), "ungrounded": patches.hexdigest()}


def freeze_record(
    data_dir: Path, out_path: Path, note: str, reads_path: Path = READS_PATH, sets: tuple[str, ...] = SEALED_SETS
) -> dict:
    """Write the hashes of the sealed pages as they are now, with a note on why (read for hashing only)."""
    pages = read_test_pages(data_dir, "ocr", reads_path, sets=sets)
    record = {
        "time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "pages": len(pages),
        "fingerprint": fingerprint(pages),
        **fingerprint_parts(pages),
        "note": note,
    }
    _write(out_path, [*load_reads(out_path), record])
    return record


def metric_reads(reads_path: Path = READS_PATH) -> int:
    return sum(1 for r in load_reads(reads_path) if r["purpose"] == "metrics")


def _write(path: Path, reads: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(reads, indent=1), encoding="utf-8")
    os.replace(temporary, path)
