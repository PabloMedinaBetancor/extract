"""The packaged decision about the second reading: whether it is on by default, which cell it uses and what
the Import panel says it costs. One JSON file, `data/second_reading.json`, is the only place the decision is written;
`Options`, the reader process and the server read it from here, so fixing the default is a data change, not a code one."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

PATH = Path(__file__).resolve().parent / "data" / "second_reading.json"
UNITS = ("line", "zone")
READERS = ("latin", "bin", "clahe", "wide", "deep")
COST_KEYS = ("reader", "seconds_per_page", "fixes_more_pct")
THOROUGH = "thorough"  # the value of `Options.second_reading` for the thorough mode: opt-in, its own guard model
MODES = (False, True, THOROUGH)
THOROUGH_MODEL = Path(__file__).resolve().parent / "data" / "guard-thorough.model"
THOROUGH_MODEL_ENV = "EXTRACT_GUARD_MODEL_THOROUGH"
THOROUGH_KEYS = (
    "lines",
    "zones",
    "readers_lines",
    "reader_zones",
    "seconds_per_page",
    "seconds_p95",
    "reader_ram_mb",
    "fixes_more_pct_vs_default",
    "fixes_more_pct_by_language",
)


def validate(data: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(data.get("default"), bool):
        raise ValueError("second_reading.json: `default` must be a bool")
    if data.get("unit") not in UNITS:
        raise ValueError(f"second_reading.json: `unit` must be one of {UNITS}")
    readers = data.get("readers")
    if (
        not isinstance(readers, list)
        or not readers
        or any(r not in READERS for r in readers)
        or len(set(readers)) != len(readers)
    ):
        raise ValueError(f"second_reading.json: `readers` must be distinct names from {READERS}")
    if any(key not in data for key in COST_KEYS):
        raise ValueError(f"second_reading.json: needs {COST_KEYS}")
    for key in ("seconds_per_page", "fixes_more_pct"):
        value = data[key]
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0):
            raise ValueError(f"second_reading.json: `{key}` must be a non-negative number or null")
    if data["reader"] is not None and not isinstance(data["reader"], str):
        raise ValueError("second_reading.json: `reader` must be a string or null")
    cap = data.get("cap")
    if cap is not None and (isinstance(cap, bool) or not isinstance(cap, int) or cap < 1):
        raise ValueError("second_reading.json: `cap` must be a positive integer or null")
    if "thorough" in data:
        validate_thorough(data["thorough"])
    if data["default"] and any(data[key] is None for key in COST_KEYS):
        raise ValueError("second_reading.json: on by default needs the cost the panel shows (reader, seconds, fixes)")
    return data


def validate_thorough(block: Any) -> dict[str, Any]:
    """The thorough mode: `lines` lines read by `readers_lines` and `zones` zones read by `reader_zones` (the most pass 1
    stores, 16 and 40), with the cost the panel shows."""
    if not isinstance(block, dict) or any(key not in block for key in THOROUGH_KEYS):
        raise ValueError(f"second_reading.json: `thorough` needs {THOROUGH_KEYS}")
    for key, top in (("lines", 16), ("zones", 40)):
        value = block[key]
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= top:
            raise ValueError(f"second_reading.json: thorough `{key}` must be an integer from 1 to {top}")
    lines = block["readers_lines"]
    if (
        not isinstance(lines, list)
        or not lines
        or any(r not in READERS for r in lines)
        or len(set(lines)) != len(lines)
    ):
        raise ValueError(f"second_reading.json: thorough `readers_lines` must be distinct names from {READERS}")
    if block["reader_zones"] not in READERS:
        raise ValueError(f"second_reading.json: thorough `reader_zones` must be one of {READERS}")
    for key in ("seconds_per_page", "seconds_p95", "reader_ram_mb", "fixes_more_pct_vs_default"):
        value = block[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"second_reading.json: thorough `{key}` must be a non-negative number")
    by_language = block["fixes_more_pct_by_language"]
    if not isinstance(by_language, dict) or not by_language:
        raise ValueError("second_reading.json: thorough `fixes_more_pct_by_language` must map languages to numbers")
    for language, value in by_language.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
            raise ValueError(f"second_reading.json: thorough `fixes_more_pct_by_language.{language}` must be a number")
    return block


def valid_mode(value: Any) -> bool:
    """`False` (off), `True` (the default cell) or `"thorough"`; nothing else (a `0`/`1` is not a mode)."""
    return any(value is mode or (isinstance(value, str) and value == mode) for mode in MODES)


def is_thorough(value: Any) -> bool:
    return isinstance(value, str) and value == THOROUGH  # only the exact word; anything else is the cell or off


@lru_cache(maxsize=1)
def load(path: str | None = None) -> dict[str, Any]:
    target = Path(path) if path else PATH
    return validate(json.loads(target.read_text(encoding="utf-8")))


def default_on() -> bool:
    return load()["default"]


def cell() -> tuple[str, tuple[str, ...]]:
    data = load()
    return data["unit"], tuple(data["readers"])


def cost() -> dict[str, Any]:
    data = load()
    return {key: data[key] for key in COST_KEYS}


def cap() -> int | None:
    """The units (lines or zones) a page reads at most, the cap that the guard's model was trained under; None: none."""
    return load().get("cap")


def thorough_installed() -> bool:
    """The guard model of the thorough mode (it travels in the wheel) and the reader models (`fetch_models.py --all`) are all on this machine."""
    from extract.engine import modelfiles, readers  # the engine loads this module: import it only when asked

    guard = Path(os.environ.get(THOROUGH_MODEL_ENV) or THOROUGH_MODEL).is_file()
    return guard and all(modelfiles.has_model(f) for f in (readers.LATIN_FILE, readers.DEEP_FILE))


def thorough() -> dict[str, Any]:
    """The thorough mode as the file writes it; raises if this install's file does not carry it."""
    block = load().get("thorough")
    if block is None:
        raise ValueError("second_reading.json has no `thorough` block")
    return block


def thorough_passes() -> tuple[tuple[str, tuple[str, ...]], ...]:
    """What pass 2 runs in the thorough mode, in order: (unit, readers) for the lines and for the zones."""
    block = thorough()
    return (("line", tuple(block["readers_lines"])), ("zone", (block["reader_zones"],)))


def thorough_readers() -> tuple[str, ...]:
    """Every reader the thorough mode loads, each once, in the order of the passes."""
    names: list[str] = []
    for _, readers in thorough_passes():
        names += [r for r in readers if r not in names]
    return tuple(names)


def thorough_cost() -> dict[str, Any]:
    """What the Import panel says about the thorough mode, all measured: the seconds a page (mean and p95), the memory the
    reader process peaks at, and the net fixes per 100 wrong words (Spanish, which the older panels read, and by language)."""
    block = thorough()
    return {
        "seconds_per_page": block["seconds_per_page"],
        "seconds_p95": block["seconds_p95"],
        "reader_ram_mb": block["reader_ram_mb"],
        "fixes_more_pct": block["fixes_more_pct_vs_default"],
        "fixes_more_pct_by_language": dict(block["fixes_more_pct_by_language"]),
    }
