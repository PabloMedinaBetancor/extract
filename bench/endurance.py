"""Criterion O9 from a job log: does a long run stay as fast and as small at its end as at its start?

    python3 bench/endurance.py JOB_LOG [OUT_JSON]

Reads the `pool.page`, `worker.ready` and `worker.recycled` entries of a `job.log` written by the batch reader. The
first five pages of every worker are left out of the means (the engine is still warming up), the first and last 100
of the rest are compared, and the peak resident memory of every worker is checked against its backend's limit.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

WARM_UP_PAGES = 5
WINDOW = 100
MAX_SLOWDOWN = 1.15
MIN_PAGES = 400
MIN_RECYCLES = 3
RSS_LIMIT_MB = {"openvino": 1600, "onnxruntime": 900}


def evaluate(entries: list[dict]) -> dict:
    seen: dict[int, int] = {}
    seconds: list[float] = []
    peaks: dict[int, int] = {}
    backends: dict[int, str] = {}
    windows: list[float] = []
    for entry in entries:
        code = entry["code"]
        if code == "worker.ready":
            backends[entry["pid"]] = entry["backend"]
        elif code == "pool.page":
            pid = entry["pid"]
            seen[pid] = seen.get(pid, 0) + 1
            peaks[pid] = max(peaks.get(pid, 0), entry["rss_mb"])
            if seen[pid] > WARM_UP_PAGES:
                seconds.append(entry["seconds"])
    pages = sum(seen.values())
    for start in range(0, len(seconds) - WINDOW + 1, WINDOW):
        windows.append(round(sum(seconds[start : start + WINDOW]) / WINDOW, 3))
    result = {
        "pages": pages,
        "recycles": sum(1 for e in entries if e["code"] == "worker.recycled"),
        "workers": len(seen),
        "seconds_per_page_by_window_of_100": windows,
        "peak_rss_mb_by_worker": {str(pid): mb for pid, mb in peaks.items()},
    }
    over = [pid for pid, mb in peaks.items() if mb > RSS_LIMIT_MB[backends.get(pid, "openvino")]]
    enough = pages >= MIN_PAGES and len(seconds) >= 2 * WINDOW
    result["slowdown"] = round(sum(seconds[-WINDOW:]) / sum(seconds[:WINDOW]), 3) if enough else None
    result["passed"] = bool(
        enough and result["recycles"] >= MIN_RECYCLES and result["slowdown"] <= MAX_SLOWDOWN and not over
    )
    result["workers_over_memory_limit"] = over
    return result


def main(log_path: str, out_path: str | None = None) -> dict:
    entries = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = evaluate(entries)
    target = Path(out_path) if out_path else Path(__file__).resolve().parent / "results" / "endurance.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=1) + "\n")
    return result


if __name__ == "__main__":
    print(main(*sys.argv[1:3]))
