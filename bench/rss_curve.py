"""Resident memory of an OCR worker as it reads pages, and how long each worker took to load, from a run's job.log.

    python3 -m bench.rss_curve JOB_LOG [OUT_JSON]

For each worker, the mean and the maximum of its resident memory over every ten pages it read, its backend, and the
seconds its engine took to load (the first worker loads cold, the ones that replace a recycled worker load with the
models already in the disk cache). Writes `bench/results/rss-curve.json` unless another path is given.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

STEP = 10


def curve(entries: list[dict]) -> dict:
    workers: dict[int, dict] = {}
    for entry in entries:
        code = entry["code"]
        if code == "worker.ready":
            worker = workers.setdefault(entry["pid"], {"samples": []})
            worker.update(backend=entry["backend"], load_seconds=entry["load_seconds"], order=len(workers) + 1)
        elif code == "pool.page":
            workers.setdefault(entry["pid"], {"samples": []})["samples"].append(entry["rss_mb"])
        elif code == "worker.recycled":
            workers.setdefault(entry["pid"], {"samples": []}).update(reason=entry.get("reason"))
    result = []
    for pid, worker in sorted(workers.items(), key=lambda item: item[1].get("order", 0)):
        samples = worker["samples"]
        tracts = [samples[start : start + STEP] for start in range(0, len(samples), STEP)]
        result.append(
            {
                "pid": pid,
                "backend": worker.get("backend"),
                "load_seconds": worker.get("load_seconds"),
                "pages": len(samples),
                "left_because": worker.get("reason"),
                "by_ten_pages": [
                    {
                        "pages": f"{i * STEP + 1}-{i * STEP + len(tract)}",
                        "mean_mb": round(sum(tract) / len(tract)),
                        "max_mb": max(tract),
                    }
                    for i, tract in enumerate(tracts)
                ],
            }
        )
    return {"step_pages": STEP, "workers": result}


def main(log_path: str, out_path: str | None = None) -> dict:
    entries = [json.loads(line) for line in Path(log_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    result = curve(entries)
    target = Path(out_path) if out_path else Path(__file__).resolve().parent / "results" / "rss-curve.json"
    target.parent.mkdir(exist_ok=True)
    target.write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    main(*sys.argv[1:3])
