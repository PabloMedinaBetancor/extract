"""End-to-end check of the two final models with the product's own code : the default model loads on the
default path with no warning and no second readings in its block; the thorough model loads with
`Guard.load(path, thorough=True)` and through `api._thorough_guard()` (which refuses a model whose `meta.fix.readings` is not
the cell of the mode); and each guard judges real pages of D, the thorough one with the readings of L and Z put together
(spec of the mode), the default one with none, with no warning.

    python3 -s -m bench.guard.model_e2e --default models/guard-new-default.model --thorough models/guard-new-thorough.model
"""

import argparse
import json
import os
from pathlib import Path

from bench.guard import dataset, pairs
from extract import api, second_reading
from extract.engine.cache import reading_from_dict
from extract.guard import Guard, readings


def check_models(default: Path, thorough: Path) -> dict:
    out = {}
    plain = Guard.load(default)
    block = plain.model.fix
    out["default_readings"] = block.get("readings") if block else "no fix block"
    if out["default_readings"] is not None:
        raise SystemExit(f"the default model must be trained without readings: {out['default_readings']}")
    deep = Guard.load(thorough, thorough=True)
    trained = readings.ReadingSpec.from_meta(deep.model.fix.get("readings"))
    out["thorough_readings"] = deep.model.fix.get("readings")
    if trained is None or not trained.same_cell(readings.product_spec(True)):
        raise SystemExit(f"the thorough model is not trained on the cell of the mode: {out['thorough_readings']}")
    os.environ[second_reading.THOROUGH_MODEL_ENV] = str(thorough)
    api._thorough_guard()  # raises EngineLoadError if the product refuses the file
    out["api_thorough_guard"] = "loaded"
    out["ids"] = {"default": plain.model.model_id(), "thorough": deep.model.model_id()}
    return out, plain, deep


def judge_pages(plain, deep, data: Path, per_set: int = 4) -> dict:
    """Both guards on the same real pages: the default on the first reading, the thorough on the first reading with the
    readings of L and Z (as the mode leaves them). Counts the words with a fix and the warnings."""
    spec = readings.product_spec(True)
    rows = []
    for set_name in ("val", "dev5en"):
        taken = 0
        for path, record in dataset.iter_page_records(data, set_name):
            reading = reading_from_dict(record["ocr"]["reading"])
            if reading.status != "ocr" or not reading.words:
                continue
            second = pairs.with_cells(reading, path, ["L", "Z"], spec)
            if not any(w.readings for w in second.words):
                continue
            a, b = plain.score_page(reading), deep.score_page(second)
            rows.append(
                {
                    "set": set_name,
                    "page": f"{record['doc_id']}/{record['page']}",
                    "words": len(reading.words),
                    "words_with_readings": sum(1 for w in second.words if w.readings),
                    "fixes_default": sum(1 for s in a if getattr(s, "fix", None)),
                    "fixes_thorough": sum(1 for s in b if getattr(s, "fix", None)),
                    "warnings_default": list(plain.last_warnings),
                    "warnings_thorough": list(deep.last_warnings),
                }
            )
            taken += 1
            if taken >= per_set:
                break
    bad = [r for r in rows if r["warnings_default"] or r["warnings_thorough"]]
    return {"pages": rows, "pages_judged": len(rows), "pages_with_warnings": len(bad)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--thorough", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--out", type=Path, default=Path("bench/results/model-e2e.json"))
    args = parser.parse_args(argv)
    checks, plain, deep = check_models(args.default, args.thorough)
    result = {**checks, **judge_pages(plain, deep, args.data)}
    args.out.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "pages"}, default=str))
    return 1 if result["pages_with_warnings"] or not result["pages_judged"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
