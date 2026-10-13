"""R10, the crop round trip: the first recogniser, shown the crop of a unit, gives back the first reading's text of the
unit's words. A crop that does not hold its words is a bug of the cutting, so the share is a floor on the quality of
the crops (at least 95 % of the zones on the pages measured; the strict share is lower than that on units of doubtful
words, because the second look is allowed to read them differently, and the share that matters for the cutting is
`context_share`, the unit's words that were not doubtful).

    python -m bench.crop_roundtrip --set train --data DIR --manifest M.json

The pages are those `run_corpus --second-reading` has read (their `<n>.crops.npz` is beside the JSON). The result goes
to `bench/results/crop-roundtrip.json` with the pages used and the units that failed (a name written after a courtesy
title is replaced by `[name removed]`).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from bench.run_corpus import pages_of
from bench.run_pass2 import study_of
from extract.engine.cache import read_crops_file, reading_from_dict
from extract.engine.crops import decode
from extract.engine.readers import ReaderSet
from extract.engine.reread import align

MIN_SHARE = 0.95
# The corpus is public administration documents, and some of their lines hold the name of a person after a courtesy
# title. A failure is kept with the shape of the error, never with that name.
NAME_AFTER_TITLE = re.compile(
    r"\b(D\.|D[aª]\.|Dña\.|Sra?\.|Don|Doña)\s+(?:[A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü'’-]+\s+)+[A-ZÁÉÍÓÚÑÜ][A-Za-zÁÉÍÓÚÑÜáéíóúñü'’-]+"
)


def squeeze(text: str) -> str:
    return " ".join(text.split())


def redact_names(text: str) -> str:
    """The text with each name written after a courtesy title (`Dª.`, `D.`, `Sr.`, `Don`...) replaced by
    `[name removed]`. A name of one word is left: it identifies nobody."""
    return NAME_AFTER_TITLE.sub(lambda match: f"{match.group(1)} [name removed]", text)


def measure(pages, readers: ReaderSet, root: Path | None = None) -> dict:
    out = {"line": [0, 0], "zone": [0, 0]}
    context = {"line": [0, 0], "zone": [0, 0]}
    failures = []
    used = 0
    for path in pages:
        ocr = json.loads(path.read_text(encoding="utf-8")).get("ocr")
        if not ocr or "reading" not in ocr or "|u:" not in ocr.get("engine", "") or ocr["reading"]["status"] != "ocr":
            continue
        config = study_of(ocr["engine"])
        crops = read_crops_file(path.with_suffix(".crops.npz"), config.key())
        if crops is None:
            continue
        page = reading_from_dict(ocr["reading"])
        used += 1
        for kind in ("line", "zone"):
            units = [u for u in config.units(page.words, kind) if f"{kind}{u.rank}" in crops]
            texts = readers.read_many("small", [decode(crops[f"{kind}{u.rank}"]) for u in units])
            for unit, (text, _) in zip(units, texts, strict=True):
                expected = " ".join(page.words[i].text for i in unit.indices)
                out[kind][1] += 1
                first = [page.words[i].text for i in unit.indices]
                held = [k for k, i in enumerate(unit.indices) if i not in unit.suspects]
                if held:
                    context[kind][1] += 1
                    aligned = align(first, text.split())
                    context[kind][0] += aligned is not None and all(aligned[k].text == first[k] for k in held)
                if squeeze(text) == expected:
                    out[kind][0] += 1
                else:
                    failures.append(
                        {
                            "page": path.name if root is None else str(path.relative_to(root)),
                            "kind": kind,
                            "rank": unit.rank,
                            "first": redact_names(expected),
                            "crop": redact_names(text),
                        }
                    )
    return {
        "pages": used,
        **{
            kind: {
                "units": n,
                "round_trip": ok,
                "share": round(ok / n, 4) if n else None,
                "context_units": context[kind][1],
                "context_round_trip": context[kind][0],
                "context_share": round(context[kind][0] / context[kind][1], 4) if context[kind][1] else None,
            }
            for kind, (ok, n) in out.items()
        },
        "note": "a unit holds words the recogniser doubted, so a second look at them may legitimately read them "
        "differently: `share` is the strict round trip of the whole unit, `context_share` the one of its words "
        "that were not doubtful",
        "min_share": MIN_SHARE,
        "failures": failures[:60],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--set", action="append", required=True)
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--manifest", type=Path, default=Path("bench/corpus/split-v5.json"))
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--result", type=Path, default=Path("bench/results/crop-roundtrip.json"))
    args = parser.parse_args()
    pages = [p for _, p in pages_of(args.manifest, args.set, args.data)][: args.limit]
    result = measure(pages, ReaderSet(("small",), threads=args.threads, backend="openvino"), args.data)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "failures"}))


if __name__ == "__main__":
    main()
