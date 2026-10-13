"""Turn the synthetic invoices into labelled pages: the same damage bands and truth as the catalogue pages.

    python -m bench.corpus.invoice_pages [--data DIR]

Reads `<data>/invoices/<split>/*.pdf`, writes `<data>/pages/<split>/<invoice id>/<n>.png` and `.json` through
`catalog.build_pdf_page`, and lists them in `<data>/invoices/pages.json`, the manifest `bench.run_corpus` takes with
`--manifest`. A page already built is built again with the same seed, so the run can be repeated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pypdfium2 as pdfium

from bench.corpus import catalog, survey5

SPLITS = ("train", "val", "test")


def pages_in(pdf: Path) -> int:
    document = pdfium.PdfDocument(str(pdf))
    try:
        return len(document)
    finally:
        document.close()


def build(data: Path, build_page=catalog.build_pdf_page, splits=SPLITS, **options) -> list[dict]:
    entries = []
    for split in splits:
        for pdf in sorted((data / "invoices" / split).glob("*.pdf")):
            for number in range(1, pages_in(pdf) + 1):
                band = build_page(pdf, number, pdf.stem, split, data, **options)
                entries.append({"set": split, "doc_id": pdf.stem, "page": number, "band": band})
    return entries


def write_manifest(data: Path, entries: list[dict], name: str = "pages.json") -> Path:
    path = data / "invoices" / name
    partial = path.with_name(path.name + ".part")
    partial.write_text(json.dumps({"pages": entries}, indent=1), encoding="utf-8")
    os.replace(partial, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--v5-test", action="store_true", help="the 60 sealed invoices of v5, into `pages-t5.json`")
    args = parser.parse_args()
    if args.v5_test:
        entries = build(args.data, splits=("test5",), seed=survey5.SEED)
        path = write_manifest(args.data, entries, "pages-t5.json")
    else:
        entries = build(args.data)
        path = write_manifest(args.data, entries)
    print(f"{len(entries)} invoice pages listed in {path}")


if __name__ == "__main__":
    main()
