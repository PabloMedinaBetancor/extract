"""The English catalogue of the v5 corpus (the OCR has to be good with English text too).

    python -m bench.corpus.discover_en candidates          # writes bench/corpus/en_candidates.csv (seeded, no request)
    python -m bench.corpus.discover_en fetch [--limit N]   # downloads them slowly, writes SOURCES_EN.csv
    python -m bench.corpus.discover_en rows                # rewrites SOURCES_EN.csv from the files already here

Only documents of public administrations with a free licence enter: the US federal government (public domain: the
Government Publishing Office's `govinfo.gov`, the IRS, NIST) and the United Kingdom (Open Government Licence:
`legislation.gov.uk`). They are PDFs with a text layer, whose truth is the layer, as for the Spanish catalogue. The
candidates are drawn from patterns that name the document (Congressional report `CRPT-118hrpt100`, public law
`PLAW-118publ10`, Act `ukpga/2010/15`), by a seeded draw, so the list is reproducible; a name that does not exist is a
404 and is left out. No request goes to a page of the site other than the document.

The limits are those of `bench.fetch_corpus`: 8 to 12 seconds between two requests to a site, 150 an hour and 800 a day
to each, and a site that answers 403 or 429 is left alone. The files go to
`<data>/en/catalog/originals/c<sha256[:10]>.pdf` with a `catalog.db` beside them, and
`bench/SOURCES_EN.csv` lists them (doc_id, source_url, sha256, licence, language, unit_id, kind) so that
`python -m bench.fetch_corpus --sources bench/SOURCES_EN.csv --out <data>/en` reproduces the folder, checking each hash.
There are no translations in this catalogue (US and UK English documents only), so a document is its own unit
(`unit_id` = `doc_id`); a bilingual PDF would be left out.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

from bench import fetch_corpus
from bench.corpus.split import stable_rng

SEED = 20261007
COLUMNS = ["url", "kind"]
SOURCES_EN = Path("bench/SOURCES_EN.csv")
SOURCES_COLUMNS = [
    "doc_id",
    "source",
    "source_url",
    "sha256",
    "licence",
    "split",
    "catalog_id",
    "language",
    "unit_id",
    "kind",
]
CANDIDATES = Path("bench/corpus/en_candidates.csv")
LICENCES = {
    "govinfo.gov": "US federal government work (public domain)",
    "irs.gov": "US federal government work (public domain)",
    "nist.gov": "US federal government work (public domain)",
    "legislation.gov.uk": "Open Government Licence v3.0",
}

IRS_PUBLICATIONS = (
    "p15 p17 p15a p15b p463 p501 p502 p503 p504 p505 p509 p510 p521 p523 p524 p525 p526 p527 p529 p530 p535 p536 p537 "
    "p538 p541 p542 p544 p547 p550 p551 p552 p554 p555 p556 p557 p559 p561 p570 p575 p583 p584 p587 p590a p590b p594 "
    "p596 p598 p925 p926 p929 p936 p946 p970 p972 p1544 p3 p4 p54 p78 p80 p225 p334"
).split()
IRS_FORMS = (
    "f1040 i1040gi f1040sc i1040sc f1040se i1040se f1040sd i1040sd f1040sse f1040sb f1040sa f1040nr f1040x i1040x f1065 "
    "i1065 f1120 i1120 f1120s i1120s f941 i941 f940 i940 f943 fw2 iw2 fw4 fw9 iw9 f1099msc i1099msc f1099int f1099div "
    "f1099nec i1099nec f4562 i4562 f8949 i8949 f8829 i8829 f2441 f8863 i8863 f8962 f990 i990 f4868 f8822 f2848 f8821 "
    "f1023 i1023 f5500 f706"
).split()
NIST = (
    "SpecialPublications/NIST.SP.800-53r5 SpecialPublications/NIST.SP.800-37r2 SpecialPublications/NIST.SP.800-61r2 "
    "SpecialPublications/NIST.SP.800-63-3 SpecialPublications/NIST.SP.800-63b SpecialPublications/NIST.SP.800-171r2 "
    "SpecialPublications/NIST.SP.800-30r1 SpecialPublications/NIST.SP.800-39 SpecialPublications/NIST.SP.800-40r3 "
    "SpecialPublications/NIST.SP.800-34r1 SpecialPublications/NIST.SP.800-12r1 SpecialPublications/NIST.SP.800-18r1 "
    "SpecialPublications/NIST.SP.800-47r1 SpecialPublications/NIST.SP.800-57pt1r5 SpecialPublications/NIST.SP.800-86 "
    "SpecialPublications/NIST.SP.800-88r1 SpecialPublications/NIST.SP.800-115 SpecialPublications/NIST.SP.800-137 "
    "SpecialPublications/NIST.SP.800-160v1r1 SpecialPublications/NIST.SP.800-218 SpecialPublications/NIST.SP.800-207 "
    "SpecialPublications/NIST.SP.800-190 SpecialPublications/NIST.SP.800-145 SpecialPublications/NIST.SP.800-146 "
    "SpecialPublications/NIST.SP.800-122 SpecialPublications/NIST.SP.800-128 SpecialPublications/NIST.SP.800-92 "
    "SpecialPublications/NIST.SP.800-94 SpecialPublications/NIST.SP.800-82r3 SpecialPublications/NIST.SP.800-83r1 "
    "SpecialPublications/NIST.SP.800-95 SpecialPublications/NIST.SP.800-100 SpecialPublications/NIST.SP.800-123 "
    "SpecialPublications/NIST.SP.800-125 SpecialPublications/NIST.SP.800-144 SpecialPublications/NIST.SP.800-175Br1 "
    "CSWP/NIST.CSWP.04162018 FIPS/NIST.FIPS.140-3 FIPS/NIST.FIPS.197 FIPS/NIST.FIPS.199 FIPS/NIST.FIPS.200 "
    "FIPS/NIST.FIPS.202 FIPS/NIST.FIPS.186-5 NISTIR/NIST.IR.8259 NISTIR/NIST.IR.8286"
).split()


def candidate_urls(seed: int = SEED) -> list[tuple[str, str]]:
    """(url, kind) of every candidate, in the order they are fetched. Pure function of the seed."""
    rng = stable_rng(seed, "en-candidates")
    gov = "https://www.govinfo.gov/content/pkg/{p}/pdf/{p}.pdf"

    def draw(label: str, count: int, congresses: tuple[int, ...], top: int, template: str, kind: str):
        pool = [(c, n) for c in congresses for n in range(1, top + 1)]
        picks = rng.permutation(len(pool))[:count]
        return [(gov.format(p=template.format(c=pool[i][0], n=pool[i][1])), kind) for i in picks]

    out: list[tuple[str, str]] = []
    out += draw("house reports", 70, (116, 117, 118), 500, "CRPT-{c}hrpt{n}", "report")
    out += draw("senate reports", 30, (116, 117, 118), 200, "CRPT-{c}srpt{n}", "report")
    out += draw("house documents", 30, (116, 117, 118), 100, "CDOC-{c}hdoc{n}", "annual report")
    out += draw("public laws", 60, (116, 117, 118), 250, "PLAW-{c}publ{n}", "legislation")
    acts = [(y, n) for y in range(2005, 2024) for n in range(1, 29)]
    for i in rng.permutation(len(acts))[:45]:
        year, number = acts[i]
        out.append(
            (
                f"https://www.legislation.gov.uk/ukpga/{year}/{number}/pdfs/ukpga_{year}{number:04d}_en.pdf",
                "legislation",
            )
        )
    instruments = [(y, n) for y in range(2015, 2023) for n in range(1, 900)]
    for i in rng.permutation(len(instruments))[:25]:
        year, number = instruments[i]
        out.append(
            (f"https://www.legislation.gov.uk/uksi/{year}/{number}/pdfs/uksi_{year}{number:04d}_en.pdf", "legislation")
        )
    out += [(f"https://www.irs.gov/pub/irs-pdf/{name}.pdf", "report") for name in IRS_PUBLICATIONS]
    out += [(f"https://www.irs.gov/pub/irs-pdf/{name}.pdf", "form") for name in IRS_FORMS]
    out += [(f"https://nvlpubs.nist.gov/nistpubs/{name}.pdf", "report") for name in NIST]
    return out


def licence_of(url: str) -> str | None:
    host = (urlparse(url).hostname or "").lower()
    for domain, name in LICENCES.items():
        if host == domain or host.endswith("." + domain):
            return name
    return None


def write_candidates(path: Path = CANDIDATES) -> int:
    rows = candidate_urls()
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)
    return len(rows)


def read_candidates(path: Path = CANDIDATES) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_name(url: str) -> str:
    return "u" + hashlib.sha1(url.encode()).hexdigest()[:12]


class Discovery(fetch_corpus.Fetcher):
    """The `fetch_corpus` pace over documents whose hash is not known yet: each file is kept under the name of the first
    10 characters of its SHA-256 (the catalogue's own naming) and a row says where it came from."""

    def __init__(self, out: Path, **kwargs) -> None:
        super().__init__(out, **kwargs)
        self.tried: dict[str, str] = {}
        self.log = out / "discover-en.log"

    def pending(self, rows: list[dict]) -> list[dict]:
        done = self.done_urls()
        return [r for r in rows if r["source_url"] not in done]

    def done_urls(self) -> set[str]:
        if not self.log.is_file():
            return set()
        return {line.split("\t")[0] for line in self.log.read_text(encoding="utf-8").splitlines() if line}

    def remember(self, url: str, outcome: str) -> None:
        self.log.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log, "a", encoding="utf-8") as handle:
            handle.write(f"{url}\t{outcome}\n")

    def fetch_one(self, row: dict) -> str:
        url = row["source_url"]
        site = fetch_corpus.site_of(url)
        self.record(site)
        request = urllib.request.Request(url, headers={"User-Agent": "extract-bench-fetch"})
        try:
            with urllib.request.urlopen(request, timeout=90) as response:
                body = response.read()
        except urllib.error.HTTPError as error:
            print(f"HTTP {error.code} from {site}: {url}", file=sys.stderr, flush=True)
            self.remember(url, f"http-{error.code}")
            return "blocked" if error.code in fetch_corpus.STOP_STATUSES else "failed"
        except (urllib.error.URLError, TimeoutError) as error:
            print(f"{url}: {error}", file=sys.stderr, flush=True)
            return "failed"
        if not body.startswith(b"%PDF"):
            self.remember(url, "not-pdf")
            return "failed"
        digest = hashlib.sha256(body).hexdigest()
        target = self.out / fetch_corpus.ORIGINALS / f"c{digest[:10]}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(target.name + ".part")
        partial.write_bytes(body)
        partial.replace(target)
        self.remember(url, f"ok\tc{digest[:10]}\t{digest}\t{row['kind']}")
        return "ok"


def rows_from_log(out: Path) -> list[dict]:
    """One row per kept document, in the order they were fetched; a file two URLs gave is listed once."""
    log = out / "discover-en.log"
    rows, seen = [], set()
    for line in log.read_text(encoding="utf-8").splitlines() if log.is_file() else []:
        parts = line.split("\t")
        if len(parts) != 5 or parts[1] != "ok" or parts[3] in seen:
            continue
        url, _, doc_id, digest, kind = parts
        if not (out / fetch_corpus.ORIGINALS / f"{doc_id}.pdf").is_file():
            continue
        seen.add(digest)
        rows.append(
            {
                "doc_id": doc_id,
                "source": "catalog-en",
                "source_url": url,
                "sha256": digest,
                "licence": licence_of(url) or "",
                "split": "",
                "catalog_id": str(900000 + len(rows)),
                "language": "en",
                "unit_id": doc_id,
                "kind": kind,
            }
        )
    return rows


def read_sources_file(path: Path = SOURCES_EN) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_sources(rows: list[dict], path: Path = SOURCES_EN) -> int:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SOURCES_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("candidates", "fetch", "rows"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")) / "en")
    args = parser.parse_args()
    if args.what == "candidates":
        print(f"{write_candidates()} candidates in {CANDIDATES}")
        return
    if args.what == "fetch":
        rows = [{"doc_id": file_name(r["url"]), "source_url": r["url"], "kind": r["kind"]} for r in read_candidates()]
        fetcher = Discovery(args.out)
        report = fetcher.run(rows, args.limit)
        print(f"{report['fetched']} fetched, {len(report['failed'])} failed or not there", flush=True)
    rows = rows_from_log(args.out)
    print(
        f"{write_sources(rows)} documents in {SOURCES_EN}; {fetch_corpus.write_catalogue(args.out, rows)} in the catalogue"
    )


if __name__ == "__main__":
    main()
