# Reproducing the numbers

The figures in the main README come from `bench/results/`, written by the scripts here. What each one needs:

**Nothing downloaded** (the tests and the synthetic invoices; a few minutes):

- `python -m pytest` runs the unit tests, the injection harness test and the checks of the result files.
- `python -m bench.inject --trials 20000 --seed 1` writes `bench/results/injection.json`: wrong digits put into synthetic
  invoice numbers, no OCR.
  Every name and identifier in the generated invoices is invented and cannot exist (tax numbers one character short, IBANs
  written `ESXX`); the harness completes them in memory to numbers that pass their check digits, so the number rules are
  still tried on every digit, and nothing valid is ever written to disk.

**The corpus** (420 documents of public administrations: 275 for development and the 145 of the Spanish sealed v5 test, listed in `SOURCES.csv` with URL, sha256 and the kind of publisher (the `licence` column); the 100 English documents of the sealed test are in `SOURCES_EN.csv`). No document is stored in this repository, only its address and hash; the reuse of each one follows the terms of its publisher:

```
python -m bench.fetch_corpus --dry-run      # the plan: documents per site and how long it takes
python -m bench.fetch_corpus --limit 3      # a first try
python -m bench.fetch_corpus                # all of them
```

The download takes about 50 minutes: 8 to 12 seconds between two requests to the same site, 150 an hour and 800 a day
at most, resumable. A document that changed since the corpus was made is reported and left out (exit status 1), so a
changed source shows in the numbers as a missing document and not as a silent difference. The files go to
`bench/data/catalog/originals/` with a `catalog.db` beside them; point the builder at them:

```
export EXTRACT_BENCH_DATA=$PWD/bench/data
export EXTRACT_CATALOG=$EXTRACT_BENCH_DATA/catalog/originals
python -m bench.corpus.catalog render            # labelled pages, from the split in bench/corpus/split.json
python -m bench.run_corpus --set train --set val # the OCR of those pages: about 3 hours, 1.5 GB per process
python -m bench.guard.run_validation             # trains and calibrates the word checker
python -m bench.measure_extraction               # invoice fields and number rules
```

`bench/run_corpus.py` refuses to start without a memory limit on the process; on Linux with systemd that is
`systemd-run --user --scope -p MemoryMax=3G python -m bench.run_corpus ...`.

Not reproducible from here: the reading of the sealed test split (`bench/read_test.py`) was done with the model and
the code frozen, and its result is `bench/results/test_reads.json`; running it again is not part of reproducing the
validation numbers.

## Names in the result files

- `T5` and `T5-EN` are the sealed Spanish and English test sets of v5; `D` is the development pool. They were read with
  the models and the code frozen (the second read of each only added report tables; see `docs/MEASURED.md`): `bench/results/test_reads_v5.json` and `test_reads_v5en.json` register every read
  (a second read needs a written reason, which is stored in the line), and `models/FROZEN-v5.sha256` lists the sha256
  of every file that ran in that read. The files of this repository were edited afterwards (wording, documentation and
  the public export), so `bench.read_test_v5` verification reports them as changed; the figures in `bench/results/` are
  the ones of that read.
- `R6` and `R7` in `pilot-cells.json` and the pilot code are the two budgets of the second reading: R6 is the time
  budget (mean and p95 seconds a page) and R7 the memory budget of the reader process.
- `L`, `Z`, `LD` and `ZC` are the cells of the second reading: line or zone units read by the cheap readers (L, ZC) or by
  the deep one (Z, LD).
