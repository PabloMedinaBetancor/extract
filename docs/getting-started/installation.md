# Installation

This page installs Extract with one command, says what that command does and where it puts things, and covers a machine
with no network.

## What you need

- 64-bit Linux with Python 3.10 or newer (`python3 --version` tells you).
- About 1.5 GB of free disk and a network connection, for the installation only. Reading never uses the network.
- A few minutes: the installer downloads about 400 MB, most of it OpenVINO, SciPy and OpenCV. On a two-core laptop it
  took about 4 minutes.

| Platform | Status |
|---|---|
| Linux, Python 3.10 | The whole test suite has been run. |
| Linux, Python 3.12 | A clean installation, the offline bundle (installed with the network cut) and a read of the sample documents work and give the same words, boxes and statuses as on 3.10. Its test suite has not been run. |
| Windows | A lock file exists and the installer knows the Windows paths (`py scripts\install.py`, then `.venv\Scripts\python -m extract.ui`). Nobody has installed or run Extract on Windows: test it on your own machine before production work. |
| macOS | Not supported: there is no lock file. |

## Install

```console
$ git clone https://github.com/PabloMedinaBetancor/extract extract
$ cd extract
$ python3 scripts/install.py
```

The installer ends by reading a sample page and printing:

```text
OK: 100% of the printed words read

Installed. Start the interface with: .../extract/.venv/bin/python -m extract.ui
```

Start the interface with `.venv/bin/python -m extract.ui`, or go on to the [quickstart](quickstart.md).

## What the installer does

`scripts/install.py` uses only the standard library, so it runs before anything is installed. In order, it:

1. creates a private environment in `.venv` (on a system without `python3-venv`, it fetches pip from a pinned wheel);
2. installs every dependency from `requirements/<platform>.txt`, each wheel pinned by its sha256 and installed with
   `pip install --require-hashes`, so a file that differs from the tested one is refused;
3. installs the Extract package into the environment;
4. downloads the OCR models, 31 MB, and checks each against the sha256 in `models/models.json`;
5. builds the word lists the guard uses, about 50 MB in about 30 seconds, from the `wordfreq` package (they are not stored
   in the repository);
6. reads a sample PDF, to prove the installation works.

Its options are `--venv DIR` (the environment somewhere else), `--offline DIR` (install from an offline bundle, below) and
`--ci` (also run `pip check` and list what was installed).

## Where things go

| What | Where |
|---|---|
| The program and its dependencies | `.venv` in the repository folder |
| OCR models | The folders of `$EXTRACT_MODELS_DIR` if it is set, otherwise the `extract/models` folder of your user data directory (`$XDG_DATA_HOME`, or `.local/share` in your home folder) |
| Word lists | `src/extract/guard/data/` in the repository |
| The work folder: page cache, `job.log` and `corrections.jsonl` | For the library, `extract` inside `$XDG_CACHE_HOME` if that is set, otherwise inside `.cache` in your home folder. The interface always uses `.cache/extract` in your home folder. `Options.work_dir` and `--work-dir` change it. |

Extract looks for a model in each folder of `$EXTRACT_MODELS_DIR` (separated like `PATH`), then in the user data folder,
then in the `rapidocr/models` folder of the installed package. Deleting the work folder removes every trace of what you
read; [Privacy](../concepts/privacy.md) says what is written where.

## The models of the thorough mode

The default installation does not fetch the reader models of the thorough
[second reading](../concepts/corrections.md#the-second-reading). Fetch them once:

```console
$ python3 scripts/fetch_models.py --all
```

That downloads the three models the default installation leaves out, 169 MB in all. The thorough mode loads two of them
(84.5 MB); the third was a candidate that lost the comparison and is never loaded. Without them, asking for the thorough
mode stops with an `EngineLoadError` that names the missing file. `fetch_models.py` resumes an interrupted download, skips
files already present, and deletes a file and fails if its hash is wrong; its flags are on the
[command line page](../reference/command-line.md#scriptsfetch_modelspy).

## If the word lists are missing

Without the word lists, `extract.read` stops with an `EngineLoadError` that says so. Build them again with the
interpreter of the environment, because `wordfreq` is installed only there:

```console
$ .venv/bin/python scripts/fetch_models.py --lists
```

Under the system Python the script stops with `wordfreq is not installed`. With no environment yet,
`python3 scripts/fetch_models.py --lists --from-release URL.tar.gz` unpacks ready-made lists instead, with the standard
library only; the `lists.tar.gz` of an offline bundle is such an archive, and a `file://` address works. Their data comes
from `wordfreq` and is CC BY-SA 4.0; [Licence and credits](../licence.md) has the full credit.

## Without a network

For a computer with no network, build an offline bundle on a connected computer where Extract is installed, copy it
across and install from it. The bundle holds everything the installation downloads: the source, every dependency as a
wheel, the OCR models and the word lists.

```console
$ .venv/bin/python scripts/make_offline_bundle.py
```

It writes `dist/extract-offline-<version>-<platform>.zip`, about 390 MB, for the platform of the computer that builds it;
`--platform` names another: `linux-x86_64-py310`, `linux-x86_64-py312`, `win-amd64-py310` or `win-amd64-py312`. The target
needs the Python version of its bundle. The Windows bundles are built, and nobody has installed from one on Windows.

On the target, unzip it, open a terminal in the `extract-offline-...` folder it makes, and run:

```console
$ python3 scripts/install.py --offline offline
```

The installer installs every wheel with `--no-index` and `--require-hashes`, so pip cannot reach out and a wheel that is
not the locked one is refused, copies the models and the lists from the bundle, and ends as online, with
`OK: 100% of the printed words read`. The bundle holds the default models only: for the thorough mode, run
`python3 scripts/fetch_models.py --all --dir extra-models` on the connected computer, carry the folder across, and run
`python3 scripts/fetch_models.py --all --source extra-models` on the target.

## Check an installation

```console
$ python3 scripts/verify_install.py
```

`verify_install.py` repeats the whole installation in a temporary folder with nothing cached and reads a sample, which is
what continuous integration runs. `.venv/bin/python scripts/verify_install.py --here` only reads the sample with the
installed Python, and works with the network cable out.

## Run the tests

```console
$ sudo apt install poppler-utils libreoffice-calc fonts-dejavu-core fonts-freefont-ttf fonts-liberation2 fonts-ubuntu
$ .venv/bin/python -m pytest
```

Extract does not need any of those packages. The tests do: they check the searchable PDFs against Poppler's `pdftotext`,
the corpus builder in `bench/` calls `pdffonts` and `pdfimages`, one test opens an exported workbook in LibreOffice, and the
synthetic invoices are drawn with fixed fonts so that every machine makes the same ones. A test that cannot find one of
them fails and names it. The browser tests are skipped when there is no Google Chrome. Continuous integration installs the
same packages on Ubuntu 24.04 before it runs the suite.

## Read the documentation as a site

The pages in `docs/` read well on their own, and they also build into a site you can browse on your machine. The site
tools are not part of Extract, so they go in their own environment, outside the repository:

```console
$ python3 -m venv ~/docs-venv
$ ~/docs-venv/bin/python -m pip install --require-hashes -r requirements/docs.txt
$ DOCS_PYTHON=~/docs-venv/bin/python .venv/bin/python scripts/docs.py serve
```

The last command regenerates the pages that are built from the code, then serves the site at `http://127.0.0.1:8000`.

## Next

The [quickstart](quickstart.md) takes you from here to a first result. [Troubleshooting](../troubleshooting.md) covers an
installer that stops with a network error or a missing model.
