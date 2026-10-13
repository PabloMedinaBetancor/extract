# Command line

The commands you run from a terminal: the interface, the scripts that install Extract and fetch its models, and the
scripts that only a person who changes Extract needs. Every flag on this page is one the script accepts, and `--help`
prints a script's own summary and does nothing else, except for the two maintainer scripts named at the end.

Run the commands from the root of the repository. `python3` is the system Python; `.venv/bin/python` is the interpreter
that `scripts/install.py` made, and the only one that has Extract installed, so `python3 -m extract.ui` fails (on Windows
it is `.venv\Scripts\python`, which has not been tested).

## The interface

```console
$ .venv/bin/python -m extract.ui [--work-dir DIR] [--open [JSON ...]] [--no-browser] [--lang {en,es}]
```

Starts the interface on your own machine and opens it in your browser; stop it with `Ctrl+C`. The address is
`http://127.0.0.1:<port>/?t=<token>`: a random port on the loopback interface, and a token that is new on every run. The
server answers only to requests that carry the token and come from the loopback host, and its folder browser shows your
home folder and nothing above it.

| Option | What it does |
|---|---|
| `--work-dir DIR` | The work folder: the page cache, `job.log` and `corrections.jsonl`. The default is `~/.cache/extract`; unlike the library, the interface does not look at `XDG_CACHE_HOME`. |
| `--open JSON [JSON ...]` | Saved readings to open at the start. Each must be a file, and the PDF must be next to the JSON for its pages to show. |
| `--no-browser` | Print the address and do not open a browser. |
| `--lang {en,es}` | The language of the interface. |

An `--open` path that is not a file, or a saved reading that cannot be read, stops the command before the server starts,
with a one-line error and exit status 2. Run the interface only with `-m`: `extract/ui/__main__.py` starts the server as
soon as it is imported. `extract.ui` has no public API: its HTTP routes are the private protocol between the page and the
server and change without notice. To read PDFs from a program, use [`extract.read`](functions.md#read).

## Install and models

`install.py` and `verify_install.py` use only the standard library and run with the system `python3`. `fetch_models.py`
also runs with it, except for `--lists`, which needs the `wordfreq` package and so the `.venv` interpreter.
`make_offline_bundle.py` needs `pip` and the lock tools and runs with `.venv/bin/python`.

### `scripts/install.py`

```console
$ python3 scripts/install.py [--venv DIR] [--offline DIR] [--ci]
```

Installs Extract from a clone with one command: a virtual environment, the packages pinned by hash, the Extract package,
the OCR models (31 MB, each checked against its SHA-256) and the word lists; then it reads a sample PDF and prints the
command that starts the interface. [Installation](../getting-started/installation.md#what-the-installer-does) lists the
steps.

| Option | What it does |
|---|---|
| `--venv DIR` | Where to create the environment. The default is `.venv` at the root of the repository. |
| `--offline DIR` | Install from an unpacked offline bundle. No network is used. |
| `--ci` | Also run `pip check` and list what was installed. |

### `scripts/fetch_models.py`

```console
$ python3 scripts/fetch_models.py [--dir DIR] [--catalogue FILE] [--all] [--only NAME] [--source DIR] [--lists] [--from-release URL]
```

Downloads the OCR models listed in `models/models.json`, checks each against its SHA-256, and skips the ones that are
already there. A download that stops resumes; a file with a wrong hash is deleted and the script fails.

| Option | What it does |
|---|---|
| `--dir DIR` | Where the models go. The default is `$EXTRACT_MODELS_DIR`, then the user data folder. |
| `--catalogue FILE` | A catalogue of models other than `models/models.json`. |
| `--all` | Every model of the catalogue, not only the two default ones. The three extra models are 169.1 MB; the thorough second reading uses two of them (84.5 MB). |
| `--only NAME` | One model by name. Repeat the option for several. |
| `--source DIR` | Copy the files from this folder instead of downloading them. |
| `--lists` | Build the word lists instead of fetching models. Run it as `.venv/bin/python scripts/fetch_models.py --lists`: `wordfreq`, which builds the lists, is installed only in that environment. |
| `--from-release URL` | With `--lists`: download the lists from a `.tar.gz` instead of building them. It needs only the standard library. |

### `scripts/verify_install.py`

```console
$ python3 scripts/verify_install.py [--here]
```

Copies the tree to a temporary folder, installs from scratch with a cold `pip` cache, downloads the models again, and
reads a sample.

| Option | What it does |
|---|---|
| `--here` | Skip the install: only read the sample with the Python that runs the script, so run it as `.venv/bin/python scripts/verify_install.py --here`. |

### `scripts/make_offline_bundle.py`

```console
$ .venv/bin/python scripts/make_offline_bundle.py [--platform {linux-x86_64-py310,linux-x86_64-py312,win-amd64-py310,win-amd64-py312}]
```

Builds `dist/extract-offline-<version>-<platform>.zip` (about 390 MB): the source tree, every wheel of the lock file, the
OCR models and the word lists, for the machine that runs the script unless `--platform` names another. On a machine with
no network, unzip it and run `python3 scripts/install.py --offline offline`. The Windows bundles are built, but nobody has
installed from one on Windows yet.

## Environment variables

Extract reads these. None is needed for a normal install.

| Variable | What it does |
|---|---|
| `EXTRACT_MODELS_DIR` | Folders to look for the OCR models in, separated like `PATH`. They are searched first, then the user data folder, then the `rapidocr` package's own `models` folder. |
| `XDG_DATA_HOME` | Where the user data folder is on Linux: models are in `$XDG_DATA_HOME/extract/models`, else `~/.local/share/extract/models`. On macOS it is `~/Library/Application Support/extract/models`, on Windows `%LOCALAPPDATA%\extract\models`. |
| `XDG_CACHE_HOME` | The default work folder of the library is `$XDG_CACHE_HOME/extract`, else `~/.cache/extract` (see [`Options.work_dir`](options.md#options)). The interface ignores it. |
| `EXTRACT_GUARD_MODEL` | A guard model file to use instead of the one that ships with the package. |
| `EXTRACT_GUARD_MODEL_THOROUGH` | The same for the guard model of the thorough second reading. |

## Scripts for maintainers

These are for a person who changes Extract or rebuilds what ships with it. A user does not need them.

| Command | What it does |
|---|---|
| `python3 scripts/lock.py [PLATFORM ...]` | Writes `requirements/<platform>.txt`, every package pinned with SHA-256 hashes; with no platform, all four. |
| `python3 scripts/third_party.py` | Writes `THIRD-PARTY.md` from the metadata of the wheels, plus the models and data files. |
| `python3 scripts/write_sums.py` | Writes `models/SHA256SUMS`: the checksums of the guard model, its copy inside the package and the guard's data files. |
| `python3 scripts/make_fixtures.py` | Rebuilds the three saved documents of `tests/fixtures`. |
| `.venv/bin/python scripts/build_dictionaries.py [--out DIR] [--typo-ratio N]` | Builds the word lists, the spelling index and the character models from `wordfreq`. `--out` defaults to `src/extract/guard/data`; `--typo-ratio` drops unaccented words that many times rarer. |
| `python3 scripts/api_docs.py [--check]` | Writes the reference pages in `docs/reference/` from the docstrings and the schema, and refreshes the measured block of `docs/MEASURED.md`. `--check` changes nothing and exits 1 if any file is stale. |
| `python3 scripts/docs.py {build,serve,check}` | Builds the documentation site, serves it for editing, or checks it (generated files, `mkdocs build --strict`, every example, every link). `check --no-nice` keeps the normal priority. |
| `python3 scripts/ui_screenshots.py` | Regenerates the prepared screenshots in `docs/screenshots`, not the `real-*` ones. It runs only when the home folder is the neutral one that its description names (the description shows how to get one with `bwrap`), and it needs Chrome. |
| `scripts/slow-tests.sh` | The slow exhaustive tests of the number rules. Its arguments go to `unittest`. |
| `.venv/bin/python -m pytest` | The test suite. The tests that need the benchmark corpus, which is not shipped, are skipped by default. |

`ui_screenshots.py` does not look at `--help`, and `slow-tests.sh` passes it to `unittest`.

---

[Back to the reference index](index.md); [JSON format](json-format.md).
