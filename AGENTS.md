# Extract: notes for AI coding assistants

Read this file before you write code that calls Extract. Every statement is checked against the code of version 5.0.7,
and every code block runs. The reasons behind the rules are in `docs/how-to/ai-assistants.md`; the full reference is
`docs/reference/index.md`.

Extract reads PDFs offline, on this machine, with OCR. It gives every word a status and says which words and numbers are
probably wrong. It never changes a number on its own authority: a digit changes only when a check that would break on any wrong
digit confirms the new one, or when a person decides. It reads PDF only (put images in a PDF first), in Spanish and
English only. Python 3.10 or newer; the suite was run on Linux, Windows is untested, and macOS is not supported (there is no lock file).

## Common tasks

For these tasks this page is enough: copy a recipe and change the file names. Open the reference only for a field that is not
listed here. Every recipe passes `Options(work_dir="work")`, so that it writes nowhere else; without it the work folder (the
page cache, `job.log` and `corrections.jsonl`) is `$XDG_CACHE_HOME/extract`, or `~/.cache/extract`, which a script of your own
may use. Pass one in tests, and wherever those files must go elsewhere. The recipes use the sample documents `invoice.pdf`,
`scan.pdf` and `damaged.pdf` (copies of `tests/fixtures/mismatch.pdf`, `doubtful.pdf` and `unreadable.pdf`), `scans/` that
holds the three, and the saved reading `invoice.extract.json`.

**Read one file.** `doc.status` is `done` or `partial`; `doc.doubtful()` lists the words nobody has decided whose status is
`doubtful` or `unscored` (it includes the unscored ones).

```python
import extract

if __name__ == "__main__":
    doc = extract.read("invoice.pdf", extract.Options(work_dir="work"))
    print(doc.status, "pages:", len(doc.pages), "to review:", len(doc.doubtful()))
```

```text
# output
done pages: 1 to review: 3
```

**Read a folder.** One `Reader` keeps the OCR workers and the models loaded for every file, where `extract.read` starts and
stops them on every call. Catch the two errors that a file causes and go on; any other `ExtractError`, such as
`EngineLoadError`, means nothing will be read, so let it stop the batch. [Batch a folder](docs/how-to/batch-a-folder.md).

```python
from pathlib import Path

import extract

if __name__ == "__main__":
    with extract.Reader(extract.Options(work_dir="work")) as reader:
        for pdf in sorted(Path("scans").glob("*.pdf")):
            try:
                doc = reader.read(pdf)
            except (extract.UnsupportedFile, extract.PasswordProtected) as error:
                print(pdf.name, "not read:", error.code)
                continue
            print(pdf.name, doc.status, "pages:", len(doc.pages), "to review:", len(doc.doubtful()))
```

```text
# output
damaged.pdf partial pages: 3 to review: 11
invoice.pdf done pages: 1 to review: 3
scan.pdf done pages: 1 to review: 0
```

**Review and decide.** Show the person `doc.word_crop(word.id)` (PNG bytes) before you decide. `accept` says the word is right,
`correct(id, text)` types the right text, `use_suggestion(id)` applies the suggestion, `reopen(id)` takes a decision back; decide
each word once. [Review doubtful words](docs/how-to/review-doubtful-words.md).

```python
import extract

# What the person answered after looking at doc.word_crop(word.id) for each word; never decide for them.
DECISIONS = {"p1w11": "accept", "p1w13": "accept", "p1w4": "accept"}

doc = extract.load("invoice.extract.json", options=extract.Options(work_dir="work"))
for word in doc.doubtful():
    print(word.id, word.text, word.suggestion)
    if DECISIONS.get(word.id) == "accept":
        doc.accept(word.id)
print(len(doc.doubtful()), "left;", [path.name for path in doc.export("out", ("json",))])
```

```text
# output
p1w11 Pelmuda None
p1w13 Esquilmar Esquivar
p1w4 F-2026-0498 None
0 left; ['invoice.extract.json']
```

**Unreadable pages and the exit code.** A document with an unreadable page is returned as `partial`, not raised. `error` is
filled only for `page.reason.render_failed`; for the other reasons `reason` says why. The page's words are `unscored` and in
`doc.doubtful()`: a person checks them against the page, or the page is scanned again and read again.
[Partial documents and errors](docs/how-to/batch-a-folder.md#partial-documents-and-errors).

```python
import extract

if __name__ == "__main__":
    doc = extract.read("damaged.pdf", extract.Options(work_dir="work"))
    for page in doc.pages:
        if page.status == "unreadable":
            print("page", page.number, page.reason, page.error)
    exit_code = 2 if doc.status == "partial" else 0  # pass it to sys.exit
    print(doc.status, doc.unreadable_pages, "exit code", exit_code)
```

```text
# output
page 2 page.reason.few_words None
partial (2,) exit code 2
```

**A number is `mismatch`.** `number.suggestion` holds the one-digit change. `doc.use_suggestion(number.word_ids[0])` applies
the word's own suggestion first, if it has one, and the number's digit fix only when it has none. A verified fix says that the
figures add up with that digit; it does not say that the digit is the printed one. Look at the page: if it shows the digit as
read, the document itself is inconsistent, the printed value is the record, and the decision is `accept`; if it shows the
suggested digit, `use_suggestion` records the fix as the person's correction. Extract applies verified digit fixes by itself
only when you read with `correct="safe"` or `"more"`, under the conditions of [Number checks](docs/concepts/number-checks.md).

```python
import extract

# The person looked at doc.word_crop(...) of the total: the page prints 101.913,81, the digit as read.
DECISIONS = {"p1w75": "accept"}

doc = extract.load("invoice.extract.json", options=extract.Options(work_dir="work"))
for number in doc.numbers:
    if number.status == "mismatch" and number.suggestion:
        fix = number.suggestion
        print(number.text, "digit", fix.position, fix.old, "to", fix.new, "verified" if fix.verified else "a guess")
        if DECISIONS.get(number.word_ids[0]) == "accept":
            doc.accept(number.word_ids[0])
print(doc.field_status(doc.invoice.total))
```

```text
# output
101.913,81 digit 8 8 to 5 verified
mismatch
```

**Invoice fields and Excel.** `doc.invoice` is read from page 1 only and is `None` for a generic document, and a field that
was not found is `None` too. `doc.field_status(None)` returns `"unverified"`, so test for `None` first: a missing total is not an
unconfirmed one. [Invoice fields](docs/how-to/batch-a-folder.md#invoice-fields).

```python
import extract

doc = extract.load("invoice.extract.json", options=extract.Options(work_dir="work"))
invoice = doc.invoice
if invoice is None or invoice.total is None:
    print("no total found")
else:
    print(invoice.total.text, doc.field_status(invoice.total), [path.name for path in doc.export("out", ("xlsx",))])
```

```text
# output
101.913,81 mismatch ['invoice.xlsx']
```

**Progress and cancel.** `on_event` receives `Started`, a `PageRead` and a `Progress` for each page, `DocumentRead` and
`Finished`. `token.cancel()` from any thread stops the read within about two seconds: `Finished("stopped")` is sent, then
`extract.Cancelled` is raised. Use a new work folder: a read of cached pages ends before the cancel arrives, and a read that
ends first returns its document as usual.
[Progress and cancel](docs/how-to/progress-and-cancel.md).

```python
import threading

import extract


def on_event(event):
    if isinstance(event, extract.Finished):
        print("event", event.status)


if __name__ == "__main__":
    token = extract.CancelToken()
    threading.Timer(0.5, token.cancel).start()
    try:
        extract.read("invoice.pdf", extract.Options(work_dir="agents-cancel"), on_event=on_event, cancel=token)
    except extract.Cancelled:
        print("raised Cancelled")
```

```text
# output
event stopped
raised Cancelled
```

**Save, load, export.** `doc.save(path)` writes a saved reading and overwrites without asking; `extract.load(path, options=...)`
reads it back with no OCR and no models, and finds the PDF next to it or at `source="where/it/is.pdf"`. `doc.export(folder,
formats)` never overwrites (a taken name gets `-2`, `-3`) and returns the paths: `<name>.searchable.pdf` (`"pdf"`, which needs the
source PDF with the same hash or raises `SourceChanged`), `<name>.xlsx`, `<name>.extract.json` and `<name>.txt` (the text of the
document), where `<name>` is the PDF's stem. Prefer `export(folder, ("json",))` to `save` for a person's saved reading.

```python
from pathlib import Path

import extract

options = extract.Options(work_dir="work")
doc = extract.load("invoice.extract.json", options=options)
doc.save(Path("mine.extract.json"))
again = extract.load("mine.extract.json", source="invoice.pdf", options=options)
print(again.status, [path.name for path in again.export("export-agents", ("xlsx", "txt"))])
```

```text
# output
done ['invoice.xlsx', 'invoice.txt']
```

**The JSON without the library.** The `.extract.json` holds `pages[].words[]`, `numbers[]` and `invoice`; a word there has no
`page` key, since its page is the one it sits under and its id carries it. A word needs a person when its `status` is
`doubtful` or `unscored` and its `review` is `null`. [JSON format](docs/reference/json-format.md).

```python
import json
from pathlib import Path

data = json.loads(Path("invoice.extract.json").read_text(encoding="utf-8"))
needs_a_person = [
    word["id"]
    for page in data["pages"]
    for word in page["words"]
    if word["status"] in ("doubtful", "unscored") and word["review"] is None
]
print(data["status"], data["page_count"], needs_a_person)
```

```text
# output
done 1 ['p1w4', 'p1w11', 'p1w13']
```

**A web service** has one thread own one `Reader` and take jobs from a queue; never share a `Reader` between threads.
[Use Extract in a web service](docs/how-to/web-service.md). **Install offline:** `.venv/bin/python scripts/make_offline_bundle.py`
on a connected machine, unzip the bundle, then `python3 scripts/install.py --offline offline`.
[Without a network](docs/getting-started/installation.md#without-a-network).

## Fields you will use

The long class names show in tracebacks and `repr`: `Page` is `PageReading`, `Number` is `NumberResult`, `Rule` is `RuleResult`
and `Invoice` is `InvoiceFields`. Write the short names.

| Object | Fields |
|---|---|
| `Document` | `status` (`done`, `partial`), `pages` (a tuple; `len(doc.pages)` is the page count), `numbers`, `invoice` (or `None`), `unreadable_pages` (1-based numbers), `doc_type` (`invoice`, `generic`), `name`, `source` |
| `Page` | `number` (1-based), `status` (`native`, `ocr`, `unreadable`), `reason`, `error`, `words`, `dpi` |
| `Word` | `id` (`"p1w4"`: page 1, word 4), `page` (1-based), `text`, `status`, `review` (`None`, `"accepted"`, `"corrected"`), `suggestion`, `suggestion_conf`, `p_wrong`, `reasons`, `original`, `box` |
| `Number` | `text`, `value` (a `Decimal`), `kind`, `status` (`checked`, `mismatch`, `unverified`), `suggestion` (`position`, the 0-based index in `text`; `old`, `new`, `verified`, `rule`), `word_ids`, `rules` |
| `Invoice` | `number`, `date`, `subtotal`, `total`, `withholding`, `currency`, `lines` (`description`, `quantity`, `unit_price`, `amount`), `vat` (`rate`, `base`, `amount`); each field is a cell with `text`, `word_ids` and `page`, or `None` |
| Events | `Started(documents, pages)`, `PageRead(doc_id, page, status, words, doubtful, fixed)`, `Progress(doc_id, done, total)`, `DocumentRead(doc_id, path, status, unreadable_pages, doubtful, numbers_checked, fixed)`, `Finished(status)` |

## Imports

```python
import extract
```

Every name of `extract.__all__` (36) is a top-level name, the event classes and the errors included: `extract.read`,
`extract.Reader`, `extract.Options`, `extract.CancelToken`, `extract.PageRead`, `extract.Finished`, `extract.Cancelled` and the
rest.

- Never write `from extract import *`: it replaces the built-in `Warning` with the event class `extract.Warning`.
- `extract.PageReading`, `extract.NumberResult`, `extract.RuleResult` and `extract.InvoiceFields` do not exist. The public
  names are the short ones: `extract.Page`, `extract.Number`, `extract.Rule`, `extract.Invoice`.
- Do not import `extract.engine`, `extract.guard`, `extract.checks`, `extract.fixes`, `extract.pipeline`, `extract.outputs` or
  `extract.ui`. They are not an API and change without notice.

## The `__main__` guard is required

The OCR runs in worker processes that import your main script again. Every script that reads must put its calls to
`extract.read` and `extract.Reader` under `if __name__ == "__main__":`. Without the guard, a scanned PDF makes `read` raise `EngineLoadError` after a few seconds (`cause="no_main_guard"`), the
worker prints one line on stderr, and the script ends with exit status 1. The error ends with:

```text
extract.errors.EngineLoadError: the script that called Extract reads a document at its top level, and each OCR worker imports that script again; put every call to `extract.read` and `extract.Reader` under `if __name__ == "__main__":`
```

A digital PDF with a good text layer, or a scan already in the work folder's cache, needs no worker, so it reads fine
without the guard: the mistake shows only on a scan that has to be read. `load` starts no worker and needs no guard.

## Read the status before you trust a word

| Value | Meaning |
|---|---|
| `Word.status == "native"` | The text came from the PDF's own text layer. It was not scored. |
| `"ok"` | The guard looked at the word and did not flag it. This is not "verified". |
| `"doubtful"` | The guard flags it: check it. |
| `"unscored"` | Nobody could judge it: no guard, an unreadable page, too many words. It is never fine. |
| `"fixed"`, `"absorbed"` | Only after a correction mode other than `off`: the word a guard fix changed, and a word joined into it. |
| `Number.status == "checked"` | A rule that fails on any wrong character passed and none failed. Only a rule can say this. |
| `"mismatch"` | A rule failed: at least one digit is wrong somewhere, not necessarily where the suggestion points. |
| `"unverified"` | Nothing contradicts it and nothing confirms it. It is not wrong; a CIF or an amount written with a space as thousands separator cannot be confirmed. |

`Number.value` is a `Decimal` or `None`: never convert it to a float.

## Review and correct

The calls are in the recipes above. What they do not say:

- A decision sets `word.review` (`"accepted"` or `"corrected"`) and leaves `word.status` alone. A word you accepted can still be
  `doubtful`: test `review` to know whether somebody decided.
- `accept` after `correct` turns `corrected` into `accepted` and keeps the typed text; a later `reopen` clears the decision but
  does not bring the OCR text back. Call one decision per word.
- A `suggestion` is the guard's best candidate, not a verdict: `word.suggestion_conf`, its probability of being right, is 0.0087
  for `Esquivar` in the sample. Show it with the crop; do not apply it unseen.
- `doc.fix_info(id)` gives the guard's proposal for a word as a dictionary: `op` (`text`, `join`, `segment`, `drop`), `kind`, `text`,
  `ids` (the words it spans), `source`, `p_fix` and `p_keep` (probabilities that the new text and the word as read are right),
  `rule` (the verifying rule of a digit fix), `modes` and `applied`. An empty `modes` means the proposal is only shown and no mode
  applies it. `doc.fixed()` lists the words whose fix is applied now.
- `accept` never makes a number `checked`; `correct` does it only when a rule passes on the new text. `use_suggestion` raises
  `ValueError` for a number written in several words (an IBAN in groups) or one with no single-digit fix that works: use
  `correct` on the word that holds the wrong digit.
- Do not accept, correct or `use_suggestion` words on your own authority to make a number pass. A digit suggestion whose
  `verified` is false is a guess, and `use_suggestion` applies it anyway. Put the crop (`doc.word_crop(id)`) or the page
  (`doc.page_image(n)`) in front of the person, or apply a rule that person gave you.

## Corrections are off by default

`Options.correct` is `"off"`: the guard changes no word. `extract.read(path, options, correct="safe")` or `"more"` lets it apply
fixes, and `doc.set_mode("more")` re-applies them without reading again. Every applied fix can be undone with `doc.undo(id)`,
`doc.undo_page(n)` or `doc.undo_all()`. A page whose fixes would change more than 8 % of its words keeps none and sends a
`guard.warning.fixes_withheld` notice. With the default languages the three sample documents get no applied fix in any mode:
that is the normal result, not a bug. `safe` applies no ordinary word in any setting.

`Options(second_reading=True)` computes extra readings that the packaged guard ignores. It costs time and memory and changes
nothing. Use `Options(second_reading="thorough")` after `python3 scripts/fetch_models.py --all`.

## Progress, cancel, errors

- `on_event` is called in the thread that called `read`; pages arrive in the order the workers finish, not always 1, 2, 3. A
  callback may call `token.cancel()` itself, and a cancelled read keeps the pages already read in the cache.
- Every error Extract raises about a file, a model or a job is an `extract.ExtractError` with a `code` and `params`:
  `UnsupportedFile` (not a PDF, or not a saved reading), `PasswordProtected`, `EngineLoadError` (a model, word list or worker
  is missing: the message says what to run), `ModelMismatch`, `OutputExists`, `SourceChanged`, `Cancelled`.
- These are not `ExtractError`: `FileNotFoundError` (a path that does not exist), `ValueError` (a bad option, mode or format),
  `KeyError` (an unknown word id), `IndexError` (a page past the end), `OSError` (a file that cannot be written).
- Page numbers are 1-based. `doc.tokens(0)` and `doc.page_image(0)` do not raise: they return the last page.
- While the models load, RapidOCR prints `[INFO]` lines on stderr. They are not errors, and Extract has no option that silences
  them: redirect stderr (`2> read.log`) to keep them out of the terminal.

## Memory, time and the work folder

- A read starts up to two worker processes of about 1.5 GB each. Never run two reads at once on a machine with 3 to 4 GB of
  memory, and never call `Reader.read` from two threads at once. On a small machine use
  `Options(processes=1, backend="onnxruntime")`.
- Time: 3.6 s a page (one process of 4 threads) or 3.2 s (two processes of two threads) on the development laptop; the thorough
  second reading adds 2.95 s a page on average over 60 pilot pages, without the memory cap. Do not promise less elsewhere.
- `extract.load` logs later corrections into the default work folder unless you pass `options=extract.Options(work_dir=...)`.

## Never

- Never invent a status, a field or a method. If it is not on `docs/reference/index.md`, it does not exist: there is no
  `doc.text`, no `extract.ocr`, no `Word.confidence`.
- Never trust a digit that a language model made, and never ask a model to repair a number. Extract's own rule is that only a
  verifying rule or a person changes a digit.
- Never call `extract.read` in a loop (make one `Reader`), or at the top level of a script (use the `__main__` guard).
- Never treat `ok` as verified, `unscored` as fine, or `partial` as an error that was raised.
- Never edit a `Document` from the outside: `doc._data`, `Word` objects and the lists you get are not for writing. `Word` is
  frozen; a change goes through `correct`, `accept`, `use_suggestion`, `reopen`, `set_mode` or `undo`, and then you ask
  `doc.word(id)` again.
- Never overwrite a person's saved reading with `doc.save` without asking. Use `export`.
- Never read the saved `.extract.json` fixtures of the test suite as the output of a fresh read: they are prepared test data.

## Where things are

- Reference: `docs/reference/index.md`. The saved file, field by field: `docs/reference/json-format.md`. Commands and scripts:
  `docs/reference/command-line.md`. How it works: `docs/concepts/how-it-works.md`. Measured results: `docs/accuracy.md`, with
  every table in `docs/MEASURED.md`.
- Install: `python3 scripts/install.py`, then `.venv/bin/python -m extract.ui` opens the interface. The three sample documents
  are `tests/fixtures/doubtful.pdf`, `mismatch.pdf` and `unreadable.pdf`; every name in them is invented.
