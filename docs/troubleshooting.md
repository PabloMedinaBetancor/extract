# Troubleshooting

The problems people meet, by what you see: the message or the symptom, the cause, the fix. The messages are copied from
real runs or from the interface's own text; a path in a message is shortened to `...`.

## `EngineLoadError`: the script reads at its top level

A script that calls `extract.read` at its top level, on a scanned PDF:

<!-- docs-test: skip - the faulty version of the script is the point of the example -->
```python
import extract

doc = extract.read("invoice.pdf", extract.Options(work_dir="work"))
print(doc.status, doc.unreadable_pages)
```

stops after a few seconds with a line from the worker, then a traceback (the middle frames are left out here) and exit
status 1:

```text
Extract: an OCR worker is importing this script, and the script reads a document at its top level; put every call to `extract.read` and `extract.Reader` under `if __name__ == "__main__":`
Traceback (most recent call last):
  File ".../script.py", line 3, in <module>
    doc = extract.read("invoice.pdf", extract.Options(work_dir="work"))
  ...
extract.errors.EngineLoadError: the script that called Extract reads a document at its top level, and each OCR worker imports that script again; put every call to `extract.read` and `extract.Reader` under `if __name__ == "__main__":`
```

The OCR workers are started with the `spawn` method, so each one imports your main script again and would run its read.
Extract stops that read in the worker, and `read` raises `EngineLoadError` with `cause="no_main_guard"` in your process.
Put every call to `extract.read` and `extract.Reader` under `if __name__ == "__main__":`. A digital PDF, or a scan already
in the work folder's cache, needs no worker, so the fault shows only on a scan that has to be read.

## `UnsupportedFile`, `PasswordProtected` or `FileNotFoundError`

```text
extract.errors.UnsupportedFile: notes.pdf is not a PDF that can be read: Failed to load document (PDFium: Data format error).
extract.errors.PasswordProtected: locked.pdf is password protected
FileNotFoundError: [Errno 2] No such file or directory: 'missing.pdf'
```

`UnsupportedFile` means the file is not a PDF that the PDF library can open, such as a text file, an image or a damaged
PDF: put an image in a PDF first ([the FAQ](faq.md#can-it-read-images-jpg-or-png) shows how), and re-export a damaged PDF
from the program that made it. `load` raises it too, with `not an Extract document of schema version 1 or 2`, when it is
given anything but the `.extract.json` that `export` or `Document.save` wrote. `PasswordProtected` means the PDF needs a
password to open: remove the password with a PDF viewer and read the new copy (a PDF with only an owner password, which
restricts copying or changes, opens normally). `FileNotFoundError` is Python's own: check the path, and the folder the
script runs from.

## `EngineLoadError`: a model or the word lists are missing

`EngineLoadError` means a piece the read needs is missing or cannot start, and its message says which. The interface
shows «The text recognition models or the word checker could not be loaded.»

| Message | Fix |
|---|---|
| `the OCR model <file>.onnx is not in any of: ...` | A `PP-OCRv6_..._small.onnx` file is a default model: run `python3 scripts/fetch_models.py`. `latin_PP-OCRv5_rec_mobile.onnx` and `PP-OCRv6_rec_medium.onnx` are the reader models of the thorough second reading: run `python3 scripts/fetch_models.py --all` once (169 MB), or set the second reading to off. |
| `the word lists for <languages> are missing; build them with: .venv/bin/python scripts/fetch_models.py --lists` | Run that command from the repository folder. The lists are built with `wordfreq`, which only the environment made by `scripts/install.py` has: under the system Python the script says that `wordfreq` is not installed. |
| `the thorough second reading needs its own guard model, <path>, ...` | Reinstall the package, which ships `guard-thorough.model`, or point `EXTRACT_GUARD_MODEL_THOROUGH` at it. |
| `rapidocr is not installed: ...` or `OpenVINO did not start: ...` | Run `scripts/verify_install.py` ([Installation](getting-started/installation.md#check-an-installation)). |

`ModelMismatch` is a close relative: the guard model file was built for another version of Extract. Reinstall the package
so that the model and the code match.

## `SourceChanged` after `load`

```text
invoice.pdf is not the file this document was read from
the source file invoice.pdf is not available
```

A saved reading keeps the hash of its PDF and looks for the PDF next to it, under the name it was read with. Whatever
needs the pages, `page_image`, `word_crop` and the `pdf` export, checks the file at its first use. The first message means
the PDF beside the JSON is another file; the second, that there is none, and `doc.source` is `None`. Words, numbers,
`save` and the `xlsx`, `json` and `txt` exports still work. Put the original PDF beside the JSON under its original name,
or pass `source=` to `load` with the path where it is. A read raises `SourceChanged` too when the file changes while it is
read: read it again.

## A document comes back `partial`

At least one page is `unreadable`; the document is returned, not raised, so a script has to look at `doc.status` and
`doc.unreadable_pages`. Each such page says why in `reason`: `page.reason.few_words` (fewer than 25 words came out: a
blank page, or a page that is mostly an image), `page.reason.low_confidence` (the mean confidence of its words was under
0.65: a page upside down, turned a quarter, or of very poor quality) or `page.reason.render_failed` (the page could not be
drawn or its worker died twice; `error` holds the text). Its words are kept as `unscored` and its numbers are never
`checked`. To read the page, check that it is the right way up, rotate it if not, and scan it again at 150 dpi or more.

## A notice on a page

A read sends `Warning` events, and the interface shows them on the page. None of them stops the read.

| Notice | What it means and what to do |
|---|---|
| `page.warning.sideways` | The lines run up or down: the page is turned a quarter. No word was changed. Rotate the PDF and read it again. |
| `page.warning.low_resolution` | The scan is under 100 dpi and was read at 100. Nothing in its words says whether it read well: scan it again at 150 dpi or more. |
| `guard.warning.fixes_withheld` | More than 8 % of the page's words would change, so no fix was applied: review the page by hand. With `cause="language"`, the page is not in a language you chose: read again with the right languages. |
| `guard.warning.unscored` | The guard could not judge some words (`no_guard`, `error`, `too_many_words` or `no_confidence`). They are `unscored`, not fine. |
| `guard.warning.readings_differ` | You set `second_reading=True`, whose readings the packaged guard ignores. Use `"thorough"`, or switch the second reading off. |
| `ram.low` | Less than 600 MB of memory is free, so no OCR worker starts until it recovers. Close other programs, or see the memory section below. |

The [events reference](reference/events.md#warning) lists every code with its parameters.

## A result that looks wrong

**Extract changed no word.** `correct` is `off` unless you ask for more, so the best fix is only suggested;
[Corrections](concepts/corrections.md) says what `safe` and `more` do.

**A word I accepted is still `doubtful`.** `accept` and `correct` set `review`, not `status`. The word left `doubtful()`
because it has a `review`: test `review` to know whether a word has been decided.

**A number is `mismatch` and it looks right.** Either a digit was read wrong, or the document's own figures do not add up
(a total off by a few cents as printed), or the VAT rate is not one Extract knows (19 % and 22 % come out `mismatch`). Look
at the number's `rules` before you suspect the OCR.

**`use_suggestion` raises `ValueError`.** The word has no suggestion: two changes tie, or the number is written in several
words, such as an IBAN in groups of four. Correct the word that holds the wrong digits with `correct(word_id, text)`.

**The export is called `invoice-2.xlsx`.** A file with that name was already there, and Extract never overwrites; the files
of one export share the number. `Document.save` is the exception: it overwrites without asking.

**`?` in the text layer of the searchable PDF.** The hidden text layer is written in Windows 1252, and any other character
becomes `?` there only. The export and the JSON keep the right characters.

## Lines that start with `[INFO]`

While a read loads the OCR models, RapidOCR prints lines that start with `[INFO]` on the error stream. They are not errors,
and Extract has no option that silences them; `2> read.log` keeps them in a file.

## Slow, or out of memory

A read loads its models before the first page, and `extract.read` pays that on every call: for more than one file, use
one `Reader`. Each OCR worker is budgeted 1,500 MB with the default backend (the measured peak was 1,330 MB), and the
thorough second reading about 1.6 GB more. On a machine with 3 or 4 GB, set `Options(processes=1, backend="onnxruntime")`,
which needs about 700 MB a worker, close other programs, and do not run two reads at once. On a small two-core laptop a
page full of text takes 18 to 42 s ([Accuracy and limits](accuracy.md#speed-and-memory)).

## The interface refuses

- «Refused: wrong or missing token. Open the address printed in the terminal.» Open the full address from the terminal,
  with the part after `?t=`: every start of Extract has a new token. «Refused: wrong address.» means a host other than
  `127.0.0.1` or `localhost`.
- «That folder is outside your home folder.» The interface opens only folders inside your home folder: move or copy the
  PDFs there.
- «Extract cannot write in its work folder ({path}): it is read-only, full or not allowed. Start Extract again with
  --work-dir and a folder of yours.»
- «A read is already running.» Wait for it to finish, or press **Stop reading** on the **Read** screen.

The [interface page](reference/interface.md) names every control. If the installer stops with a network error, run it
again when you are connected, or use the [offline bundle](getting-started/installation.md#without-a-network).
