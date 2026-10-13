# Privacy

Extract reads your documents on your computer and sends nothing anywhere. This page says what that rests on, what it
writes to disk and where, and how to remove it. Read it if the documents you read are confidential.

## Nothing leaves the computer

- Reading uses no network. The only downloads are the software and the OCR models, once, at installation (`install.py`,
  `fetch_models.py` and `make_offline_bundle.py`), and each model is checked against a published sha256. After that
  Extract works with the network unplugged, and a test fails if reading opens a network connection.
- One of the libraries Extract uses, OpenVINO, would send an anonymous usage event to Google Analytics each time it is
  imported. Extract imports it with that reporting switched off, and a test fails if it is ever reached. No document,
  page, word or statistic is sent.
- The interface, `.venv/bin/python -m extract.ui`, serves a page on `127.0.0.1` at a random port. Every request must carry
  the token printed when the interface starts, which changes at each start, and a loopback host name, or it is refused.
  It browses only folders inside your home folder, and it answers only on the computer that runs it.

## What is written to disk

| What | Where | When |
|---|---|---|
| One JSON file per page read (the page cache) | `pages/<sha256 of the PDF>/p0001.json` in the work folder | While reading. A cached page is not read again. |
| A log of the engine's decisions | `job.log` in the work folder | While reading. |
| Every correction, acceptance, undo and mode change | `corrections.jsonl` in the work folder | At the first edit, then at each one. |
| The files of an export | The destination folder you choose | On `export`. |
| The file `Document.save` writes | The path you give | On `save`. |
| A hidden partial file, `.<name>.partial-<pid>.<ext>` | The export's destination folder | Briefly, during an export. A failed or cancelled export leaves none. |

Extract writes nothing in the folder of the source PDF. The work folder is `$XDG_CACHE_HOME/extract` or `.cache/extract`
in your home folder for the library, and `.cache/extract` for the interface; `Options.work_dir` and `--work-dir` change
it. It holds the words read from every document, as the page cache, and the original and corrected text of every edit, so
treat it as you treat the documents themselves. Deleting it removes every trace of what you read, and the next read of the
same file starts from scratch; to keep a reading, use `Document.save` or an export.

## What is not covered

Extract does not encrypt anything it writes. The searchable PDF is written without the restrictions of the original: a
PDF with an owner password that restricts copying or changes is read like any other, and the new file is not locked, so
keep the original if the restrictions matter and do not share the new file as if it were protected. A PDF that needs a
password to open is refused.

## Next

[Installation](../getting-started/installation.md) says where each download comes from. [Licence and
credits](../licence.md) lists the models and word lists Extract installs.
