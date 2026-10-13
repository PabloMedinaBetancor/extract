# Quickstart

This page takes you from an installed Extract to a first result, first in the browser interface and then in ten lines of
Python. It assumes you installed Extract as described in [Installation](installation.md) and are in the repository
folder. If you are writing code, or an AI assistant is, `AGENTS.md` at the root of the repository has the calls for the
common tasks on one page.

## With the interface

```console
$ .venv/bin/python -m extract.ui
```

Your browser opens on the Extract page. On a machine without a desktop, add `--no-browser` and open the address it
prints; the address carries a token that changes at every start, and the page refuses requests without it. Leave the
terminal open while you work: `Ctrl+C` there closes Extract.

1. On **Import**, type the folder that holds your PDFs and press **Browse**. For a first try, type the path of the
   `tests/fixtures` folder of the repository: it has three small sample PDFs with invented content. The interface opens
   only folders inside your home folder.
2. Press **Add** next to each PDF, or **Add all**.
3. Tick **Spanish**, **English** or both, whichever the documents are in, and press **Read**.
4. When the bar is full, press **Go to Review**.
5. On **Review**, yellow boxes are the doubtful words and red ones the numbers whose rules fail. `N` jumps to the next
   one, `Enter` accepts it as it is, `Tab` takes the suggestion when there is one, and `E` lets you type the right word.
6. On **Export**, choose a destination folder and press **Export**. A box at the top says how many files were written
   and where.

Each document gives four files:

| File | What it is |
|---|---|
| `name.searchable.pdf` | Your original pages, with selectable and searchable text. |
| `name.xlsx` | The numbers, the invoice fields, the words to check and your corrections. |
| `name.extract.json` | Every word with its box, confidence and status. |
| `name.txt` | The plain text. |

An existing file is never overwritten: a second export of the same document writes `name-2.searchable.pdf` and so on, all
four with the same number. The [interface page](../reference/interface.md) describes each screen and every key.

## With Python

The same work in code, on the sample invoice whose total is 30 cents off. Every name and number in it is invented.

```python
import extract

if __name__ == "__main__":
    doc = extract.read("tests/fixtures/mismatch.pdf")
    for word in doc.doubtful():
        print(word.id, word.text, word.suggestion)
    for path in doc.export("out/"):
        print(path)
```

```text
# output
p1w11 Pelmuda None
p1w13 Esquilmar Esquivar
p1w4 F-2026-0498 None
out/mismatch.searchable.pdf
out/mismatch.xlsx
out/mismatch.extract.json
out/mismatch.txt
```

`extract.read` returns a `Document`. `doubtful()` lists the words to check, most doubtful first, and `export` writes the
four files above into `out/`.

The `if __name__ == "__main__":` line is required. The OCR runs in worker processes that Python starts by importing your
script again, so a script that reads at its top level stops with an `EngineLoadError`; [Troubleshooting](../troubleshooting.md)
shows what that looks like. While the models load, RapidOCR prints `[INFO]` lines on the error stream: they are not
errors ([Troubleshooting](../troubleshooting.md#lines-that-start-with-info) says how to keep them out of the terminal).

`extract.read` starts the workers and loads the models for that one call. To read several files, use one `Reader`, so
that start-up is paid once ([Batch a folder](../how-to/batch-a-folder.md)). With no options, a read keeps its page cache,
`job.log` and `corrections.jsonl` in the work folder, `$XDG_CACHE_HOME/extract` or `~/.cache/extract`;
`extract.Options(work_dir="work")` puts them in a folder of your own.

## The samples

`tests/fixtures` holds three PDFs, all scans, with invented names, addresses and identifiers that cannot exist:

| File | What it is |
|---|---|
| `doubtful.pdf` | One page, an invoice stub of eight lines. It reads without any word to review. |
| `mismatch.pdf` | One page, an invoice whose total is 30 cents off. It has three words to review and a `mismatch` total. |
| `unreadable.pdf` | Three pages; page 2 is nearly empty and comes out `unreadable`, so the document is `partial`. |

Next to each PDF there is a saved reading, `*.extract.json`. Those files are prepared test data for the interface, not
the result of a read; read a PDF yourself to get a real one.

## Next

The [tutorial](../tutorials/invoice.md) follows the sample invoice from its doubtful words to the failing total and the
workbook. [How it works](../concepts/how-it-works.md) defines `ok`, `doubtful`, `checked` and the other statuses.
