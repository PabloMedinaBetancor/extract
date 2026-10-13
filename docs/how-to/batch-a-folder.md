# Batch a folder

This page reads every PDF of a folder with one `Reader`, exports each document, and writes one workbook of your own with a
row for each, with the invoice number and the total when there is one. It is for anyone with more than a few files to
read, unattended. `openpyxl`, which writes the workbook, is a dependency of Extract, so it is installed.

```python
from pathlib import Path

import openpyxl

import extract

HEADER = [
    "File",
    "Status",
    "Unreadable pages",
    "Words to review",
    "Invoice number",
    "Total",
    "Total (value)",
    "Total status",
]


def value_of(doc, cell):
    """The exact value of an invoice field as a Decimal: the cell holds the text, its number holds the value."""
    return next((number.value for number in doc.numbers if number.word_ids == cell.word_ids), None)


def row_of(doc):
    invoice = doc.invoice
    total = invoice.total if invoice else None
    return [
        doc.name,
        doc.status,
        ", ".join(str(page) for page in doc.unreadable_pages) or None,
        len(doc.doubtful()),
        invoice.number.text if invoice and invoice.number else None,
        total.text if total else None,
        value_of(doc, total) if total else None,
        doc.field_status(total) if total else "not found",
    ]


def read_folder(folder, destination):
    """Read the PDFs of `folder` in name order, export each into its own subfolder, and write one workbook."""
    book = openpyxl.Workbook()
    sheet = book.active
    sheet.title = "Documents"
    sheet.append(HEADER)
    with extract.Reader(extract.Options(work_dir="work")) as reader:
        for pdf in sorted(Path(folder).glob("*.pdf")):
            try:
                doc = reader.read(pdf)
            except (extract.UnsupportedFile, extract.PasswordProtected) as error:
                print(f"{pdf.name}: not read ({error.code})")
                continue
            doc.export(Path(destination) / pdf.stem)
            sheet.append(row_of(doc))
    Path(destination).mkdir(parents=True, exist_ok=True)
    book.save(Path(destination) / "documents.xlsx")


if __name__ == "__main__":
    read_folder("tests/fixtures", "out")
    for row in openpyxl.load_workbook("out/documents.xlsx")["Documents"].iter_rows(values_only=True):
        print(row)
```

```text
# output
('File', 'Status', 'Unreadable pages', 'Words to review', 'Invoice number', 'Total', 'Total (value)', 'Total status')
('doubtful.pdf', 'done', None, 0, '2026-0042', None, None, 'not found')
('mismatch.pdf', 'done', None, 3, 'F-2026-0498', '101.913,81', 101913.81, 'mismatch')
('unreadable.pdf', 'partial', '2', 11, '2026-0042', None, None, 'not found')
```

This is `examples/batch_folder.py`; run it from the repository folder. Each document's four files go into their own
subfolder of `out/`. A second run exports each document under the next free copy number, and replaces `documents.xlsx`.

## One `Reader` for the batch

`extract.read` starts the OCR workers and loads the models on every call; a `Reader` does it once, on its first read, and
keeps the workers until the `with` block ends, even if a read raises. It reads one document at a time: do not call
`reader.read` from two threads at once ([Use Extract in a web service](web-service.md) shows a service). The
`if __name__ == "__main__":` line is required, because the workers import your script again.

## Partial documents and errors

A page that cannot be read is not an exception. The document comes back with `doc.status == "partial"`, and
`doc.unreadable_pages` gives the page numbers, counting from 1: page 2 of `unreadable.pdf` above. The other pages are
complete. The words of the unreadable page are kept as `unscored` and counted among those to review, its numbers are never
`checked`, and the page says why in `reason`: `page.reason.few_words` (fewer than 25 words came out),
`page.reason.low_confidence` (mean confidence under 0.65) or `page.reason.render_failed` (it could not be drawn, or its
worker died twice; `error` then holds the text). Decide what partial means for your batch: the script exports such a
document and its row says so.

An exception means the file as a whole could not be read. Every error Extract raises about a file, a model or a job is an
`ExtractError`, with a `code` such as `error.password_protected` and `params`:

| Error | Raised when | What to do |
|---|---|---|
| `UnsupportedFile` | The file is not a PDF, or `load` was given something that is not a saved reading. | Skip the file. Images must be put in a PDF first. |
| `PasswordProtected` | The PDF needs a password to open. A PDF with only an owner password opens normally. | Skip the file and ask for an unprotected copy. |
| `EngineLoadError` | A model, a word list or a worker is missing or could not start. The message says which, and what to run. | Stop the batch: nothing will be read until the installation is fixed. |
| `ModelMismatch` | The guard's model does not match this version of Extract. | Stop and reinstall the version you run. |
| `SourceChanged` | The PDF changed while it was read. | Read it again. |
| `Cancelled` | A `CancelToken` was set. | Nothing is wrong ([Progress and cancel](progress-and-cancel.md)). |

The script catches the two that a file causes and lets the others stop the batch. Some errors are not `ExtractError`:
`FileNotFoundError` for a path that does not exist (a glob gives only paths that exist), `ValueError` for a bad option,
mode or format, and `OSError` for a file that cannot be written. The [errors reference](../reference/errors.md) lists every
class.

## Invoice fields

An invoice field is a `Cell`: the text as read and the ids of the words it came from, not a number. The `Total` column
holds that text, as printed; `Total (value)` holds the number, the one in `doc.numbers` whose `word_ids` are the cell's,
whose `value` is a `Decimal`, exact to the cent. `openpyxl` writes a `Decimal` as a number; do not convert it to a `float`
to sum money. (`openpyxl` gives numbers back as floats when it reads the file, which is why the output shows
`101913.81`.)

A field that Extract did not find is `None`, never a guess, and `doc.invoice` is `None` for a document that is not an
invoice: test each before you use it. `doc.field_status(None)` returns `"unverified"`, which would read a missing total as
an unconfirmed one, so the script asks for the status only when the cell is there. Fields are read from page 1 only, so a
multi-page invoice is not covered. A `mismatch` total keeps the printed value: do not book it before a person has looked at
the page (the [tutorial](../tutorials/invoice.md#why-the-total-fails) says why). `doc.invoice.lines` holds the lines and
`doc.invoice.vat` one `VatPair` for each VAT rate; the [data types](../reference/data-types.md#invoice-invoicefields) page
lists their fields.

## The workbook of each document

The export writes, among its four files, a workbook with four sheets. **Numbers** lists every number of the document: its
text, value, kind, status, the rules in words and the page. **Fields** lists the invoice fields with their value, the text
as read and the status, and exists only for an invoice. **Review** lists the words nobody has decided yet, with their
reasons and suggestion. **Corrections** lists what was changed, by you or by the guard. Status cells are coloured, and the
workbook holds values, not formulas. `doc.export(folder, ("xlsx",))` writes the workbook alone.

## Many files, little memory

A read starts up to two workers, each budgeted 1,500 MB with the default backend: on a machine with 3 to 4 GB, run one
read at a time with `Options(processes=1, backend="onnxruntime")`. Pages read before come from the page cache in the work
folder, so a second run over the same folder costs no OCR.

## Next

[Review doubtful words](review-doubtful-words.md) decides the words and the failing totals before you export.
[Progress and cancel](progress-and-cancel.md) reports how far a long batch has got and stops it.
