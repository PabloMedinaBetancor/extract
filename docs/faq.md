# Frequently asked questions

The questions people ask first, with short answers and a link to the page that explains each. If something has gone
wrong, [troubleshooting](troubleshooting.md) is the better page.

## Does it send my documents anywhere?

No. Everything runs on your computer. The downloads happen once, at installation: the software and the OCR models. After
that Extract works with the network unplugged, and a test fails if reading a document opens a connection. The interface is
a local page at `127.0.0.1`, reachable only from your computer and only with the token printed when it starts.
[Privacy](concepts/privacy.md) lists what is written where.

## Which languages does it read?

Spanish and English. The OCR models are the same for both; the language choice selects the word lists the guard uses and
how numbers are read. Other languages are not supported and were not measured, and the figures for the two languages are
reported apart in [Accuracy and limits](accuracy.md).

## Can it read images (JPG or PNG)?

Not directly: Extract reads PDF files. Put the image in a PDF first. With Pillow, which Extract already installs, one line
does it, `Image.open("scan.png").convert("RGB").save("scan.pdf", resolution=200)`, with `resolution` set to the real
resolution of the scan, because Extract uses it to choose the resolution it reads the page at.

## Which operating systems does it run on?

Linux is what was tested: the full test suite has been run on Linux with Python 3.10 only, and Python 3.12 on Linux was
installed and a document was read with it. Windows has a prepared lock file that was never run, and macOS has none and is
not supported. [Installation](getting-started/installation.md) has the steps.

## What does it produce?

For `invoice.pdf`, four files: `invoice.searchable.pdf` (your pages, untouched, with an invisible text layer),
`invoice.xlsx` (the sheets Numbers, Fields, Review and Corrections), `invoice.extract.json` (everything, to open again
later) and `invoice.txt`. Extract never overwrites: a second export is called `invoice-2.xlsx` and so on. Your PDF is not
changed. The [JSON format](reference/json-format.md) describes the saved file field by field.

## Does `ok` mean a word is right?

No. `ok` means the guard did not flag it. Only a person can verify a word, and Extract does not pretend otherwise. About 3
in 10 wrong words in Spanish and 2 in 10 in English are not flagged even when you review the 5 % that are.
[How it works](concepts/how-it-works.md#statuses) defines every status.

## Why is a number `unverified`, or `mismatch` when it looks right?

`unverified` means no rule that would catch any wrong digit passed, and none failed: nothing confirms the number and
nothing contradicts it. A CIF, a French or Italian IBAN, the withholding amount and an amount written with a space as the
thousands separator are never `checked`. `mismatch` means such a rule failed: a digit was read wrong, or the document's own
figures do not add up (the sample invoice is 30 cents off as printed), or the VAT rate is not one Extract knows. Look at
the `rules` of the number before you suspect the OCR. [Number checks](concepts/number-checks.md) lists the rules.

## Why did Extract change nothing?

Because `correct` is `off` unless you ask for more: the best fix is only suggested. `safe` applies the fixes that almost
never break a word, which in practice are the check digits of numbers, and `more` changes ordinary words as well. Every
fix can be undone. [Corrections](concepts/corrections.md) explains the three modes.

## How many words do I have to review?

About 5 % of them with the default review budget, which is a threshold set on the development pages, not a quota: a
damaged page gets more flags and a clean one almost none. `fewer` and `more` move the threshold to about 3 and 8 %.

## Is it fast?

3.6 s a page with one process of 4 threads, and 3.2 s with two processes of two threads, on the development laptop; 7 to
14 s a page for the short sample documents on a two-core Celeron laptop, and 18 to 42 s for a page full of text there. The
thorough second reading adds about 3 s a page. [Accuracy and limits](accuracy.md#speed-and-memory) has the conditions.

## How do I read many files?

With one `Reader`, so the workers and models load once: [Batch a folder](how-to/batch-a-folder.md) reads a whole folder
into one workbook.

## Where does it write files, and can I go on later?

Into its work folder, which holds the page cache, a log and the record of your corrections, and into the folder you export
to; nothing goes into the folder of the source PDF. Delete the work folder to remove every trace. To go on later,
`Document.save` writes the document with your decisions as an `.extract.json` file, and `extract.load` opens it again, as
does the interface with its `--open` option. Keep the original PDF beside the JSON under its original name, or pass
`source=` to `load`.

## Can I use it with an AI coding assistant?

Yes. `AGENTS.md` in the repository gives an assistant runnable recipes, the exact imports and the mistakes models make most
often; [Use Extract from an AI coding assistant](how-to/ai-assistants.md) is for the person who checks what it wrote.

## What is the licence?

Extract is released under the MIT licence. The OCR models are Apache-2.0 and the word lists derive from wordfreq under
CC BY-SA 4.0. [Licence and credits](licence.md) lists each one.

## How do I get help?

Questions and bug reports: open an issue on GitHub. Work requests go through the author's Upwork profile.
