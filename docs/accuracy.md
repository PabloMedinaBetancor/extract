# Accuracy and limits

This page gives the figures that decide whether Extract fits a job, each with the conditions it was measured under, and
then what Extract does not do, with the reason. Read it before you rely on the output, and again when a result surprises
you. Every table behind these figures, with its method and its intervals, is in `docs/MEASURED.md` in the repository.

## How the figures were measured

Most figures come from one sealed test of 772 pages, 456 Spanish and 316 English, from 245 documents of public
administrations and 60 synthetic invoices: about 347,000 words, many of them on pages damaged on purpose to look like poor
scans. The test was put aside before development began and was read for the published figures on code frozen beforehand.
Every interval is 95 % and computed by document, and Spanish and English are reported apart. The invoices are made by the
project's own generator, so they measure the number rules and the field reader on that generator's layouts, not real
invoices. A figure from the earlier v4 test, which is development data, says so.

## The figures

| What | Spanish | English | Conditions |
|---|---|---|---|
| Words read wrong (word error rate) | 2.78 % (2.07 to 3.65) | 2.85 % (2.20 to 3.61) | As the OCR reads them, before any review or correction. |
| Wrong words among the flagged ones (recall) | 68.6 % (49.7 to 86.5) | 77.1 % (72.0 to 81.3) | The default review budget, `balanced`, which flags about 5 % of the words. |
| Flagged words that are wrong (precision) | 37.1 % (29.5 to 45.4) | 42.5 % (32.5 to 53.6) | Same budget. |
| `checked` numbers that were wrong | 0 of 232 (95 % upper bound 1.29 %) | 0 of 287 (1.05 %) | 28 and 32 synthetic invoices, generated before 5.0.7 when their tax numbers and IBANs still passed their check digits: a measure of the rules, not of real invoices. |
| Wrong words removed by `correct="more"`, net, per 100 | 19.5 (14.1 to 26.8) | 10.9 (7.1 to 15.6) | All pages, the default model. |
| The same with the thorough second reading | 21.2 (15.3 to 29.0) | 13.0 (8.6 to 18.3) | `second_reading="thorough"`, all pages. |
| Words `more` broke, as a share of those it fixed | 4.99 % (57 of 1,143) | 7.62 % (38 of 499) | The default model. The limit set before the test was 7.5 %. |

In plain words, the OCR reads about 97 of every 100 words right. Reviewing the flagged 5 % of a Spanish document finds
between 5 and 9 of every 10 wrong words, about 7 at the point estimate, and about 4 of every 10 words you look at turn out
to be wrong; the wrong words that are not flagged are not pointed out. `safe` changed no word of the test, with either
model, so the measured gain is all in `more`. How good the page is matters more than the language: the Spanish word error
rate is 0.77 % (0.55 to 1.08) on lightly damaged documents and 10.51 % (8.08 to 13.59) on heavily damaged ones, and `more`
with the default model removed 6.8 (3.5 to 10.9) wrong Spanish words per 100 on the light ones and 25.9 (17.2 to 36.6) on
the heavy ones.

## Where it fell short

Three criteria set before the test were missed. They are kept in, as misses:

- In English with the default model, `more` broke 7.62 % as many words as it fixed, just over the 7.5 % limit. The thorough
  second reading, which uses its own guard model, stayed inside it at 6.14 % (36 of 586).
- On the English invoice pages, flagging caught 78.0 % (63.1 to 100.0) of the wrong words, against 85.4 % (65.7 to 100.0)
  for version 4, from 41 wrong words.
- On the Spanish invoice pages, the thorough mode is not shown to help: -3.75 wrong words per 100 against the default model
  (-11.31 to +5.48). The test holds too few wrong words on invoice pages (80 in Spanish, 46 in English) to say whether
  `more` helps there at all.

## Speed and memory

| What | Measured | Conditions |
|---|---|---|
| OCR | 3.616 s a page with one process of 4 threads; 3.245 s with two processes of two threads, the default when the memory allows | 20 pages of public-administration documents, `openvino` backend, development laptop |
| A small laptop | 7.4 to 13.7 s a page for the sample documents (5 pages in about 50 s); 18 to 42 s for a page full of text | Intel Celeron N4020, 2 cores, 04/10/2026 |
| Thorough second reading | 2.95 s a page more on average, 95th percentile 5.46 s, up to 6.8 s; 2.64 s on average with the product's memory cap | 60 pilot pages (a fixed set of pages of the corpus) through the real reader process, a clean run on a quiet machine |
| First page of a thorough batch | 17.6 s at the product's memory cap, 7.6 s without a cap, 22.5 s on a loaded machine | The three models load |
| Endurance | The last 100 pages took 0.784 times as long as the first 100 | 553 pages, 4 recycled workers |
| Memory | An OCR worker at most 1,330 MB (the bound is 1,600 MB); the reader process of the thorough mode 1,586 MB at its peak, besides the workers | The 553-page run; the 60 pilot pages |

Plan a batch by its page count and by how much text each page holds, not by its file count. Portability was checked once,
on Linux: on a second laptop (an i3 with 2 cores and 3.3 GB of memory), a new environment with the models downloaded again
took 2 min 16 s with `scripts/verify_install.py`, and its reading of 5 development pages equals, value for value, the one the
development machine stored. Nothing was measured on Windows or macOS.

## Limits

### Files and languages

- PDF only: put a JPG or PNG into a PDF first ([the FAQ](faq.md#can-it-read-images-jpg-or-png) shows how). Any other file
  raises `UnsupportedFile`.
- Spanish and English only, with one OCR engine. `languages` chooses the word lists the guard uses and how numbers are
  read, so a Spanish page read with only English selected is compared with English words: select the languages of your
  documents. Other languages were not measured.
- A PDF that needs a password to open is refused with `PasswordProtected`. One with an owner password that restricts
  copying or changes is read like any other, and its searchable PDF is written without those restrictions.

### What flagging cannot find

- A word the OCR reads wrong with high confidence, and that is a real word in its place, passes as fine: it fails none of
  the guard's tests of confidence, spelling and neighbours. `ok` means "not flagged", not "verified".
- A word merged with its neighbour or split in two is the hardest case: on the v4 test the flagged 5 % caught 30.8 % of the
  merged words and 28.1 % of the split ones.
- The figures are for public-administration documents and invoices. Other kinds of document were not measured.

### Numbers

- Only Spanish DNI and NIE numbers, IBANs whose national layout fixes every position as a letter or a digit, and the
  arithmetic of an invoice can be `checked`. A CIF, a French, Italian or Swiss IBAN, the withholding amount and an amount
  written with a space as the thousands separator stay `unverified`.
- The VAT rates that pass are Spain's VAT and the Canary Islands' IGIC: 0, 2, 3, 4, 5, 7, 9.5, 10, 15, 20 and 21 %. A rate
  such as 19 % or 22 % comes out `mismatch`, so invoices from other countries show failures that are not misreadings.
- `mismatch` says that a digit is wrong somewhere, not which one, and an IBAN written in groups often has no suggestion. A
  `checked` number can be written with another thousands separator than the printed one: the rules check digits, not
  punctuation.

### Corrections

- `correct` is `off` unless you set it. `more` breaks words too and rarely repairs a merged or split word: of 1,296 merged
  Spanish words it fixed 12 with the default model and 38 with the thorough mode, and none of 169 split ones.
- A page where the planned fixes would change more than 8 % of its words keeps none (the sealed test issued 72 such
  warnings over 772 pages), and a page whose language is not selected gets none. Both say so in a warning.

### Layout and invoices

- The reading order follows the lines across the page, so a line of a two-column page runs from one column into the other.
- Invoice fields are read from page 1 only, so a multi-page invoice is not supported. They come from Spanish and English
  labels, and the currencies Extract knows are EUR, USD and GBP. A layout it has not seen can leave a field `None`: a field
  is never guessed.

### Scans and text layers

- A digital PDF's own text layer is trusted when it reads as text. If it differs from what is printed, that is not
  detected.
- A page upside down or turned a quarter one way (90 degrees in the tests) comes out `unreadable`. Turned a quarter the
  other way (270 degrees in the tests), it is read word by word but its lines come out in the wrong order, with a page note.
  Rotate the PDF and read it again.
- A scan under 100 dpi is read at 100 dpi, as it is, with a page note: it may read well or badly, and nothing in its words
  says which. Scan it again at 150 dpi or more.
- In the searchable PDF only, characters outside the Western European set (Windows 1252) appear as `?` in the hidden text
  layer. The export and the JSON keep the right characters.

### Memory and platforms

- The OCR runs in up to 2 worker processes, each budgeted 1,500 MB with the default `openvino` backend and 700 MB with
  `onnxruntime`. With less than 600 MB free no worker starts until memory recovers, and the read says so with `ram.low`. The
  thorough second reading adds a reader process of up to 1.6 GB. On a machine with 3 or 4 GB, set
  `Options(processes=1, backend="onnxruntime")` and do not run two reads at once.
- The full test suite has been run on Linux with Python 3.10 only. Python 3.12 on Linux was installed and a document was
  read with it, but the suite was not run there. Windows has a lock file that was never run, and nothing was measured on
  Windows; macOS has no lock file and is not supported.
- The interface listens on `127.0.0.1` and opens only folders inside your home folder.

## Next

[Troubleshooting](troubleshooting.md) covers the failures you can hit, with the exact messages, and the [FAQ](faq.md)
answers the questions people ask first.
