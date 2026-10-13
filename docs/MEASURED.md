# Measured figures

The accuracy figures and the main speed figures of Extract, with their 95 % interval. They come from `bench/results/`,
written there by the scripts in `bench/`, and the block below is generated from those files by `scripts/api_docs.py`; a
test fails if this file and the result files disagree. The same figures in plain words, each with its conditions, are on
the [accuracy page](accuracy.md), which also has the figures for the small laptop and the portability check, with the two
files they come from.

T5 is the sealed test of 772 pages. Its public-administration pages are the pages of documents published by public administrations, in Spanish
and English (`bench/README.md` describes the corpus), as against its invoice pages and its unseen-damage pages. The speed rows use 20 pages of public-administration documents.

Method notes: the sealed test was put aside before development and read for the published figures on code frozen
beforehand. A second read of it only added the report tables, after the whole report of the first read had been
recomputed and compared number by number with the stored one (they were equal); the registers `test_reads_v5.json` and
`test_reads_v5en.json` in `bench/results/` list both reads. The v4 test is development data for v5. Intervals are 95 %
by document.

Invoices: the 60 synthetic invoices of the sealed test were generated before 5.0.7, when their tax numbers and IBANs still
passed their check digits. Since 5.0.7 the generator writes invented identifiers that cannot exist (`bench/corpus/invoices.py`),
so a new read of freshly generated invoices has fewer `checked` numbers (the tax numbers and the IBAN are no longer checkable);
the figures below were not measured again, and the number-rule tests complete the invented identifiers in memory so that
the check digits are still tried on every invoice.

<!-- measured:start -->
- OCR speed on 20 pages of public-administration documents (openvino): 3.616 s a page with one process of 4 threads.
- Two processes of two threads: 3.245 s a page (10 % less); this is the default when the memory allows.
- Endurance run of 553 pages with 4 recycled workers: the last 100 pages took 0.784 times the first 100; the largest memory of any worker was 1330 MB, within the 1,600 MB bound.
- The next lines are the v4 test, the detector and the number rules (from the v4 release; that test is development data for v5): the numbers of the sealed test of v5 follow them.
- The guard, 120 test documents of public administrations damaged on purpose (235 pages, 3,400 wrong words among 103,408): flagging 5 % of the words catches 70.3 % (60.1-80.1) of the wrong ones; logistic regression on the same signals 69.5 % (60.0-78.1); the lowest mean OCR confidence 58.9 % (50.4-66.7). The difference with logistic regression is +0.9 points and its interval includes zero.
- Of the words it flags, 46.2 % (35.1-59.4) are wrong. Calibration error 0.006 (heavily damaged pages 0.017).
- Suggestions are shown for 11.3 % (8.0-16.2) of the doubtful plain words and are right 91.1 % (87.5-94.4) of the time.
- Recall at 5 % flagged by damage band: light 40.6 % (22.3-83.1) (658 wrong words), medium 68.8 % (48.9-83.0) (520 wrong words), heavy 42.9 % (32.6-53.4) (2,222 wrong words). The 95 % interval of the pooled recall is ±10.0 points (the design aimed at ±4): 120 documents are not enough to say more.
- What it catches at that setting: 82.4 % of the words with a wrong character (2,455), 92.5 % of the invented words (120), but only 30.8 % of merged words (598) and 28.1 % of split words (210).
- Labels, audited by hand on 120 words: 0 of 60 labelled right and 4 of 60 labelled wrong were mislabelled, all of them artefacts of the text layer.
- Judging a page of 600 words takes a median of 42 ms (95th percentile 208 ms), on top of the OCR.
- Number rules: 702 numbers of 60 read invoices came out `checked`, none with a value different from the printed one (95 % upper bound 0.43 %). A `checked` number can still be written with another thousands separator than the original.
- Invoice fields read from OCR (the generator's own layouts, so not a measure of real invoices): total 100 %, date 98 %, number 93 %, subtotal 92 %, VAT amount 88 %.
- Sealed test T5, read twice on the frozen code (the first read gave the numbers of this block, the second only added the tables below); Spanish and English are reported apart, every interval is 95 % by document, `d WER` is the change of the word error rate in percentage points (negative is better) and `net` is the right words made right minus the right words broken, per 100 wrong words.
- Spanish, public-administration pages: 130 documents, 384 pages, 178,447 words, 4,966 wrong as read (WER 2.78 % (1.97 to 3.78)). With `more`, default model: WER -0.53 (-0.66 to -0.40) points, fixed 995, broken 52, applied 1,277, precision 77.9 % (72.2 to 83.4), net 19.0 (13.2 to 26.8) per 100 wrong words; thorough mode: WER -0.58 (-0.71 to -0.44) points, fixed 1,086, broken 55, applied 1,369, precision 79.3 % (73.9 to 84.4), net 20.8 (14.3 to 29.3) per 100 wrong words; thorough minus default, paired by document: WER -0.049 (-0.072 to -0.028) points, net +1.77 (+0.98 to +2.87) per 100 wrong words.
- Spanish, all pages of the test: 173 documents, 456 pages, 199,646 words, 5,557 wrong as read (WER 2.78 % (2.07 to 3.65)). With `more`, default model: WER -0.54 (-0.67 to -0.43) points, fixed 1,143, broken 57, applied 1,446, precision 79.0 % (73.8 to 83.9), net 19.5 (14.1 to 26.8) per 100 wrong words; thorough mode: WER -0.59 (-0.73 to -0.46) points, fixed 1,241, broken 64, applied 1,549, precision 80.1 % (75.1 to 84.5), net 21.2 (15.3 to 29.0) per 100 wrong words; thorough minus default, paired by document: WER -0.046 (-0.067 to -0.026) points, net +1.64 (+0.88 to +2.64) per 100 wrong words.
- Spanish, invoice pages: 28 documents, 28 pages, 2,299 words, 80 wrong as read (WER 3.48 % (1.65 to 5.77)). With `more`, default model: WER -0.43 (-0.85 to -0.09) points, fixed 10, broken 0, applied 13, precision 76.9 % (58.3 to 100.0), net 12.5 (4.6 to 18.7) per 100 wrong words; thorough mode: WER -0.30 (-0.65 to +0.00) points, fixed 9, broken 2, applied 13, precision 69.2 % (36.4 to 100.0), net 8.8 (0.0 to 18.7) per 100 wrong words; thorough minus default, paired by document: WER +0.130 (-0.132 to +0.479) points, net -3.75 (-11.31 to +5.48) per 100 wrong words.
- Spanish, unseen-damage pages: 15 documents, 44 pages, 18,900 words, 511 wrong as read (WER 2.70 % (1.28 to 5.05)). With `more`, default model: WER -0.70 (-1.11 to -0.38) points, fixed 138, broken 5, applied 156, precision 88.5 % (80.2 to 93.4), net 26.0 (16.6 to 43.6) per 100 wrong words; thorough mode: WER -0.74 (-1.16 to -0.41) points, fixed 146, broken 7, applied 167, precision 87.4 % (78.4 to 93.9), net 27.2 (17.6 to 46.3) per 100 wrong words; thorough minus default, paired by document: WER -0.032 (-0.077 to +0.015) points, net +1.17 (-0.57 to +4.20) per 100 wrong words.
- Spanish, what never changed: `safe` applied 0 fixes to words, with either model, on all the pages; 232 numbers of 28 invoices came out `checked` and 0 of them were false (95 % upper bound 1.29 %); the arithmetic of the invoices made 0 suggestions.
- Spanish, floor of `more` (its WER interval below zero and the words it broke at most 7.5 % of the ones it fixed), all pages: default model 4.99 % (57 of 1143) meets it; thorough mode 5.16 % (64 of 1241) meets it.
- Spanish: the 95 % interval of the thorough net on the public-administration pages is about ±36 % of its value; read the figures as ranges, not as points.
- English, public-administration pages: 100 documents, 284 pages (2 unreadable, counted apart), 145,382 words, 4,172 wrong as read (WER 2.87 % (2.22 to 3.64)). With `more`, default model: WER -0.32 (-0.43 to -0.21) points, fixed 495, broken 35, applied 661, precision 74.9 % (71.0 to 78.5), net 11.0 (7.2 to 15.3) per 100 wrong words; thorough mode: WER -0.38 (-0.51 to -0.25) points, fixed 579, broken 32, applied 746, precision 77.6 % (73.2 to 81.3), net 13.1 (8.6 to 18.0) per 100 wrong words; thorough minus default, paired by document: WER -0.060 (-0.085 to -0.035) points, net +2.09 (+1.21 to +3.07) per 100 wrong words.
- English, all pages of the test: 132 documents, 316 pages (2 unreadable, counted apart), 147,800 words, 4,218 wrong as read (WER 2.85 % (2.20 to 3.61)). With `more`, default model: WER -0.31 (-0.43 to -0.20) points, fixed 499, broken 38, applied 668, precision 74.7 % (70.7 to 78.1), net 10.9 (7.1 to 15.6) per 100 wrong words; thorough mode: WER -0.37 (-0.51 to -0.25) points, fixed 586, broken 36, applied 759, precision 77.2 % (72.6 to 80.8), net 13.0 (8.6 to 18.3) per 100 wrong words; thorough minus default, paired by document: WER -0.060 (-0.087 to -0.037) points, net +2.11 (+1.26 to +3.12) per 100 wrong words.
- English, invoice pages: 32 documents, 32 pages, 2,418 words, 46 wrong as read (WER 1.90 % (0.58 to 4.12)). With `more`, default model: WER -0.04 (-0.46 to +0.24) points, fixed 4, broken 3, applied 7, precision 57.1 % (0.0 to 100.0), net 2.2 (-26.7 to 12.5) per 100 wrong words; thorough mode: WER -0.12 (-0.65 to +0.20) points, fixed 7, broken 4, applied 13, precision 53.8 % (0.0 to 71.4), net 6.5 (-20.0 to 17.6) per 100 wrong words; thorough minus default, paired by document: WER -0.083 (-0.286 to +0.122) points, net +4.35 (-8.33 to +21.05) per 100 wrong words.
- English, what never changed: `safe` applied 0 fixes to words, with either model, on all the pages; 287 numbers of 32 invoices came out `checked` and 0 of them were false (95 % upper bound 1.05 %); the arithmetic of the invoices made 0 suggestions.
- English, floor of `more` (its WER interval below zero and the words it broke at most 7.5 % of the ones it fixed), all pages: default model 7.62 % (38 of 499) misses it; thorough mode 6.14 % (36 of 586) meets it.
- English: the 95 % interval of the thorough net on the public-administration pages is about ±36 % of its value; read the figures as ranges, not as points.
- T5 read 772 pages (456 Spanish, 316 English; the English invoices are counted as English): 13 pages had no doubtful unit and so no second reading to make, 0 lost theirs, 2 English pages were unreadable, and the guard issued 72 `fixes_withheld` warnings (a page whose fixes would change more than 8 % of its words keeps none).
- Thorough mode cost (`second_reading="thorough"`, 16 lines and 40 zones, its own guard model), 60 pilot pages (a fixed set of pages of the corpus) through the real reader process, in a clean run on a quiet machine without the memory cap: 2.95 s a page on average, 95th percentile 5.46 s, up to 6.8 s; the first page of a batch 7.6 s (the three models loading). With the product's memory cap of 1,600 MB the same pages took 2.64 s on average (95th percentile 5.07 s, up to 5.8 s) and the first page 17.6 s; in a run on a loaded machine, with a cap of 1,700 MB, the first page took 22.5 s. The reader process peaked at 1,586 MB over the runs, besides the OCR workers; a page that goes over its cap of 1,600 MB loses its second reading and is listed in `doc.fixes["second_reading_failed_pages"]`.
- The first attempt to read T5 was stopped by a gate that took the pages with no doubtful unit for lost second readings; the frozen tree was frozen again before the first read, and no number had been seen.
- `more` is off by default: `correct` is `off` unless it is asked for, and `safe` applies no word in the sealed test (digits change only through the control rule of a number).
- Development pages, out of fold (not the test): net gain of the thorough mode over no second reading, Spanish 163 fixes (98-241), 2.28 per 100 wrong words; English 123 fixes (82-166), 2.96 per 100 wrong words. On T5 the thorough mode gained 1.64 (Spanish) and 2.11 (English) per 100 wrong words over the default model: the two figures compare different things (no second reading, the default model) and the development one was the more optimistic.
- Second read of the sealed test: it added the tables below and nothing else. Before anything was written the whole report of the first read was recomputed and compared number by number with the stored one; they were equal, so the headline numbers above are the first read's. Why a second read: the tool of the first read did not keep the tables fixed in advance (by damage band, page source, kind of component, edit operation, kind and source, and the detector against v4); the omission was the tool's and no number was chosen after seeing one. Both reads are in the registers `test_reads_v5.json` and `test_reads_v5en.json`: 2 and 2 metrics reads, the second with the reason second-read-tables.

Spanish, `more`, all pages of the test; intervals are 95 % by document.

| damage band | documents | WER as read | default: d WER (points) | default: net per 100 wrong | thorough: d WER (points) | thorough: net per 100 wrong | thorough minus default: net per 100 wrong |
|---|---|---|---|---|---|---|---|
| heavy | 46 | 10.51 % (8.08 to 13.59) | -2.72 (-3.17 to -2.20) | 25.9 (17.2 to 36.6) | -2.95 (-3.46 to -2.36) | 28.0 (18.5 to 39.8) | 2.2 (0.9 to 3.8) |
| light | 130 | 0.77 % (0.55 to 1.08) | -0.05 (-0.08 to -0.03) | 6.8 (3.5 to 10.9) | -0.07 (-0.10 to -0.05) | 9.4 (5.4 to 14.3) | 2.6 (0.8 to 4.7) |
| medium | 98 | 2.92 % (1.46 to 5.01) | -0.42 (-0.55 to -0.30) | 14.3 (7.7 to 28.3) | -0.44 (-0.58 to -0.31) | 15.0 (8.1 to 29.3) | 0.7 (-0.4 to 1.9) |
| unseen | 15 | 2.70 % (1.28 to 5.05) | -0.70 (-1.11 to -0.38) | 26.0 (16.6 to 43.6) | -0.74 (-1.16 to -0.41) | 27.2 (17.6 to 46.3) | 1.2 (-0.6 to 4.2) |

| page source | documents | WER as read | default: d WER (points) | default: net per 100 wrong | thorough: d WER (points) | thorough: net per 100 wrong | thorough minus default: net per 100 wrong |
|---|---|---|---|---|---|---|---|
| public-administration pages | 145 | 2.78 % (2.02 to 3.66) | -0.55 (-0.67 to -0.43) | 19.6 (14.1 to 27.0) | -0.59 (-0.73 to -0.46) | 21.4 (15.4 to 29.5) | 1.7 (0.9 to 2.8) |
| invoice pages | 28 | 3.48 % (1.65 to 5.77) | -0.43 (-0.85 to -0.09) | 12.5 (4.6 to 18.7) | -0.30 (-0.65 to 0.00) | 8.8 (0.0 to 18.7) | -3.8 (-11.3 to 5.5) |

| component kind | documents | WER as read | default: d WER (points) | default: net per 100 wrong | thorough: d WER (points) | thorough: net per 100 wrong | thorough minus default: net per 100 wrong |
|---|---|---|---|---|---|---|---|
| pair | 173 | 1.98 % (1.45 to 2.63) | -0.53 (-0.65 to -0.42) | 26.8 (19.5 to 37.0) | -0.56 (-0.70 to -0.44) | 28.5 (20.7 to 39.3) | 1.7 (0.7 to 2.9) |
| merge | 122 | 100.00 % (100.00 to 100.00) | -0.93 (-2.21 to -0.17) | 0.9 (0.2 to 2.2) | -2.93 (-5.57 to -1.32) | 2.9 (1.3 to 5.6) | 2.0 (0.9 to 3.9) |
| split | 24 | 100.00 % (100.00 to 100.00) | 0.00 (0.00 to 0.00) | 0.0 (0.0 to 0.0) | 0.00 (0.00 to 0.00) | 0.0 (0.0 to 0.0) | 0.0 (0.0 to 0.0) |
| mixed | 4 | 100.00 % (100.00 to 100.00) | 0.00 (0.00 to 0.00) | 0.0 (0.0 to 0.0) | 0.00 (0.00 to 0.00) | 0.0 (0.0 to 0.0) | 0.0 (0.0 to 0.0) |
| invented | 53 | 100.00 % (100.00 to 100.00) | -14.19 (-22.66 to -7.97) | 14.2 (8.0 to 22.7) | -14.19 (-21.88 to -8.28) | 14.2 (8.3 to 21.9) | 0.0 (-4.4 to 4.4) |

A document is counted in every damage band it has pages in, so the band table adds up to 289 for the 173 documents of the language. In the component-kind table `WER as read` is 100 % for merge, split, mixed and invented by definition (every word in those components is wrong): read their net columns.

Rows with fewer than 50 fixed words show counts only; `documents` in a family's row is the language's.

| default, `more`: applied edits by family | fixed | broken | changed, still wrong | precision | broken as a share of fixed | net per 100 wrong |
|---|---|---|---|---|---|---|
| kind=accent | 656 | 25 | 55 | 89.1 % (82.4 to 93.9) | 3.8 % (2.1 to 6.0) | 11.4 (8.2 to 15.4) |
| kind=accent_case | 0 | 0 | 1 | - | - | - |
| kind=case | 1 | 0 | 2 | - | - | - |
| kind=letters | 414 | 27 | 139 | 71.4 % (64.7 to 77.8) | 6.5 % (4.0 to 9.5) | 7.0 (4.8 to 10.0) |
| kind=punct | 39 | 3 | 35 | - | - | - |
| kind=structure | 33 | 2 | 14 | - | - | - |
| op=drop | 21 | 1 | 8 | - | - | - |
| op=join | 0 | 1 | 2 | - | - | - |
| op=segment | 12 | 0 | 4 | - | - | - |
| op=text | 1,110 | 55 | 232 | 79.5 % (73.9 to 84.4) | 5.0 % (3.5 to 6.6) | 19.0 (13.6 to 26.0) |
| source=ctc_alt | 8 | 5 | 35 | - | - | - |
| source=dictionary | 1,063 | 45 | 163 | 83.6 % (78.9 to 87.9) | 4.2 % (2.8 to 5.8) | 18.3 (13.2 to 25.0) |
| source=drop | 21 | 1 | 8 | - | - | - |
| source=fold | 0 | 0 | 1 | - | - | - |
| source=join_edit | 0 | 0 | 1 | - | - | - |
| source=joined | 0 | 1 | 1 | - | - | - |
| source=page | 1 | 2 | 0 | - | - | - |
| source=punct | 38 | 3 | 33 | - | - | - |
| source=segment | 12 | 0 | 4 | - | - | - |

| thorough, `more`: applied edits by family | fixed | broken | changed, still wrong | precision | broken as a share of fixed | net per 100 wrong |
|---|---|---|---|---|---|---|
| kind=accent | 656 | 28 | 43 | 90.2 % (83.3 to 94.7) | 4.3 % (2.6 to 6.3) | 11.3 (8.2 to 15.4) |
| kind=accent_case | 0 | 0 | 1 | - | - | - |
| kind=case | 8 | 0 | 1 | - | - | - |
| kind=letters | 438 | 19 | 126 | 75.1 % (68.1 to 81.1) | 4.3 % (2.6 to 6.7) | 7.5 (5.1 to 10.9) |
| kind=punct | 80 | 15 | 39 | 59.7 % (43.5 to 74.6) | 18.8 % (9.1 to 31.9) | 1.2 (0.7 to 1.8) |
| kind=structure | 59 | 2 | 34 | 62.1 % (48.6 to 75.8) | 3.4 % (0.0 to 10.4) | 1.0 (0.6 to 1.6) |
| op=drop | 21 | 0 | 8 | - | - | - |
| op=join | 0 | 1 | 4 | - | - | - |
| op=segment | 38 | 1 | 22 | - | - | - |
| op=text | 1,182 | 62 | 210 | 81.3 % (76.3 to 85.6) | 5.2 % (3.9 to 6.8) | 20.2 (14.5 to 27.6) |
| source=ctc_alt | 18 | 9 | 17 | - | - | - |
| source=dictionary | 1,073 | 37 | 141 | 85.8 % (81.0 to 89.8) | 3.4 % (2.2 to 4.9) | 18.6 (13.4 to 25.5) |
| source=drop | 21 | 0 | 8 | - | - | - |
| source=fold | 1 | 0 | 0 | - | - | - |
| source=join_edit | 0 | 0 | 1 | - | - | - |
| source=page | 3 | 0 | 2 | - | - | - |
| source=punct | 72 | 13 | 34 | 60.5 % (43.3 to 75.8) | 18.1 % (8.5 to 31.0) | 1.1 (0.6 to 1.7) |
| source=reading:deep | 21 | 3 | 17 | - | - | - |
| source=reading:latin | 16 | 1 | 6 | - | - | - |
| source=reading:wide | 0 | 0 | 1 | - | - | - |
| source=segment | 16 | 1 | 17 | - | - | - |

Spanish invoice pages hold 80 wrong words: too few to tell whether `more` helps them, so no claim is made for them.
Spanish, kinds of edit whose broken words pass 7.5 % of the fixed ones (the floor is judged on all the pages, not per kind; rows with at least 50 fixed words): thorough kind=punct 18.8 % (15 of 80).
Spanish, what structure fixing reaches (`more`): merge: 12 fixed with the default model and 38 with the thorough mode of 1,296 wrong words; split: 0 fixed with the default model and 0 with the thorough mode of 169 wrong words; mixed: 0 fixed with the default model and 0 with the thorough mode of 17 wrong words; invented: 21 fixed with the default model and 21 with the thorough mode of 148 wrong words.

| Spanish, review budget 5 % of the words | recall | precision | minus v4: recall | criterion (recall at most the allowed loss below v4) |
|---|---|---|---|---|
| v4 detector | 63.7 % (45.8 to 81.5) | 34.4 % (27.4 to 42.6) | - | - |
| default model's detector | 68.6 % (49.7 to 86.5) | 37.1 % (29.5 to 45.4) | 4.88 points (-0.05 to 9.68) | met (v5 4.9 points higher than v4; allowed loss 1.0) |
| thorough model's detector | 68.6 % (49.7 to 86.5) | 37.1 % (29.5 to 45.4) | 4.88 points (-0.05 to 9.68) | met (v5 4.9 points higher than v4; allowed loss 1.0) |

Spanish detector: not worse than the v4 detector (the criterion is met; the interval of the difference includes zero, so not better either). The default model and the thorough mode use the same detector: their figures are identical.
By slice: The criterion is met on public-administration pages, unseen-damage pages, invoice pages by the point estimate; the interval of the difference clears the allowed loss on public-administration pages, unseen-damage pages and not on invoice pages.

The raw OCR's word error rate on the same Spanish pages: 2.78 % (2.07 to 3.65) (173 documents, 199,475 rows, 5,386 wrong words).


English, `more`, all pages of the test; intervals are 95 % by document.

| damage band | documents | WER as read | default: d WER (points) | default: net per 100 wrong | thorough: d WER (points) | thorough: net per 100 wrong | thorough minus default: net per 100 wrong |
|---|---|---|---|---|---|---|---|
| heavy | 35 | 14.08 % (10.79 to 18.19) | -1.84 (-2.34 to -1.26) | 13.1 (7.4 to 20.8) | -2.08 (-2.64 to -1.42) | 14.8 (8.2 to 23.5) | 1.7 (0.6 to 3.3) |
| light | 112 | 1.00 % (0.76 to 1.30) | -0.05 (-0.07 to -0.03) | 5.3 (3.0 to 8.2) | -0.06 (-0.09 to -0.04) | 6.5 (3.9 to 9.9) | 1.2 (0.0 to 2.5) |
| medium | 77 | 1.38 % (1.04 to 1.79) | -0.12 (-0.17 to -0.08) | 8.9 (5.8 to 12.6) | -0.19 (-0.25 to -0.14) | 13.8 (10.3 to 18.4) | 5.0 (2.9 to 7.3) |

| page source | documents | WER as read | default: d WER (points) | default: net per 100 wrong | thorough: d WER (points) | thorough: net per 100 wrong | thorough minus default: net per 100 wrong |
|---|---|---|---|---|---|---|---|
| public-administration pages | 100 | 2.87 % (2.22 to 3.64) | -0.32 (-0.43 to -0.21) | 11.0 (7.2 to 15.3) | -0.38 (-0.51 to -0.25) | 13.1 (8.6 to 18.0) | 2.1 (1.2 to 3.1) |
| invoice pages | 32 | 1.90 % (0.58 to 4.12) | -0.04 (-0.46 to 0.24) | 2.2 (-26.7 to 12.5) | -0.12 (-0.65 to 0.20) | 6.5 (-20.0 to 17.6) | 4.3 (-8.3 to 21.1) |

| component kind | documents | WER as read | default: d WER (points) | default: net per 100 wrong | thorough: d WER (points) | thorough: net per 100 wrong | thorough minus default: net per 100 wrong |
|---|---|---|---|---|---|---|---|
| pair | 132 | 2.23 % (1.65 to 2.91) | -0.31 (-0.44 to -0.20) | 14.0 (8.9 to 20.6) | -0.36 (-0.50 to -0.24) | 16.3 (10.6 to 23.6) | 2.3 (1.3 to 3.5) |
| merge | 84 | 100.00 % (100.00 to 100.00) | 0.00 (0.00 to 0.00) | 0.0 (0.0 to 0.0) | -2.23 (-4.24 to -0.63) | 2.2 (0.6 to 4.2) | 2.2 (0.6 to 4.2) |
| split | 49 | 100.00 % (100.00 to 100.00) | 0.00 (0.00 to 0.00) | 0.0 (0.0 to 0.0) | 0.00 (0.00 to 0.00) | 0.0 (0.0 to 0.0) | 0.0 (0.0 to 0.0) |
| mixed | 7 | 100.00 % (100.00 to 100.00) | 0.00 (0.00 to 0.00) | 0.0 (0.0 to 0.0) | 0.00 (0.00 to 0.00) | 0.0 (0.0 to 0.0) | 0.0 (0.0 to 0.0) |
| invented | 36 | 100.00 % (100.00 to 100.00) | 0.00 (0.00 to 0.00) | 0.0 (0.0 to 0.0) | 0.00 (0.00 to 0.00) | 0.0 (0.0 to 0.0) | 0.0 (0.0 to 0.0) |

A document is counted in every damage band it has pages in, so the band table adds up to 224 for the 132 documents of the language. In the component-kind table `WER as read` is 100 % for merge, split, mixed and invented by definition (every word in those components is wrong): read their net columns.

Rows with fewer than 50 fixed words show counts only; `documents` in a family's row is the language's.

| default, `more`: applied edits by family | fixed | broken | changed, still wrong | precision | broken as a share of fixed | net per 100 wrong |
|---|---|---|---|---|---|---|
| kind=accent | 70 | 3 | 9 | 85.4 % (69.5 to 97.0) | 4.3 % (0.0 to 11.1) | 1.6 (1.1 to 2.3) |
| kind=case | 1 | 2 | 4 | - | - | - |
| kind=letters | 367 | 28 | 95 | 74.9 % (70.6 to 78.8) | 7.6 % (4.8 to 10.2) | 8.0 (4.9 to 11.7) |
| kind=punct | 61 | 5 | 23 | 68.5 % (54.5 to 81.3) | 8.2 % (1.6 to 19.2) | 1.3 (0.7 to 2.1) |
| op=text | 499 | 38 | 131 | 74.7 % (70.7 to 78.1) | 7.6 % (5.2 to 10.1) | 10.9 (7.1 to 15.6) |
| source=ctc_alt | 6 | 2 | 20 | - | - | - |
| source=dictionary | 434 | 32 | 88 | 78.3 % (74.8 to 81.5) | 7.4 % (5.0 to 9.9) | 9.5 (6.1 to 13.6) |
| source=fold | 0 | 0 | 1 | - | - | - |
| source=page | 4 | 0 | 0 | - | - | - |
| source=punct | 55 | 4 | 22 | 67.9 % (52.4 to 82.3) | 7.3 % (0.0 to 19.2) | 1.2 (0.6 to 2.0) |

| thorough, `more`: applied edits by family | fixed | broken | changed, still wrong | precision | broken as a share of fixed | net per 100 wrong |
|---|---|---|---|---|---|---|
| kind=accent | 70 | 2 | 10 | 85.4 % (69.2 to 97.0) | 2.9 % (0.0 to 8.5) | 1.6 (1.1 to 2.3) |
| kind=case | 9 | 6 | 5 | - | - | - |
| kind=letters | 402 | 24 | 89 | 78.1 % (73.3 to 82.2) | 6.0 % (3.9 to 7.7) | 9.0 (5.6 to 13.0) |
| kind=punct | 91 | 4 | 25 | 75.8 % (64.0 to 85.3) | 4.4 % (0.9 to 9.7) | 2.1 (1.3 to 3.0) |
| kind=structure | 14 | 0 | 8 | - | - | - |
| op=segment | 14 | 0 | 8 | - | - | - |
| op=text | 572 | 36 | 129 | 77.6 % (73.3 to 81.0) | 6.3 % (4.5 to 8.5) | 12.7 (8.5 to 17.7) |
| source=ctc_alt | 22 | 1 | 14 | - | - | - |
| source=dictionary | 463 | 28 | 86 | 80.2 % (75.9 to 83.7) | 6.0 % (4.2 to 8.1) | 10.3 (6.7 to 14.6) |
| source=fold | 1 | 2 | 3 | - | - | - |
| source=page | 3 | 0 | 0 | - | - | - |
| source=punct | 75 | 3 | 21 | 75.8 % (62.1 to 86.8) | 4.0 % (0.0 to 9.1) | 1.7 (1.0 to 2.6) |
| source=reading:deep | 10 | 2 | 8 | - | - | - |
| source=reading:latin | 6 | 0 | 1 | - | - | - |
| source=reading:wide | 0 | 0 | 2 | - | - | - |
| source=segment | 6 | 0 | 2 | - | - | - |

English invoice pages hold 46 wrong words: too few to tell whether `more` helps them, so no claim is made for them.
English, kinds of edit whose broken words pass 7.5 % of the fixed ones (the floor is judged on all the pages, not per kind; rows with at least 50 fixed words): default kind=letters 7.6 % (28 of 367); default kind=punct 8.2 % (5 of 61).
English, what structure fixing reaches (`more`): merge: 0 fixed with the default model and 14 with the thorough mode of 627 wrong words; split: 0 fixed with the default model and 0 with the thorough mode of 158 wrong words; mixed: 0 fixed with the default model and 0 with the thorough mode of 35 wrong words; invented: 0 fixed with the default model and 0 with the thorough mode of 116 wrong words.

| English, review budget 5 % of the words | recall | precision | minus v4: recall | criterion (recall at most the allowed loss below v4) |
|---|---|---|---|---|
| v4 detector | 72.2 % (66.2 to 77.7) | 39.8 % (30.3 to 50.6) | - | - |
| default model's detector | 77.1 % (72.0 to 81.3) | 42.5 % (32.5 to 53.6) | 4.89 points (1.87 to 7.94) | met (v5 4.9 points higher than v4; allowed loss 1.0) |
| thorough model's detector | 77.1 % (72.0 to 81.3) | 42.5 % (32.5 to 53.6) | 4.89 points (1.87 to 7.94) | met (v5 4.9 points higher than v4; allowed loss 1.0) |

English detector: better than the v4 detector (the criterion is met and the interval of the difference excludes zero). The default model and the thorough mode use the same detector: their figures are identical.
By slice, the criterion is not met on: invoice pages (41 wrong words): v5 78.0 % (63.1 to 100.0) against v4 85.4 % (65.7 to 100.0), difference -7.3 points (-14.5 to 2.2). The criterion is met on public-administration pages by the point estimate; the interval of the difference clears the allowed loss on public-administration pages.

The raw OCR's word error rate on the same English pages: 2.85 % (2.20 to 3.61) (132 documents, 147,653 rows, 4,071 wrong words).

- Portability, on a second, smaller laptop (i3, 2 cores, 3.3 GB): `scripts/verify_install.py` (a new environment, a cold pip cache, the models downloaded again) took 2 min 16 s; the first reading of 5 development pages (set clean5en) made there with the `openvino` backend equals, value for value (words, boxes, confidences, alternatives), the one the development machine stored on 5 of 5 pages (identical; the sha256 of each reading is in `bench/results/portability.json`).
<!-- measured:end -->
