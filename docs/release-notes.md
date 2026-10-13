# Release notes

The changelog gives one line for each change. This page is the long form of the same changes: for each bug, the symptom, the
steps that show it, the cause, the fix and the test that keeps it fixed. The newest version comes first, and the entries of one version
are in the order they were written up. The numbers, such as B46, are the numbers the entries had when they were found.

## Changes since 5.0.6 (version 5.0.7)

Found by the blind review before the public release: the synthetic invoices of the repository carried a tax number, a
company tax number and an IBAN that pass their check digits, so any of them could be a real person's, a real company's or a
real account. Extract goes public, and nothing in what it ships or generates may belong to anyone. The reading engine
did not move: a read of the three sample documents as they were, and of five pages of the test corpus, is byte-identical before and after. The sample documents themselves were then made again.

### B53. Sample data and generated invoices carried identifiers that could exist

- **Symptom.** The saved invoice of `tests/fixtures` (and the Review screenshots) showed a personal tax number, a company one and an IBAN that
  pass their check digits; the invoice generator drew every tax number and IBAN so that they pass; the sample letter named
  a person and a street; the tests used well-known valid identifiers written out.
- **Cause.** The generator wanted the check-digit rules to have something to check, and made identifiers that satisfy them;
  a valid identifier cannot be told from a real one.
- **Fix.** Every identifier written in the repository, and every one the generator writes, has a form that cannot exist:
  a personal tax number is seven digits and a letter and a company one a letter and seven digits (a Spanish tax number has
  nine characters), the IBAN is `ESXX` and eighteen digits (a Spanish one has two check digits there and twenty digits after
  them), and the company and person names, streets and towns come from lists of invented ones that are not on any map.
  The random draws are the same as before, so the amounts, dates and layouts of the corpus do not move. Extract takes
  these for what they are: nothing is flagged and nothing is checked on them. The tests that try the number rules
  complete them in memory to numbers that pass (`bench/corpus/sidecar_inputs.py`, `tests/checks_oracle.py`) and never
  write them down; the examples of the check-digit tests are built when the tests load, not written out. The sample
  letter, the three fixtures (read again with the shipped engine) and the screenshots were remade, and the README, the
  user guide and the generator say that everything in the samples is invented.
- **Test.** `tests/test_public_identifiers.py` reads every text file and PDF text layer of the repository and what the
  generator makes, and fails on any tax number, IBAN, card, phone number, plate or e-mail address that passes its own
  check digits; `tests/test_invoices.py` asserts the generated forms cannot exist. The figures of `docs/MEASURED.md` come from
  invoices generated before this change and were not measured again.

## Changes since 5.0.5 (version 5.0.6)

Found on the day of the release, taking screenshots of a real read of three documents: the Read screen named one document
over the page of another. The cause was not one line but a rule the screens did not keep (what is shown next to a name
comes from that name's document), so every place that shows a document's name, page, words, fixes or notes was
checked, during a read and after it, with several documents, after a reload and with two windows open. One entry per
bug: symptom, steps, cause, fix, test. The tests are in `tests/test_qa_v506.py` (headless Chrome); each fails on 5.0.5.
The reading engine did not move: a read of the three sample documents and of five pages of the test corpus is
byte-identical before and after.

### B46. The Read screen named one document over the page of another

- **Symptom.** After a read of three documents the Read screen said `three-pages.pdf` above a page that was the second
  document's (an invoice). The same happened whenever the page opened or reloaded after the read, or a second window
  was opened on it.
- **Steps.** Read three PDFs; open the interface again in another window (or reload it) and go to the Read tab.
- **Cause.** Two things. The name, the progress and the page tiles are those of the last document the read began, while
  the page below is the last one that finished, so they differ whenever the last document fails or the read is still
  going. And the page was drawn by a function that waits for the page to load and then paints on "the current view":
  a page opened late gets all the "document read" events at once, asks for the first page of each document, and the page
  that arrives last is painted last, whichever document it belongs to.
- **Fix.** Once the read is over, the name, the card that is marked, the page tiles and the page are all those of the last
  document that was read. While the read goes on the page keeps being the previous document's, and a caption above it
  says so ("invoice-scan.pdf - page 1"). A page that arrives is painted on the view it was asked for and only if no newer
  drawing of the screen has asked since. A card of a file whose content is that of a document already open says "Same
  content as ...", because pressing its button opens that document.
- **Test.** `ReadScreen` (3 tests: the pages arrive in the reverse order of the events; the last document fails; two
  files with the same content).

### B47. Choosing a document with fewer pages left the page of the other one on screen

- **Symptom.** In Review, with page 3 of a three-page document open, clicking a one-page document gave "The page image
  could not be loaded" and left the new document selected over the old document's page, with the page buttons of one and
  the words of the other.
- **Steps.** Read a three-page and a one-page document; in Review open page 3 of the first and click the second.
- **Cause.** The page number was kept when the document changed, so page 3 of a one-page document was asked for. The
  request failed and the screen went on drawing the page it held before.
- **Fix.** A different document opens at its first page; a page that the document does not have is its first. The page
  data says which document and page it is, and a screen draws it only next to that document and page.
- **Test.** `ReviewScreen.test_B47_...`. The screen test of the page notes (`section=pagenotes`, 5.0.3) asked for page 2 of a
  one-page document and passed only because the screen then kept page 2 of nothing; it now uses a document with
  three pages.

### B48. Open in Review could draw the page of the document that was open before

- **Symptom.** The "Open in Review" button of a document on the Read screen opened Review with the previous document's
  page on the canvas until the new page loaded, and for good if it could not be loaded; a page that failed to load was
  also remembered as failed until the document was read again.
- **Steps.** Open page 3 of a document in Review, go to the Read screen and press Open in Review on another document
  (with the page of the second one unavailable).
- **Cause.** The new screen drew whatever page it had last shown, and the failed request stayed in the page cache.
- **Fix.** The canvas stays empty until the page of the chosen document is there, the failure is reported, and a page
  that could not be loaded is asked for again the next time.
- **Test.** `ReviewScreen.test_B48_...`.

### B49. A document read again was shown as its old reading; new documents were not listed

- **Symptom.** With Review open, a read of the same document started from another window (or by this one) refreshed the
  numbers and the counts but left the words on the page as the old reading had them: the right panel and the list
  disagreed. A document read while Review was open did not appear in its list until the screen was drawn again, and one
  read while Export was open did not appear there.
- **Steps.** Open a document in Review, accept a word in a second window, read the document again in that window, and look
  at the first one.
- **Cause.** The handler of the "document read" event refreshed the data of the document but not the page being shown.
- **Fix.** Review refreshes its list of documents and, if the document read is the one on screen, its page; Export
  redraws its list.
- **Test.** `ReviewScreen.test_B49_...`.

### B50. The filter of fixed words and the rule picked were carried to another document

- **Symptom.** With "Fixed" switched on in one document, opening a document with no fixed words showed an empty page, and
  the button that would switch the filter off was disabled. A rule picked in one document's number panel stayed picked
  in the next.
- **Steps.** Switch on Fixed in a document with fixes, click a document without; pick a rule and change document.
- **Cause.** Both choices were kept for the whole session, not for the document they were made on.
- **Fix.** The filter is switched off when the document on screen has no fixed words, and the picked rule is dropped when
  the document changes. The button shows whether the filter is on.
- **Test.** `FixedFilter` (1 test).

### B51. A document read after the Export screen was first opened was left unticked there

- **Symptom.** Documents read after the first visit to Export were in its list but not ticked, so exporting "everything"
  left them out without a word.
- **Steps.** Open Export, go back, read another document, open Export, press Export.
- **Cause.** The ticks were set once, from the documents there were at the time.
- **Fix.** A document read later is ticked like the others (a document the person unticked stays unticked).
- **Test.** `ExportList` (1 test).

### B52. A document that failed to read said "Reading" or "Waiting" for ever

- **Symptom.** When a file could not be read (a folder named like a PDF, a file that went away, a PDF that fails to open)
  its card on the Read screen stayed on "Reading" with an empty bar, or on "Waiting", after the read had finished.
- **Steps.** Read three files, one of them a folder named `x.pdf`.
- **Cause.** The error of a file that could not even be opened did not say which file of the read it was about, and the
  screen did not mark a document as failed when its error came.
- **Fix.** Every error of a file carries its position in the read; the card of that file says "Not read (failed)", and a
  read that ends well leaves no card waiting or reading.
- **Test.** `ReadScreen.test_B46_when_the_last_document_could_not_be_read...` (the card), `ErrorsNameTheirDocument` (the
  position in the error).

## Changes since 5.0.4 (version 5.0.5)

Found by a review of the public repository before its release: the connections a read makes were watched with `strace`
in a clean home folder, and every statement of the documents was compared with a run. The reading engine did not move: a
read of the three sample documents and of five pages of the test corpus is byte-identical before and after.

### B45. OpenVINO sent an anonymous usage event to Google Analytics each time it was imported

- **Symptom.** `README.md` and the guide said that nothing is sent anywhere while reading. In a clean home folder, `python -c
  "import openvino"` (and so every read, which imports it in the process that loads the models) looked up
  `www.google-analytics.com` and opened a connection to it. The event holds a random client id, the OpenVINO version, the
  operating system and a count of uses; it never holds a document, a page or a word. OpenVINO also writes
  `~/intel/openvino_ga_cid` and `~/intel/stats` in the home folder.
- **Steps.** With an empty home folder: `strace -f -e trace=connect python -c "import openvino, time; time.sleep(3)"`, or
  replace `openvino_telemetry.utils.sender.TelemetrySender.send` by a function that records its argument and import
  `openvino`: it is called once, with the event.
- **Cause.** `openvino/tools/ovc/__init__.py` sends `("ov", "import", "general_import")` when `openvino` is imported, and
  with no consent file in the home folder its checker takes the absence as consent. Earlier checks looked at Extract's own
  imports and at reads with the network cut, where the send fails without a word; a process that ends within a second
  also ends before the connection.
- **Fix.** `extract.engine.quiet_openvino.import_openvino` imports OpenVINO with `CI=true`, the variable its usage
  reporting honours, and puts the old value back at once; `engine.ocr._patch_rapidocr`, which every process runs before it
  loads a model (the main process, each OCR worker, the reader process of the second reading), calls it first. Nothing is
  written in the home folder and no consent file is created. No other library acts on `CI` while OpenVINO is imported. The
  text read is untouched (parity read byte-identical). `README.md`, the guide and `THIRD-PARTY.md` say so.
- **Test.** `tests/test_openvino_quiet.py` (5 tests): in a subprocess with an empty home folder the engine code is loaded
  with the sender replaced and it is never called, no `intel` folder appears and `CI` has its old value; a control that
  imports `openvino` with no guard must reach the sender, so the test cannot pass for the wrong reason.

### Documents

- The screenshots are captioned as scripted samples (made with a test engine and three prepared fixes), not as live readings.
- The guide no longer says that a tablet on the same network can open Extract (it answers on `127.0.0.1` only); the tablet
  and phone layouts were checked in a browser window of those sizes.
- The five sample pages took about 50 s, not 27 s, on the small two-core laptop (`bench/results/ui-estimate-seconds.json`);
  the README speed line gives the figure of the small laptop beside the development one.
- The sample PDFs are described as a real read shows them.
- The corpus downloaders describe their pace and identity as they are (slow, limited, named in the `User-Agent`).

<!-- end of the 5.0.5 entries -->

## Changes since 5.0.3 (version 5.0.4)

Found by the fourth client QA round of 05/10/2026: installation with Python 3.12, long and heavy use, odd PDFs,
every export opened in what a client uses. One entry per bug: symptom, steps, cause, fix, test. The tests are in
`tests/test_qa_v504.py`; each fails on 5.0.3. The reading engine did not move: a read of the three sample documents and of five pages of the test corpus is
byte-identical before and after.

### B34. A note that holds for every page of a document made one box per page on the Read screen

- **Symptom.** Reading a document whose language is not one of the ticked ones (or any note that holds for each page), the
  Read screen showed one amber box per page ("Page 5: no fix was applied here because the language of the page ..."),
  in the order the workers finished. At 10 pages of 300 there were ten boxes above the progress; at 300 pages the
  progress grid, the bar and the page being read were many screens down, behind a wall of identical text.
- **Steps.** Add a 300-page scan of text that the word lists cannot judge (or tick only the wrong language) and press
  **Read**; look at the Read screen after a minute.
- **Cause.** Each `job.warning` event was pushed as its own box, and the boxes come before the progress.
- **Fix.** Notes with the same text and only the page different are one box, with the lowest page, and a second line that
  counts the other pages and lists the first twelve ("The same note applies to 39 more pages: 2, 3, 4, ..."), in English
  and Spanish. The box list scrolls inside a third of the window. A note with no page (two files with the same content)
  is shown as before.
- **Test.** `WarningsOfManyPages` (1 test in a headless Chrome, 6 checks, `section=warninggroup` of the screen self-test).

### B35. The guide's `doc.field_status(doc.invoice.total)` crashed when the invoice had no total

- **Symptom.** The example of `docs/USAGE.md` (`print(doc.invoice.total, doc.field_status(doc.invoice.total))`) ended in
  `AttributeError: 'NoneType' object has no attribute 'word_ids'` for any invoice whose layout gave no total (the docs say
  such a field is left empty, never guessed), and the same call on any other empty field.
- **Steps.** `doc = extract.load("tests/fixtures/doubtful.extract.json")`; `doc.field_status(doc.invoice.total)`.
- **Cause.** `field_status` took the words of the cell without asking whether there was one.
- **Fix.** A field the invoice does not have is `unverified` (no rule covers it), in the function and in its description in
  `docs/API.md`. The status of a field that exists is worked out as before.
- **Test.** `FieldStatusOfAMissingField` (1 test).

### B36. Two of the three Python examples of the README did not run as written

- **Symptom.** Copying the second or the third example of the README ("several documents", "Corrections") into a file
  gave `NameError: name 'extract' is not defined`: only the first of the three had `import extract`.
- **Steps.** Paste each Python block of `README.md` into its own file and run it (with `scan.pdf` and `mixed.pdf` there).
- **Cause.** The blocks were written to follow one another.
- **Fix.** Each block starts with `import extract`. The examples of the guide and of `docs/API.md` already did.
- **Test.** `ReadmeExamplesAreComplete` (1 test: every Python example of the README, the guide and the API reference
  that uses `extract` imports it). All the examples were also run, one by one, in a clean installation.

### B37. The README never said how to ask for the second reading from Python

- **Symptom.** The README names `second_reading=True` and `second_reading="thorough"` and shows `correct="safe"` as an
  argument of `extract.read`; a client who writes `extract.read("scan.pdf", second_reading="thorough")` gets
  `TypeError: read() got an unexpected keyword argument 'second_reading'`.
- **Steps.** Run that call in a clean installation.
- **Cause.** `second_reading` is a field of `Options` (as `docs/API.md` says); the README did not show a call.
- **Fix.** The Second reading section of the README has the example (`extract.read("scan.pdf", extract.Options(second_reading=
  "thorough", correct="more"))`), run in a clean installation with the reader models fetched: it read and exported.
  The test of B36 covers its import.
- **Test.** `ReadmeExamplesAreComplete.test_B37_...` (the section has the call); the example itself was run as said.

### B39. `--help` of four maintenance scripts ran them, and one of them emptied THIRD-PARTY.md

- **Symptom.** `python3 scripts/third_party.py --help` (the first thing one types to see what a script does) printed the path
  of `THIRD-PARTY.md` and replaced the licence list of the repository with a file that has the models table and no package
  table (16 lines instead of 77). `write_sums.py`, `api_docs.py` and `make_fixtures.py` also ran: they rewrote
  `models/SHA256SUMS`, `docs/API.md` and `docs/MEASURED.md`, and read three PDFs with the OCR to rewrite the saved sample
  documents of `tests/fixtures/`. Separately, `python3 scripts/install.py --help` described the program as "a virtual
  environment, every dependency pinned by hash, the package," and stopped there, and `--venv` had no text.
- **Steps.** In a clone, `python3 scripts/third_party.py --help`, then `git diff --stat`.
- **Cause.** These scripts have no argument parser and run whatever the arguments are; `third_party.py` writes its file
  from the wheels in `dist/wheelhouse`, which only a run of `lock.py` leaves there, so with none it wrote an empty list. The
  installer's help used only the first line of its docstring, which wraps.
- **Fix.** `--help` and `-h` print the script's own text and stop. `third_party.py` stops with "run `python3 scripts/lock.py`
  first" and leaves the file alone when there is no wheel to list. The installer and the bundle builder show their
  whole text (with the commands) and `--venv` says what it does.
- **Test.** `MaintenanceScriptsAndHelp` (3 tests).

### B40. The documents called Python 3.12 untested, although it installs, works offline and reads like 3.10

- **Symptom.** The README, the quick start and the guide said Extract was "tested on Python 3.10 only" and that the
  Python 3.12 lock file was "generated, not installed: nobody has run it", while the installer itself accepts 3.12.
  A client with Python 3.12 (the version of most current Linux distributions) could not tell whether to try.
- **Steps.** In a clean home, with a standalone Python 3.12: `python3 scripts/install.py` (257 s, ends with "OK: 100 % of the
  printed words read"), `python3 scripts/make_offline_bundle.py` and its installation with the network cut (`unshare -rn`, 144 s),
  then a read of the three sample documents and of five pages of the test corpus: every word, box and status is the same as
  on 3.10 (only the last digit of a few confidences differs, below 1e-9).
- **Cause.** The statement was written before anybody had tried it.
- **Fix.** The documents now say what was checked and what was not: the whole test suite has been run on Python 3.10 only; on
  Python 3.12 the installation, the offline bundle and the reading of the samples were checked. Windows is still said to be
  untested, and macOS unsupported.
- **Test.** `PythonVersionsAreSaidAsTested` (1 test: none of the old sentences is left).

### B41. After an export of some formats, an export of all of them numbered each file on its own

- **Symptom.** Export only the text and the JSON of a document, then export all four formats: the PDF and the workbook came
  out as `invoice.searchable.pdf` and `invoice.xlsx` while the text and the JSON were `invoice-2.txt` and
  `invoice-2.extract.json`. The four files of one export did not go together, and the next export of all four was `-3` for
  some and `-2` for others.
- **Steps.** Export `("txt", "json")`, then `("pdf", "xlsx", "json", "txt")` into the same folder.
- **Cause.** Each file looked for the first free name for itself.
- **Fix.** An export takes the first number (the plain name, `-2`, `-3`...) for which none of its files exists and gives it to
  all of them. A file that appears while the export runs is still never overwritten (that file moves to the next number).
  Exports of all four formats, the usual case, are named as before.
- **Test.** `FilesOfOneExportShareTheirNumber` (2 tests).

### B42. "1 values involved are outlined on the page"

- **Symptom.** Select a word that Extract joined from two pieces (Corrections On), or a number whose rule involves one other
  value: the Review panel said "1 values involved are outlined on the page." (and its Spanish
  equivalent, with the plural "valores" for one).
- **Steps.** Open a document with a join fix and select the joined word.
- **Cause.** The text had no singular form; the other counts of the screens (documents, pages, fixes) do.
- **Fix.** `ui.review.involved_one` in English and Spanish, used for a count of 1.
- **Test.** `OneValueInvolved` (1 test: the key exists in both languages and the screen asks for it).

### B43. A reader process that stopped answering was waited for for ever

- **Symptom.** A full run of the test suite on a computer at load 16 hung for 47 minutes in the test of the reader that goes
  over its memory cap: the parent waited for the answer of a reader process that was asleep (0 % of the CPU) and never
  killed it. A reader of the second reading that deadlocks in a real read would stop the whole read the same way.
- **Steps.** Give the reader of the second reading a page whose work never ends (the test makes the fake reader sleep 600 s)
  without letting its memory pass the cap, and drain it.
- **Cause.** The process was killed when it died on its own or when its memory passed the hard cap; a process that was
  neither was waited for in `collect_pending`/`drain`, one second at a time, without end.
- **Fix.** The process that has a page waiting and has sent nothing for `STALL_SECONDS` (300 s; the longest page of the
  measurements took under 10 s, 60 s with the memory under pressure) is asked to stop, then killed, each wait bounded;
  its page comes back without readings (as for a crash), a new process takes the pages that wait, and the run goes on.
  The kill at the memory cap now also forces the end of a process that ignores the first request. The text read is untouched
  (a page without readings is the page of the first reading); parity read byte-identical.
- **Test.** `StalledReaderProcess` (1 test: the stall is 3 s there; the page comes back bare in under 40 s).

<!-- end of the 5.0.4 entries -->

## Changes since 5.0.2 (version 5.0.3)

Found by the third client QA of 04/10/2026: the three items left open by the second QA, then a new pass
(install from zero with a clean home, every screen and size, bad paths, privacy watched with `ss`). One entry per bug:
symptom, steps, cause, fix, test. The tests are in `tests/test_qa_v503.py`; each fails on 5.0.2. The reading engine did
not move: a read of the three sample documents and of five pages of the test corpus is byte-identical before and after (the
text of every page is unchanged).

### B44. A word without digits was called "a number not written in a valid way"

- **Symptom.** In the Review panel some doubtful words with no digit at all (an accented word misread, for instance) gave as their
  reason "The number is not written in a valid way." (10 of 58 such reasons in a 300-page read).
- **Steps.** Read a scan with a doubtful accented word and select it in Review.
- **Cause.** The reasons come from how much each signal pushed the word's score, and the model can credit the digit signals
  with a push on a word that has none.
- **Fix.** The two reasons about digits ("number not written in a valid way", "letters and digits mixed") are given only to
  words that hold a digit; the next reason takes their place. Which words are doubtful does not change, only the words of
  the reason.
- **Test.** `test_a_word_without_digits_is_never_a_badly_written_number` and `test_a_word_with_digits_keeps_the_number_reasons`
  in `tests/test_guard_reasons.py` (the first fails on 5.0.3).

### B16. A PDF with no pages was called password protected after a locked file

- **Symptom.** After a password protected PDF was refused, a PDF with no pages was refused as "protected by a password"
  too (and every one after it, until another file failed with a code of its own). The advice, "remove the password",
  cannot help with that file.
- **Steps.** Add `locked.pdf`, then a PDF with zero pages (`pikepdf.new().save("zero.pdf")`), in one session or in one
  `extract.inspect` loop.
- **Cause.** pdfium keeps the code of the last failed load, a successful load does not reset it, and a PDF with no pages
  fails without setting a code, so `open_pdf` read the stale one.
- **Fix.** `open_pdf` first makes pdfium forget it with a load of bytes that are not a PDF (format error). The pages of
  every other file are read exactly as before.
- **Test.** `StalePdfiumError` (2 tests).

### B17. The invoice number (and the date) could hold a whole sentence

- **Symptom.** In a letter that begins "Factura numero 2026-0042 emitida el 14/03/2026 ...", the Fields tab, the Fields
  sheet of the xlsx and the JSON gave the invoice `number` as the whole row after the label (`2026-0042 emitida el
  14/03/2026 ...`, eight words). The `date` took the same shape after "Fecha".
- **Steps.** Read `tests/fixtures/doubtful.pdf` and open Review, Fields; or `extract.read(...).invoice.number.text`.
- **Cause.** After the label, the field took every word of the row up to a wide gap, and only asked for one digit
  anywhere in them.
- **Fix.** The number is the leading words that can be one (a digit in them, or only signs, or a short capital prefix
  such as `F` or `INV`; at most 4 words and 30 characters) and the date the leading words that can be one (numbers,
  month names, `de`/`of`); the first word of a sentence ends the value, and no value is left when none of the words
  is one. Words with a misread character ( `.09/27/2025`, `,a-2025-7230`, `25de`) still count. The OCR text is not
  touched, only how much of the row is taken. The two saved documents made from that letter (`doubtful`,
  `unreadable`) carry the number `2026-0042`. On the validation and training invoices the extraction figures are
  identical before and after (`bench.measure_extraction`, byte for byte).
- **Test.** `InvoiceFieldsAreNotSentences` (6 tests).

### B18. A scan under 100 dpi and a page turned a quarter read as "done", against the README

- **Symptom.** The README and the guide said a page turned 90, 180 or 270 degrees, or scanned under about 100 dpi, "comes
  out unreadable". In fact a 70 dpi scan and a page turned 270 degrees came out `done` with no word about it. The 270 page
  was read word by word rightly, but its lines run as columns, so the text file had the last line first.
- **Steps.** `extract.read` a scan of 70 dpi, or a page whose image is turned 270 degrees (the 90 and 180 degree pages did
  come out `unreadable`, with a mean confidence of 0.48 and 0.55).
- **Cause.** The reader raises any scan to 100 dpi and measures nothing else about it, and nothing looked at the
  direction of the lines; the README described what was expected, not what the measurements showed.
- **Fix.** The text the engine reads is the same (parity read, byte-identical). Two notes are added to the page, never
  to a word: `page.warning.low_resolution` (the real resolution of the scan, from `scan_dpi`, which only reports) and
  `page.warning.sideways` (40 % or more of the words of four letters or more are at least 1.5 times taller than wide; none of 580 pages of the development set that are the right way up reaches it, and the 3 it flags are really turned: a table and a Boletin page printed sideways). They are events
  (the Read screen shows them), `doc.page_notes`, `options.page_notes` of the JSON and a note in the Review panel, in
  English and Spanish. README, USAGE and the guide now say what happens: the upside down and the 90 degree page are
  `unreadable` through their low confidence, the 270 degree page is read as columns, a low scan is read as it is.
- **Test.** `LowResolutionAndSideways` (6 tests).

### B19. Text from a PDF that starts with "=" became a formula in the Excel file

- **Symptom.** A word, an invoice field or a corrected text such as `=HYPERLINK("http://...","click")` was written to the
  Review, Fields or Corrections sheet as a formula, which Excel runs when the file is opened (a PDF made by anybody could
  send a cell value to a web address, or ask for a confirmation to run a program).
- **Steps.** Read a PDF whose text holds `=HYPERLINK(...)` (or type that as a correction), export the xlsx, open it in
  Excel or `openpyxl.load_workbook`: the cell has `data_type == "f"`.
- **Cause.** `openpyxl` stores any string that starts with `=` as a formula; the sheets were filled with the text as read.
- **Fix.** After the sheets are written, any formula cell is made a text cell, so the file shows exactly the text. Nothing
  in these sheets is meant to be a formula.
- **Test.** `WorkbookHoldsNoFormulas` (1 test).

### B20. A typed folder was searched where the server was started; quotes, spaces and odd paths broke it

- **Symptom.** Typing `Documents/invoices` (or any relative name) in the Folder box answered "not found" with a path inside
  the installation folder; `..` listed the parent of the installation. A path pasted with quotes (the "copy as path" of
  Windows and of many file managers) or with a space at the end was "not found". A path of 300 characters in one name, or
  `~someone` who does not exist, showed "An internal error happened (OSError ... File name too long)".
- **Steps.** In Import type `Documents`, then `"~/Documents"` (with the quotes), then a very long name.
- **Cause.** `State.path` resolved a relative path against the working folder of the server and did not catch the
  `OSError` and `RuntimeError` of `resolve`, `exists` and `expanduser`.
- **Fix.** Spaces around the text and one pair of quotes are dropped, a relative path starts from the home folder, and a
  path the system cannot look up is "not found". Nothing gets outside the home folder: `..`, `../..` and `/etc` still
  answer "outside your home folder".
- **Test.** `TypedPaths` (4 tests).

### B21. The fix of B3 made poppler (Evince, Okular, `pdftotext`) join 5 to 8 % of the neighbouring words

- **Symptom.** In the searchable PDF of a real letter, `pdftotext -raw` gave `Facturanumero`, `el14/03/2026`, `EUR,IVA`: 8 of
  95 words on one page and 5 of 81 on another were joined to a neighbour, so copied text and a search for the pair
  failed in the viewers built on poppler. Chrome and Edge (pdfium) were right, which is what B3 had measured.
- **Steps.** Export a read of `tests/fixtures/doubtful.pdf`, then `pdftotext -raw doubtful.searchable.pdf -`.
- **Cause.** B3 added a space after every word, scaled to the width of the word without it, so the space stuck out over
  the start of the next word and poppler dropped it. The test of B3 used a grid of evenly spaced words, which never
  overlap.
- **Fix.** The scale is worked out for the word and its space together, so the space ends inside the box of its word and
  nothing overlaps. Both viewers now give the words as they are (95 of 95 and 81 of 81, and every pair found by pdfium).
- **Test.** `SearchableTextLayerOfRealWords` (1 test, on the real boxes of the letter).

### B22. A warning on the Read screen did not say which document it was about

- **Symptom.** With several documents in one read, a line such as "Page 1: ..." appeared on the Read screen with no file
  name, so the user could not tell which of five documents it meant.
- **Steps.** Read five documents of which one has a page that gets a note or a withheld fix.
- **Cause.** The warning events carry the index of their document, but the screen showed only the translated text.
- **Fix.** With more than one document the line starts with the name of the document. The "same content" warning, which
  names its own two files, is left as it was. (The Review panel notes of B18 are tested in the same section.)
- **Test.** `PageNotesOnScreen` (1 test in a headless Chrome, 6 checks).

### B23. The time estimate said "on this PC" and was measured on another one

- **Symptom.** Import showed "Roughly 1 min 57 s to 5 min 50 s for 106 pages on this PC". On a two-core Celeron N4020 (the
  small laptop of the guide) the same 106 pages took 11 minutes, 6.1 s a page against an upper figure of 3.3 s.
- **Steps.** Install on a slow computer, add 100 scanned pages, look at the line under **Read**, read, compare.
- **Cause.** The two figures per page (1.1 s and 3.3 s) are constants measured on the development laptop; nothing
  adapted them, and the text claimed they were for this computer.
- **Fix.** The server learns the speed of the computer from the job log of each read (median OCR seconds of a page,
  shared among the workers, against the mean of the two constants; at least 3 OCR pages; kept in `speed.json` of the work
  folder, averaged with earlier reads). The estimate scales with it, and the line says "on this PC" only after that; before
  the first read it says "on a typical laptop; this PC may be slower or faster, and the figure adjusts after your first
  read". The time left on the Read screen, which comes from the last pages, was already right.
- **Test.** `EstimateFollowsTheComputer` (4 tests, one in a headless Chrome).

### B24. The installer's last line could not be pasted when the folder has a space in its name

- **Symptom.** In a folder such as `My Extract/extract`, the installer ended with `Start the interface with:
  /srv/My Extract/extract/.venv/bin/python -m extract.ui`, which a shell reads as a command named `/srv/My`.
  The `+ command` lines it echoes while it works had the same problem, and so did the failure message.
- **Steps.** Clone or unpack into a folder with a space or an accent, run `python3 scripts/install.py`, copy the last line.
- **Cause.** The commands were joined with plain spaces.
- **Fix.** `shown()` quotes what needs it (`shlex.join`, `list2cmdline` on Windows); a plain path comes out as before.
  The installation itself worked in such a folder (checked on a clean home).
- **Test.** `InstallerNamesPathsYouCanPaste` (2 tests).

### B25. Reading a file again threw away the user's corrections without a word

- **Symptom.** After accepting or typing over words of `doubtful.pdf` in Review, adding the same file again and pressing
  **Read** (the guide recommends exactly that after a wrong language choice or a poor reading) replaced the open document
  by the new reading: the typed word was the OCR's again and the correction mode went back to the one on screen. No
  question, no note.
- **Steps.** Read a file, correct a word, Import, add the same file, Read, open Review.
- **Cause.** `State.add` replaces the document of the same content, as it should, but nothing looked at what the user had
  done in the old one.
- **Fix.** Before a read starts, the server finds the files that are open as documents with a word accepted or typed
  over (or a number accepted) and refuses with `error.edits_would_be_lost` and their names. The screen asks, naming the
  files, and sends the read again with `replace` only after the yes; after a no nothing is read and the work stays. A
  document with nothing changed is read again without a question, as before.
- **Test.** `ReadingAgainKeepsTheUsersWork` (4 tests, one of them in a headless Chrome).

### B26. A second tab on the same document kept showing, and acting on, what it had seen when it opened

- **Symptom.** With Extract open in two tabs, switching the correction mode, or accepting words, in tab A left tab B
  showing the old mode and the old list of words to check. In B, pressing the button for the mode A had left (for example
  **More** when A had gone to **Safe** and B still showed **More**) did nothing, because B believed it was already there.
  Everything the server exports was right; only what B showed and sent was stale.
- **Steps.** Open the page twice, read and open a document, press **Safe** in one tab, go to the other and look at it.
- **Cause.** A tab loads its documents when it opens and after its own actions; nothing told it about another tab's.
- **Fix.** When a tab gets the focus back it reads the documents from the server, drops the page images and word lists of
  those that differ, and redraws only if something differs (so a typed correction is not lost for nothing).
- **Test.** `TwoTabsOnOneDocument` (1 test in a headless Chrome, 4 checks).

### B27. The README of the offline bundle showed two backslashes in the Windows command

- **Symptom.** The text file inside `extract-offline-*.zip` said `(.venv\\Scripts\\python -m extract.ui on Windows)`, which
  is not a command that runs.
- **Steps.** Build a bundle (`scripts/make_offline_bundle.py`), unzip it, read its README.
- **Cause.** The backslashes were escaped twice in a Python string that is not raw.
- **Fix.** One escape each; the text now has one backslash between the names.
- **Test.** `OfflineBundleReadme` (1 test).

### B28. Killing Extract (or closing its terminal) in the middle of a read left the OCR workers running for ever

- **Symptom.** `kill` (SIGTERM) of the interface, closing the terminal window (SIGHUP) or a crash of a script that reads
  left two `python ... spawn_main` processes of about 830 MB each, reparented to the system, still alive minutes later and
  until the next reboot: 1.7 GB of memory on a computer the guide calls small. (Ctrl+C is a different case: Python tidies
  its daemon children then.)
- **Steps.** Start `python -m extract.ui`, press **Read** on 100 pages, `kill <pid of the interface>`, `ps` the workers.
- **Cause.** The workers are children of a `spawn` context and wait on a queue with no time limit: when the parent dies
  without running its exit code nothing tells them to stop.
- **Fix.** A worker, and the reader process of the second reading, look at their parent (`multiprocessing.parent_process`)
  every 2 seconds while they wait for a task and before every page, and leave when it is gone. The pages they read, and the
  text, are the same (parity read); this is process handling, in `engine/orphan.py` and the two loops that wait.
- **Test.** `WorkersDoNotOutliveTheirParent` (1 test: a script that reads is killed with SIGKILL, its workers must be gone
  in 30 s; it fails on 5.0.2).

### B29. A read request with the wrong shape was an internal error

- **Symptom.** `POST /api/read` with `paths` as a string, `null` or a list of numbers, or with `options` as a string,
  answered 404, 403 or 500 ("An internal error happened: AttributeError ...") depending on the value. The screen never
  sends these, but a script that talks to the interface does.
- **Steps.** `curl -X POST .../api/read -d '{"paths": "/x.pdf", "options": "a"}'` with the token.
- **Cause.** The body was passed on as it came.
- **Fix.** `paths` must be a list of strings and `options` an object, or the answer is 400 `error.bad_request`.
- **Test.** `ReadRequestShape` (1 test, 7 bodies).

### B30. On a computer that writes paths as `D:\Work\me\...`, the folder buttons and the file names on screen were wrong

- **Symptom (by reading the code, not run on Windows: there was no Windows machine to run it on).** The screen split every
  path on `/` only. On Windows, `D:\Work\me\Documents` became one name, the folder buttons under the Folder box held
  that whole text and a click on one asked the server for `/C:\Users...`; the title, the names of the documents on the Read
  screen and the file named in an error showed the whole path instead of the last part.
- **Steps.** Run the interface on Windows, Browse a folder under the home folder, look at the buttons under the box.
- **Cause.** `crumbs`, `projectName`, the first Read screen and the error line used `split("/")`; the export screen
  already used `/[\\/]/`.
- **Fix.** One helper, `pathParts`, splits a path written with `/`, with `\` and a drive letter, or as `\\server\share`,
  and gives the path up to each folder in the same style; the names use `fileName`. Linux and macOS paths come out as before.
- **Test.** `WindowsPathsOnScreen` (1 test in a headless Chrome, 8 checks with Windows-style paths). The server side
  already used `pathlib`; the lock files for Windows were only checked for their wheels (see the report).

### B31. A control character in one word made the whole Excel export fail

- **Symptom.** Typing a correction that holds a control character (a bell pasted from a terminal, a null), or reading a
  digital PDF whose broken font gives the code 0x1f for a ligature, made **Export** answer "An internal error happened
  (IllegalCharacterError ...)" for the document. The searchable PDF, written first, stayed; the Excel, JSON and text
  files of that document were not written.
- **Steps.** In Review press `E`, paste text with a control character, Enter, Export with Excel ticked. Or in Python
  `doc.correct(word_id, "x\x07y"); doc.export("out")`.
- **Cause.** `openpyxl` refuses a string with a character the XLSX format cannot hold.
- **Fix.** Every cell goes through one function that turns such a character into the replacement character, so the cell
  shows that something was there; the JSON and the text keep the text exactly.
- **Test.** `WorkbookWithControlCharacters` (1 test: Review, Corrections and Fields sheets).

### B32. The searchable PDF of a restricted original was written without its restrictions, and nothing said so

- **Symptom.** A PDF that opens without a password but has an owner password that forbids copying text and changing the
  pages (`pikepdf` shows `extract=False`, `modify_assembly=False`) was read normally, and the searchable PDF made from it
  allowed both: no restriction, no password. The README and the guide did not mention it.
- **Steps.** Read such a PDF, export the searchable PDF, open it in a viewer or with `pikepdf`.
- **Cause.** The text layer is added with `pikepdf`, which writes a new, unencrypted file.
- **Fix (documentation).** The Limits of the README and of the guide now say it: the restrictions
  are not carried over and the new file is not locked; keep the original if they matter. No lock is added: a new owner
  password would shut the owner of the document out of the file. If a notice on screen at export time is wanted, it is a
  small change in `outputs/export.py` (the unreadable-pages notice is the model).
- **Test.** `OwnerRestrictedPdf` (1 test: the output is unencrypted, the README says it).

### B33. Does Extract start a browser or touch the network while it reads? It does not

- **Symptom.** None found. The check was whether a helper browser, a background update or any connection could start while
  Extract reads or exports.
- **Steps.** Read the source for network modules, programs started and browser names; run a whole read with every
  connection, name lookup and process start refused; the earlier reads under `unshare -rn` (no network at all) and the
  install watched with `ss` agree.
- **Cause.** Not applicable. Extract imports no network module, its page loads nothing from outside (`/static/` only), the
  PDF is written with `pikepdf` (no browser), and the only program it starts is the file manager (`xdg-open`, `open`,
  `os.startfile`) when a folder button is pressed. `webbrowser.open` shows the local address in the user's own browser
  and is off with `--no-browser`.
- **Fix.** Nothing in the product. Two guards so that it stays so, and the test browser of the suites now runs with its
  own background traffic (updates, sync, safe browsing) turned off, so a suite on a machine with a network does not
  contact anyone either. (Version 5.0.5 found one connection that this check missed, in a library: see B45.)
- **Test.** `NoNetworkNoBrowserAtRunTime` (2 tests: the source, and a whole read of a scan with sockets, name lookups and
  `Popen` refused).

<!-- end of the 5.0.3 entries -->

## Changes since 5.0.1

Found by the second client QA of 04/10/2026 (install from zero, every screen and size, bad paths, privacy
watched with `ss`). One entry per bug: symptom, steps, cause, fix, test. The tests are in
`tests/test_ui_v502.py` and each fails on 5.0.1. The reading engine is untouched (a read of the sample documents is byte-identical).

### B1. A broken link or a folder without permission broke the Import screen; a read could hang for ever

- **Symptom.** Browsing a folder that holds a link to nothing named `x.pdf` (a dangling symlink) showed "An internal error
  happened (FileNotFoundError ...)" and no file of the folder could be added. A folder without read permission showed
  the same raw error; a PDF without read permission was reported as "not a PDF that can be read"; and a path that had
  become a folder after it was added left the Read screen on "Reading" with no end.
- **Steps.** `ln -s /nonexistent/x.pdf in/x.pdf`, Browse `in`. Or `chmod 000` on a folder or on one PDF of a folder, then
  Browse / Add all. Or Add a PDF, replace it by a folder of the same name, press Read.
- **Cause.** `api_browse` called `entry.stat()` on every `*.pdf` and `iterdir()` without catching `OSError`; the inspect
  loop at the start of the read thread caught only `ExtractError`, so any other exception killed the thread before
  `job.finished`.
- **Fix.** `listing()` leaves out an entry it cannot stat and turns a folder it cannot read into `error.cannot_read_folder`;
  `inspect_readable()` says `error.file_not_readable` for a file that is not a file or not readable, and the read thread
  reports any failure of that step as `job.error` and always finishes. Two texts, in English and Spanish.
- **Test.** `FileSystemTrouble` (5 tests).

### B2. Two files with the same content became one document, with no word about it

- **Symptom.** Adding `doubtful.pdf` and a copy of it named `Factura (copia).pdf` (or any two files with the same bytes) and
  pressing Read gave one document in Review and Export, named after the last file read: 10 files read, 8 documents
  listed, and the first name gone from the screen and from the exported files.
- **Steps.** Copy a PDF next to itself, Add both, Read, open Review.
- **Cause.** The interface keys its documents by the first 12 digits of the SHA-256 of the PDF; `State.add` replaced
  the open document of that key.
- **Fix.** The server remembers the file each document was read from. A second file with the same content is not
  opened again: the read says so in a warning that names both files (`job.warning.same_content`, English and Spanish),
  and its card on the Read screen opens the first one. Reading the same file again still replaces it.
- **Test.** `SameContent` (4 tests).

### B3. In Chrome and Edge, a search in the searchable PDF missed some two-word phrases

- **Symptom.** In a searchable PDF made by Extract, about 4 % of the neighbouring words were read as one word by pdfium (the
  viewer of Chrome and Edge): two neighbouring words came out as one word, so a search for the pair found nothing. Copied
  text lost the same spaces. `pdftotext` (poppler) was right.
- **Steps.** Export `doubtful.pdf`, open the `.searchable.pdf` in Chrome and search for two neighbouring words of line 1; or read it with
  `pypdfium2` (`get_text_range()`: 92 words of 95 on the first fixture, 77 of 81 on the second).
- **Cause.** The invisible layer writes each word with no space after it and relies on the viewer to guess word breaks
  from the gap; pdfium does not guess a break when the gap is below about 0.11 of the line height, and OCR boxes of
  neighbouring words are often that close.
- **Fix.** Every word of the layer ends with a space (the scale still uses the word's own width, so nothing moves).
  Measured on the three fixtures: all 334 adjacent pairs found by a pdfium search (15 missed before); poppler unchanged.
- **Test.** `SearchableTextLayer`.

### B4. A page opened or reloaded during a read hid it: no Read tab, no progress, no Stop

- **Symptom.** Reloading the page (or opening Extract in a second tab) while a read was going on showed Import with the
  Read tab greyed out and the Read button off. The read went on, but its progress and the Stop button could not be
  reached, and documents read meanwhile did not enable Review or Export until some click redrew the bar. Pressing
  Read from a tab that did not know of the running read showed the error "A read is already running."
- **Steps.** Add a 100-page PDF, Read, reload the page after 10 seconds (or open the same address in another tab).
- **Cause.** The events of the read are replayed when the page opens, but only the Read screen redrew on them; the tab
  bar was built before they arrived and never again, and nothing sent the user to the screen of a read in progress.
- **Fix.** The tab bar is rebuilt when a read starts or a document is read while another screen is open; a page that
  opens with a read in progress (and no `tab=` in its link) goes to the Read screen; pressing Read while the server has
  a read running opens the Read screen instead of the error.
- **Test.** `ReadInProgress` (2 tests, in headless Chrome, with a reader held by the test).

### B6. A double click on Read showed the error "A read is already running."

- **Symptom.** Pressing Read twice quickly started the read and then showed a red error banner on the Read screen.
- **Steps.** Add a PDF, double click Read.
- **Cause.** Both clicks sent the order; the server refused the second (correctly) and the page showed that refusal.
- **Fix.** `startRead` ignores a click while its order is on the way.
- **Test.** `ReadInProgress.test_B6_...` (one `job.started`, no banner).

### B5. The Tab key was a keyboard trap in Review

- **Symptom.** On the Review screen Tab never moved the focus: a keyboard-only user could not reach the tabs (so not
  Export), the zoom, Only doubtful or the buttons of the word panel, and Shift+Tab did nothing either.
- **Steps.** Open Review, click on the page background, press Tab: the focus stays on the page (`document.activeElement`
  is `body`) however often it is pressed.
- **Cause.** The key handler of Review called `preventDefault()` on every Tab, to use the suggestion, even with no word
  selected or a word with no suggestion.
- **Fix.** Tab takes the suggestion only when a word with a suggestion is selected and the key is a plain Tab pressed
  outside a control; Shift+Tab, and Tab on a button or link, always move the focus. The guide says so.
- **Test.** `KeyboardOnly` (headless Chrome, `section=tabkey`, 9 checks).

### B10. Reading a document a second time showed the first reading's words, and kept what had been accepted

- **Symptom.** Read `doubtful.pdf` with Corrections Off and open Review; go back to Import, choose Corrections More and
  Read again. Review then said "Fixed (1)" and listed the fix, but the page showed the word as it was in the first
  reading (no blue box), and pressing More did not change it. A number accepted in the first reading stayed out of the
  list of words to check in the second.
- **Steps.** As above (the Corrections choice of Import applies to the next Read of the same file).
- **Cause.** A document is keyed by the hash of its PDF, so the second reading replaced the first, but the page cache of
  the Review screen (words per page) was not dropped, and the server kept the set of accepted numbers of the old reading.
- **Fix.** The page cache of a document is dropped when the document is read; the server starts a new reading with
  nothing accepted.
- **Test.** `ReadAgain` (2 tests).

### B9. A folder without write permission, or a very long file name, ended in "An internal error happened"

- **Symptom.** Exporting to a folder the user may not write to showed "An internal error happened. (PermissionError:
  [Errno 13] Permission denied: '.../.doubtful.partial-60825.txt')", with the name of a temporary file. A file name pattern
  of 300 letters gave "OSError: [Errno 36] File name too long". An empty pattern was silently replaced by `{name}`, so the
  preview showed nothing and the files got the default name; a pattern with a slash said only "The request is not valid."
- **Steps.** `chmod 555` a folder, export into it; or type a very long name in File name; or empty the File name box.
- **Cause.** `api_export` caught only `ExtractError` from the writer, so any `OSError` reached the generic 500 handler;
  `body.get("pattern") or "{name}"` turned an empty pattern into the default.
- **Fix.** A write that fails is a failed document with `error.cannot_write` (naming the folder) or
  `error.name_too_long`, and the other documents still go on; a bad pattern is `error.bad_pattern` with a text that
  says what a pattern may hold. Three texts, in English and Spanish.
- **Test.** `ExportErrors` (4 tests).

### B8. A missing word list was reported without its reason, with a command the guide does not give

- **Symptom.** With the word lists (or a model) missing, Read ended with "The text recognition models or the word checker
  could not be loaded." and nothing else, although the guide says "the details say which file is missing". The Python
  error offered `python3 scripts/build_dictionaries.py`, while the README and the guide say
  `python3 scripts/fetch_models.py --lists`.
- **Steps.** Start the interface from a tree whose `src/extract/guard/data` has no word lists (a clone before
  `install.py`) and press Read.
- **Cause.** `wire_error` sends only the `params` of an error, and `EngineLoadError` carries its reason in its message.
- **Fix.** The reason travels as `params.detail` (the screen already prints it in brackets); the message names
  `fetch_models.py --lists`.
- **Test.** `EngineLoadMessage` (2 tests).

### B11. A work folder that cannot be written ended in "An internal error happened"

- **Symptom.** Starting Extract with `--work-dir` on a folder it may not write (or with a home whose `.cache` is
  read-only) and pressing Read gave "An internal error happened. (PermissionError: [Errno 13] Permission denied: '...')".
- **Steps.** `chmod 555 parent; python -m extract.ui --work-dir parent/work`, Add a PDF, Read.
- **Cause.** The read thread turned any exception that was not an `ExtractError` into `error.internal`.
- **Fix.** An `OSError` that names the work folder is `error.cannot_write_work`: the text gives the folder and says to
  start Extract again with `--work-dir` and a folder of the user's. Other errors stay internal.
- **Test.** `WorkFolder` (2 tests).

### B12. `--open` with a wrong file printed a Python traceback

- **Symptom.** `python -m extract.ui --open typo.extract.json` (a file that does not exist, is not JSON or is not an Extract
  document) ended in a 20-line traceback.
- **Cause.** Nothing between `argparse` and the loading of the saved documents caught their errors.
- **Fix.** A missing file is refused by the argument check ("--open: typo.extract.json is not a file"); a file that
  cannot be read as a saved document is one line, `python -m extract.ui: <reason>`, and exit code 2.
- **Test.** `CommandLine` (2 tests).

### B13. A document read with the wrong language ticked showed "{share} %" and "NaN %"

- **Symptom.** Reading a Spanish document with only English ticked (or the other way round) with Corrections Safe or More
  gave, on the Read screen, "Page 1: {share} % of its words would have changed, so no fix was applied here" with the
  placeholder unresolved, and in the Review panel the same sentence with "NaN %".
- **Steps.** Languages: untick Spanish, Corrections: More, Read `mismatch.pdf`.
- **Cause.** The guard withholds the fixes of a page for two causes and sends one event for both; the language cause has
  no `share`, and the screens always printed the text of the other cause.
- **Fix.** `withheldCode()` picks `guard.warning.fixes_withheld_language` (the page's language is not one of the chosen
  ones: check the Languages setting) for that cause, on both screens.
- **Test.** `WithheldForLanguage` (2 tests; the first drives both screens in Chrome).

### B14. The folders of the Import list and the rows of the Fields table could not be used with the keyboard

- **Symptom.** With the keyboard only, Tab never reached the folders listed on Import (so a sub-folder could only be
  entered by typing its whole path) nor the rows of the Fields table of an invoice.
- **Steps.** Press Tab on Import: the focus goes through the tabs, the language button and the path box, never to a folder.
- **Cause.** They were `div`/`tr` elements with a click handler and nothing else.
- **Fix.** `pressable()` gives them `tabindex`, `role="button"` and Enter / Space (which stop there and do not reach the
  Review shortcuts: Enter on a row must not accept the word that was selected before).
- **Test.** `KeyboardLists` (9 checks in Chrome).

### B15. The image of a page that does not exist was an internal error

- **Symptom.** `/api/page/<document>/0.png` or `/99999.png` answered 500 "PdfiumError: Failed to load page." or
  "IndexError: tuple index out of range"; the words of the same page answered 404.
- **Cause.** `api_page_png` did not check the page number the way `api_words` does. The screens never ask for such a page;
  it matters for a link typed by hand and for the tests of the interface.
- **Fix.** The same check: 404 `error.not_found`.
- **Test.** `PageNumbers`.
