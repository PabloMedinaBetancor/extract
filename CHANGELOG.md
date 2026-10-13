# Changelog

All notable changes to Extract are listed here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Each change has one line, and a number such as B46 that refers to
its long write-up (symptom, steps, cause, fix and test) in the [release notes](docs/release-notes.md).

In every release from 5.0.2 to 5.0.7 the reading engine gave the same text as before on the sample documents (from 5.0.3 on
also on five pages of the test corpus); in 5.0.7 the sample documents themselves were then made again. The dates are those
of the release tags.

## [5.0.7] - 2026-10-05

### Changed

- The sample documents, the tests and the invoice generator no longer carry a tax number, a company tax number or an IBAN that
  passes its check digits: every identifier now has a form that cannot exist, so none can belong to a real person, company or
  account. The three sample documents and the screenshots were made again (B53).

## [5.0.6] - 2026-10-05

### Fixed

- The Read screen named one document over the page of another (B46).
- Choosing a document with fewer pages left the page of the other one on screen (B47).
- Open in Review could draw the page of the document that was open before (B48).
- A document that was read again was shown as its old reading, and new documents were not listed (B49).
- The filter of fixed words and the rule picked were carried to another document (B50).
- A document read after the Export screen was first opened was left unticked there (B51).
- A document that failed to read said "Reading" or "Waiting" for ever (B52).

## [5.0.5] - 2026-10-05

### Fixed

- Importing OpenVINO, which every read does, sent an anonymous usage event to Google Analytics, against what the documents
  said. The event held no document, page or word. Extract now imports the library with the report switched off, and nothing
  is written to the home folder (B45).

### Changed

- The documents describe the screenshots as scripted samples, say that the interface answers on `127.0.0.1` only, give the
  measured time of the small laptop next to the one of the development laptop, and describe the sample PDFs as a real read
  shows them.

## [5.0.4] - 2026-10-05

### Fixed

- A note that holds for every page of a document made one box for each page on the Read screen (B34).
- `doc.field_status(doc.invoice.total)` crashed when the invoice had no total (B35).
- Two of the three Python examples of the README did not run as written (B36).
- `--help` of four maintenance scripts ran them, and one of them emptied `THIRD-PARTY.md` (B39).
- After an export of some formats, an export of all of them numbered each file on its own (B41).
- The Review screen said "1 values involved are outlined on the page" (B42).
- A reader process that stopped answering was waited for for ever (B43).

### Changed

- The README shows how to ask for the second reading from Python (B37).
- The documents say what was checked on Python 3.12: it installs, works offline and reads like 3.10, and its test suite has not
  been run (B40).

## [5.0.3] - 2026-10-04

### Added

- Tests that fail if a read starts a browser or touches the network (B33).

### Changed

- The documents say that the searchable PDF of a PDF with an owner password does not keep its restrictions (B32).

### Fixed

- A scan under 100 dpi and a page turned a quarter were read as `done` with no word about it: each now gets a note (B18).
- The time estimate said "on this PC" and was measured on another one: it now learns the speed of the computer from its own
  reads (B23).
- Reading a file again threw away the user's corrections without a word: Extract now asks first and names the files (B25).
- A word without digits was called "a number not written in a valid way" (B44).
- A PDF with no pages was called password protected after a locked file (B16).
- The invoice number and the date could hold a whole sentence (B17).
- Text from a PDF that starts with `=` became a formula in the Excel file (B19).
- A control character in one word made the whole Excel export fail (B31).
- A typed folder was searched where the server was started, and quotes, spaces and odd paths broke it (B20).
- The searchable PDF made poppler viewers (Evince, Okular, `pdftotext`) join 5 to 8 % of the neighbouring words (B21).
- A warning on the Read screen did not say which document it was about (B22).
- The installer's last line could not be pasted when the folder had a space in its name (B24).
- A second tab on the same document kept showing what it had seen when it opened (B26).
- The README of the offline bundle showed two backslashes in the Windows command (B27).
- Killing Extract, or closing its terminal, in the middle of a read left the OCR workers running for ever (B28).
- A read request with the wrong shape was an internal error (B29).
- On a computer that writes paths as `D:\Work\me\...`, the folder buttons and the file names were wrong (B30).

## [5.0.2] - 2026-10-04

### Fixed

- A broken link or a folder without permission broke the Import screen, and a read could hang for ever (B1).
- Two files with the same content became one document, with no word about it (B2).
- In Chrome and Edge, a search in the searchable PDF missed some two-word phrases (B3).
- A page opened or reloaded during a read hid it: no Read tab, no progress, no Stop (B4).
- The Tab key was a keyboard trap in Review (B5).
- A double click on Read showed the error "A read is already running." (B6).
- A missing word list was reported without its reason, with a command the guide did not give (B8).
- A folder without write permission, or a very long file name, ended in "An internal error happened" (B9), and so did a work
  folder that cannot be written (B11).
- Reading a document a second time showed the first reading's words, and kept what had been accepted (B10).
- `--open` with a wrong file printed a Python traceback (B12).
- A document read with the wrong language ticked showed "{share} %" and "NaN %" (B13).
- The folders of the Import list and the rows of the Fields table could not be used with the keyboard (B14).
- The image of a page that does not exist was an internal error (B15).

## [5.0.1] - 2026-10-04

Nothing of this version was recorded in this file.

## [5.0.0] - 2026-10-03

The first release of version 5. What came before it is not recorded here.
