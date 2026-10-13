# Use Extract from an AI coding assistant

An assistant that writes code against Extract from memory gets a few things wrong, always the same few, and some of them
fail silently. This page is for the person who reads what the assistant wrote; the rules for the assistant itself are in
`AGENTS.md`, at the root of the repository.

## Give the assistant the right files

- `AGENTS.md` is written for a coding assistant that has nothing else: runnable recipes for the common tasks, the fields
  it will use, and what never to do. Many coding assistants look for it on their own when they work inside the
  repository; if yours does not, name it in your request. For a project that only installs Extract, copy it into that
  project.
- `llms.txt` is an index of the documentation, with one line for each page.
- `llms-full.txt` is every page of this site as one plain file. Attach it when the assistant cannot browse the
  repository.

## The mistakes, and how they show

| What the assistant wrote | What you see | Why, and the fix |
|---|---|---|
| Reads at the top level of the script, with no `__main__` guard | A scan raises `EngineLoadError` after a few seconds (`cause="no_main_guard"`) and the script exits with status 1 | The workers import the script again. Put every call to `extract.read` and `extract.Reader` under `if __name__ == "__main__":` ([Troubleshooting](../troubleshooting.md#engineloaderror-the-script-reads-at-its-top-level)). |
| Calls `extract.read` for each file in a loop | Every file waits several seconds before its first page | `read` starts the workers and loads the models on every call. One [`Reader`](../reference/reader.md#reader) does it once. |
| Treats `ok` as checked, `unscored` as fine, or a returned document as read | Wrong words in the output that nobody was asked about | `ok` means the guard did not flag the word; `unscored` means nobody could judge it; a document with an unreadable page comes back `partial`, not raised. |
| Tests `word.status == "doubtful"` to ask whether a word is still open | Decided words come back | A decision sets `review` and leaves `status` alone: test `word.review is None`. |
| Calls `accept` or `use_suggestion` on every doubtful word, or `correct` with a guessed text | A clean document that nobody reviewed | A decision belongs to a person who has seen `Document.word_crop`. `use_suggestion` applies whatever is offered, even a digit change whose `verified` is false. |
| Asks a language model to fix a digit, or edits a number itself | A total that is right on screen and wrong in the books | Extract changes a digit only when a verifying rule confirms the new one, or when a person types it. |
| Uses `accept` to turn a `mismatch` into `checked` | The number stays `mismatch` | Only a rule makes a number `checked`. |
| Calls `use_suggestion` on an IBAN written in groups | `ValueError: ... has no suggestion` | The method handles a number made of one word, and a tie gives no suggestion. Use `Document.correct` on the word that holds the wrong digit. |
| Expects corrected words after a plain `read`, or sets `second_reading=True` for better text | Nothing changes, or a slower read that changes nothing | `correct` is `off` unless you ask for `safe` or `more`. `second_reading=True` computes readings that the packaged guard ignores: use `"thorough"` after fetching its models. |
| Saves with `Document.save` over a file it should keep | The old file is gone | `save` overwrites without asking; `export` never overwrites. |
| Exports `"pdf"` from a loaded reading | `SourceChanged` | The searchable PDF needs the source PDF with the same hash. Pass `source=` to `load`. |
| Reads in tests and examples with the default work folder | Files in the user's cache folder, and a cancel test that runs instantly the second time | Pass `Options(work_dir=<a temporary folder>)`: pages read before come from the cache with no OCR. |
| Starts two reads at once, or shares one `Reader` between threads | `ram.low` notices and a read that waits for memory | A read starts up to two workers of about 1.5 GB each. On 3 to 4 GB, use `Options(processes=1, backend="onnxruntime")` and one read at a time. |
| Writes `from extract import *`, imports `extract.engine`, or calls `doc.text` | `Warning` stops being the built-in, or an `AttributeError` | Write `import extract`. Only the 36 names of `extract.__all__` are public, and there is no `Document.text`: export `"txt"` instead. |
| Passes a JPEG, a Word file or a French document | `UnsupportedFile`, or many `doubtful` words | PDF only, Spanish and English only. Put images in a PDF first. |
| Expects invoice fields from a page after the first | `doc.invoice` is `None` or lacks fields | Fields are read from page 1 only, and a field that was not found is `None`. Test each one. |

## Check the code you were given

- Every call to `extract.read` or `extract.Reader` is under `if __name__ == "__main__":`.
- Several files go through one `Reader`, in one thread.
- The code looks at `doc.status`, and at `doc.unreadable_pages` when it is `partial`.
- Nothing decides a doubtful word, or changes a number, without a person or a rule you can name.
- `Options(work_dir=...)` is set in tests and examples.
- `export` is used where a file must not be lost, and `save` only where overwriting is meant.
- Every name used is on the [reference index](../reference/index.md). If it is not there, it does not exist.

## A request that works

```text
Read AGENTS.md first. Write a script that reads every PDF in ./incoming with one extract.Reader and
Options(work_dir="work"), prints each document's status, writes a workbook for each into ./out, and lists the
words that need a decision with the page number. Do not accept or correct any word.
```
