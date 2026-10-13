# Review doubtful words

This page reviews a document in code: it saves a reading, loads it back in a later process, decides the words and the
failing numbers, and exports. It is for a tool that puts a person in front of the words that need a decision, or applies
decisions that a person has given. In the interface, the same work is the **Review** screen.

Extract does not decide for you. A word is `doubtful` because the guard estimates that it may be wrong, and a number is
`mismatch` because a rule failed: in both cases a person looks at the page. The code below takes its decisions from a
table, `DECISIONS`, where a real tool would take them from a screen. The table accepts the total as the page prints it,
which is what the [tutorial](../tutorials/invoice.md#why-the-total-fails) recommends for a real invoice.

```python
from pathlib import Path

import extract

SOURCE = "tests/fixtures/mismatch.pdf"
SAVED = "work/mismatch.extract.json"
DECISIONS = {"p1w11": "accept", "p1w13": "accept", "p1w4": "accept", "p1w75": "accept"}


def main():
    options = extract.Options(work_dir="work")
    doc = extract.read(SOURCE, options=options)
    Path(SAVED).parent.mkdir(exist_ok=True)
    doc.save(SAVED)
    print("read:", len(doc.doubtful()), "words to review; saved", SAVED)

    doc = extract.load(SAVED, source=SOURCE, options=options)
    print("loaded:", doc.source, "-", len(doc.doubtful()), "words to review")

    crops = Path("work/crops")
    crops.mkdir(exist_ok=True)
    for word in doc.doubtful():
        (crops / f"{word.id}.png").write_bytes(doc.word_crop(word.id))
        action = DECISIONS.get(word.id)
        print(word.id, word.text, "->", action)
        if action == "accept":
            doc.accept(word.id)

    for number in doc.numbers:
        if number.status == "mismatch" and len(number.word_ids) == 1:
            word_id = number.word_ids[0]
            if DECISIONS.get(word_id) == "accept":
                doc.accept(word_id)
                print(word_id, number.text, "->", doc.word(word_id).text, doc.word(word_id).review)

    left = [number.text for number in doc.numbers if number.status == "mismatch"]
    print("to review:", len(doc.doubtful()), "- mismatches:", left)
    print([path.name for path in doc.export("out/", ("json",))])


if __name__ == "__main__":
    main()
```

```text
# output
read: 3 words to review; saved work/mismatch.extract.json
loaded: tests/fixtures/mismatch.pdf - 3 words to review
p1w11 Pelmuda -> accept
p1w13 Esquilmar -> accept
p1w4 F-2026-0498 -> accept
p1w75 101.913,81 -> 101.913,81 accepted
to review: 0 - mismatches: ['84.226,04', '17.687,47', '101.913,81']
['mismatch.extract.json']
```

This is `examples/review_words.py`.

## Save and load

`doc.save(path)` writes the reading as a `.extract.json` file, with the words, boxes, statuses, numbers and your decisions
so far, and `extract.load(path)` reads it back in well under a second: it loads no model and does no OCR. `load` looks for
the PDF next to the JSON, under the name it was read with; when the PDF is somewhere else, pass `source=`, as above.
Without a reachable PDF you can still read, correct, save and export to `xlsx`, `json` and `txt`, but `word_crop`,
`page_image` and the `pdf` export raise `SourceChanged`, as they do when the PDF is not the one the reading came from.

`save` overwrites an existing file without asking: to keep a checked, never-overwritten copy, use
`doc.export(folder, ("json",))`. Pass `options=extract.Options(work_dir=...)` to `load`, as above, so that the log of your
decisions goes where you want it and not to the default folder.

## The words

`doc.doubtful()` lists the words that nobody has decided, the likeliest to be wrong first, and after them the `unscored`
ones, such as the words of a page that could not be read: nobody judged those, so the person checks them like the others,
or has the page scanned again and read again. Show the person the picture of each word, which `doc.word_crop(word.id)`
gives as PNG bytes, and decide it once:

| Call | Use it when |
|---|---|
| `doc.accept(id)` | The word is right as it stands. |
| `doc.correct(id, text)` | The word is wrong, and you have the right text. |
| `doc.use_suggestion(id)` | The word carries a `suggestion`, and the person agrees with it. |
| `doc.reopen(id)` | A decision has to be taken back. |

A suggestion is the guard's best candidate, not a verdict: the one for `Esquilmar` was `Esquivar`, with a probability of
0.009 of being right, and the right decision was `accept`. Show it next to the picture, and do not apply it unseen.

A decision sets `word.review` to `accepted` or `corrected` and leaves `word.status` alone, so test `review` to know whether
anyone decided. A `Word` is a snapshot: ask `doc.word(id)` again after a change to see it. `accept` after `correct` keeps
the typed text and turns `corrected` into `accepted`, and a later `reopen` then does not bring the OCR text back, which is
rarely what you meant.

## The numbers

A failing number is not in `doubtful()`: its words are usually `ok`, and the number is `mismatch`. Loop over
`doc.numbers` and show those whose `status` is `mismatch`, with the page and `number.suggestion`, the one-digit change that
makes the failing rule pass when exactly one does. `doc.use_suggestion(word_id)` applies it whether its `verified` is true
or not, so show `verified` too. A verified fix says that the figures add up with that digit, not that the digit is the
printed one: if the page shows the digit as read, the decision is `accept`, as in the script.

`accept` never makes a number `checked`, and `correct` does only when a rule passes on the new text. A number written in
several words, such as an IBAN in groups of four, has no suggestion that `use_suggestion` can apply, and a tie gives no
suggestion at all, which is common for IBANs: correct the word that holds the wrong digit with `correct`.

## What is written where

Every decision is appended as one line to `corrections.jsonl` in the work folder, `work/corrections.jsonl` here, with the
word, the original, the new text, the action and the time. The log is a record, not what `load` reads: the decisions
travel in the saved or exported JSON, so save or export after you decide, as the last line of the script does.

## Next

[Batch a folder](batch-a-folder.md) builds one workbook from many documents. [Corrections and the second
reading](../concepts/corrections.md) explains the modes in which the guard applies fixes by itself, and how to undo them.
