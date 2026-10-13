# Corrections and the second reading

Extract can change the words it is sure about, and it can read the doubtful lines a second time before it decides. This
page explains the three correction modes, what a fix is and how to take one back, and what the second reading costs and
gains. Read it before you set `correct` to anything but `off`, or when you want to know why Extract changed nothing.

## Three modes

`correct` is `off`, `safe` or `more`. It is an `Options` field and also the one option `read` takes directly.

| Mode | What it does |
|---|---|
| `off` (default) | Nothing is changed. Every word stays as read, and the best fix is only suggested. Choose it when the text must match the paper exactly, or when you will review every flagged word. |
| `safe` | Applies only the fixes that almost never break a word. In practice that is the check digits of numbers (DNI, NIE, IBAN) and little else: on the sealed test it changed no ordinary word at all. |
| `more` | Applies the fixes with the best balance, without the precision floor that `safe` keeps. Choose it for large batches you will not review word by word, such as an archive you want searchable. |

On the sealed test, `more` removed 19.5 to 21.2 of every 100 wrong words in Spanish and 10.9 to 13.0 in English, depending
on the model. In English with the default model it broke 7.6 % as many words as it fixed, just over the 7.5 % limit set
before the test, a miss that [Accuracy and limits](../accuracy.md) reports with the intervals.

## What a fix is

The guard proposes fixes while it judges a page, and Extract stores every proposal with its reasons. A fix has an
operation: `text` (a different text for the word), `join` (two or more words become one), `segment` (one word becomes
several) or `drop` (a word is removed). It has a kind, `accent`, `case`, `accent_case`, `letters`, `punct`, `structure` or
`number`, the probabilities that the new text and the old one are right, and the modes that apply it. A proposal whose
list of modes is empty is only a proposal: no mode applies it, but `Document.fix_info` shows it.

Some words are never touched: any word you already decided, and protected tokens, which are words with digits, `@`, `/`
or `_`, and words with two inner capitals. When a mode applies a fix, the word it belongs to becomes `fixed`, the words a
join swallowed become `absorbed`, and the text before the fix stays in `original`.

## A fix, applied and undone

The three sample documents give no applied fix with the default settings. One setting gives one: reading `doubtful.pdf`
in Spanish only, with `correct` set to `more`. The page has the word `dias`, written without its accent.

```python
import extract

if __name__ == "__main__":
    options = extract.Options(languages=("es",), correct="more", work_dir="work")
    doc = extract.read("tests/fixtures/doubtful.pdf", options=options)
    print(doc.mode, doc.fixes["applied"], doc.fixes["by_source"])
    word = doc.fixed()[0]
    print(word.id, word.text, word.status, word.original, word.corrected_by)
    fix = doc.fix_info(word.id)
    print(fix["op"], fix["kind"], fix["source"], fix["modes"], fix["applied"])
```

```text
# output
more 1 {'dictionary': 1}
p1w55 días fixed dias guard
text accent dictionary ['more'] True
```

One fix was applied, and it came from the dictionary. The word is now `días`, its status is `fixed`, `original` keeps
`dias`, and `corrected_by` says that the guard, not you, made the change. `fix_info` gives the proposal: an operation
`text`, a kind `accent`, and the modes that apply it, here only `more`. Take it back:

```python
doc.undo(word.id)
word = doc.word(word.id)
print(word.text, word.status, word.review)
doc.set_mode("more")
print(doc.fixes["applied"], doc.word(word.id).text)
```

```text
# output
dias ok accepted
0 dias
```

The word reads `dias` again, with the status the guard had given it, `ok`. Its `review` is `accepted` on purpose: an
undone word counts as decided, so `set_mode("more")`, which applies the stored proposals again, does not put the same fix
back.

The language is part of the reading. In the same session, with the default, Spanish and English together, the same read applies nothing:

```python
both = extract.Options(correct="more", work_dir="work")
doc = extract.read("tests/fixtures/doubtful.pdf", options=both)
print(doc.fixes["applied"], doc.word("p1w55").text, doc.word("p1w55").status)
```

```text
# output
0 dias ok
```

With both languages on, the guard does not flag `dias` and proposes no fix, while Spanish alone flags it and proposes
`días`. Tick the languages the documents are written in, and no others.

## Undoing and switching modes

Proposals are stored, so `Document.set_mode` switches the mode without reading the page again. Any applied fix can be
undone, and the original text, id and box come back: `Document.undo(word_id)` for one word or its group,
`Document.undo_page(number)` for a page, and `Document.undo_all()` for the document, whose text is then that of the `off`
reading. An undone word is marked `review="accepted"`, so a later `set_mode` never applies the same fix again; the proposal
stays in the document.

If the planned fixes would change more than 8 % of the words of a page, none of that page's fixes is applied, and the
warning `guard.warning.fixes_withheld` says so. That much change usually means the page itself is the problem: a poor scan,
an unusual layout, or a language that was not selected. Review such a page by hand; `doc.fixes["withheld"]` lists those
pages, each with its share of changed words. A page whose language is not among the `languages` you chose gets no fixes
either.

## The second reading

The second reading reads the doubtful lines of a page again with other recognisers, and lets a dedicated model weigh the
readings before it proposes fixes. It is off by default.

| `Options(second_reading=...)` | What happens |
|---|---|
| `False` (default) | Each page is read once. |
| `"thorough"` | For each page with doubtful words, 16 lines are read again with two light recognisers and 40 zones with a deeper one, in a separate reader process, and the thorough guard model judges them together with the first reading. |
| `True` | The doubtful lines are read again, but the packaged guard model was trained without second readings, so it discards them and sends a `guard.warning.readings_differ` warning for each page that has some. The time and the memory are spent and nothing changes. |

Where the evidence supports a change, the thorough mode proposes a fix: joining two words, cutting one in two, or a
different text. It is applied only in the `more` mode; with `off` or `safe` the second reading changes no word and only
costs time.

```python
import extract

if __name__ == "__main__":
    options = extract.Options(second_reading="thorough", correct="more")
    doc = extract.read("tests/fixtures/mismatch.pdf", options)
    print(doc.fixes["thorough"], doc.fixes["applied"], doc.fixes["second_reading_failed_pages"])
```

```text
# output
True 0 []
```

On this sample the thorough mode applies no fix, and no page lost its second reading. That is not a failure: the sample's
three flagged words have no fix that passes the thresholds.

The thorough mode needs its own guard model, `guard-thorough.model`, which ships inside the package, and two reader models
that the installation does not fetch: `python3 scripts/fetch_models.py --all` downloads them once, and without them the read
stops with an `EngineLoadError` that names the missing file. It also needs memory. The reader process starts only when
1,600 MB are free and is replaced when it passes its cap of 1,600 MB; without the memory it is skipped with a
`reader.skipped` notice, and a page that goes over the cap keeps its first reading and is listed in
`doc.fixes["second_reading_failed_pages"]`. A reader process that sends nothing for 300 s while a page waits is killed too, with the same result.

On 60 pilot pages, a fixed set of pages of the corpus, it added 2.95 s a page on average (5.46 s at the 95th percentile)
in a clean run on a quiet machine without the memory cap, and the first page of a batch took 17.6 s at the product's cap
while the three models load. On the sealed test, with `correct="more"`, it removed 1.64 more wrong words per 100 in Spanish
and 2.11 more in English than `more` with the default model, and in English it stayed within the limit on broken words
(6.1 %), where the default model was just over it. Choose it for documents where every word counts and time is not the
constraint; [Accuracy and limits](../accuracy.md#speed-and-memory) has the conditions.

## Next

[Review doubtful words](../how-to/review-doubtful-words.md) turns this into a loop. The
[Document reference](../reference/document.md) documents `set_mode`, `undo`, `accept`, `correct` and `reopen`, and
[Options](../reference/options.md) lists every setting.
