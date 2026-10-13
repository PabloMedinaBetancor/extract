# Extract

Extract reads PDFs, scanned or digital, on your own computer and gives you their text. Unlike plain OCR, it also tells
you which words and numbers are probably wrong, so you review a few flagged words instead of the whole page.

It is for people who need the text of a scanned letter, a council ordinance or an invoice and cannot afford a wrong digit
going unnoticed: finance teams, archives, anyone who feeds OCR output into something else. It comes as a local web
interface and as a Python library. If you write code against it, yourself or with an AI assistant, start with `AGENTS.md`
at the root of the repository: the calls for the common tasks, on one page.

## What it does

Every word comes out with a status. `ok` means the guard did not flag it, `doubtful` means it is probably wrong, and
`unscored` means nothing could judge it; words taken from the text layer of a digital page are `native`, and two more,
`fixed` and `absorbed`, appear only when corrections are on. A doubtful word carries its reasons and sometimes a
suggestion.

Every number is checked with rules, not with the OCR's confidence: the letter of a Spanish DNI, the check digits of an
IBAN, the arithmetic of an invoice. Only a rule that would break on any wrong digit can make a number `checked`. When one
fails, Extract says so and, if exactly one digit change would fix it, suggests that digit.

Nothing is changed behind your back. By default Extract proposes and you decide; if you switch corrections on, every fix
is kept with its reasons and can be undone. The result is a searchable PDF, a JSON file with every word and its box, an
Excel workbook and plain text. Nothing is uploaded: reading works with the network unplugged.

## An example

From the root of the repository, on a sample invoice whose total is 30 cents off (every name and number in it is
invented):

```python
import extract

if __name__ == "__main__":
    doc = extract.read("tests/fixtures/mismatch.pdf")
    for number in doc.numbers:
        if number.status == "mismatch":
            print(number.text, number.status)
    total = [n for n in doc.numbers if n.text == "101.913,81"][0]
    print(total.suggestion)
```

```text
# output
84.226,04 mismatch
17.687,47 mismatch
101.913,81 mismatch
DigitSuggestion(position=8, old='8', new='5', verified=True, rule='checks.rule.total')
```

The invoice prints a total of `101.913,81`. The lines add up and the VAT is right, so the total should be `101.913,51`:
the character at position 8 of the text, an `8`, would have to be a `5`. The subtotal and the VAT amount are `mismatch` as
well, because the total rule involves them. The [tutorial](tutorials/invoice.md) works through this sample.

## What it does not do

It reads PDFs only, in Spanish and English, with one OCR engine. It was tested on Linux with Python 3.10; Windows is
untested and macOS is not supported. It cannot confirm a number that has no rule, and `doubtful` is a likelihood, not a
verdict. [Accuracy and limits](accuracy.md) gives every figure with the conditions it was measured under, and every limit
with its reason.

## Where to go next

- New here: [install it](getting-started/installation.md), then follow the [quickstart](getting-started/quickstart.md)
  and the [tutorial](tutorials/invoice.md).
- Understanding it: [how it works](concepts/how-it-works.md) and [number checks](concepts/number-checks.md).
- Solving a task: the [how-to guides](how-to/batch-a-folder.md), or [troubleshooting](troubleshooting.md) when
  something fails.
- Looking something up: the [API reference](reference/index.md) and the [JSON format](reference/json-format.md).

More in the repository: `CHANGELOG.md` (what changed in each version), `docs/release-notes.md` (the long write-up of
each fix it lists) and `docs/MEASURED.md` (every measured table, with its method and intervals).
