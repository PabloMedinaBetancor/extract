# Number checks

Extract checks numbers with rules, not with the OCR's confidence. This page explains what a rule is, which numbers can be
confirmed and when a digit suggestion appears. Read it before you trust a `checked` number or wonder why another is
`unverified`.

## The idea

A wrong digit in an IBAN breaks its check digits; a wrong digit in a line of an invoice breaks the sum. A rule is a
calculation that either passes or fails on the number as read, with exact decimal arithmetic and no confidence involved.
Rules come in two kinds.

- A **verifying** rule would fail on any single wrong digit of the number it covers. If it passes, the number is very
  likely right.
- A **plausibility** rule can only fail. It catches what cannot be right, such as a date that does not exist, and passing
  proves nothing.

A number is `checked` when at least one verifying rule passes and none fails. It is `mismatch` when any rule fails, and
`unverified` otherwise. A number on an `unreadable` page is never `checked`.

## The rules

| Rule | What it checks | Can confirm a number |
|---|---|---|
| `dni_letter`, `nie_letter` | The control letter of a Spanish DNI or NIE. | Yes |
| `iban_mod97` | The check digits of an IBAN. | Only when the national layout fixes every position as a letter or a digit |
| `iban_length` | The country code and the length for that country. | No |
| `cif_control` | The control character of a Spanish CIF, which may be a letter or a digit. | No |
| `line_amount` | Quantity times unit price equals the line amount. | Yes, when a change of one in the last printed decimal moves the result by at least a cent |
| `lines_sum` | The lines add up to the subtotal. | Same condition |
| `vat_amount` | The subtotal times the VAT rate equals the VAT amount. | Same condition |
| `withholding_amount` | The subtotal times the withholding rate equals the withholding. | Never confirms; see below |
| `total` | Subtotal plus VAT minus withholding equals the total. | Same condition |
| `number_format` | The number is written in the style of the document's locale. | No |
| `date_valid` | The date exists. | No |
| `vat_rate_plausible` | The rate is one of 0, 2, 3, 4, 5, 7, 9.5, 10, 15, 20 and 21. | No |

Sums accept both rounding conventions, half up and half even, so a total that differs only by how a cent was rounded
passes. The locale decides how amounts are written, Spanish `1.234,56` or English `1,234.56`: Extract takes it from the
majority of the unambiguous numbers on the document, and falls back to the first language of `Options.languages` on a tie.

## Which numbers can be confirmed

`checked` is possible for Spanish DNI and NIE numbers, for IBANs whose national layout fixes every position as a letter
or a digit (Spanish, German, British, Portuguese, Dutch, Austrian, Belgian and Czech among others), and for the arithmetic
of an invoice. These stay `unverified`:

- A CIF, and an IBAN whose layout allows either a letter or a digit in some position (French, Italian, Swiss and others),
  even when their control passes: the control cannot tell a wrong letter from a wrong digit there.
- The withholding amount, because a document may print it with or without a minus sign. A wrong digit in it is still
  caught, since the total and the withholding rules fail.
- An amount written with a space as the thousands separator, `1 234,56`, which the OCR delivers as two words.

A VAT rate outside the list above comes out `mismatch`: a rate of 19 % or 22 % can only fail. The list is Spain's VAT and
the IGIC of the Canary Islands, which is a limit when you read invoices from elsewhere.

## A number that fails

On the sample invoice whose total is 30 cents off, three numbers are `mismatch`, and the same rule failed for each:

```python
import extract

doc = extract.load("invoice.extract.json", options=extract.Options(work_dir="work"))
for number in doc.numbers:
    if number.status == "mismatch":
        failed = [rule.key for rule in number.rules if rule.result == "fail"]
        print(number.text, failed)
```

```text
# output
84.226,04 ['checks.rule.total']
17.687,47 ['checks.rule.total']
101.913,81 ['checks.rule.total']
```

The subtotal, the VAT amount and the total all take part in the `total` rule, so all three are flagged. The rule found
`101913.81` where the subtotal and the VAT call for `101913.51`. `mismatch` says that a digit is wrong somewhere among
them, not which one. The [tutorial](../tutorials/invoice.md#why-the-total-fails) follows this case to its end.

## Digit suggestions

When a rule fails, Extract looks for a single-digit change that would make it pass. The suggestion is `verified` only when
a verifying rule fails now, passes with that one change and with no other single change, and no plausibility rule objects.
If two changes would both pass, there is no suggestion at all, and IBANs often tie because their check digits have few
possible values.

A verified fix says that the figures add up with that digit; it does not say that the digit is the printed one. Look at the
page: if it shows the digit as read, the document itself is inconsistent, the printed value is the record, and the
decision is `accept`.

With `correct="off"`, a suggestion is only offered, and `Document.use_suggestion` applies it as your action. With `safe` or
`more`, a digit is changed automatically only when all of these hold: the suggestion is verified, the new digit is one the
recogniser had among its alternatives for that position, you have not decided the word, and the word is not an identifier
whose letter could be the mistaken one. In `safe` the arithmetic rules stay suggestions; they apply only in `more`. No model
ever changes a number: a digit changes through these conditions or by your hand.

`use_suggestion` handles only a number made of one word, and when that word carries a suggestion of its own, it applies
the word's suggestion, not the digit fix. For a number written in groups, such as an IBAN in blocks of four, correct the
word that holds the wrong digits with `Document.correct`.

On the sealed test, none of the 519 numbers that came out `checked` on 60 synthetic invoices was wrong;
[Accuracy and limits](../accuracy.md) gives the conditions of that figure.

## Next

[Corrections and the second reading](corrections.md) covers what changes a word on its own. The
[data types](../reference/data-types.md) page documents `Number`, `Rule` and `DigitSuggestion`.
