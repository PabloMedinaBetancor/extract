"""Data shared by the engine, the guard, the number rules, the outputs and the interface.

These are the objects you read from a `Document`: `Word`, `Page`, the invoice types and the number types.
All of them are frozen dataclasses: you do not change them, and a change made through a `Document` gives you new
objects.

A box is a tuple `(left, top, right, bottom)` of floats. The boxes of a word are in pixels of the page as it was drawn
for the OCR, at `Page.dpi`; `Page.box_to_points` turns one into PDF points.

Four of the types have a shorter public name: `extract.Page` is `PageReading`, `extract.Number` is `NumberResult`,
`extract.Rule` is `RuleResult` and `extract.Invoice` is `InvoiceFields`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Mapping

Box = tuple[float, float, float, float]

# The values a field can take. The meaning of each is in the docstring of the type that has the field.
WORD_STATUSES = ("native", "ok", "doubtful", "fixed", "absorbed", "unscored")  # Word.status
FIXED_BY = ("guard", "user")  # Word.corrected_by
FIX_OPS = ("text", "join", "segment", "drop")  # Fix.op
FIX_KINDS = ("accent", "case", "accent_case", "letters", "punct", "structure", "number")  # Fix.kind
PAGE_STATUSES = ("native", "ocr", "unreadable")  # PageReading.status
NUMBER_STATUSES = ("checked", "mismatch", "unverified")  # NumberResult.status
REVIEW_STATES = ("accepted", "corrected")  # Word.review (None while undecided)

_WORD_ID = re.compile(r"^p(\d+)w(\d+)(?:#(\d+))?$")


def word_id(page: int, index: int) -> str:
    """Build the id of a word from its page and its position.

    Args:
        page: The page number, counting from 1.
        index: The position of the word on the page in reading order, counting from 0.

    Returns:
        The id, `p{page}w{index}`: `word_id(3, 17)` is `"p3w17"`.

    Example:
        ```python
        from extract import types

        print(types.word_id(3, 17), types.base_id("p3w17#1"), types.page_of("p3w17"))
        ```

        ```text
        # output
        p3w17 p3w17 3
        ```

    See also:
        `base_id`, `page_of`, `Word`.
    """
    return f"p{page}w{index}"


def base_id(token: str) -> str:
    """Return the id of the word a token comes from.

    A word that the guard segmented is split into tokens whose ids have the form `p3w17#1`; `p3w17` is their base. An
    id without `#` is its own base.

    Args:
        token: A word id or a token id.

    Returns:
        The word id, without the `#` part.

    See also:
        `word_id`, `Document.tokens`, `Document.word`.
    """
    return token.split("#", 1)[0]


def page_of(word: str) -> int:
    """Return the page number that a word id belongs to.

    Args:
        word: A word id such as `"p3w17"`, or a token id such as `"p3w17#1"`.

    Returns:
        The page number, counting from 1.

    Raises:
        ValueError: `word` is not a word id.

    See also:
        `word_id`, `base_id`.
    """
    match = _WORD_ID.match(word)
    if match is None:
        raise ValueError(f"not a word id: {word!r}")
    return int(match.group(1))


@dataclass(frozen=True)
class Reason:
    """Why a word is doubtful, or why a page could not be read: an i18n key and the numbers its text needs.

    Attributes:
        key: The key, from one of three families. `guard.reason.*` explains a doubtful word: `not_in_dictionary`,
            `close_to` (params `word`), `low_char_conf` (params `char`, `conf`), `mixed_letters_digits`,
            `odd_chars`, `bad_number_format`, `neighbours_unusual`, `unusual_height`, `unusual_width` and
            `low_page_quality`. `guard.fix.*` explains a proposed fix: `accent_restored`, `case_restored`,
            `dictionary_nearest` (params `word`), `engine_alternative` (params `char`, `conf`), `page_vocabulary`
            (params `count`), `punctuation_restored`, `readers_agree` (params `n`, `of`), `speck_dropped`,
            `word_split` and `words_joined`. `page.reason.*` explains an unreadable page: `few_words`,
            `low_confidence` and `render_failed`.
        params: The values the text needs, such as `{"char": "l", "conf": 0.98}`. Empty when the key needs none.

    See also:
        `Word`, `Fix`, `Page`.
    """

    key: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Reading:
    """What one reader of the second reading made of a word of a line it read again.

    Engine detail: it is kept on `Word.readings` while the document is in memory and is never saved in the JSON.

    Attributes:
        source: The name of the reader that made this reading.
        text: What the reader read for the word, or `None` when its line could not be aligned with the first reading.
        conf: The reader's confidence in the word, from 0 to 1, or `None`.
        char_conf: One confidence per character, or `None`.
        group: The ids of the words of the first reading that the reader saw as one: the evidence for a join. A reader
            that cut the word in two has a space in `text`.
        rank: The rank of the unit (a line or a zone) the word was read in, 0 being the most doubtful of its page, or
            `None`.
        context: True when the word is in the unit only as context: a neighbour, or a word of a line that did not pass
            the selection rule.

    See also:
        `Word`, `Options`.
    """

    source: str
    text: str | None
    conf: float | None
    char_conf: tuple[float, ...] | None
    group: tuple[str, ...] = ()
    rank: int | None = None
    context: bool = False


@dataclass(frozen=True)
class Fix:
    """One proposal of the guard for a word: what to change, how sure it is, and which correction modes apply it.

    A proposal exists whether or not it is applied. `modes` says which modes of `Options.correct` apply it; an empty
    tuple means the proposal is only shown, as a suggestion. The proposal stays on the word after `Document.undo`.

    Attributes:
        op: What the fix does: `text` (replace the text of one word), `join` (join the words in `ids` into one),
            `segment` (split one word into several) or `drop` (remove a speck that is not a word).
        kind: The family of the edit: `accent`, `case`, `accent_case`, `letters`, `punct`, `structure` or `number`.
        text: The new text.
        ids: The ids of the words the fix spans.
        source: Where the candidate text came from: `dictionary`, `fold`, `punct`, `page`, `ctc_alt`, `masked`,
            `reading`, `joined`, `join_edit`, `segment` or `drop`; a digit fix has `rule:` followed by the key of
            the rule that verified it. It is the first of `sources`.
        sources: Every source that proposed the same text.
        p_fix: The calibrated probability that the new text is the truth. It can be very small: it is a probability,
            not a score of how good the text looks.
        p_keep: The calibrated probability that the word as it was read is right.
        rule: The key of the verifying rule of a digit fix, or `None`.
        modes: The modes that apply the fix: a subset of `("safe", "more")`.
        reasons: Up to three `guard.fix.*` reasons.

    See also:
        `Document.fix_info`, `Document.set_mode`, `Reason`, `Word`.
    """

    op: str
    kind: str
    text: str
    ids: tuple[str, ...]
    source: str
    sources: tuple[str, ...]
    p_fix: float
    p_keep: float
    rule: str | None
    modes: tuple[str, ...]
    reasons: tuple[Reason, ...]


@dataclass(frozen=True)
class Word:
    """One word of a page, with what the engine and the guard know about it.

    A `Word` is a snapshot. A `Document` replaces it when you correct or accept it, so ask `Document.word` again to
    see the change.

    The `status` says how far to trust the word as read: `ok` means that the guard looked at the word and did not
    flag it, not that the word is verified, and `unscored` means that nobody could judge it. Your decisions do not change
    `status`: they set `review`, so a word you accepted can still be `doubtful`.

    Attributes:
        id: `p{page}w{index}`: the page counts from 1, the index from 0 in reading order. `"p3w17"` is the 18th word
            of page 3. Token ids such as `"p3w17#1"` exist only in `Document.tokens`.
        page: The page number, counting from 1.
        text: The current text: what was read, or what a fix or a correction made of it. Empty for an `absorbed` word.
        box: `(left, top, right, bottom)` in pixels of the page as drawn at `Page.dpi`.
        conf: The OCR confidence of the word itself, from 0 to 1: the mean of its characters' confidences. For an
            unaligned word (see `Page.unaligned_words`) it is the mean of the word's line. `None` on `native`
            pages.
        char_conf: One confidence per character, or `None` on `native` pages and for words whose characters could not
            be told apart from their line's (`Page.unaligned_words`).
        line_id: The number of the text line the word is on, from 0.
        idx_in_line: The position of the word inside its line, from 0.
        status: `native` (from the PDF's text layer, not scored), `ok`, `doubtful`, `fixed` (the owner of a fix the
            guard applied), `absorbed` (joined into another word or dropped by a fix) or `unscored`.
        p_wrong: The guard's calibrated probability that the word is wrong, or `None` for `native` and `unscored`
            words.
        reasons: Why the word is doubtful, as `Reason` objects, in the order the guard lists them.
        suggestion: A replacement text, set only on a `doubtful` word whose proposed fix is a plain change of text.
            `Document.use_suggestion` applies it.
        suggestion_conf: The probability that the suggestion is right (`Fix.p_fix`), which can be small.
        review: `None` while nobody has decided, `"accepted"` after `Document.accept`, `"corrected"` after
            `Document.correct`. A word with a `review` is not in `Document.doubtful`.
        original: The text before the first correction or applied fix, or `None`.
        corrected_by: `"guard"` for an applied fix, `"user"` for a correction, `None` otherwise.
        char_alts: For each character, the recogniser's alternatives as `(character, probability)` pairs. Engine
            detail, not saved in the JSON.
        masked: The reading of the word with a restricted alphabet, used as evidence by the guard. Engine detail, not
            saved.
        readings: The second readings of the word, as `Reading` objects, or `None`. Engine detail, not saved.
        fix: The guard's proposal as a `Fix`, applied or not, or `None`.
        absorbed_by: The id of the word that took this one in, for an `absorbed` word.
        guard_status: The status the guard gave the word before a fix changed it; `Document.undo` restores it.
        orig_box: The box the word had before a join widened it; `Document.undo` restores it.

    Example:
        ```python
        import extract

        doc = extract.load("invoice.extract.json")
        word = doc.word("p1w13")
        print(word.id, word.text, word.status, round(word.p_wrong, 2), word.reasons[0].key)
        ```

        ```text
        # output
        p1w13 Esquilmar doubtful 0.12 guard.reason.not_in_dictionary
        ```

    See also:
        `Document.word`, `Document.doubtful`, `Page`, `Fix`, `Reason`.
    """

    id: str
    page: int
    text: str
    box: Box
    conf: float | None = None
    char_conf: tuple[float, ...] | None = None
    line_id: int = 0
    idx_in_line: int = 0
    status: str = "unscored"
    p_wrong: float | None = None
    reasons: tuple[Reason, ...] = ()
    suggestion: str | None = None
    suggestion_conf: float | None = None
    review: str | None = None
    original: str | None = None
    corrected_by: str | None = None
    char_alts: tuple[tuple[tuple[str, float], ...], ...] | None = None
    masked: str | None = None
    readings: tuple[Reading, ...] | None = None
    fix: Fix | None = None
    absorbed_by: str | None = None
    guard_status: str | None = None
    orig_box: Box | None = None  # the box the word had before a join widened it (restored by undo)


@dataclass(frozen=True)
class PageReading:
    """What the engine gives for one page: its size, its status and its words.

    A page is `unreadable` when fewer than 25 words came out (`page.reason.few_words`), when the mean confidence is
    under 0.65 (`page.reason.low_confidence`), or when it could not be drawn or its worker died twice
    (`page.reason.render_failed`, with the text of the failure in `error`). It keeps whatever words were read, as
    `unscored`, and none of its numbers is ever `checked`. A page whose worker failed has `dpi` 0, sizes 0 and no words.

    Attributes:
        number: The page number, counting from 1.
        width_px: Width of the page as drawn for the OCR, in pixels.
        height_px: Height of the page as drawn for the OCR, in pixels.
        width_pt: Width of the page in PDF points (1/72 inch).
        height_pt: Height of the page in PDF points.
        dpi: The resolution the page was drawn at: the scan's own resolution limited to 100-300, or 200 when the
            page has no scan image, and lowered further so that the longer side of the page stays within 3,000 pixels.
        words: The words of the page, in reading order, as `Word` objects.
        status: `native` (the PDF's own text layer was legible and was used), `ocr` or `unreadable`.
        reason: For an `unreadable` page, the `page.reason.*` key that says why. `None` otherwise.
        error: The text of the failure, when something failed while the page was read. `None` otherwise.
        mean_conf: The mean confidence of the words, from 0 to 1, or `None` on `native` pages.
        unaligned_words: How many words have no `char_conf` because their characters could not be matched to their
            line's. For these words `conf` is the confidence of the line.
        rereads: For each line that was read again by the second reading, the readers that ran on it. A line that is
            absent was not read again. Engine detail, not saved.
        reread_report: Counts of what the second reading did on the page. Engine detail, not saved.

    Example:
        ```python
        import extract

        doc = extract.load("invoice.extract.json")
        page = doc.pages[0]
        print(page.number, page.status, page.dpi, page.width_px, page.height_px)
        print([round(value, 1) for value in page.box_to_points(page.words[0].box)])
        ```

        ```text
        # output
        1 ocr 200 1654 2339
        [52.9, 42.1, 153.0, 61.9]
        ```

    See also:
        `Document.pages`, `Document.unreadable_pages`, `Word`.
    """

    number: int
    width_px: int
    height_px: int
    width_pt: float
    height_pt: float
    dpi: int
    words: tuple[Word, ...]
    status: str = "ocr"
    reason: str | None = None
    error: str | None = None
    mean_conf: float | None = None
    unaligned_words: int = 0
    rereads: Mapping[int, tuple[str, ...]] = field(default_factory=dict)
    reread_report: Mapping[str, int] = field(default_factory=dict)

    def box_to_points(self, box: Box) -> Box:
        """Convert a box from pixels to PDF points.

        Args:
            box: `(left, top, right, bottom)` in pixels of this page, such as `Word.box`.

        Returns:
            The same box in points (1/72 inch), scaled by `width_pt / width_px` and `height_pt / height_px`. The
            JSON stores both forms as `box_px` and `box_pt`.

        Raises:
            ZeroDivisionError: The page has `width_px` 0 (a page whose rendering failed).

        See also:
            `Word.box`, `Page.dpi`.
        """
        sx = self.width_pt / self.width_px
        sy = self.height_pt / self.height_px
        return (box[0] * sx, box[1] * sy, box[2] * sx, box[3] * sy)


@dataclass(frozen=True)
class Scored:
    """The guard's verdict on one word. Internal: the guard returns these and the engine copies them onto `Word`."""

    id: str
    p_wrong: float | None
    status: str
    reasons: tuple[Reason, ...] = ()
    suggestion: str | None = None
    suggestion_conf: float | None = None
    fix: Fix | None = None


@dataclass(frozen=True)
class Cell:
    """A field of an invoice: its text and the words it was read from.

    Attributes:
        text: The text of the field as read, such as `"101.913,81"`. Not a number: the number is in
            `Document.numbers`.
        word_ids: The ids of the words the text was made from.
        page: The page number the field is on. Fields are read from page 1.

    See also:
        `Invoice`, `Document.field_status`.
    """

    text: str
    word_ids: tuple[str, ...]
    page: int


@dataclass(frozen=True)
class VatPair:
    """A VAT rate with its base and amount, each a `Cell` or `None`.

    Attributes:
        rate: The `Cell` of the rate, as printed.
        amount: The VAT amount.
        base: The taxable base next to the rate, when the invoice prints one. `None` otherwise.

    See also:
        `Invoice`, `Cell`.
    """

    rate: Cell | None
    amount: Cell | None
    base: Cell | None = None


@dataclass(frozen=True)
class InvoiceLine:
    """One line of an invoice. A column that was not found is `None`.

    Attributes:
        description: The description of the line.
        quantity: The quantity.
        unit_price: The price of one unit.
        amount: The amount of the line, which the rule `checks.rule.line_amount` compares with quantity times price.

    See also:
        `Invoice`, `Cell`.
    """

    description: Cell | None = None
    quantity: Cell | None = None
    unit_price: Cell | None = None
    amount: Cell | None = None


@dataclass(frozen=True)
class InvoiceFields:
    """The fields of an invoice. A field that was not found is `None`, never a guess.

    Only page 1 is read for fields, so a multi-page invoice is not covered. A layout that Extract has not seen can
    leave a field empty. Each field is a `Cell`; its trust is `Document.field_status(cell)`.

    Attributes:
        number: The invoice number.
        date: The date of the invoice.
        subtotal: The amount before taxes, labelled `base imponible`, `subtotal` or similar.
        vat: The VAT lines, as `VatPair` objects. Empty when none was found.
        withholding: The withholding amount. It is never `checked`, because a document may print it with or without
            a minus sign.
        total: The total.
        currency: `"EUR"`, `"USD"` or `"GBP"`, from a symbol or a code on the page, or `None`. A plain string, not a
            `Cell`.
        lines: The lines of the invoice, as `InvoiceLine` objects. Empty when no table was found.
        withholding_rate: The `Cell` of the withholding rate, as printed.

    Example:
        ```python
        import extract

        doc = extract.load("invoice.extract.json")
        invoice = doc.invoice
        print(invoice.number.text, invoice.date.text, invoice.currency, len(invoice.lines), len(invoice.vat))
        total = invoice.total
        print(total.text, total.word_ids, total.page, doc.field_status(total), invoice.withholding)
        ```

        ```text
        # output
        F-2026-0498 20/01/2025 EUR 4 1
        101.913,81 ('p1w75',) 1 mismatch None
        ```

    See also:
        `Document.invoice`, `Cell`, `VatPair`, `InvoiceLine`, `Document.field_status`.
    """

    number: Cell | None = None
    date: Cell | None = None
    subtotal: Cell | None = None
    vat: tuple[VatPair, ...] = ()
    withholding: Cell | None = None
    total: Cell | None = None
    currency: str | None = None
    lines: tuple[InvoiceLine, ...] = ()
    withholding_rate: Cell | None = None


@dataclass(frozen=True)
class RuleResult:
    """One rule that involves a number, and whether the number passed it.

    A rule is `verifying` when it would fail for any wrong character in the number. Only a verifying rule that passes
    can make a number `checked`; the plausibility rules can only fail.

    The keys, all in the family `checks.rule.*`: `dni_letter`, `nie_letter` (verifying), `cif_control`, `iban_length`,
    `number_format`, `date_valid` and `vat_rate_plausible` (not verifying), `iban_mod97` (verifying only for countries
    whose layout has no letter-or-digit position), and the arithmetic rules `line_amount`, `lines_sum`, `vat_amount`,
    `withholding_amount` and `total` (verifying when a change of one in the last printed decimal moves the result by at
    least a cent).

    Attributes:
        key: The rule, such as `"checks.rule.total"`.
        params: The values the rule used. For `checks.rule.total` they are `expected` and `found`. Decimals are strings.
        result: `"pass"`, `"fail"` or `"not_applicable"`.
        verifying: Whether the rule is a verifying one.

    See also:
        `Number`, `Document.numbers`.
    """

    key: str
    params: Mapping[str, Any]
    result: str
    verifying: bool


@dataclass(frozen=True)
class DigitSuggestion:
    """A one-digit change that the rules offer for a number: the character at `position` goes from `old` to `new`.

    A tie between two changes gives no suggestion at all, and only a number written in a single word can be applied
    with `Document.use_suggestion`.

    Attributes:
        position: The index in `Number.text` of the character to change, counting from 0.
        old: The character that is there now.
        new: The character to put there.
        verified: True only when a verifying rule fails with the number as read and passes with this one change
            and with no other, and no plausibility rule objects.
        rule: The key of the verifying rule when `verified` is true, or of the failing rule the change would satisfy
            when it is not.

    See also:
        `Number`, `Document.use_suggestion`.
    """

    position: int
    old: str
    new: str
    verified: bool = False
    rule: str = ""


@dataclass(frozen=True)
class NumberResult:
    """A number of the document and what the rules say about it.

    A number on an unreadable page is never `checked`, and `Document.accept` never changes the status: only the
    rules do. `unverified` is not an error: a CIF, a French IBAN or an amount written with a space as thousands
    separator cannot be confirmed.

    Attributes:
        word_ids: The ids of the words the number is made of: one, or the two to five words of an IBAN written in
            groups.
        text: The number as written, such as `"1.893,85"`; the words of a grouped IBAN are joined with a space.
        value: The exact value as a `Decimal` for `amount`, `percent`, `integer` and `quantity`; `None` for `dni`,
            `nie`, `cif`, `iban`, `date` and `other`, and for an amount that does not parse.
        kind: What kind of number it is: `dni`, `nie`, `cif`, `iban`, `percent`, `date`, `integer`, `amount`,
            `quantity` or `other`.
        status: `checked` (a verifying rule passes and none fails), `mismatch` (a rule fails: some digit is wrong,
            not necessarily the one suggested) or `unverified` (nothing confirms it and nothing contradicts it).
        rules: The `Rule` of every rule that applies to the number.
        suggestion: A `DigitSuggestion`, or `None` when there is none.

    Example:
        ```python
        import extract

        doc = extract.load("invoice.extract.json")
        total = next(number for number in doc.numbers if number.text == "101.913,81")
        print(total.kind, total.value, total.status, total.word_ids)
        for rule in total.rules:
            print(rule.key, rule.result, rule.verifying)
        fix = total.suggestion
        print(fix.position, fix.old, fix.new, fix.verified)
        ```

        ```text
        # output
        amount 101913.81 mismatch ('p1w75',)
        checks.rule.number_format pass False
        checks.rule.total fail True
        8 8 5 True
        ```

    See also:
        `Document.numbers`, `Rule`, `DigitSuggestion`, `Document.use_suggestion`.
    """

    word_ids: tuple[str, ...]
    text: str
    value: Decimal | None
    kind: str
    status: str
    rules: tuple[RuleResult, ...] = ()
    suggestion: DigitSuggestion | None = None


@dataclass(frozen=True)
class DocumentData:
    """Everything a document holds, as one immutable snapshot: what the outputs are written from and what `save`
    stores. Corrections make a new snapshot. Internal: a `Document` wraps one, and you do not make it yourself."""

    source: str
    sha256: str
    pages: tuple[PageReading, ...]
    engine: str
    guard: str | None = None
    options: Mapping[str, Any] = field(default_factory=dict)
    doc_type: str = "generic"
    locale: str | None = None
    numbers: tuple[NumberResult, ...] = ()
    invoice: InvoiceFields | None = None

    @property
    def unreadable_pages(self) -> tuple[int, ...]:
        return tuple(p.number for p in self.pages if p.status == "unreadable")

    @property
    def status(self) -> str:
        return "partial" if self.unreadable_pages else "done"
