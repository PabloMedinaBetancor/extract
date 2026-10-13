"""Applying the guard's proposals to a page: modes, page brake, overlaps, structure fixes, undo, the token
view and the verified digit fixes of the number rules.

The guard only proposes: every word it has a candidate for carries a `Fix` (`Word.fix`), applied or not. Everything
here is a pure function of those stored proposals, the user's decisions (`Word.review`) and the mode, so a mode
switch needs no engine and no guard. The statistics live in the guard (`Fix.modes`); this module holds none."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import replace
from typing import Any

from .events import Warning as JobWarning
from .outputs.helvetica import WIDTHS
from .types import (
    Box,
    DocumentData,
    Fix,
    NumberResult,
    PageReading,
    Word,
    base_id,
)

BRAKE_SHARE = 0.08  # more than this share of a page's words changed: no fix of that page is applied
RULE_PREFIX = "rule:"
DIGIT_MODES = ("safe", "more")
# In `safe` only the fixes by a check digit apply alone; a fix by invoice arithmetic (line, sum, VAT,
# withholding, total) stays a marked suggestion on the number (the interface shows it, a click applies it); `more` applies both.
ARITHMETIC_RULES = frozenset(
    f"checks.rule.{name}" for name in ("line_amount", "lines_sum", "vat_amount", "withholding_amount", "total")
)
_INNER_CAPITALS = re.compile(r"[A-Z]")


def _text_rules(text: str) -> bool:
    """The part of the guard's protection rules (`PageContext.is_site`) that depends on the token alone, with
    the guard's own constants and splitter so there is one definition: a token that is not a plain word, is longer
    than `MAX_LENGTH`, holds `@ / _ # \\`, has a hyphen or apostrophe at either end, a colon that is not last or follows
    a URL scheme, or two capitals in a word that is not all capitals. What needs the whole page (a repeated acronym)
    stays with the guard."""
    from .guard import candidates, forms

    parts = forms.split_word(text)
    if parts is None or len(text) > candidates.MAX_LENGTH or candidates.BLOCKED_CHARACTERS & set(text):
        return True
    prefix, core, suffix = parts
    marks = forms.normalise(prefix + suffix)
    if "-" in marks or "'" in marks or candidates.FINAL_MARK in prefix or candidates.FINAL_MARK in suffix[:-1]:
        return True
    if candidates.FINAL_MARK in suffix and core.casefold() in candidates.URL_SCHEMES:
        return True
    return sum(1 for ch in core if ch.isupper()) >= 2 and not core.isupper()


def _mark_token_protected(text: str) -> bool:
    """A token the guard's splitter does not read as a word (marks and symbols only) is still a speck that `drop` may
    remove, unless it holds a protected character, a hyphen or an apostrophe (`structure.drop_ok`)."""
    from .guard import candidates, forms

    if len(text) > candidates.MAX_LENGTH or candidates.BLOCKED_CHARACTERS & set(text):
        return True
    if any(c.isalnum() for c in text):
        return True
    marks = forms.normalise(text)
    return "-" in marks or "'" in marks


def is_protected(text: str, op: str = "text") -> bool:
    """A token the guard never touches by structure or spelling: it has a digit, `@`, `/`, `_` or two inner capitals
    (A2.4/A2.5), or the guard's own text rules exclude it. A digit changes only through the number rules. For `drop`
    a token of marks only is not protected by the word rules (the guard drops `·`, `|`, `.`, `•`), only by its
    characters."""
    if any(ch.isdigit() or ch in "@/_" for ch in text):
        return True
    if op == "drop":
        try:
            from .guard import forms

            if forms.split_word(text) is None:
                return _mark_token_protected(text)
        except ImportError:
            pass
    if not text.isupper() and len(_INNER_CAPITALS.findall(text[1:])) >= 2:
        return True
    try:
        return _text_rules(text)
    except ImportError:  # the guard is not installed: the basic rules above are all there is
        return False


def is_applied(word: Word) -> bool:
    """True for the word that carries an applied fix (a join's first word, a dropped word, a changed text); the words
    a join absorbed are not owners."""
    return (
        word.fix is not None
        and word.corrected_by == "guard"
        and word.status in ("fixed", "absorbed")
        and word.absorbed_by is None
    )


def _is_rule(fix: Fix | None) -> bool:
    return fix is not None and fix.source.startswith(RULE_PREFIX)


def _union(boxes: Iterable[Box]) -> Box:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _with_proposal(word: Word) -> Word:
    """The word as the guard gave it, its proposal shown as a suggestion when it is a doubtful word and the fix is a
    plain text change (v4's button keeps working)."""
    fix = word.fix
    if fix is None or fix.op != "text" or word.status != "doubtful" or _is_rule(fix):
        return word
    return replace(word, suggestion=fix.text, suggestion_conf=fix.p_fix)


def _restore_group(words: list[Word], index: dict[str, int], owner: Word, *, keep_mark: bool) -> None:
    """Put the words an applied fix touched back as the guard gave them. `keep_mark` leaves `guard_status` on the
    owner (what an undo does, so the report can count it as undone) instead of clearing it."""
    for word_id in owner.fix.ids if owner.fix is not None and owner.fix.ids else (owner.id,):
        i = index.get(word_id)
        if i is None:
            continue
        word = words[i]
        if word.id == owner.id:
            continue
        words[i] = replace(
            word,
            text=word.original if word.original is not None else word.text,
            original=None,
            status=word.guard_status or "ok",
            absorbed_by=None,
            guard_status=None,
            corrected_by=None,
        )
    i = index[owner.id]
    status = owner.guard_status or "ok"
    restored = replace(
        owner,
        text=owner.original if owner.original is not None else owner.text,
        box=owner.orig_box or owner.box,
        orig_box=None,
        original=None,
        status=status,
        corrected_by=None,
        absorbed_by=None,
        guard_status=owner.guard_status if keep_mark else None,
    )
    words[i] = _with_proposal(replace(restored, suggestion=None, suggestion_conf=None))


def _valid_group(fix: Fix, owner: Word, words: Sequence[Word], index: dict[str, int]) -> list[Word] | None:
    """The words a proposal spans, if it can be applied to them; None when it cannot."""
    ids = fix.ids or (owner.id,)
    if ids[0] != owner.id or any(i not in index for i in ids):
        return None
    group = [words[index[i]] for i in ids]
    if fix.op != "join" and len(group) != 1:
        return None
    if fix.op == "join":
        positions = [index[i] for i in ids]
        if len(group) < 2 or positions != list(range(positions[0], positions[0] + len(group))):
            return None
        if len({w.line_id for w in group}) != 1:
            return None
    if any(w.review is not None or is_applied(w) or w.status in ("absorbed", "fixed") for w in group):
        return None
    if any(is_protected(w.text, fix.op) for w in group):
        return None
    if fix.op != "drop" and not fix.text.strip():
        return None
    if fix.op == "segment" and " " not in fix.text.strip():
        return None
    return group


def apply_fixes(
    page: PageReading, mode: str, warn: Callable[[JobWarning], None] | None = None
) -> tuple[PageReading, int]:
    """The page with the proposals of the guard applied for `mode` (`off`, `safe` or `more`) and how many were.

    Words the user decided (`review` set) never move. Fixes are applied in reading order; one that touches a word an
    applied fix already used is skipped. If the fixes of the mode would change more than 8 % of the page's words, none
    is applied and `guard.warning.fixes_withheld` is sent. Calling it again with another mode rebuilds the page from
    the stored proposals."""
    if not page.words:
        return page, 0
    words = list(page.words)
    index = {w.id: i for i, w in enumerate(words)}
    for word in tuple(words):
        if is_applied(word) and word.review is None:
            _restore_group(words, index, word, keep_mark=False)
        elif word.fix is not None and not is_applied(word) and word.status not in ("absorbed", "fixed"):
            words[index[word.id]] = _with_proposal(word)
    if mode == "off":
        return replace(page, words=tuple(words)), 0
    plan: list[tuple[Fix, list[Word]]] = []
    used: set[str] = set()
    for word in words:
        fix = word.fix
        if fix is None or _is_rule(fix) or mode not in fix.modes:
            continue
        group = _valid_group(fix, word, words, index)
        if group is None or any(w.id in used for w in group):
            continue
        used.update(w.id for w in group)
        plan.append((fix, group))
    changed = sum(len(group) for _, group in plan)
    if plan and changed / len(words) > BRAKE_SHARE:
        if warn is not None:
            warn(
                JobWarning(
                    "guard.warning.fixes_withheld", {"page": page.number, "share": round(changed / len(words), 4)}
                )
            )
        return replace(page, words=tuple(words)), 0
    for fix, group in plan:
        owner = group[0]
        if fix.op == "drop":
            words[index[owner.id]] = replace(
                owner,
                text="",
                original=owner.text,
                status="absorbed",
                corrected_by="guard",
                suggestion=None,
                suggestion_conf=None,
                guard_status=owner.status,
            )
            continue
        box = _union(w.box for w in group) if fix.op == "join" else owner.box
        words[index[owner.id]] = replace(
            owner,
            text=fix.text,
            box=box,
            orig_box=owner.box if box != owner.box else None,
            original=owner.text,
            status="fixed",
            corrected_by="guard",
            suggestion=None,
            suggestion_conf=None,
            guard_status=owner.status,
        )
        for member in group[1:]:
            words[index[member.id]] = replace(
                member,
                text="",
                original=member.text,
                status="absorbed",
                absorbed_by=owner.id,
                guard_status=member.status,
                suggestion=None,
                suggestion_conf=None,
            )
    return replace(page, words=tuple(words)), len(plan)


def undo_words(page: PageReading, word_id: str | None = None) -> tuple[PageReading, int]:
    """Undo the applied fix of `word_id` (or of every word of the page when None): the text, id, box and status come
    back as the guard gave them, the word is marked `accepted` so that a mode switch never applies it again, and the
    proposal stays on it. Returns the page and how many fixes were undone."""
    words = list(page.words)
    index = {w.id: i for i, w in enumerate(words)}
    owners = [w for w in words if is_applied(w)]
    if word_id is not None:
        base = words[index[base_id(word_id)]]
        owner = base if is_applied(base) else words[index[base.absorbed_by]] if base.absorbed_by in index else base
        owners = [owner] if is_applied(owner) else []
    count = 0
    for owner in owners:
        fresh = words[index[owner.id]]
        _restore_group(words, index, fresh, keep_mark=True)
        words[index[owner.id]] = replace(words[index[owner.id]], review="accepted")
        count += 1
    return replace(page, words=tuple(words)), count


def _helvetica_width(text: str) -> float:
    total = 0
    for char in text:
        try:
            total += WIDTHS[char.encode("cp1252")[0]]
        except (UnicodeEncodeError, IndexError):
            total += WIDTHS[ord("?")]
    return float(total)


def tokens(page: PageReading) -> tuple[Word, ...]:
    """The words as the PDF layer, the extraction and the number rules see them: absorbed words removed and a
    segmented word split into tokens `id#k` whose boxes share the word's box in proportion to the Helvetica widths of
    their characters. A word with no structure fix is its own token."""
    out: list[Word] = []
    for word in page.words:
        if word.status == "absorbed":
            continue
        parts = word.text.split()
        if is_applied(word) and word.fix is not None and word.fix.op == "segment" and len(parts) > 1:
            gap = _helvetica_width(" ")
            widths = [_helvetica_width(part) for part in parts]
            total = sum(widths) + gap * (len(parts) - 1)
            left, top, right, bottom = word.box
            span = right - left
            cursor = 0.0
            for k, (part, width) in enumerate(zip(parts, widths, strict=True), 1):
                x0 = left + span * cursor / total
                x1 = left + span * (cursor + width) / total
                out.append(replace(word, id=f"{word.id}#{k}", text=part, box=(x0, top, x1, bottom)))
                cursor += width + gap
        else:
            out.append(word)
    return tuple(out)


def token_page(page: PageReading) -> PageReading:
    """The page with its words replaced by the token view; the page itself when nothing is absorbed or segmented."""
    if not any(w.status == "absorbed" or (w.fix is not None and w.fix.op == "segment") for w in page.words):
        return page
    return replace(page, words=tokens(page))


def token_pages(pages: Sequence[PageReading]) -> tuple[PageReading, ...]:
    return tuple(token_page(p) for p in pages)


def _letter_may_be_the_error(kind: str, word: Word) -> bool:
    """True when the identifier would pass its own rule by changing one LETTER to one of the recogniser's alternatives
    for that position: the error is then as likely in the letter as in the digit, and a digit fix of the number as read
    would turn a good digit into a bad one (N7, letter errors: 5,6 % of DNI with a wrong letter got a false verified fix)."""
    from .checks import IDENTIFIER_CHECKS

    check = IDENTIFIER_CHECKS.get(kind)
    alts = word.char_alts
    if check is None or alts is None:
        return False
    text = word.text
    for i, char in enumerate(text):
        if char.isdigit() or i >= len(alts):
            continue
        for alternative, _ in alts[i]:
            if alternative.upper() != char.upper() and check(text[:i] + alternative.upper() + text[i + 1 :]):
                return True
    return False


def apply_digit_fixes(
    pages: Sequence[PageReading], numbers: Sequence[NumberResult], mode: str
) -> tuple[tuple[PageReading, ...], int]:
    """Step 7b: a digit changes only when a verifying rule fails now and passes with that one change and only with it
    (`DigitSuggestion.verified`), the new digit is among the recogniser's alternatives for that position and the mode
    is `safe` or `more` (A2.5). Returns the pages and how many digit fixes were applied."""
    if mode not in DIGIT_MODES:
        return tuple(pages), 0
    by_page = {p.number: p for p in pages}
    edits: dict[str, tuple[Fix, str]] = {}
    for number in numbers:
        suggestion = number.suggestion
        if suggestion is None or not suggestion.verified or len(number.word_ids) != 1:
            continue
        if mode == "safe" and suggestion.rule in ARITHMETIC_RULES:
            continue
        word_id = number.word_ids[0]
        if word_id in edits or "#" in word_id:
            continue
        page = by_page.get(int(re.match(r"p(\d+)w", word_id).group(1)))
        word = next((w for w in page.words if w.id == word_id), None) if page else None
        if word is None or word.review is not None or word.corrected_by is not None or word.status == "absorbed":
            continue
        position = suggestion.position
        if word.text[position : position + 1] != suggestion.old:
            continue
        if _letter_may_be_the_error(number.kind, word):
            continue
        text = word.text[:position] + suggestion.new + word.text[position + 1 :]
        stored = word.fix if _is_rule(word.fix) and word.fix.text == text else None
        alts = word.char_alts
        in_alts = alts is not None and position < len(alts) and any(c == suggestion.new for c, _ in alts[position])
        if not in_alts and stored is None:
            continue
        fix = stored or Fix(
            op="text",
            kind="number",
            text=text,
            ids=(word_id,),
            source=f"{RULE_PREFIX}{suggestion.rule}",
            sources=(f"{RULE_PREFIX}{suggestion.rule}",),
            p_fix=1.0,
            p_keep=0.0,
            rule=suggestion.rule,
            modes=("more",) if suggestion.rule in ARITHMETIC_RULES else DIGIT_MODES,
            reasons=(),
        )
        edits[word_id] = (fix, text)
    if not edits:
        return tuple(pages), 0
    out = []
    for page in pages:
        words = tuple(
            replace(
                w,
                text=edits[w.id][1],
                original=w.text,
                status="fixed",
                corrected_by="guard",
                fix=edits[w.id][0],
                guard_status=w.status,
                suggestion=None,
                suggestion_conf=None,
            )
            if w.id in edits
            else w
            for w in page.words
        )
        out.append(replace(page, words=words) if any(w.id in edits for w in page.words) else page)
    return tuple(out), len(edits)


def _undone(word: Word) -> bool:
    return word.fix is not None and word.guard_status is not None and not is_applied(word) and word.status != "absorbed"


def summary(data: DocumentData) -> dict[str, Any]:
    """The `fixes` block of the document: the mode, whether the second reading ran, how many fixes are applied and how
    many were undone, and the applied ones by source and by operation; `second_reading_failed_pages` are the pages that
    did not get their second reading (the reader process was not there or was cut by its memory cap)."""
    applied = [w for p in data.pages for w in p.words if is_applied(w)]
    undone = sum(1 for p in data.pages for w in p.words if _undone(w))
    by_source = Counter(w.fix.source for w in applied)
    by_op = Counter(w.fix.op for w in applied)
    withheld = list(data.options.get("fixes_withheld", ()))
    return {
        "mode": data.options.get("correct", "off"),
        "second_reading": bool(data.options.get("second_reading", False)),
        "thorough": data.options.get("second_reading") == "thorough",
        "applied": len(applied),
        "undone": undone,
        "by_source": dict(sorted(by_source.items())),
        "by_op": dict(sorted(by_op.items())),
        "withheld": withheld,
        "second_reading_failed_pages": list(data.options.get("second_reading_failed_pages", ())),
    }


def corrections(data: DocumentData) -> list[tuple[int, Word, str]]:
    """Every change to the text, in reading order, with its state: `applied` or `undone` for a fix of the guard, `user`
    for a word whose text the user typed over (the text differs from what was read)."""
    rows = []
    for page in data.pages:
        for word in page.words:
            if is_applied(word):
                rows.append((page.number, word, "applied"))
            elif _undone(word):
                rows.append((page.number, word, "undone"))
            elif word.review == "corrected" and word.corrected_by == "user" and word.original != word.text:
                rows.append((page.number, word, "user"))
    return rows
