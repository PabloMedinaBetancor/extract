"""Structure candidates: joins, segmentations and drops.

A join site is a run of two or three adjacent plain tokens of one line; a segment site and a drop site are one token.
Each family is a site of its own with the tokens as read as its first candidate, so the chooser decides every
structure edit against "leave it" alone and the selection (`select.py`) settles the overlaps afterwards. Only
digit-free, unprotected tokens take part (`PageContext.is_site`), a span never crosses a line and never bridges a gap
wider than `JOIN_GAP` line heights (columns, tables).
"""

import itertools
from collections.abc import Sequence

from rapidfuzz.distance import Levenshtein

from extract.guard.candidates import BLOCKED_CHARACTERS, KEEP, MAX_CANDIDATES, Candidate, PageContext, source_rank
from extract.guard.edits import weighted_distance
from extract.guard.forms import apply_case, normalise, split_word

FAMILIES = ("join", "segment", "drop")
JOIN_GAP = 1.2
MAX_JOIN = 3
JOIN_EDIT_WORDS = 3
SEGMENT_MIN_LETTERS = 5
SEGMENT_PARTS = 3
SEGMENT_CANDIDATES = 6
SEGMENT_EDITS = 2
SEGMENT_EDIT_COST = 1.0
SEGMENT_LENGTH_BONUS = 0.5
PIECE_MIN_EDIT_LETTERS = 3
SINGLE_LETTER_PARTS = frozenset("ayoeu")
ROMAN_PARTS = frozenset("IVX")
# Which joined texts are proposed as written: only a listed word or what a reader read ("listed"), those and the words
# the page itself repeats ("listed_or_page"), or any ("any", for measuring the cost of the restriction).
JOIN_AS_WRITTEN = "listed_or_page"
DROP_MAX_CHARACTERS = 2
DROP_CONFIDENCE = 0.99
DROP_SMALL_BOX = 0.7


def gap_ratio(ctx: PageContext, left: int, right: int) -> float:
    """The empty space between two boxes of a line, in line heights (0 when they touch or overlap)."""
    height = max(ctx.line_height(left), 1e-6)
    return max(ctx.words[right].box[0] - ctx.words[left].box[2], 0.0) / height


def _next_to(ctx: PageContext, left: int) -> bool:
    """Whether word `left + 1` may be joined to `left`: the same line, the next place on it, both plain, nothing
    attached between them and a small gap."""
    right = left + 1
    if right >= len(ctx.words):
        return False
    first, second = ctx.words[left], ctx.words[right]
    if first.line_id != second.line_id or second.idx_in_line != first.idx_in_line + 1:
        return False
    if not (ctx.is_site(left) and ctx.is_site(right)) or ctx.parts[left][2] or ctx.parts[right][0]:
        return False
    return gap_ratio(ctx, left, right) <= JOIN_GAP


def join_spans(ctx: PageContext) -> list[tuple[int, ...]]:
    """Every run of two or three words that a join could replace, in reading order, shorter runs first."""
    spans = []
    for start in range(len(ctx.words)):
        for length in range(2, MAX_JOIN + 1):
            if not all(_next_to(ctx, start + k) for k in range(length - 1)):
                break
            spans.append(tuple(range(start, start + length)))
    return spans


def segment_ok(ctx: PageContext, index: int) -> bool:
    return ctx.is_site(index) and len(ctx.parts[index][1]) >= SEGMENT_MIN_LETTERS


def drop_ok(ctx: PageContext, index: int) -> bool:
    """A speck: one or two characters, no digit and none of the protected ones, read with doubt or drawn small against
    the line. A token of letters must be plain (a listed letter or two is no proof: a speck is often read as an `o` or
    an `e`); one of marks and symbols only qualifies when it holds no hyphen or apostrophe (a lone dash or bullet that
    is a real one is for the chooser to keep: it sees the box and the confidence)."""
    word = ctx.words[index]
    text = word.text
    if len(text) > DROP_MAX_CHARACTERS or index in ctx.fragments or BLOCKED_CHARACTERS & set(text):
        return False
    marks = normalise(text)
    if ctx.parts[index] is not None:
        if not ctx.is_site(index):
            return False
    elif any(c.isalnum() for c in text) or "-" in marks or "'" in marks:
        return False
    small = (word.box[3] - word.box[1]) <= DROP_SMALL_BOX * ctx.line_height(index)
    return small or (word.conf is not None and word.conf < DROP_CONFIDENCE)


def listed(ctx: PageContext, letters: str) -> bool:
    return any(book.rank_of(letters) >= 0 for book in ctx.books)


def frequency(ctx: PageContext, letters: str) -> float:
    return max((book.log_frequency(letters) for book in ctx.books), default=0.0)


def reader_groups(ctx: PageContext, span: Sequence[int]) -> list[tuple[str, str]]:
    """(source, text) of every reading that read exactly this span as one word."""
    ids = {ctx.words[i].id for i in span}
    found = []
    for reading in getattr(ctx.words[span[0]], "readings", None) or ():
        if reading.text is not None and set(reading.group) == ids and split_word(reading.text) is not None:
            found.append((reading.source, reading.text))
    return found


def reader_cuts(ctx: PageContext, index: int) -> list[tuple[str, str]]:
    """(source, text) of every reading that cut this word, with the spaces its reader put."""
    own = {ctx.words[index].id}
    found = []
    for reading in getattr(ctx.words[index], "readings", None) or ():
        if reading.text is None or " " not in reading.text.strip() or set(reading.group) != own:
            continue
        if all(split_word(part) is not None for part in reading.text.split()):
            found.append((reading.source, reading.text))
    return found


def join_texts(ctx: PageContext, span: Sequence[int]) -> dict[str, list[str]]:
    """The texts that replace the run: its letters as written, the nearest listed words within one edit of them
    and what a reader read the run as, with the attached punctuation of the run's ends."""
    prefix, suffix = ctx.parts[span[0]][0], ctx.parts[span[-1]][2]
    letters = "".join(ctx.parts[i][1] for i in span)
    out: dict[str, list[str]] = {}
    if (
        JOIN_AS_WRITTEN == "any"
        or listed(ctx, letters)
        or (JOIN_AS_WRITTEN == "listed_or_page" and _on_page(ctx, letters))
    ):
        out["joined"] = [prefix + letters + suffix]
    near = []
    for book in ctx.books:
        for rank, distance in book.candidates(letters, 1):
            if distance == 1:
                near.append(book.words[rank])
    # two listed words side by side are no sign of a split (`de la`): the edit is offered only when a piece is unlisted
    if not all(listed(ctx, ctx.parts[i][1]) for i in span):
        near = sorted(set(near), key=lambda w: (weighted_distance(letters, w), -frequency(ctx, w), w))
        restored = sorted(ctx.fold_group(letters) - {letters})
        edits = [prefix + apply_case(word, letters) + suffix for word in [*restored, *near[:JOIN_EDIT_WORDS]]]
        if edits:
            out["join_edit"] = list(dict.fromkeys(edits))
    for source, text in reader_groups(ctx, span):
        out.setdefault(f"reading:{source}", []).append(text)
    return out


def _on_page(ctx: PageContext, letters: str) -> bool:
    return letters.casefold() in ctx.vocabulary


def cuts(letters: str) -> list[tuple[str, ...]]:
    """Every way to cut `letters` into two or three parts that are each a listed word or one of `ayoeu`."""
    n = len(letters)
    found = [(letters[:i], letters[i:]) for i in range(1, n)]
    found += [(letters[:i], letters[i:j], letters[j:]) for i in range(1, n) for j in range(i + 1, n)]
    return [parts for parts in found if len(parts) <= SEGMENT_PARTS]


def piece_words(ctx: PageContext, piece: str) -> tuple[tuple[str, int, str], ...]:
    """The words a piece of a cut token may stand for, as (text written, cost in edits, the listed word it stands for): itself when listed (or one of `ayoeu`,
    or a capital Roman numeral), itself when it is listed once its accents are folded (the layer drops the accents of
    capitals), and the nearest listed word within one edit, written in the piece's case, for a piece of at least
    `PIECE_MIN_EDIT_LETTERS` letters. Empty when no word fits."""
    if piece in ctx.piece_words:
        return ctx.piece_words[piece]
    found: list[tuple[str, int, str]] = []
    if len(piece) == 1:
        if piece.casefold() in SINGLE_LETTER_PARTS or piece in ROMAN_PARTS:
            found.append((piece, 0, piece))
    elif listed(ctx, piece):
        found.append((piece, 0, piece))
    else:
        restored = ctx.fold_group(piece)
        if restored:
            found.append((piece, 0, max(restored, key=lambda w: (frequency(ctx, w), w))))
        if len(piece) >= PIECE_MIN_EDIT_LETTERS:
            near = {
                w for book in ctx.books for rank, d in book.candidates(piece, 1) if d == 1 for w in (book.words[rank],)
            }
            near |= set(ctx.fold_group(piece))
            if near:
                best = min(near, key=lambda w: (weighted_distance(piece, w), -frequency(ctx, w), w))
                found.append((apply_case(best, piece), 1, best))
    ctx.piece_words[piece] = tuple(found)
    return ctx.piece_words[piece]


def segment_texts(ctx: PageContext, index: int) -> dict[str, list[str]]:
    """The token with the marks attached to its ends set apart (`·Planes` as `· Planes`, `Terreno,` as `Terreno ,`), the
    token cut into dictionary words, each piece as it is or with one dictionary edit and at most
    `SEGMENT_EDITS` edits in all: the likeliest `SEGMENT_CANDIDATES` (the mean frequency of the words, a bonus for the
    shortest piece being long and a price for every edit) and the cuts a reader made."""
    prefix, core, suffix = ctx.parts[index]
    scored = []
    for pieces in cuts(core):
        options = [piece_words(ctx, piece) for piece in pieces]
        if not all(options):
            continue
        for chosen in itertools.product(*options):
            cost = sum(c for _, c, _ in chosen)
            if cost > SEGMENT_EDITS:
                continue
            written = [w for w, _, _ in chosen]
            score = sum(frequency(ctx, w) for _, _, w in chosen) / len(chosen)
            score += SEGMENT_LENGTH_BONUS * min(min(len(p) for p in pieces), 4) - SEGMENT_EDIT_COST * cost
            scored.append(((-score, cost, written), written))
    scored.sort(key=lambda item: item[0])
    out = {"segment": [prefix + " ".join(parts) + suffix for _, parts in scored[:SEGMENT_CANDIDATES]]}
    if prefix or suffix:
        out["segment"].insert(0, (prefix + " " if prefix else "") + core + (" " + suffix if suffix else ""))
    for source, text in reader_cuts(ctx, index):
        out.setdefault(f"reading:{source}", []).append(text)
    return out


def drop_texts(ctx: PageContext, index: int) -> dict[str, list[str]]:
    return {"drop": [""]}


def span_text(ctx: PageContext, span: Sequence[int]) -> str:
    return " ".join(ctx.words[i].text for i in span)


def generate_structure(ctx: PageContext, family: str, span: Sequence[int]) -> list[Candidate]:
    """The candidates of a structure site: the tokens as read first, then at most `MAX_CANDIDATES` - 1 others, one
    per normalised final text, each with every source that proposed it, best supported first."""
    span = tuple(span)
    if family == "join":
        proposals = join_texts(ctx, span)
    elif family == "segment":
        proposals = segment_texts(ctx, span[0])
    elif family == "drop":
        proposals = drop_texts(ctx, span[0])
    else:
        raise ValueError(f"unknown structure family {family!r}")
    token = span_text(ctx, span)
    found: dict[str, tuple[str, list[str]]] = {}
    for source, texts in proposals.items():
        for text in texts:
            sources = found.setdefault(normalise(text), (text, []))[1]
            if source not in sources:
                sources.append(source)
                sources.sort(key=source_rank)
    # more sources first, then the order in which the family listed them (likeliest first), then the text
    order: dict[str, int] = {}
    for texts in proposals.values():
        for text in texts:
            order.setdefault(text, len(order))
    ranked = sorted(found.values(), key=lambda item: (-len(item[1]), order[item[0]], item[0]))
    out = [Candidate(token, "keep", KEEP, (KEEP,), span)]
    for text, sources in ranked[: MAX_CANDIDATES - 1]:
        out.append(Candidate(text, family, "structure", tuple(sources), span))
    return out


def distance(token: str, text: str) -> int:
    """Characters that change, spaces not counted."""
    return Levenshtein.distance(token.replace(" ", ""), text.replace(" ", ""))
