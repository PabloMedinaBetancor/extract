"""Which proposals of a page are applied when some of them touch the same words.

Proposals are taken in decreasing gain (`p_fix - p_keep`); one that shares a word with a proposal already taken is
kept as a suggestion only. The order of ties is fixed (span, op, text), so the result never depends on the order the
proposals arrive in. The application in reading order that skips what overlaps an applied fix (the app's rule) then
finds nothing to skip.
"""

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Proposal:
    span: tuple[int, ...]
    op: str
    gain: float
    text: str = ""


def select(proposals: Iterable[Proposal]) -> tuple[list[Proposal], list[Proposal]]:
    """(applied in reading order, suggested in the order they lost)."""
    taken: set[int] = set()
    applied: list[Proposal] = []
    suggested: list[Proposal] = []
    for proposal in sorted(proposals, key=lambda p: (-p.gain, p.span, p.op, p.text)):
        if taken.intersection(proposal.span):
            suggested.append(proposal)
        else:
            taken.update(proposal.span)
            applied.append(proposal)
    return sorted(applied, key=lambda p: (p.span, p.op, p.text)), suggested


def in_reading_order(proposals: Iterable[Proposal]) -> list[Proposal]:
    """The application rule of the app: reading order, skipping a proposal that overlaps one already applied."""
    taken: set[int] = set()
    applied = []
    for proposal in sorted(proposals, key=lambda p: (p.span, p.op, p.text)):
        if not taken.intersection(proposal.span):
            taken.update(proposal.span)
            applied.append(proposal)
    return applied
