"""A stand-in for the real guard's proposals (the guard's `Fix`): it proposes a fix of every kind (text, join of two and of three,
segment, drop) over the words of a page, so the application of fixes, the modes, undo and the outputs can be tested
before the real guard hands out `Fix`es. Nothing here is a statistic."""

from __future__ import annotations

from dataclasses import replace

from extract.types import Fix, PageReading, Reason, Scored

REASON = Reason("guard.fix.dictionary_nearest", {"word": "x"})


def fix(op, kind, text, ids, *, source="dictionary", modes=("safe", "more"), p_fix=0.97, p_keep=0.02):
    return Fix(op, kind, text, tuple(ids), source, (source,), p_fix, p_keep, None, tuple(modes), (REASON,))


def with_fixes(page: PageReading, proposals: dict[str, Fix], status: str = "doubtful") -> PageReading:
    """The page with each proposal hung from the word whose id is the key, those words marked as the guard marks a
    processed word (`status`)."""
    words = tuple(
        replace(w, status=status, p_wrong=0.7, fix=proposals[w.id])
        if w.id in proposals
        else replace(w, status="ok", p_wrong=0.01)
        for w in page.words
    )
    return replace(page, words=words)


class FixGuard:
    """Gives every word `ok` and hangs `proposals` (word id -> Fix) on the page they belong to."""

    info = {"model_id": "fix-guard"}

    def __init__(self, proposals: dict[str, Fix], status: str = "doubtful"):
        self.proposals = dict(proposals)
        self.status = status
        self.last_warnings: list[dict] = []

    def score_page(self, page, *, languages, locale, budget):
        out = []
        for word in page.words:
            proposal = self.proposals.get(word.id)
            if proposal is None:
                out.append(Scored(word.id, 0.01, "ok"))
            else:
                out.append(Scored(word.id, 0.7, self.status, (REASON,), None, None, proposal))
        return out
