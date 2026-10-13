"""Stand-ins for the guard in the tests of the pipeline: one that finds every word fine, one that doubts chosen words."""

from extract.types import Reason, Scored


class FakeGuard:
    info = {"model_id": "fake-guard"}

    def __init__(self):
        self.calls = []

    def score_page(self, page, *, languages, locale, budget):
        self.calls.append({"languages": languages, "locale": locale, "budget": budget})
        return [Scored(w.id, 0.01, "ok") for w in page.words]


class DoubtingGuard:
    """Doubts every word whose text is in `doubts`, with a suggestion when one is given."""

    info = {"model_id": "doubting-guard"}

    def __init__(self, doubts):
        self.doubts = dict(doubts)

    def score_page(self, page, *, languages, locale, budget):
        scored = []
        for word in page.words:
            if word.text in self.doubts:
                reason = Reason("guard.reason.not_in_dictionary")
                scored.append(Scored(word.id, 0.8, "doubtful", (reason,), self.doubts[word.text], 0.7))
            else:
                scored.append(Scored(word.id, 0.01, "ok"))
        return scored


class FailingGuard:
    info = {"model_id": "failing-guard"}

    def score_page(self, page, *, languages, locale, budget):
        raise RuntimeError("the guard broke")


class PartialGuard:
    """Gives a verdict for the first `count` words only, and calls the next one `unscored`."""

    info = {"model_id": "partial-guard"}

    def __init__(self, count):
        self.count = count

    def score_page(self, page, *, languages, locale, budget):
        out = [Scored(w.id, 0.01, "ok") for w in page.words[: self.count]]
        if len(page.words) > self.count:
            out.append(Scored(page.words[self.count].id, None, "unscored"))
        return out
