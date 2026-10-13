"""Pass 2 of the second reading for one page: the readers of a cell read the crops pass 1 stored, their text is aligned
to the first reading's words inside each unit, and each word of a read unit gets one `Reading` for every reader that
ran on it (those that agree with the first reading included)."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping, Sequence

from ..types import PageReading, Reading
from .crops import decode
from .reread import RereadConfig, align

REPORT_KEYS = ("units_read", "units_skipped", "reread_unaligned", "reread_words", "reader_failed")


def second_pass(
    page: PageReading,
    crops: Mapping[str, bytes],
    readers,
    unit: str,
    config: RereadConfig,
    names: Sequence[str] | None = None,
    append: bool = False,
) -> PageReading:
    """`page` with `readings` on the words of the units that have a stored crop and `rereads` and the pass 2 counts in
    `reread_report` (`units_read`, `units_skipped` for a unit without a crop, `reread_unaligned` for each reader and
    unit that could not be aligned). `readers` has `names` and `read_many(name, crops)`. With `append` the page already
    has the readings, `rereads` and counts of an earlier pass (the thorough mode runs the lines, then the zones): the new
    ones are added to them, nothing is replaced."""
    names = tuple(names if names is not None else readers.names)
    words = page.words
    units = config.units(words, unit)
    todo = [(u, crops[f"{unit}{u.rank}"]) for u in units if f"{unit}{u.rank}" in crops]
    images = [decode(data) for _, data in todo]
    by_word: dict[int, list[Reading]] = {}
    unaligned = 0
    for name in names:
        outputs = readers.read_many(name, images)
        for (u, _), (text, score) in zip(todo, outputs, strict=True):
            first = [words[i].text for i in u.indices]
            aligned = align(first, text.split())
            if aligned is None:
                unaligned += 1
            for position, index in enumerate(u.indices):
                if aligned is None:
                    reading = Reading(name, None, None, None, (), u.rank, index not in u.suspects)
                else:
                    a = aligned[position]
                    group = tuple(words[u.indices[g]].id for g in a.group)
                    reading = Reading(
                        name, "" if a.text is None else a.text, score, None, group, u.rank, index not in u.suspects
                    )
                by_word.setdefault(index, []).append(reading)
    before = (lambda w: tuple(w.readings or ())) if append else (lambda w: ())
    new_words = tuple(
        dataclasses.replace(w, readings=(*before(w), *by_word[i])) if i in by_word else w for i, w in enumerate(words)
    )
    rereads: dict[int, tuple[str, ...]] = dict(page.rereads) if append else {}
    for u, _ in todo:
        line = words[u.indices[0]].line_id
        rereads[line] = (*(n for n in rereads.get(line, ()) if n not in names), *names)
    old = page.reread_report if append else {}
    report = {
        **{k: v for k, v in page.reread_report.items() if k not in REPORT_KEYS},
        "units_read": old.get("units_read", 0) + len(todo),
        "units_skipped": old.get("units_skipped", 0) + len(units) - len(todo),
        "reread_unaligned": old.get("reread_unaligned", 0) + unaligned,
        "reread_words": sum(1 for w in new_words if w.readings) if append else len(by_word),
    }
    return dataclasses.replace(page, words=new_words, rereads=rereads, reread_report=report)


def without_readings(page: PageReading, skipped: int | None = None) -> PageReading:
    """A page the reader process could not read: no readings, and the units it skipped counted."""
    report = {k: v for k, v in page.reread_report.items() if k not in REPORT_KEYS}
    report["units_skipped"] = skipped if skipped is not None else report.get("units_line", 0)
    report["units_read"] = 0
    report["reader_failed"] = 1
    return dataclasses.replace(page, rereads={}, reread_report=report)


def second_pass_cells(
    page: PageReading, crops: Mapping[str, bytes], readers, passes, config: RereadConfig
) -> PageReading:
    """The passes of a mode in order, `passes` being `((unit, readers), ...)`: the thorough mode reads the lines with the
    cheap readers and then the zones with the deep one, and every word ends with the readings of both."""
    out = page
    for index, (unit, names) in enumerate(passes):
        out = second_pass(out, crops, readers, unit, config, names, append=index > 0)
    return out
