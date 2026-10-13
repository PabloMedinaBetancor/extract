"""What the chooser sees of the second readings: one description of the cell, written in the model when it is trained
and compared with what the product reads when the model is used, and one function that shows the words as the cell leaves
them (the readers of the cell, the units under the cap) to the training loader and to the runtime alike.

The training rows and the rows of the product come from the same words through `apply`, so a model never meets columns
(`readers_*`, `strong_*`, `zone_rank`...) that mean something else than what it learnt.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

from extract import second_reading
from extract.engine import reread
from extract.engine.readerproc import THOROUGH_UNIT

# the cells of pass 2 that the corpus has read: (unit, readers) -> name of the cell
CELL_OF = {
    ("line", frozenset({"latin", "wide"})): "L",
    ("zone", frozenset({"deep"})): "Z",
    ("zone", frozenset({"latin", "wide"})): "ZC",
    (THOROUGH_UNIT, frozenset({"latin", "wide", "deep"})): "LZ",  # the thorough mode: both cells on one page
}


@dataclass(frozen=True)
class ReadingSpec:
    """The unit (`line` or `zone`), the readers whose readings count and the cap of units a page reads (None: no cap)."""

    unit: str
    readers: tuple[str, ...]
    cap: int | None = None

    def as_meta(self) -> dict:
        return {"unit": self.unit, "readers": sorted(self.readers), "cap": self.cap}

    @classmethod
    def from_meta(cls, data: dict | None) -> ReadingSpec | None:
        if not data:
            return None
        return cls(data["unit"], tuple(sorted(data["readers"])), data.get("cap"))

    def same_cell(self, other: ReadingSpec | None) -> bool:
        return (
            other is not None
            and self.unit == other.unit
            and set(self.readers) == set(other.readers)
            and self.cap == other.cap
        )

    @property
    def label(self) -> str:
        """`line latin+wide` or `zone deep @4` (with a cap), for the warnings."""
        return f"{self.unit} {'+'.join(sorted(self.readers))}" + (f" @{self.cap}" if self.cap is not None else "")

    @property
    def cell(self) -> str:
        """The cell of pass 2 that holds these readings in the corpus."""
        try:
            return CELL_OF[(self.unit, frozenset(self.readers))]
        except KeyError:
            raise ValueError(f"no cell of pass 2 reads {self.unit}s with {sorted(self.readers)}") from None


def product_spec(thorough: bool = False) -> ReadingSpec:
    """What the product reads, from the packaged decision of the second reading (`data/second_reading.json`); in the
    thorough mode the lines and the zones together, the readers of both and no cap (pass 1 stores what it reads)."""
    if thorough:
        return ReadingSpec(THOROUGH_UNIT, tuple(sorted(second_reading.thorough_readers())), None)
    unit, readers = second_reading.cell()
    return ReadingSpec(unit, tuple(sorted(readers)), second_reading.cap())


def apply(words: Sequence, spec: ReadingSpec | None) -> tuple:
    """The words as the model of `spec` sees them: only the readings of its readers, and of the units under its cap (a
    word left with no reading has `readings = None`). With no spec the readings are all hidden."""
    if spec is None:
        return tuple(dataclasses.replace(w, readings=None) if getattr(w, "readings", None) else w for w in words)
    kept = []
    for word in words:
        readings = getattr(word, "readings", None)
        if readings:
            mine = tuple(r for r in readings if r.source in spec.readers)
            if len(mine) != len(readings):
                word = dataclasses.replace(word, readings=mine or None)
        kept.append(word)
    return reread.view(kept, spec.cap) if spec.cap is not None else tuple(kept)
