"""Pages of words for the guard tests, with the extras the second reading adds (alternatives, readings, masked)."""

from dataclasses import dataclass

from extract.types import Word


@dataclass(frozen=True)
class Reading:
    source: str
    text: str | None
    conf: float | None = 0.9
    char_conf: tuple[float, ...] | None = None
    group: tuple[str, ...] = ()
    rank: int | None = None
    context: bool = False


@dataclass(frozen=True)
class EngineWord(Word):
    char_alts: tuple | None = None
    masked: str | None = None
    readings: tuple | None = None


def page(*lines: str, conf: float = 0.95, gap: float | None = None, **extras) -> list[Word]:
    """One word per space-separated token, one line per argument; `extras` maps a token's position on the page to the
    fields of its `EngineWord`. Words sit 60 pixels apart, or `gap` pixels apart (boxes 8 pixels per character, 12
    high) when a test needs to tell a small gap from a wide one."""
    words = []
    for line_id, text in enumerate(lines):
        x = 10.0
        for idx, token in enumerate(text.split()):
            if gap is None:
                x = 10.0 + 60 * idx
            fields = extras.get(str(len(words)), {})
            words.append(
                EngineWord(
                    f"p1w{len(words)}",
                    1,
                    token,
                    (x, 20.0 * line_id, x + 8.0 * len(token), 20.0 * line_id + 12.0),
                    conf,
                    (conf,) * len(token),
                    line_id,
                    idx,
                    **fields,
                )
            )
            if gap is not None:
                x += 8.0 * len(token) + gap
    return words
