from __future__ import annotations

from ..fixes import tokens
from ..types import DocumentData


def to_text(data: DocumentData) -> str:
    """Plain text in reading order: a line per text line, a form feed between pages, corrections included."""
    pages = []
    for page in data.pages:
        lines, current, last = [], [], None
        for word in tokens(page):
            if last is not None and word.line_id != last:
                lines.append(" ".join(current))
                current = []
            current.append(word.text)
            last = word.line_id
        if current:
            lines.append(" ".join(current))
        pages.append("\n".join(lines))
    return "\f".join(pages) + "\n"
