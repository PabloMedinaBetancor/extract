"""Truth words from a born-digital PDF and the labelling of OCR words against them.

The truth is the text layer, read with pdfium (the library that renders the page and reads the
native text in the engine). An OCR word is wrong when it does not equal its truth word after
NFKC normalisation. OCR and truth words are linked by the overlap of their boxes, not by reading
order. Truth words that no OCR word covers are missed words: the guard cannot flag what is not
there, so they are counted apart and never become labels.

The truth boxes are the tight glyph boxes of pdfium. The boxes `pdftotext -bbox` gives follow the
font metrics, and for some fonts they are three times the height of the ink and sit on the
neighbouring lines (measured on 30 pilot pages, 2026-10-02).
"""

import unicodedata
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
import pypdfium2 as pdfium

from bench.corpus.degrade import map_box
from extract.guard.forms import normalise

OVERLAP_THRESHOLD = 0.5
SAME_LINE = 0.5
STRONG_OVERLAP = 0.8
SIDE_SLACK = 0.3
GROUND_DILATION = 3
INK_LEVEL = 192
VERTICAL_ASPECT = 1.5
MIN_VERTICAL_LETTERS = 3
MERGE_KERNEL = 9
MIN_PATCH = 14
MIN_PATCH_FILL = 0.05
GLYPH_SIZE = (3, 80)
MIN_GLYPHS = 6
HYPHEN_MARK = "\ufffe"


@dataclass(frozen=True)
class TruthWord:
    id: str
    text: str
    box: tuple[float, float, float, float]


@dataclass(frozen=True)
class Label:
    ocr_id: str
    wrong: bool | None
    kind: str
    detail: str | None
    truth_text: str | None
    truth_ids: tuple[str, ...]


@dataclass
class Alignment:
    labels: list[Label]
    missed: list[TruthWord]
    boundary_only: int = 0
    kinds: Counter = field(default_factory=Counter)


def read_page_truth(pdf_path: str, page_number: int) -> tuple[float, float, list[TruthWord]]:
    """Size of the rendered page in points and the words of one page (1-based).

    Boxes are in points of the render, origin top left, rotation and crop box applied.
    """
    document = pdfium.PdfDocument(str(pdf_path))
    try:
        page = document[page_number - 1]
        try:
            textpage = page.get_textpage()
            try:
                return words_from_textpage(page, textpage)
            finally:
                textpage.close()
        finally:
            page.close()
    finally:
        document.close()


def words_from_textpage(page, textpage) -> tuple[float, float, list[TruthWord]]:
    left, bottom, right, top = page.get_cropbox()
    rotation = page.get_rotation()
    text = textpage.get_text_range()
    if len(text) != textpage.count_chars():
        raise ValueError("text and character boxes disagree")
    width_pt, height_pt = right - left, top - bottom
    boxes = [_to_render(textpage.get_charbox(i), rotation, left, bottom, width_pt, height_pt) for i in range(len(text))]
    words: list[TruthWord] = []
    pieces: list[str] = []
    extent: list[float] | None = None

    def flush() -> None:
        nonlocal pieces, extent
        word = normalise("".join(pieces))
        if word and extent is not None:
            words.append(TruthWord(f"t{len(words)}", word, tuple(extent)))
        pieces, extent = [], None

    for index, char in enumerate(text):
        box = boxes[index]
        if char.isspace():
            flush()
            continue
        if char == HYPHEN_MARK:
            pieces.append("-")
            extent = _union(extent, box)
            if _next_line(boxes, index):
                flush()
            continue
        if ord(char) < 32:
            continue
        pieces.append(char)
        if box[2] > box[0] and box[3] > box[1]:
            extent = _union(extent, box)
    flush()
    render_w, render_h = (height_pt, width_pt) if rotation in (90, 270) else (width_pt, height_pt)
    return render_w, render_h, words


def _to_render(box, rotation, left, bottom, width, height):
    """Move a pdfium character box (bottom-left origin, user space) to the render (top-left origin)."""
    x0, y0, x1, y1 = box[0] - left, box[1] - bottom, box[2] - left, box[3] - bottom
    if rotation == 0:
        return (x0, height - y1, x1, height - y0)
    if rotation == 90:
        return (y0, x0, y1, x1)
    if rotation == 180:
        return (width - x1, y0, width - x0, y1)
    return (height - y1, width - x1, height - y0, width - x0)


def _union(extent, box):
    if extent is None:
        return list(box)
    return [min(extent[0], box[0]), min(extent[1], box[1]), max(extent[2], box[2]), max(extent[3], box[3])]


def _next_line(boxes, index) -> bool:
    """Whether the character after a hyphen mark sits on another line (the hyphen ends its line)."""
    here = boxes[index]
    for later in boxes[index + 1 :]:
        if later[2] > later[0] and later[3] > later[1]:
            centre_gap = abs((later[1] + later[3]) - (here[1] + here[3])) / 2
            return centre_gap > 0.6 * (later[3] - later[1])
    return False


def ungrounded_boxes(
    clean: np.ndarray, truth_boxes: list[tuple[float, float, float, float]], scale: float
) -> list[tuple[float, float, float, float]]:
    """Patches of ink in the clean render that no truth word accounts for: logos, stamps, text inside drawings.

    `clean` is the grey render, `scale` its pixels per point and `truth_boxes` the truth in points. Ink
    within a few pixels of a truth box is the word's own. What is left is merged into patches. Thin lines and
    sparse grids (table rules) are not patches, but a sparse area with letter-sized pieces of ink, such as a
    stamp, is. Boxes come back in points.
    """
    ink = clean < INK_LEVEL
    own = np.zeros_like(ink)
    height, width = ink.shape
    for x0, y0, x1, y1 in truth_boxes:
        left, top = max(0, int(x0 * scale) - GROUND_DILATION), max(0, int(y0 * scale) - GROUND_DILATION)
        own[
            top : min(height, int(y1 * scale) + GROUND_DILATION + 1),
            left : min(width, int(x1 * scale) + GROUND_DILATION + 1),
        ] = True
    rest = (ink & ~own).astype(np.uint8)
    merged = cv2.dilate(rest, np.ones((MERGE_KERNEL, MERGE_KERNEL), np.uint8))
    count, merged_labels, stats, _ = cv2.connectedComponentsWithStats(merged)
    glyphs = np.zeros(count, dtype=int)
    _, _, pieces, centres = cv2.connectedComponentsWithStats(rest)
    low, high = GLYPH_SIZE
    for (_, _, w, h, _), (cx, cy) in zip(pieces[1:], centres[1:], strict=True):
        if low <= w <= high and low <= h <= high:
            glyphs[merged_labels[int(round(cy)), int(round(cx))]] += 1
    patches = []
    for index, (left, top, w, h, _) in enumerate(stats):
        if index == 0 or w < MIN_PATCH or h < MIN_PATCH:
            continue
        sparse = rest[top : top + h, left : left + w].sum() / (w * h) < MIN_PATCH_FILL
        if sparse and glyphs[index] < MIN_GLYPHS:
            continue
        patches.append((left / scale, top / scale, (left + w) / scale, (top + h) / scale))
    return patches


def truth_to_pixels(words: list[TruthWord], dpi: float, matrix: np.ndarray) -> list[TruthWord]:
    """Move truth boxes to the pixels of a damaged render."""
    return [TruthWord(w.id, w.text, map_box(w.box, dpi, matrix)) for w in words]


def label_words(
    ocr_words: list[tuple[str, str, tuple[float, float, float, float]]],
    truth: list[TruthWord],
    ungrounded: list[tuple[float, float, float, float]] = (),
) -> Alignment:
    """Label OCR words (id, text, box in pixels) against truth words in the same pixels.

    Truth text is normalised here too, so pages written under an older rule are compared like new ones.

    `ungrounded` holds boxes of ink that the text layer does not describe (logos, stamps, text inside
    drawings, words of the layer that are turned a quarter turn). An OCR word with no truth under it that sits on such ink is `ungrounded`: it may be read
    right from the picture and the layer cannot say so, so its `wrong` is None and it stays out of
    training, calibration and every metric. A word on blank paper is `invented`.
    """
    truth = [TruthWord(t.id, normalise(t.text), t.box) for t in truth]
    turned = [t for t in truth if _is_turned(t)]
    if turned:
        truth = [t for t in truth if not _is_turned(t)]
        ungrounded = [*ungrounded, *(t.box for t in turned)]
    links = _links(ocr_words, truth)
    groups = _components(len(ocr_words), len(truth), links)
    labels: dict[int, Label] = {}
    missed: list[TruthWord] = []
    alignment = Alignment(labels=[], missed=missed)
    for ocr_idx, truth_idx in groups:
        if not ocr_idx:
            missed.extend(truth[t] for t in truth_idx)
        elif not truth_idx:
            for o in ocr_idx:
                grounded = _on_ink(ocr_words[o][2], ungrounded)
                kind = "ungrounded" if grounded else "invented"
                labels[o] = Label(ocr_words[o][0], None if grounded else True, kind, None, None, ())
        elif len(ocr_idx) == len(truth_idx) and (len(ocr_idx) == 1 or _one_line(ocr_words, ocr_idx, truth, truth_idx)):
            for o, t in zip(
                sorted(ocr_idx, key=lambda i: ocr_words[i][2][0]),
                sorted(truth_idx, key=lambda i: truth[i].box[0]),
                strict=True,
            ):
                labels[o] = _pair_label(ocr_words[o], truth[t])
        else:
            _group_labels(ocr_words, truth, ocr_idx, truth_idx, labels, alignment)
    alignment.labels = [labels[i] for i in range(len(ocr_words))]
    alignment.kinds = Counter(label.kind for label in alignment.labels)
    return alignment


def _is_turned(word: TruthWord) -> bool:
    """A word of text turned a quarter turn (taller than wide): the engine reads each such word as pieces, so
    the layer cannot say what a piece should be, and its box is treated like ink the layer does not describe."""
    width, height = word.box[2] - word.box[0], word.box[3] - word.box[1]
    return len(word.text) >= MIN_VERTICAL_LETTERS and height > VERTICAL_ASPECT * width


def _pair_label(ocr_word, truth_word: TruthWord) -> Label:
    right = normalise(ocr_word[1]) == truth_word.text
    detail = None if right else _difference(normalise(ocr_word[1]), truth_word.text)
    return Label(ocr_word[0], not right, "ok" if right else "substitution", detail, truth_word.text, (truth_word.id,))


def _group_labels(ocr_words, truth, ocr_idx, truth_idx, labels, alignment) -> None:
    """Words that are not paired one to one. An OCR word whose text is exactly one truth word of the group is
    right: its box only reached into the neighbours. Every other OCR word of the group is wrong."""
    kind = "merge" if len(ocr_idx) == 1 else "split" if len(truth_idx) == 1 else "mixed"
    ocr_text = "".join(normalise(ocr_words[o][1]) for o in ocr_idx)
    truth_text = [truth[t].text for t in truth_idx]
    if ocr_text == "".join(truth_text):
        alignment.boundary_only += len(ocr_idx)
    for o in ocr_idx:
        same = [t for t in truth_idx if truth[t].text == normalise(ocr_words[o][1])]
        if len(same) == 1:
            labels[o] = _pair_label(ocr_words[o], truth[same[0]])
            continue
        labels[o] = Label(
            ocr_words[o][0], True, kind, None, " ".join(truth_text), tuple(truth[t].id for t in truth_idx)
        )


def _one_line(ocr_words, ocr_idx, truth, truth_idx) -> bool:
    """Whether all the words of a group sit on one line: their vertical centres differ by less than half a line."""
    boxes = [ocr_words[o][2] for o in ocr_idx] + [truth[t].box for t in truth_idx]
    centres = [(b[1] + b[3]) / 2 for b in boxes]
    heights = sorted(b[3] - b[1] for b in boxes)
    return max(centres) - min(centres) < 0.5 * heights[len(heights) // 2]


def _on_ink(box, ungrounded) -> bool:
    area = (box[2] - box[0]) * (box[3] - box[1])
    if area <= 0:
        return False
    covered = 0.0
    for other in ungrounded:
        width = min(box[2], other[2]) - max(box[0], other[0])
        height = min(box[3], other[3]) - max(box[1], other[1])
        if width > 0 and height > 0:
            covered += width * height
    return covered >= OVERLAP_THRESHOLD * area


def _difference(ocr: str, truth: str) -> str:
    if ocr.casefold() == truth.casefold():
        return "case"
    if _fold_accents(ocr) == _fold_accents(truth):
        return "accent"
    if _letters_only(ocr) == _letters_only(truth):
        return "punctuation"
    if any(c.isdigit() for c in ocr + truth):
        return "digit"
    return "letters"


def _fold_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", text.casefold()) if not unicodedata.combining(c))


def _letters_only(text: str) -> str:
    return "".join(c for c in _fold_accents(text) if c.isalnum())


def _links(ocr_words, truth) -> list[tuple[int, int]]:
    """Pairs (OCR word, truth word) that sit on the same text.

    A pair needs each box to overlap the other along both axes and their vertical centres to be less than half
    the taller height apart. The truth box is widened by a third of its height on each side, because a narrow
    word is easily shifted by a pixel or two. A strong pair has at least 80 % of the overlap in both axes; a weak
    one at least half, and it counts only when neither of its words has a strong partner. The OCR box of a line
    reaches into the line above and below, so a neighbouring word often overlaps it by half, but the word that
    really sits in the box is inside it almost entirely.
    """
    if not ocr_words or not truth:
        return []
    a = np.array([w[2] for w in ocr_words], dtype=np.float64)
    b = np.array([w.box for w in truth], dtype=np.float64)
    height_a, height_b = a[:, 3] - a[:, 1], b[:, 3] - b[:, 1]
    slack = SIDE_SLACK * height_b
    overlap_w = np.minimum(a[:, None, 2], (b[:, 2] + slack)[None, :]) - np.maximum(
        a[:, None, 0], (b[:, 0] - slack)[None, :]
    )
    overlap_h = np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1])
    narrower = np.minimum((a[:, 2] - a[:, 0])[:, None], (b[:, 2] - b[:, 0] + 2 * slack)[None, :])
    shorter = np.minimum(height_a[:, None], height_b[None, :])
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(
            (narrower > 0) & (shorter > 0),
            np.clip(overlap_w, 0, None) / narrower * np.clip(overlap_h, 0, None) / shorter,
            0.0,
        )
    centre_gap = np.abs((a[:, None, 1] + a[:, None, 3]) - (b[None, :, 1] + b[None, :, 3])) / 2
    same_line = centre_gap < SAME_LINE * np.maximum(height_a[:, None], height_b[None, :])
    strong = (share >= STRONG_OVERLAP) & same_line
    weak = (share >= OVERLAP_THRESHOLD) & same_line & ~strong
    weak &= ~strong.any(axis=1)[:, None] & ~strong.any(axis=0)[None, :]
    rows, cols = np.nonzero(strong | weak)
    return list(zip(rows.tolist(), cols.tolist(), strict=True))


def _components(n_ocr: int, n_truth: int, links) -> list[tuple[list[int], list[int]]]:
    parent = list(range(n_ocr + n_truth))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for o, t in links:
        parent[find(o)] = find(n_ocr + t)
    groups: dict[int, tuple[list[int], list[int]]] = {}
    for node in range(n_ocr + n_truth):
        ocr_side, truth_side = groups.setdefault(find(node), ([], []))
        (ocr_side if node < n_ocr else truth_side).append(node if node < n_ocr else node - n_ocr)
    return list(groups.values())
