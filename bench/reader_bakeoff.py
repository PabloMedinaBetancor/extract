"""The reader bake-off (SR-0): which single recogniser reads the doubtful zones of a page best.

It needs no engine and reads no page: it takes the cached first reading of 40 development pages (train set only),
cuts a seeded list of zone crops from the first reading's boxes (never from the truth), and gives the same crops to
each candidate in a process of its own, so memory and time are measured per model. The ranking figure is `net100`:
the wrong words a model reads right minus the right words it reads wrong, per 100 wrong words.

    python3 -s -m bench.reader_bakeoff --seed 20261002 --out bench/results/reader-bakeoff.json
"""

import argparse
import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from bench.corpus import sealed
from bench.corpus.labels import TruthWord
from bench.guard import artifacts, pool_recall
from extract.engine import reread
from extract.engine.cache import reading_from_dict
from extract.guard.forms import normalise

SEED = 20261002
PAGES = 40
MAX_CROPS = 1800
SCREEN_CROPS = 150
WARMUP = 5
SLOW_SECONDS = 1.0
HORIZONTAL_PAD = 0.25
VERTICAL_PAD = -0.025
MIN_SPAN_PX = 100
MAX_TILT = 5.0
MIN_NET_GAIN_VLM = 10.0
FIXABLE = ("letters", "accent", "punctuation", "case")
SPANISH_CHARACTERS = "áéíóúüñÁÉÍÓÚÜÑ¿¡"
HAND_WORDS = ("camión", "señor", "pingüino", "¿Qué?", "¡Hola!", "árbol", "último", "MÁLAGA", "NIÑO", "avión")
FONT_FILES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
)
RAPIDOCR_BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.9.2/onnx"

CANDIDATES = {
    "ppocrv6-small": {
        "role": "control",
        "file": "PP-OCRv6_rec_small.onnx",
        "url": f"{RAPIDOCR_BASE}/PP-OCRv6/rec/PP-OCRv6_rec_small.onnx",
        "sha256": "6f327246b50388f3c176ae304bd95767ea6dc0c9ae92153ef8cbe210b3c14884",
        "licence": "Apache-2.0",
    },
    "ppocrv6-medium": {
        "role": "candidate",
        "file": "PP-OCRv6_rec_medium.onnx",
        "url": f"{RAPIDOCR_BASE}/PP-OCRv6/rec/PP-OCRv6_rec_medium.onnx",
        "sha256": "eef444829dbbe18d7fea59a3f6eb75647518d2b3a9568d27c92e42940204894b",
        "licence": "Apache-2.0",
    },
    "ppocrv5-server": {
        "role": "candidate",
        "file": "ch_PP-OCRv5_rec_server.onnx",
        "url": f"{RAPIDOCR_BASE}/PP-OCRv5/rec/ch_PP-OCRv5_rec_server.onnx",
        "sha256": "e09385400eaaaef34ceff54aeb7c4f0f1fe014c27fa8b9905d4709b65746562a",
        "licence": "Apache-2.0",
    },
    "latin-ppocrv5-mobile": {
        "role": "candidate",
        "file": "latin_PP-OCRv5_rec_mobile.onnx",
        "url": f"{RAPIDOCR_BASE}/PP-OCRv5/rec/latin_PP-OCRv5_rec_mobile.onnx",
        "sha256": "b20bd37c168a570f583afbc8cd7925603890efbcdc000a59e22c269d160b5f5a",
        "licence": "Apache-2.0",
    },
}

# OnnxTR's published recognisers (parseq, crnn_vgg16_bn, vitstr) are all trained on its "french" vocabulary, the
# English one plus these letters (onnxtr 0.8.1, onnxtr/utils/vocabs.py). A model that cannot write most of our
# accent errors is out before it runs.
VOCABULARY_EXCLUDED = {
    "onnxtr-parseq": "latin letters, digits and punctuation of English plus àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ",
    "onnxtr-crnn_vgg16_bn": "latin letters, digits and punctuation of English plus àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ",
    "onnxtr-vitstr_small": "latin letters, digits and punctuation of English plus àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ",
}
FRENCH_EXTRA = "àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ"


@dataclass(frozen=True)
class Zone:
    doc: str
    page: int
    band: str
    first: int
    last: int
    suspects: tuple[int, ...]
    geometry: tuple[float, float, float, float, float]
    words: tuple[str, ...]
    ids: tuple[str, ...]


def list_fingerprint(zones) -> str:
    rows = [[z.doc, z.page, z.first, z.last, list(z.suspects), list(z.geometry)] for z in zones]
    return hashlib.sha256(json.dumps(rows, separators=(",", ":")).encode()).hexdigest()


def _record_words(record: dict):
    reading = reading_from_dict(record["ocr"]["reading"])
    return reading, reading.words


def page_angle(words) -> float:
    """The tilt of the page's text in degrees (positive when lines run down to the right), the median over the lines of
    three words or more of the slope of their box centres. Zero when no line is long enough."""
    slopes = []
    for line in reread.lines_of(words):
        if len(line) < 3:
            continue
        centres = np.array(
            [[(words[i].box[0] + words[i].box[2]) / 2, (words[i].box[1] + words[i].box[3]) / 2] for i in line]
        )
        if np.ptp(centres[:, 0]) >= MIN_SPAN_PX:
            slopes.append(np.polyfit(centres[:, 0], centres[:, 1], 1)[0])
    if not slopes:
        return 0.0
    return float(np.clip(np.degrees(np.arctan(np.median(slopes))), -MAX_TILT, MAX_TILT))


def zone_geometry(words, indices, angle: float) -> tuple[float, float, float, float, float]:
    """(centre x, centre y, width, height, tilt) of the straightened slice that holds the zone's words, from the first
    reading's boxes alone: the words' span widened by a quarter of a line height on each side."""
    boxes = [words[i].box for i in indices]
    height = float(np.median([b[3] - b[1] for b in boxes]))
    left, right = min(b[0] for b in boxes), max(b[2] for b in boxes)
    centre_y = float(np.mean([(b[1] + b[3]) / 2 for b in boxes]))
    width = (right - left) + 2 * HORIZONTAL_PAD * height
    return (
        round((left + right) / 2, 1),
        round(centre_y, 1),
        round(width, 1),
        round(height * (1 + 2 * VERTICAL_PAD), 1),
        round(angle, 2),
    )


def page_zones(record: dict) -> list[Zone]:
    """Every zone of the page that holds a suspect word without a digit, from the first reading alone (the truth in
    the record is never looked at)."""
    reading, words = _record_words(record)
    if reading.status != "ocr":
        return []
    angle = page_angle(words)
    zones = []
    for unit in reread.select_zones(words, limit=10**6):
        suspects = tuple(i for i in unit.suspects if not any(c.isdigit() for c in words[i].text))
        if not suspects:
            continue
        zones.append(
            Zone(
                record["doc_id"],
                record["page"],
                record["band"],
                unit.indices[0],
                unit.indices[-1],
                suspects,
                zone_geometry(words, unit.indices, angle),
                tuple(words[i].text for i in unit.indices),
                tuple(words[i].id for i in unit.indices),
            )
        )
    return zones


def choose_pages(records, seed: int, count: int = PAGES):
    """`count` readable pages, the bands in turn and a document once before it repeats, in a fixed order."""
    rng = random.Random(seed)
    by_band: dict[str, list] = {}
    for path, record in sorted(records, key=lambda item: str(item[0])):
        if record.get("ocr") and record["ocr"]["reading"]["status"] == "ocr":
            by_band.setdefault(record["band"], []).append((path, record))
    for pages in by_band.values():
        rng.shuffle(pages)
    chosen, seen, bands = [], set(), sorted(by_band)
    for lap in range(2):
        position = 0
        while len(chosen) < count and any(by_band.values()):
            band = bands[position % len(bands)]
            position += 1
            pool = by_band[band]
            pick = next((i for i, (_, r) in enumerate(pool) if lap == 1 or r["doc_id"] not in seen), None)
            if pick is None:
                if all(not any(r["doc_id"] not in seen for _, r in p) for p in by_band.values()):
                    break
                continue
            path, record = pool.pop(pick)
            seen.add(record["doc_id"])
            chosen.append((path, record))
    return chosen


def build_zones(records, seed: int, max_crops: int = MAX_CROPS):
    """The seeded zone list over the chosen pages, with the page each came from."""
    pages = choose_pages(records, seed)
    zones = [(path, zone) for path, record in pages for zone in page_zones(record)]
    random.Random(seed + 1).shuffle(zones)
    return pages, zones[:max_crops]


def cut(image: np.ndarray, geometry) -> np.ndarray:
    """The straightened slice of the page: rotated about its centre by the tilt and cut to the zone's size."""
    centre_x, centre_y, width, height, tilt = geometry
    size = (max(1, round(width)), max(1, round(height)))
    matrix = cv2.getRotationMatrix2D((centre_x, centre_y), tilt, 1.0)
    matrix[0, 2] += size[0] / 2 - centre_x
    matrix[1, 2] += size[1] / 2 - centre_y
    return cv2.warpAffine(image, matrix, size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def assert_development(pages, manifests) -> None:
    """Every page is listed as train in a manifest (the catalogue split and the invoice pages) and no sealed set is
    read."""
    listed = {}
    for manifest in manifests:
        for entry in json.loads(Path(manifest).read_text())["pages"]:
            listed[(entry["doc_id"], entry["page"])] = entry["set"]
    for _, record in pages:
        sealed.refuse_sealed_set(record["set"])
        if listed.get((record["doc_id"], record["page"])) != "train":
            raise ValueError(
                f"{record['doc_id']}/{record['page']} is not listed as train in {list(map(str, manifests))}"
            )


def label_targets(record: dict):
    """For one page: {word index: (truth text or None, kind, beyond pool)} for the words the bake-off measures."""
    reading, words = _record_words(record)
    truth = [TruthWord(w["id"], w["text"], tuple(w["box"])) for w in record["truth"]["words"]]
    alignment, _ = artifacts.align([(w.id, w.text, w.box) for w in words], truth, record.get("ungrounded", []))
    case = pool_recall.PageCase(record["doc_id"], record["page"], record["band"], words, alignment.labels, truth)
    beyond = set(pool_recall.beyond_pool_ids(case))
    out = {}
    for i, (word, label) in enumerate(zip(words, alignment.labels, strict=True)):
        kind = pool_recall.error_kind(label, word)
        if kind == "right" or kind in FIXABLE:
            out[i] = (label.truth_text if label.wrong else word.text, kind, word.id in beyond)
    return out


def score(zones, labels, texts) -> dict:
    """The metrics of one model over the zones. `texts[k]` is the reader's text for zone k (None if it failed).
    A target counts only when the reader's words align one to one with the zone's; otherwise the word is unchanged."""
    wrong = right = fixed = harmed = changed = changed_right = beyond = beyond_fixed = 0
    unaligned = 0
    for zone, text in zip(zones, texts, strict=True):
        aligned = reread.align(list(zone.words), (text or "").split()) if text else None
        if aligned is None:
            unaligned += 1
        for index in zone.suspects:
            info = labels[(zone.doc, zone.page)].get(index)
            if info is None:
                continue
            truth, kind, beyond_pool = info
            token = zone.words[index - zone.first]
            is_wrong = kind != "right"
            wrong += is_wrong
            right += not is_wrong
            beyond += is_wrong and beyond_pool
            read = None
            if aligned is not None:
                item = aligned[index - zone.first]
                if item.text is not None and item.group == (index - zone.first,) and " " not in item.text:
                    read = item.text
            if read is None:
                continue
            differs = normalise(read) != normalise(token)
            correct = normalise(read) == normalise(truth)
            if differs:
                changed += 1
                changed_right += correct
            if is_wrong and correct:
                fixed += 1
                beyond_fixed += beyond_pool
            if not is_wrong and differs:
                harmed += 1
    ratio = lambda a, b: round(a / b, 4) if b else None  # noqa: E731
    return {
        "zones": len(zones),
        "zones_unaligned": unaligned,
        "wrong_words": wrong,
        "right_words": right,
        "fixed": fixed,
        "harmed": harmed,
        "changed": changed,
        "reader_recall": ratio(fixed, wrong),
        "recall_beyond_pool": ratio(beyond_fixed, beyond),
        "wrong_beyond_pool": beyond,
        "change_precision": ratio(changed_right, changed),
        "reader_harm": ratio(harmed, right),
        "net100": round(100 * (fixed - harmed) / wrong, 2) if wrong else None,
    }


def reproduction(zones, texts) -> float:
    """The share of zones whose text a reader gives back exactly: for the first recogniser it says how faithful the
    crops are to what it saw on the page (a crop that does not hold its words is a bug, but doubtful words are the
    unstable ones, so it is not 1)."""
    return round(sum(t.split() == list(z.words) for z, t in zip(zones, texts, strict=True)) / max(1, len(zones)), 4)


def skipped_targets(zones, labels) -> dict:
    """The suspect words the bake-off does not measure, by kind (a number is never changed by a reading)."""
    return dict(
        Counter("measured" if labels[(z.doc, z.page)].get(i) else "not measured" for z in zones for i in z.suspects)
    )


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def peak_rss_mb() -> float:
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM"):
            return round(int(line.split()[1]) / 1024, 1)
    raise RuntimeError("no VmHWM in /proc/self/status")


def hand_crops() -> list[tuple[str, np.ndarray]]:
    """Rendered words with the accents and signs of Spanish, to see whether a model can write them at all."""
    from PIL import Image, ImageDraw, ImageFont

    font_file = next((f for f in FONT_FILES if Path(f).is_file()), None)
    if font_file is None:
        raise RuntimeError("none of the fonts for the character test is installed")
    font = ImageFont.truetype(font_file, 40)
    out = []
    for word in HAND_WORDS:
        image = Image.new("RGB", (len(word) * 34 + 40, 80), "white")
        ImageDraw.Draw(image).text((20, 14), word, fill="black", font=font)
        out.append((word, cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)))
    return out


def load_reader(name: str, threads: int):
    from rapidocr import EngineType, RapidOCR

    spec = CANDIDATES[name]
    from extract.engine.modelfiles import find_model
    from extract.engine.quiet_openvino import import_openvino

    import_openvino()  # OpenVINO reports a usage event when it is imported; this imports it with that off
    path = find_model(spec["file"])  # the folders of EXTRACT_MODELS_DIR, the user's data folder, the package's
    if file_sha256(path) != spec["sha256"]:
        raise RuntimeError(f"{path} does not match the sha256 of {name}")
    import logging

    ocr = RapidOCR(
        params={
            "Rec.engine_type": EngineType.OPENVINO,
            "Rec.model_path": str(path),
            "Det.model_path": str(find_model("PP-OCRv6_det_small.onnx")),
            "EngineConfig.openvino.inference_num_threads": threads,
            "EngineConfig.openvino.num_streams": 1,
            "EngineConfig.openvino.enable_cpu_pinning": False,
        }
    )
    logging.getLogger("RapidOCR").setLevel(logging.ERROR)
    return ocr, path


def read_crop(ocr, crop: np.ndarray) -> str:
    from rapidocr.ch_ppocr_rec.typings import TextRecInput

    result = ocr.text_rec(TextRecInput(img=[crop]))
    return result.txts[0] if result.txts else ""


def worker(args) -> int:
    """One model in this process: read the crop files in order, time each call after a warm-up, report peak memory."""
    crops_dir = Path(args.crops)
    names = json.loads((crops_dir / "crops.json").read_text())["files"][: args.limit]
    ocr, path = load_reader(args.model, args.threads)
    images = [cv2.imread(str(crops_dir / name)) for name in names]
    for image in images[:WARMUP]:
        read_crop(ocr, image)
    texts, seconds = [], []
    for image in images:
        started = time.perf_counter()
        texts.append(read_crop(ocr, image))
        seconds.append(time.perf_counter() - started)
    hand = {word: read_crop(ocr, image) for word, image in hand_crops()}
    character = list(ocr.text_rec.postprocess_op.character)
    result = {
        "texts": texts,
        "seconds": seconds,
        "peak_rss_mb": peak_rss_mb(),
        "hand_test": hand,
        "emits": {c: c in set(character) for c in SPANISH_CHARACTERS},
        "model_bytes": path.stat().st_size,
    }
    Path(args.result).write_text(json.dumps(result, ensure_ascii=False))
    return 0


def run_worker(model: str, crops_dir: Path, threads: int, limit: int, scratch: Path) -> dict:
    result_path = scratch / f"{model}-{threads}t-{limit}.json"
    command = [
        sys.executable,
        "-s",
        "-m",
        "bench.reader_bakeoff",
        "worker",
        "--model",
        model,
        "--crops",
        str(crops_dir),
        "--threads",
        str(threads),
        "--limit",
        str(limit),
        "--result",
        str(result_path),
    ]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(f"{model} failed ({threads} threads):\n{done.stderr[-2000:]}")
    return json.loads(result_path.read_text())


def summarise_seconds(seconds: list[float]) -> dict:
    ordered = sorted(seconds)
    return {
        "median": round(statistics.median(ordered), 4),
        "p95": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 4),
    }


def write_crops(pages, zones, crops_dir: Path) -> None:
    crops_dir.mkdir(parents=True, exist_ok=True)
    images: dict[Path, np.ndarray] = {}
    files = []
    for k, (path, zone) in enumerate(zones):
        png = path.with_suffix(".png")
        if png not in images:
            images = {png: cv2.imread(str(png))}
        name = f"{k:05d}.png"
        cv2.imwrite(str(crops_dir / name), cut(images[png], zone.geometry))
        files.append(name)
    (crops_dir / "crops.json").write_text(json.dumps({"files": files}))


READER_PARSE = "NFC, then every run of white space (line breaks included) becomes one space"


def reader_parse(text: str) -> str:
    """How a generative reader's answer is reduced to words (READER_PARSE)."""
    return " ".join(unicodedata.normalize("NFC", text).split())


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command")
    work = sub.add_parser("worker")
    work.add_argument("--model", required=True)
    work.add_argument("--crops", required=True)
    work.add_argument("--threads", type=int, required=True)
    work.add_argument("--limit", type=int, required=True)
    work.add_argument("--result", required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--data", default=os.environ.get("EXTRACT_BENCH_DATA", "bench/data"))
    parser.add_argument("--split", default="bench/corpus/split.json")
    parser.add_argument("--out", default="bench/results/reader-bakeoff.json")
    parser.add_argument("--models", nargs="*", default=list(CANDIDATES))
    parser.add_argument("--screen", type=int, default=SCREEN_CROPS)
    parser.add_argument("--max-crops", type=int, default=MAX_CROPS)
    args = parser.parse_args(argv)
    if args.command == "worker":
        return worker(args)

    from bench.guard import dataset

    data = Path(args.data)
    records = dataset.page_records(data, "train")
    pages, zones = build_zones(records, args.seed, args.max_crops)
    assert_development(pages, [Path(args.split), data / "invoices" / "pages.json"])
    fingerprint = list_fingerprint([z for _, z in zones])
    scratch = data / "bakeoff" / fingerprint[:12]
    write_crops(pages, zones, scratch)
    page_records = {(r["doc_id"], r["page"]): r for _, r in pages}
    labels = {key: label_targets(record) for key, record in page_records.items()}
    zone_list = [z for _, z in zones]
    models = {}
    for name in args.models:
        spec = CANDIDATES[name]
        screen = run_worker(name, scratch, 4, args.screen, scratch)
        entry = {
            "role": spec["role"],
            "url": spec["url"],
            "licence": spec["licence"],
            "sha256": spec["sha256"],
            "size_bytes": screen["model_bytes"],
            "screening": {"crops": args.screen, **score(zone_list[: args.screen], labels, screen["texts"])},
            "screening_seconds_4_threads": summarise_seconds(screen["seconds"]),
            "emits": screen["emits"],
            "hand_test": screen["hand_test"],
        }
        slow = entry["screening_seconds_4_threads"]["median"] > SLOW_SECONDS
        entry["admissible"] = not slow
        if not slow:
            full4 = run_worker(name, scratch, 4, len(zone_list), scratch)
            full1 = run_worker(name, scratch, 1, args.screen, scratch)
            entry["full"] = score(zone_list, labels, full4["texts"])
            entry["reproduces_first_reading"] = reproduction(zone_list, full4["texts"])
            entry["seconds_4_threads"] = summarise_seconds(full4["seconds"])
            entry["seconds_1_thread"] = summarise_seconds(full1["seconds"])
            entry["peak_rss_mb"] = full4["peak_rss_mb"]
        models[name] = entry
    ranked = sorted(
        (n for n, m in models.items() if m["admissible"] and m["role"] == "candidate"),
        key=lambda n: (-models[n]["full"]["net100"], models[n]["seconds_4_threads"]["median"]),
    )
    result = {
        "seed": args.seed,
        "crop_fingerprint": fingerprint,
        "pages": [f"{r['doc_id']}/{r['page']}" for _, r in pages],
        "crops": len(zone_list),
        "crop_method": "slice of the first reading's word boxes straightened by the page's tilt, 0.25 line heights wider on each side",
        "suspect_rule": {"tau_char": reread.TAU_CHAR, "tau_word": reread.TAU_WORD},
        "targets_skipped": skipped_targets(zone_list, labels),
        "reader_parse": READER_PARSE,
        "vocabulary_excluded": {
            name: {
                "vocabulary": text,
                "missing": "".join(c for c in SPANISH_CHARACTERS if c not in FRENCH_EXTRA + _english()),
            }
            for name, text in VOCABULARY_EXCLUDED.items()
        },
        "not_run": NOT_RUN,
        "models": models,
        "deep": ranked[0] if ranked else None,
        "ranking_by_net100": ranked,
    }
    Path(args.out).write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n")
    return 0


NOT_RUN = {
    name: {
        "reason": "Not measured on this machine (2026-10-02; the reason is given here). (1) torch and transformers are not "
        "installed and a VLM needs them, outside the repo, plus its weights; (2) the run asks for a 4.5 GB memory scope with "
        "more than 5.5 GB free and the machine has 7.8 GB shared with other processes (about 5.4 GB available when "
        "asked); (3) it would go on only with 10 points of net100 over the best recogniser (PP-OCRv6 medium), the "
        "catalogue marks every VLM as inventing text, and the pilot (gate P) showed that all second readers together "
        "add at most 1.8 points of pool recall, so the engine was frozen without it.",
        "status": "not_run",
    }
    for name in ("florence-2-base", "got-ocr-2.0", "glm-ocr", "paddleocr-vl")
}


def _english() -> str:
    import string

    return string.ascii_letters + string.digits + string.punctuation


if __name__ == "__main__":
    sys.exit(main())
