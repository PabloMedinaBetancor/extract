"""Chooses how aggressively the dictionary drops unaccented spellings.

Social-media text puts words without their accent in the frequency lists ("articulo"). A word is dropped when its
accented spelling is at least RATIO times as frequent. The dictionaries for each ratio are built, the detector of a
fixed cell is trained on each, and the choice is made on validation by one rule written beforehand:

    8 wins only if the recall at 5 % of the errors labelled `accent` rises by at least 2 points and the average
    precision of validation falls by at most 0.5 points; otherwise 20.

    python -m bench.guard.dictionary_ratio --leaves 15 --trees 100
"""

import argparse
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from bench.guard import analysis, dataset, metrics
from extract.guard import dictionary, lexicon, model

RATIOS = (20, 8)
MIN_ACCENT_GAIN_POINTS = 2.0
MAX_AP_LOSS_POINTS = 0.5


def decide(results: dict[int, dict]) -> dict:
    """The ratio the rule picks from the validation numbers of each (`accent_recall`, `ap`, as shares)."""
    base, strict = results[20], results[8]
    gain = 100 * (strict["accent_recall"] - base["accent_recall"])
    loss = 100 * (base["ap"] - strict["ap"])
    chosen = 8 if gain >= MIN_ACCENT_GAIN_POINTS and loss <= MAX_AP_LOSS_POINTS else 20
    return {
        "chosen": chosen,
        "accent_gain_points": gain,
        "ap_loss_points": loss,
        "rule": " ".join(next(p for p in __doc__.split("\n\n") if "8 wins" in p).split()),
    }


def accent_recall(validation_rows, score: np.ndarray, share: float = analysis.REVIEW_SHARE) -> float:
    """Recall at the review share of the wrong words whose error is only a lost accent, tilde or diaeresis."""
    flagged = metrics.flag_top(score, share)
    accent = (validation_rows.y == 1) & (validation_rows.detail == "accent")
    return float(flagged[accent].mean()) if accent.any() else float("nan")


def build_data(ratio: int, out: Path) -> None:
    script = Path(__file__).resolve().parents[2] / "scripts" / "build_dictionaries.py"
    spec = importlib.util.spec_from_file_location("build_dictionaries", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    import sys

    old = sys.argv
    sys.argv = ["build_dictionaries", "--out", str(out), "--typo-ratio", str(ratio)]
    try:
        module.main()
    finally:
        sys.argv = old


def use_data(folder: Path) -> None:
    """Point the lexicon and the dictionaries at another data folder and forget what was loaded."""
    lexicon.DATA_DIR = folder
    dictionary.DATA_DIR = folder
    lexicon._SETS.clear()
    dictionary.load.cache_clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("EXTRACT_BENCH_DATA", "bench/data")))
    parser.add_argument("--leaves", type=int, default=15)
    parser.add_argument("--trees", type=int, default=100)
    parser.add_argument("--out", type=Path, default=Path("bench/results/dictionary-ratio.json"))
    args = parser.parse_args()
    shipped = lexicon.DATA_DIR
    results = {}
    try:
        for ratio in RATIOS:
            with tempfile.TemporaryDirectory() as folder:
                build_data(ratio, Path(folder))
                (Path(folder) / "confusions.json").write_bytes((shipped / "confusions.json").read_bytes())
                use_data(Path(folder))
                train_rows = dataset.load_rows(args.data_dir, "train")
                validation_rows = dataset.load_rows(args.data_dir, "val")
                booster = model.train_detector(train_rows.x, train_rows.y, args.leaves, args.trees)
                score = model.raw_scores(booster, validation_rows.x)
                results[ratio] = {
                    "ap": metrics.average_precision(validation_rows.y, score),
                    "recall_at_5": metrics.recall_at_share(validation_rows.y, score, analysis.REVIEW_SHARE),
                    "accent_recall": accent_recall(validation_rows, score),
                    "accent_errors": int(((validation_rows.y == 1) & (validation_rows.detail == "accent")).sum()),
                }
    finally:
        use_data(shipped)
    report = {
        "results": {str(r): v for r, v in results.items()},
        "decision": decide(results),
        "cell": [args.leaves, args.trees],
    }
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
