"""The learner: the only module that imports LightGBM.

A model file is a zip with the LightGBM text model of the detector, optionally that of the ranker
that orders spelling candidates, and `meta.json`: the signal schema, the calibration table, the
review thresholds, the training settings and a summary of the training data. A file written for
another signal schema or another LightGBM major version is refused. Swapping the learner means
rewriting this file and nothing else.
"""

import hashlib
import io
import json
import zipfile
from pathlib import Path

import lightgbm as lgb
import numpy as np

from extract.errors import ModelMismatch
from extract.guard import features, signals

DETECTOR_ENTRY = "detector.txt"
RANKER_ENTRY = "ranker.txt"
CHOOSER_ENTRY = "chooser.txt"
META_ENTRY = "meta.json"
NUMBER_DIVISOR = 3.0
BUDGETS = (0.03, 0.05, 0.08)
DETECTOR_FIXED = {
    "objective": "binary",
    "learning_rate": 0.1,
    "min_data_in_leaf": 50,
    "lambda_l2": 1.0,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


def train_detector(
    x: np.ndarray,
    y: np.ndarray,
    num_leaves: int,
    n_estimators: int,
    seed: int = 1,
    threads: int = 1,
    monotone: bool = True,
) -> lgb.Booster:
    """The binary classifier "is this word wrong?" on the signal matrix, with the signs of the schema as constraints."""
    params = {
        **DETECTOR_FIXED,
        "num_leaves": num_leaves,
        "seed": seed,
        "num_threads": threads,
        "monotone_constraints": list(signals.SIGNS) if monotone else [0] * len(signals.SIGNS),
    }
    data = lgb.Dataset(x, label=y, feature_name=list(signals.COLUMNS), free_raw_data=False)
    return lgb.train(params, data, num_boost_round=n_estimators)


def train_ranker(
    x: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    n_estimators: int = 100,
    num_leaves: int = 15,
    seed: int = 1,
    threads: int = 1,
) -> lgb.Booster:
    """The lambdarank model that orders the candidates of one word (`groups` are the query sizes)."""
    params = {
        "objective": "lambdarank",
        "learning_rate": 0.1,
        "num_leaves": num_leaves,
        "min_data_in_leaf": 20,
        "lambda_l2": 1.0,
        "deterministic": True,
        "force_row_wise": True,
        "verbosity": -1,
        "seed": seed,
        "num_threads": threads,
    }
    return lgb.train(params, lgb.Dataset(x, label=y, group=groups), num_boost_round=n_estimators)


CHOOSER_FIXED = {
    "objective": "binary",
    "learning_rate": 0.1,
    "min_data_in_leaf": 20,
    "lambda_l2": 1.0,
    "feature_fraction": 0.8,
    "deterministic": True,
    "force_row_wise": True,
    "verbosity": -1,
}


def train_chooser(
    x: np.ndarray, y: np.ndarray, num_leaves: int, n_estimators: int, seed: int = 1, threads: int = 1
) -> lgb.Booster:
    """The binary classifier "is this candidate the truth?" on the rows of (site, candidate) pairs, the token as
    read included. The edit kind and the operation are categorical."""
    params = {**CHOOSER_FIXED, "num_leaves": num_leaves, "seed": seed, "num_threads": threads}
    data = lgb.Dataset(
        x,
        label=y,
        feature_name=list(features.FEATURES),
        categorical_feature=list(features.CATEGORICAL),
        free_raw_data=False,
    )
    return lgb.train(params, data, num_boost_round=n_estimators)


def raw_scores(booster: lgb.Booster, x: np.ndarray, threads: int = 1) -> np.ndarray:
    return booster.predict(x, raw_score=True, num_threads=threads)


def contributions(booster: lgb.Booster, x: np.ndarray, threads: int = 1) -> np.ndarray:
    """Per-signal contributions to the raw score (TreeSHAP); the last column is the bias."""
    return booster.predict(x, pred_contrib=True, num_threads=threads)


def tree_count(booster: lgb.Booster) -> int:
    return booster.num_trees()


class GuardModel:
    """A detector (and optionally a ranker) with the calibration and thresholds that go with it."""

    def __init__(
        self,
        detector: lgb.Booster,
        meta: dict,
        ranker: lgb.Booster | None = None,
        chooser: lgb.Booster | None = None,
    ) -> None:
        self.detector = detector
        self.ranker = ranker
        self.chooser = chooser
        self.meta = meta
        calibration = meta["calibration"]
        self._scores = np.asarray(calibration["scores"], dtype=np.float64)
        self._probs = np.asarray(calibration["probs"], dtype=np.float64)

    def raw(self, x: np.ndarray, threads: int = 1) -> np.ndarray:
        return raw_scores(self.detector, x, threads)

    @property
    def fix(self) -> dict | None:
        """The `fix` block of the model file (the thresholds and tables of the chooser), None for a model without one."""
        return self.meta.get("fix") if self.chooser is not None else None

    def choose(self, x: np.ndarray, threads: int = 1) -> np.ndarray:
        """Raw chooser scores of the rows of (site, candidate) pairs."""
        return raw_scores(self.chooser, x, threads)

    def rank(self, features: np.ndarray, threads: int = 1) -> np.ndarray:
        """Raw ranker scores of one word's candidates."""
        return raw_scores(self.ranker, features, threads)

    def probability(self, raw: np.ndarray) -> np.ndarray:
        """The calibrated probability that a word is wrong, from its raw score."""
        return np.interp(raw, self._scores, self._probs)

    def thresholds(self, budget: float) -> tuple[float, float]:
        """(word threshold, number threshold) for one of the review budgets."""
        table = self.meta["thresholds"]
        key = f"{budget:.2f}"
        if key not in table:
            raise ValueError(f"no thresholds for budget {budget}; known budgets are {sorted(table)}")
        word = table[key]
        return word, word / self.meta["number_divisor"]

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            _add(archive, DETECTOR_ENTRY, self.detector.model_to_string().encode("utf-8"))
            if self.ranker is not None:
                _add(archive, RANKER_ENTRY, self.ranker.model_to_string().encode("utf-8"))
            if self.chooser is not None:
                _add(archive, CHOOSER_ENTRY, self.chooser.model_to_string().encode("utf-8"))
            _add(archive, META_ENTRY, json.dumps(self.meta, indent=1, sort_keys=True).encode("utf-8"))
        path.write_bytes(buffer.getvalue())

    @classmethod
    def load(cls, path: Path) -> "GuardModel":
        try:
            with zipfile.ZipFile(path) as archive:
                meta = json.loads(archive.read(META_ENTRY))
                detector_text = archive.read(DETECTOR_ENTRY).decode("utf-8")
                ranker_text = archive.read(RANKER_ENTRY).decode("utf-8") if RANKER_ENTRY in archive.namelist() else None
                chooser_text = (
                    archive.read(CHOOSER_ENTRY).decode("utf-8") if CHOOSER_ENTRY in archive.namelist() else None
                )
        except (OSError, KeyError, zipfile.BadZipFile, ValueError) as error:
            raise ModelMismatch(f"cannot read the guard model {path}: {error}") from error
        if meta.get("schema") != signals.schema_fingerprint():
            raise ModelMismatch("the guard model was trained on another signal schema", path=str(path))
        if str(meta.get("lightgbm_major")) != lgb.__version__.split(".")[0]:
            raise ModelMismatch(
                "the guard model was written by another LightGBM major version",
                path=str(path),
                model=meta.get("lightgbm_major"),
                installed=lgb.__version__,
            )
        if chooser_text and meta.get("fix", {}).get("feature_fingerprint") != features.feature_fingerprint():
            raise ModelMismatch("the guard model was trained on another feature schema", path=str(path))
        return cls(
            lgb.Booster(model_str=detector_text),
            meta,
            lgb.Booster(model_str=ranker_text) if ranker_text else None,
            lgb.Booster(model_str=chooser_text) if chooser_text else None,
        )

    def model_id(self) -> str:
        """A short id that changes whenever the file's content does."""
        digest = hashlib.sha256(self.detector.model_to_string().encode("utf-8"))
        if self.chooser is not None:
            digest.update(self.chooser.model_to_string().encode("utf-8"))
        digest.update(json.dumps(self.meta, sort_keys=True).encode("utf-8"))
        return digest.hexdigest()[:12]


def new_meta(calibration: dict, thresholds: dict, settings: dict, summary: dict, seed: int) -> dict:
    """The `meta.json` of a model, with the fields every loader checks."""
    return {
        "schema": signals.schema_fingerprint(),
        "extractor_version": signals.EXTRACTOR_VERSION,
        "columns": list(signals.COLUMNS),
        "lightgbm_version": lgb.__version__,
        "lightgbm_major": lgb.__version__.split(".")[0],
        "calibrated": True,
        "calibration": calibration,
        "thresholds": thresholds,
        "number_divisor": NUMBER_DIVISOR,
        "settings": settings,
        "seed": seed,
        "training": summary,
    }


def _add(archive: zipfile.ZipFile, name: str, payload: bytes) -> None:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    archive.writestr(info, payload)
