"""Where the OCR model files are. One lookup for the whole product, so that no path of any machine is written anywhere
else: the folders named by `EXTRACT_MODELS_DIR` (several, separated like `PATH`), then the user's data folder (where `scripts/fetch_models.py` puts
them), then the folder inside the installed `rapidocr` package."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

from ..errors import EngineLoadError

ENV = "EXTRACT_MODELS_DIR"
DETECTION = "PP-OCRv6_det_small.onnx"
RECOGNITION = "PP-OCRv6_rec_small.onnx"


def user_models_dir() -> Path:
    """The folder `fetch_models.py` fills when it is not told another one."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "extract" / "models"


def search_folders() -> list[Path]:
    folders = []
    folders.extend(Path(part) for part in os.environ.get(ENV, "").split(os.pathsep) if part)
    folders.append(user_models_dir())
    spec = importlib.util.find_spec("rapidocr")
    if spec and spec.origin:
        folders.append(Path(spec.origin).parent / "models")
    return folders


def has_model(file: str) -> bool:
    return any((folder / file).is_file() for folder in search_folders())


def find_model(file: str) -> Path:
    """The first copy of `file` in the search folders. A model nobody can find stops the load with the folders tried."""
    folders = search_folders()
    for folder in folders:
        if (folder / file).is_file():
            return folder / file
    tried = ", ".join(str(folder) for folder in folders)
    raise EngineLoadError(
        f"the OCR model {file} is not in any of: {tried}. Run `python3 scripts/fetch_models.py` to download the default models, or `python3 scripts/fetch_models.py --all` for the reader models of the second reading (PP-OCRv6 medium, Latin mobile).",
        file=file,
    )
