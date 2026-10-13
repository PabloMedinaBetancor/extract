"""The word checker: says how likely each OCR word is wrong, and never changes a word.

`Guard` lives in `checker.py` and is imported on first use, so that `extract.guard.lexicon` can be imported
by the engine's main process without loading numpy, LightGBM or rapidfuzz.
"""

_LAZY = {"Guard", "MAX_WORDS", "BUDGETS", "MODEL_ENV", "default_model_path"}


def __getattr__(name: str):
    if name in _LAZY:
        from extract.guard import checker

        return getattr(checker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
