"""Offline OCR for PDFs that says which words and numbers are probably wrong.

Extract reads a scanned or digital PDF on your own machine, with no network. It gives every word a status, checks the
numbers with the rules a number has to obey (check letters, sums, VAT), and exports a searchable PDF, a workbook, a JSON
file and a text file. Import the package as `import extract` and use the names below. Nothing else is public, apart
from three helpers for word ids in `extract.types` (`word_id`, `base_id`, `page_of`).

The names, by group:

- Reading: `read`, `Reader`, `load`, `inspect`, `Options`, `CancelToken`, `FileInfo`.
- The document: `Document`, with its parts `Page` (`PageReading`), `Word`, `Reason`, `Fix`, `Reading`, `Number`
  (`NumberResult`), `Rule` (`RuleResult`), `DigitSuggestion`, `Invoice` (`InvoiceFields`), `InvoiceLine`, `VatPair`
  and `Cell`.
- Events, for the `on_event` callback: `Event`, `Started`, `PageRead`, `Progress`, `DocumentRead`, `Exported`, `Warning`
  and `Finished`.
- Errors: `ExtractError` and its subclasses `UnsupportedFile`, `PasswordProtected`, `EngineLoadError`,
  `ModelMismatch`, `OutputExists`, `SourceChanged` and `Cancelled`.

Four names are shorter than the classes they stand for: `Page` is `PageReading`, `Number` is `NumberResult`, `Rule` is
`RuleResult` and `Invoice` is `InvoiceFields`; `repr` and `__name__` show the long name. `extract.Warning` is an event
class, not the built-in `Warning`: write `import extract` and use `extract.Warning`, never `from extract import *`,
which rebinds `Warning` in your module.

The names are imported on first use, so that importing a light submodule does not load the PDF, image and spreadsheet
libraries. The package is typed (`py.typed`), and `extract.__version__` is its version. Any script that reads a PDF
must put its calls under `if __name__ == "__main__":`, because the OCR runs in worker processes that import the script
again; see `read`.

Example:
    ```python
    import extract

    print(extract.__version__)
    print(extract.Page.__name__, extract.Number.__name__, extract.Rule.__name__, extract.Invoice.__name__)
    print(len(extract.__all__))
    ```

    ```text
    # output
    5.0.7
    PageReading NumberResult RuleResult InvoiceFields
    36
    ```
"""

from importlib import import_module

__version__ = "5.0.7"

_EXPORTS = {
    "api": ("CancelToken", "Document", "FileInfo", "Reader", "inspect", "load", "read"),
    "errors": (
        "Cancelled",
        "EngineLoadError",
        "ExtractError",
        "ModelMismatch",
        "OutputExists",
        "PasswordProtected",
        "SourceChanged",
        "UnsupportedFile",
    ),
    "events": ("DocumentRead", "Event", "Exported", "Finished", "PageRead", "Progress", "Started", "Warning"),
    "options": ("Options",),
    "types": ("Cell", "DigitSuggestion", "Fix", "InvoiceLine", "Reading", "Reason", "VatPair", "Word"),
}
_ALIASES = {
    "Invoice": ("types", "InvoiceFields"),
    "Number": ("types", "NumberResult"),
    "Page": ("types", "PageReading"),
    "Rule": ("types", "RuleResult"),
}
_SOURCES = {name: (module, name) for module, names in _EXPORTS.items() for name in names} | _ALIASES

__all__ = sorted(_SOURCES)


def __getattr__(name: str):
    if name not in _SOURCES:
        raise AttributeError(f"module 'extract' has no attribute {name!r}")
    module, attribute = _SOURCES[name]
    value = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
