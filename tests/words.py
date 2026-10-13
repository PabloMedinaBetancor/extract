"""Spanish words with accents, kept in a JSON file with escaped characters so that the tests stay ASCII."""

import json
from pathlib import Path

WORDS = json.loads((Path(__file__).parent / "data" / "spanish_words.json").read_text(encoding="utf-8"))
