"""Mutations of the digit path of `src/extract/fixes.py` (C3): each must fail a test. Run it in a COPY of the tree (it edits
the file and restores it at the end):  python3 -m bench.mutate_digit_path"""

import subprocess
import sys

P = "src/extract/fixes.py"
MUTS = [
    ("off applies", "if mode not in DIGIT_MODES:", "if False:"),
    (
        "unverified applies",
        "if suggestion is None or not suggestion.verified or len(number.word_ids) != 1:",
        "if suggestion is None or len(number.word_ids) != 1:",
    ),
    (
        "several words apply",
        "if suggestion is None or not suggestion.verified or len(number.word_ids) != 1:",
        "if suggestion is None or not suggestion.verified:",
    ),
    ("arithmetic in safe", 'if mode == "safe" and suggestion.rule in ARITHMETIC_RULES:', "if False:"),
    ("segment token", 'if word_id in edits or "#" in word_id:', "if word_id in edits:"),
    (
        "reviewed word",
        'if word is None or word.review is not None or word.corrected_by is not None or word.status == "absorbed":',
        "if word is None:",
    ),
    ("stale position", "if word.text[position : position + 1] != suggestion.old:", "if False:"),
    ("letter gate", "if _letter_may_be_the_error(number.kind, word):", "if False:"),
    ("alts gate off", "if not in_alts and stored is None:", "if False:"),
    ("alts any digit", "any(c == suggestion.new for c, _ in alts[position])", "True"),
    ("source not rule", 'source=f"{RULE_PREFIX}{suggestion.rule}",', 'source="guard",'),
    (
        "modes of arithmetic",
        'modes=("more",) if suggestion.rule in ARITHMETIC_RULES else DIGIT_MODES,',
        "modes=DIGIT_MODES,",
    ),
    (
        "wrong position",
        "text = word.text[:position] + suggestion.new + word.text[position + 1 :]",
        "text = word.text[:position] + suggestion.new + word.text[position + 2 :]",
    ),
    (
        "stored fix any text",
        "stored = word.fix if _is_rule(word.fix) and word.fix.text == text else None",
        "stored = word.fix if _is_rule(word.fix) else None",
    ),
    ("already edited", "if word_id in edits or", "if"),
]
original = open(P).read()
survived = []
try:
    for name, old, new in MUTS:
        if old not in original:
            print("MISSING", name, flush=True)
            survived.append(name + " (missing)")
            continue
        open(P, "w").write(original.replace(old, new, 1))
        run = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-x",
                "-q",
                "-p",
                "no:cacheprovider",
                "tests/test_fixes.py",
                "tests/test_c3_property.py",
                "tests/test_inject_fixes.py",
            ],
            capture_output=True,
            text=True,
        )
        caught = run.returncode != 0
        print(("CAUGHT   " if caught else "SURVIVES ") + name, flush=True)
        if not caught:
            survived.append(name)
finally:
    open(P, "w").write(original)
print("SURVIVORS:", survived)
