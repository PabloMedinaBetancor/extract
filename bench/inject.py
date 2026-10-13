"""The injection harness for the number rules: wrong characters put into the true numbers of the synthetic invoices, with no OCR.

A false check is a number that is still `checked` while its value is not the true one. One wrong digit, or any other
one-character edit, must never give one. Two wrong digits in the same equation can cancel; their rate is measured
here and published, not assumed to be zero.

    python -m bench.inject --trials 20000 --seed 1 --out bench/results/injection.json
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
from collections.abc import Iterable, Sequence
from pathlib import Path

from bench.corpus.sidecar_inputs import CONFUSABLE, Inputs, build, load_truths
from extract import checks
from extract.types import NumberResult

SUM_ROLES = ("line_amount", "subtotal")
TOTAL_ROLES = ("subtotal", "vat_amount", "withholding", "total")


EDIT_CHARS = "0123456789.,-()%€ xO"


def checked_results(inputs: Inputs, locale: str) -> dict[str, NumberResult]:
    """The checked numbers of a document, each by its first word id (an IBAN has several words, the others one)."""
    results = checks.run(inputs.words, inputs.fields, locale, suggest=False)
    return {r.word_ids[0]: r for r in results if r.status == "checked"}


def edits(text: str) -> Iterable[str]:
    """Every text that differs from `text` by one character: a substitution, a deletion or an insertion."""
    for i, char in enumerate(text):
        for other in EDIT_CHARS:
            if other != char:
                yield text[:i] + other + text[i + 1 :]
        yield text[:i] + text[i + 1 :]
    for i in range(len(text) + 1):
        for other in EDIT_CHARS:
            yield text[:i] + other + text[i:]


def is_false_checked(before: NumberResult, after: NumberResult | None) -> bool:
    """A number that is still `checked` but no longer stands for the true value, or for the true identifier."""
    if after is None or after.status != "checked":
        return False
    if before.value is not None:
        return after.value != before.value
    return identifier_of(after.text) != identifier_of(before.text)


def identifier_of(text: str) -> str:
    """What an identifier stands for: spaces, case and the punctuation that ends a sentence do not change it."""
    return text.replace(" ", "").upper().rstrip(".,;:")


def result_of(results: Iterable[NumberResult], word_id: str) -> NumberResult | None:
    return next((r for r in results if word_id in r.word_ids), None)


def substitute(text: str, rng: random.Random) -> str | None:
    positions = [i for i, c in enumerate(text) if c.isdigit()]
    if not positions:
        return None
    i = rng.choice(positions)
    return text[:i] + rng.choice(CONFUSABLE[text[i]]) + text[i + 1 :]


def substitute_two_in_one(text: str, rng: random.Random) -> str | None:
    positions = [i for i, c in enumerate(text) if c.isdigit()]
    if len(positions) < 2:
        return None
    chars = list(text)
    for i in rng.sample(positions, 2):
        chars[i] = rng.choice(CONFUSABLE[chars[i]])
    changed = "".join(chars)
    return None if changed == text else changed


def equation_members(inputs: Inputs, roles: Sequence[str]) -> list[str]:
    members = []
    for key, word_id in inputs.ids.items():
        name = key[0] if isinstance(key, tuple) else key
        if name in roles and isinstance(word_id, str):
            members.append(word_id)
    return members


def trial(inputs: Inputs, locale: str, baseline: dict[str, NumberResult], mode: str, rng: random.Random) -> bool | None:
    """True if a wrong number stayed checked, False if every wrong number was caught, None if nothing could be done."""
    pool = sorted(baseline)
    changes: dict[str, str] = {}
    if mode in ("one_digit", "one_edit", "two_digits_one_number"):
        if not pool:
            return None
        target = rng.choice(pool)
        text = inputs.text_of(target)
        if mode == "one_edit":
            wrong = rng.choice([e for e in edits(text) if e != text])
        else:
            wrong = (substitute if mode == "one_digit" else substitute_two_in_one)(text, rng)
        if wrong is None:
            return None
        changes[target] = wrong
    else:
        members = [m for m in equation_members(inputs, rng.choice((SUM_ROLES, TOTAL_ROLES))) if m in baseline]
        if len(members) < 2:
            return None
        for target in rng.sample(members, 2):
            wrong = substitute(inputs.text_of(target), rng)
            if wrong is None:
                return None
            changes[target] = wrong
    mutated = inputs.with_texts(changes)
    after = checks.run(mutated.words, mutated.fields, locale, suggest=False)
    return any(is_false_checked(baseline[wid], result_of(after, wid)) for wid in changes)


def sweep_one_digit(truths: Sequence[dict], seed: int) -> dict:
    """Every checked number of every invoice, every digit of it, once with a random confusable digit."""
    rng = random.Random(seed)
    done = false_checked = 0
    for truth in truths:
        inputs = build(truth)
        locale = truth["language"]
        for target, before in checked_results(inputs, locale).items():
            text = inputs.text_of(target)
            for i, char in enumerate(text):
                if char not in "0123456789":
                    continue
                mutated = inputs.with_text(target, text[:i] + rng.choice(CONFUSABLE[char]) + text[i + 1 :])
                done += 1
                after = checks.run(mutated.words, mutated.fields, locale, suggest=False)
                false_checked += is_false_checked(before, result_of(after, target))
    return {"invoices": len(truths), "trials": done, "false_checked": false_checked}


def run_harness(truths: Sequence[dict], trials: int, seed: int) -> dict:
    rng = random.Random(seed)
    prepared = []
    for truth in truths:
        inputs = build(truth)
        prepared.append((inputs, truth["language"], checked_results(inputs, truth["language"])))
    report = {"seed": seed, "invoices": len(truths), "one_digit_sweep": sweep_one_digit(truths, seed)}
    for mode in ("one_digit", "one_edit", "two_digits_one_number", "two_digits_same_equation"):
        done = false_checked = 0
        for _ in range(trials):
            inputs, locale, baseline = rng.choice(prepared)
            outcome = trial(inputs, locale, baseline, mode, rng)
            if outcome is None:
                continue
            done += 1
            false_checked += outcome
        report[mode] = {
            "trials": done,
            "false_checked": false_checked,
            "rate": false_checked / done if done else None,
            "upper_bound_95": (3 / done if false_checked == 0 else None) if done else None,
        }
    return report


def main() -> None:
    from bench.corpus import invoices

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--out", type=Path, default=Path("bench/results/injection.json"))
    args = parser.parse_args()
    folder = Path(tempfile.mkdtemp(prefix="injection-invoices-"))
    invoices.generate(folder, args.seed)
    report = run_harness(load_truths(folder), args.trials, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
