"""The injection harness for the verified digit fix: wrong digits put into the true numbers of the
synthetic invoices, no OCR.

Each trial puts one or two wrong digits into a number that its rules vouch for, runs `checks.run`, applies the
verified suggestions the way step 7b of the pipeline does (`verified` and, as a second gate, a new digit
that is among the recogniser's alternatives for that position), runs the rules again and compares with the truth.

A *false fix* is an applied fix that leaves the number different from the true one. A *wrong check* is a number that is
`checked` after the fix while it is not the true one. One wrong digit must give neither; the rate with two wrong digits
is measured and published with its rule-of-three bound (or a Wilson bound when something was found).

    python -m bench.inject_fixes --trials 20000 --seed 20261006 --out bench/results/injection-fixes.json

The alternatives of the recogniser are modelled by the digit pairs a CTC reader mixes up (1/7, 0/8, 5/6, 3/8, 4/9):
the alternatives of a digit are its partners; `one_letter_ctc` misreads the control letter of a DNI or NIE and lists the true letter among its alternatives, `one_letter_any` lists none. `one_digit_ctc` and `two_digits_ctc` draw their errors from those pairs
(the misreads an engine really makes); `one_digit_any` and `two_digits_any` draw any other digit, which the gate
would refuse, so they test the verified rule alone.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import random
import tempfile
from collections import Counter
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from bench.corpus.sidecar_inputs import Inputs, build, load_truths
from bench.inject import SUM_ROLES, TOTAL_ROLES, equation_members, is_false_checked, result_of
from extract import checks, fixes
from extract.types import PageReading

CTC_PAIRS = (("1", "7"), ("0", "8"), ("5", "6"), ("3", "8"), ("4", "9"))
PARTNERS: dict[str, tuple[str, ...]] = {}
for _a, _b in CTC_PAIRS:
    PARTNERS[_a] = (*PARTNERS.get(_a, ()), _b)
    PARTNERS[_b] = (*PARTNERS.get(_b, ()), _a)
MODES = ("one_digit_any", "one_digit_ctc", "two_digits_any", "two_digits_ctc", "one_letter_any", "one_letter_ctc")
VARIANTS = ("verified", "verified_and_alts")


def alternatives(char: str) -> tuple[str, ...]:
    """The digits the modelled recogniser lists as alternatives to a read digit."""
    return PARTNERS.get(char, ())


def upper_bound_95(failures: int, trials: int) -> float | None:
    """The rule of three when nothing was found, the upper end of the Wilson 95 % interval otherwise."""
    if not trials:
        return None
    if failures == 0:
        return 3 / trials
    z, p = 1.96, failures / trials
    centre = p + z * z / (2 * trials)
    spread = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    return (centre + spread) / (1 + z * z / trials)


def digit_sites(inputs: Inputs, word_ids: Sequence[str]) -> list[tuple[str, int]]:
    """(word id, index in that word) of every digit of a number, in reading order."""
    sites = []
    for word_id in word_ids:
        sites += [(word_id, i) for i, c in enumerate(inputs.text_of(word_id)) if c.isdigit()]
    return sites


def replace_at(inputs: Inputs, changes: dict[str, str]) -> Inputs:
    return inputs.with_texts(changes)


def corrupt(inputs: Inputs, baseline: dict, mode: str, rng: random.Random) -> dict[str, str] | None:
    """The new texts of the words that get wrong digits, or None when the draw cannot be made."""
    ctc = mode.endswith("ctc")
    two = mode.startswith("two")
    if mode.startswith("one_letter"):
        return corrupt_letter(inputs, baseline, rng)
    pool = sorted(baseline)
    if not pool:
        return None
    changes: dict[str, str] = {}
    if two and rng.random() < 0.5:
        members = [m for m in equation_members(inputs, rng.choice((SUM_ROLES, TOTAL_ROLES))) if m in baseline]
        if len(members) < 2:
            return None
        targets = [(word_id, 1) for word_id in rng.sample(members, 2)]
    else:
        word_id = rng.choice(pool)
        targets = [(word_id, 2 if two else 1)]
    for word_id, count in targets:
        sites = digit_sites(inputs, baseline[word_id].word_ids)
        if len(sites) < count:
            return None
        for site_word, index in rng.sample(sites, count):
            text = changes.get(site_word, inputs.text_of(site_word))
            new = _misread(text[index], ctc, rng)
            if new is None:
                return None
            changes[site_word] = text[:index] + new + text[index + 1 :]
    return changes if any(changes[w] != inputs.text_of(w) for w in changes) else None


LETTERS = "TRWAGMYFPDXBNJZSQVHLCKE"


def corrupt_letter(inputs: Inputs, baseline: dict, rng: random.Random) -> dict[str, str] | None:
    """A DNI or NIE (letters of an IBAN are never fixed by the product: it skips the IBAN in groups) read with another
    control letter and all its digits right: a wrong letter that the verified digit rule could mistake for a wrong digit."""
    pool = sorted(w for w, r in baseline.items() if r.kind in ("dni", "nie"))
    if not pool:
        return None
    word_id = rng.choice(pool)
    text = inputs.text_of(word_id)
    if not text or not text[-1].isalpha():
        return None
    return {word_id: text[:-1] + rng.choice([c for c in LETTERS if c != text[-1].upper()])}


def _misread(char: str, ctc: bool, rng: random.Random) -> str | None:
    if ctc:
        partners = alternatives(char)
        return rng.choice(partners) if partners else None
    return rng.choice([d for d in "0123456789" if d != char])


def with_alternatives(inputs: Inputs, truth: Inputs | None, use_alts: bool, letters: bool) -> Inputs:
    """The words with the `char_alts` the modelled recogniser would give: a digit's alternatives are its CTC partners
    (every other digit when the gate is off, which tests the verified rule alone); when `letters` is on, a letter that
    was misread lists the true letter among its alternatives, as a digit's partners list the true digit."""
    words = []
    for word in inputs.words:
        true_text = truth.text_of(word.id) if truth is not None else word.text
        alts = []
        for i, char in enumerate(word.text):
            if char.isdigit():
                options = alternatives(char) if use_alts else tuple(d for d in "0123456789" if d != char)
            elif letters and i < len(true_text) and true_text[i] != char:
                options = (true_text[i],)
            else:
                options = ()
            alts.append(tuple((o, 0.2) for o in options))
        words.append(dataclasses.replace(word, char_alts=tuple(alts)))
    return Inputs(words, inputs.fields, inputs.ids)


def apply_verified(
    inputs: Inputs, results, use_alts: bool, truth: Inputs | None = None, letter_alts: bool = True, mode: str = "safe"
) -> tuple[Inputs, list]:
    """Step 7b of the pipeline, by the pipeline's own code: `fixes.apply_digit_fixes` in `mode` (`safe`, the control that
    applies alone; `more` for the arithmetic suggestions) over a page built
    from the words, whose `char_alts` are the modelled alternatives. Its gates (one word, no review, the new digit among
    the alternatives, the letter that could be the error) are the product's; with `use_alts` off the digit alternatives
    are all the digits and the letter gate is off, so only the verified rule decides."""
    page = PageReading(
        number=1,
        width_px=1,
        height_px=1,
        width_pt=1.0,
        height_pt=1.0,
        dpi=72,
        words=tuple(with_alternatives(inputs, truth, use_alts, use_alts and letter_alts).words),
    )
    (fixed_page,), count = fixes.apply_digit_fixes([page], results, mode)
    if not count:
        return inputs, []
    changes = {w.id: w.text for w, old in zip(fixed_page.words, inputs.words, strict=True) if w.text != old.text}
    applied = [r for r in results if len(r.word_ids) == 1 and r.word_ids[0] in changes]
    return inputs.with_texts(changes), applied


def trial(inputs: Inputs, locale: str, baseline: dict, mode: str, rng: random.Random) -> dict | None:
    changes = corrupt(inputs, baseline, mode, rng)
    if changes is None:
        return None
    wrong = inputs.with_texts(changes)
    results = checks.run(wrong.words, wrong.fields, locale)
    outcome = {}
    for variant in VARIANTS:
        fixed, applied = apply_verified(
            wrong, results, variant == "verified_and_alts", truth=inputs, letter_alts=mode != "one_letter_any"
        )
        after = checks.run(fixed.words, fixed.fields, locale, suggest=False) if applied else results
        touched = set(changes) | {w for r in applied for w in r.word_ids}
        number_of = {w: next(b for b in baseline.values() if w in b.word_ids) for w in touched if _in(baseline, w)}
        false_fix = any(_text_of(fixed, r.word_ids) != _text_of(inputs, r.word_ids) for r in applied)
        wrong_checked = any(is_false_checked(before, result_of(after, w)) for w, before in number_of.items())
        # What `safe` leaves as a suggestion is what `more` applies and `safe` did not: counted apart.
        more, applied_more = apply_verified(
            wrong,
            results,
            variant == "verified_and_alts",
            truth=inputs,
            letter_alts=mode != "one_letter_any",
            mode="more",
        )
        alone = {id(r) for r in applied}
        suggested = [r for r in applied_more if id(r) not in alone and r.suggestion.rule in fixes.ARITHMETIC_RULES]
        outcome[variant] = {
            "suggested": [
                (r.suggestion.rule, _text_of(more, r.word_ids) != _text_of(inputs, r.word_ids)) for r in suggested
            ],
            "applied": len(applied),
            "false_fix": false_fix,
            "wrong_checked": wrong_checked,
            "all_corrected": all(fixed.text_of(w) == inputs.text_of(w) for w in changes),
            "rules": [
                (r.suggestion.rule, _text_of(fixed, r.word_ids) != _text_of(inputs, r.word_ids)) for r in applied
            ],
        }
    return outcome


def _in(baseline: dict, word_id: str) -> bool:
    return any(word_id in b.word_ids for b in baseline.values())


def _text_of(inputs: Inputs, word_ids: Sequence[str]) -> str:
    return " ".join(inputs.text_of(w) for w in word_ids)


def run_mode(args: tuple[str, list[dict], int, int]) -> tuple[str, dict]:
    mode, truths, trials, seed = args
    rng = random.Random(f"{seed}:{mode}")
    prepared = []
    for truth in truths:
        inputs = build(truth)
        checked = {
            r.word_ids[0]: r
            for r in checks.run(inputs.words, inputs.fields, truth["language"], suggest=False)
            if r.status == "checked"
        }
        prepared.append((inputs, truth["language"], checked))
    counts = {v: Counter() for v in VARIANTS}
    rules = {v: {} for v in VARIANTS}
    suggested = {v: {} for v in VARIANTS}
    done = 0
    while done < trials:
        inputs, locale, baseline = rng.choice(prepared)
        outcome = trial(inputs, locale, baseline, mode, rng)
        if outcome is None:
            continue
        done += 1
        for variant, found in outcome.items():
            c = counts[variant]
            c["trials"] += 1
            c["applied_trials"] += found["applied"] > 0
            c["false_fix"] += found["false_fix"]
            c["wrong_checked"] += found["wrong_checked"]
            c["all_corrected"] += found["all_corrected"]
            for key, false in found["rules"]:
                entry = rules[variant].setdefault(key, {"applied": 0, "false": 0})
                entry["applied"] += 1
                entry["false"] += false
            for key, false in found["suggested"]:
                entry = suggested[variant].setdefault(key, {"suggested": 0, "false": 0})
                entry["suggested"] += 1
                entry["false"] += false
    return mode, {v: summary(counts[v], rules[v], suggested[v]) for v in VARIANTS}


def summary(counts: Counter, rules: dict, suggested: dict | None = None) -> dict:
    n = counts["trials"]
    return {
        "trials": n,
        "trials_with_a_fix": counts["applied_trials"],
        "all_corrected": counts["all_corrected"],
        "false_fixes": counts["false_fix"],
        "false_fix_rate": counts["false_fix"] / n if n else None,
        "false_fix_upper_bound_95": upper_bound_95(counts["false_fix"], n),
        "false_fix_rate_among_trials_with_a_fix": (
            counts["false_fix"] / counts["applied_trials"] if counts["applied_trials"] else None
        ),
        "wrong_checked": counts["wrong_checked"],
        "wrong_checked_upper_bound_95": upper_bound_95(counts["wrong_checked"], n),
        "by_rule": {
            key: {
                **entry,
                "false_rate": entry["false"] / entry["applied"],
                "upper_bound_95": upper_bound_95(entry["false"], entry["applied"]),
            }
            for key, entry in sorted(rules.items())
        },
        # arithmetic fixes that `safe` leaves as suggestions, applied here in `more` to measure them
        "suggestions_by_rule": {
            key: {
                **entry,
                "false_rate": entry["false"] / entry["suggested"],
                "upper_bound_95": upper_bound_95(entry["false"], entry["suggested"]),
            }
            for key, entry in sorted((suggested or {}).items())
        },
    }


def run_harness(truths: Sequence[dict], trials: int, seed: int, workers: int = 1) -> dict:
    jobs = [(mode, list(truths), trials, seed) for mode in MODES]
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            done = dict(pool.map(run_mode, jobs))
    else:
        done = dict(map(run_mode, jobs))
    return {"seed": seed, "invoices": len(truths), "trials_per_mode": trials, "modes": {m: done[m] for m in MODES}}


def main() -> None:
    from bench.corpus import invoices

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261006)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--out", type=Path, default=Path("bench/results/injection-fixes.json"))
    args = parser.parse_args()
    folder = Path(tempfile.mkdtemp(prefix="injection-fixes-invoices-"))
    invoices.generate(folder, args.seed)
    report = run_harness(load_truths(folder), args.trials, args.seed, args.workers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({m: r["verified_and_alts"]["false_fixes"] for m, r in report["modes"].items()}))


if __name__ == "__main__":
    main()
