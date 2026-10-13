"""C3: for every token with a digit, the final text equals the original unless the fix's source is a rule (`rule:`),
the suggestion was verified and the new digit was among the recogniser's alternatives for that position. A property test
over generated invoices with misreads injected at random (the harness of N7): whatever the misreads, in every mode."""

import dataclasses
import random
import unittest

from bench import inject_fixes
from bench.corpus.sidecar_inputs import build, generated_truths
from extract import checks, fixes
from extract.types import PageReading

TRUTHS = [t for t in generated_truths() if len(t["vat"]) == 1][:10]
SEEDS = range(8)


def misreads(inputs, rng):
    """One to three digit words with a digit changed: a partner of the digit (what a recogniser confuses) or any other."""
    wanted = [w for w in inputs.words if any(c.isdigit() for c in w.text) and "#" not in w.id]
    changes = {}
    for word in rng.sample(wanted, min(len(wanted), rng.randint(1, 3))):
        at = rng.choice([i for i, c in enumerate(word.text) if c.isdigit()])
        new = inject_fixes._misread(word.text[at], rng.random() < 0.7, rng)
        if new is not None:
            changes[word.id] = word.text[:at] + new + word.text[at + 1 :]
    return changes


def baseline_of(inputs, language):
    """The numbers the rules read as `checked` on the true page: where the harness injects its misreads."""
    return {
        r.word_ids[0]: r
        for r in checks.run(inputs.words, inputs.fields, language, suggest=False)
        if r.status == "checked"
    }


def cases():
    """(truth, page with the misreads, the rules' results on it): the harness's misreads inside the numbers (digits that
    a recogniser confuses or any digit, one or two of them, or a wrong control letter) and, as a second kind, a digit
    changed in any word with digits."""
    for truth in TRUTHS:
        truth_inputs = build(truth)
        baseline = baseline_of(truth_inputs, truth["language"])
        for seed in SEEDS:
            rng = random.Random(f"{truth['language']}{truth['vat']}{seed}")
            for kind in ("number", "any"):
                changes = (
                    inject_fixes.corrupt(truth_inputs, baseline, rng.choice(inject_fixes.MODES), rng)
                    if kind == "number"
                    else misreads(truth_inputs, rng)
                )
                if not changes:
                    continue
                wrong = truth_inputs.with_texts(changes)
                # the true digit is among the alternatives only when it is a CTC partner of the one that was read
                page = PageReading(
                    number=1,
                    width_px=1,
                    height_px=1,
                    width_pt=1.0,
                    height_pt=1.0,
                    dpi=72,
                    words=tuple(inject_fixes.with_alternatives(wrong, truth_inputs, True, True).words),
                )
                yield truth, wrong, page, checks.run(wrong.words, wrong.fields, truth["language"])


class DigitsNeverChangeUnlessVerifiedTest(unittest.TestCase):
    def check_page(self, before, after, mode, results):
        suggestions = {
            r.word_ids[0]: r.suggestion for r in results if r.suggestion is not None and len(r.word_ids) == 1
        }
        changed = 0
        for old, new in zip(before.words, after.words, strict=True):
            if new.text == old.text:
                self.assertIsNone(new.fix if new.corrected_by == "guard" else None, (mode, old.id))
                continue
            changed += 1
            where = (mode, old.id, old.text, new.text)
            self.assertTrue(any(c.isdigit() for c in old.text), where)  # nothing else is touched here
            self.assertEqual(len(old.text), len(new.text), where)
            diff = [i for i, (a, b) in enumerate(zip(old.text, new.text, strict=True)) if a != b]
            self.assertEqual(len(diff), 1, where)  # one digit, no more
            position = diff[0]
            fix = new.fix
            self.assertIsNotNone(fix, where)
            self.assertTrue(fix.source.startswith(fixes.RULE_PREFIX), where)
            suggestion = suggestions[old.id]
            self.assertTrue(suggestion.verified, where)
            self.assertEqual(
                (fix.rule, suggestion.position, suggestion.new), (suggestion.rule, position, new.text[position]), where
            )
            self.assertIn(new.text[position], [c for c, _ in old.char_alts[position]], where)
            self.assertNotIn(mode, ("off",), where)
            if mode == "safe":
                self.assertNotIn(fix.rule, fixes.ARITHMETIC_RULES, where)
        return changed

    def test_C3_a_digit_changes_only_through_a_verified_rule_with_the_digit_among_the_alternatives(self):
        applied = {"safe": 0, "more": 0}
        pages = 0
        for _, _wrong, page, results in cases():
            pages += 1
            for mode in ("off", "safe", "more"):
                (fixed,), count = fixes.apply_digit_fixes([page], results, mode)
                changed = self.check_page(page, fixed, mode, results)
                self.assertEqual(changed, count, mode)
                if mode == "off":
                    self.assertEqual(changed, 0)
                else:
                    applied[mode] += changed
        self.assertGreater(pages, 40)
        # the property is not vacuous: the arithmetic of an invoice fixes misreads in `more` (in `safe` it only suggests)
        self.assertGreater(applied["more"], 5)
        self.assertGreaterEqual(applied["more"], applied["safe"])

    def test_C3_the_same_misreads_with_no_alternatives_change_nothing(self):
        for _, _wrong, page, results in cases():
            bare = dataclasses.replace(page, words=tuple(dataclasses.replace(w, char_alts=None) for w in page.words))
            for mode in ("safe", "more"):
                self.assertEqual(fixes.apply_digit_fixes([bare], results, mode)[1], 0, mode)

    def test_C3_a_page_where_every_digit_is_read_right_is_never_changed(self):
        for truth in TRUTHS:
            inputs = build(truth)
            page = PageReading(
                number=1, width_px=1, height_px=1, width_pt=1.0, height_pt=1.0, dpi=72,
                words=tuple(inject_fixes.with_alternatives(inputs, inputs, True, True).words),
            )  # fmt: skip
            results = checks.run(inputs.words, inputs.fields, truth["language"])
            for mode in ("safe", "more"):
                self.assertEqual(fixes.apply_digit_fixes([page], results, mode)[1], 0, (truth["vat"], mode))


if __name__ == "__main__":
    unittest.main()
