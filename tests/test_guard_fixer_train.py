import random
import unittest
from types import SimpleNamespace

import numpy as np

from bench.corpus.labels import _difference
from bench.guard import fixer_train
from extract.guard import dictionary, signals
from extract.guard.signals import INDEX

CONFUSED = {"c": "e", "a": "o", "u": "v", "n": "h", "i": "l", "t": "f", "e": "c", "o": "a"}


def simulated_rows(count=500, seed=1):
    """Dictionary words misread the way an OCR does: a confusable letter, a lost letter, an extra one."""
    rng = random.Random(seed)
    book = dictionary.load("es")
    pool = [w for w in book.words[800:30000] if len(w) >= 5]
    texts, truths, chars, details = [], [], [], []
    while len(texts) < count:
        truth = rng.choice(pool)
        position = rng.randrange(len(truth))
        kind = rng.choice(("confusion", "confusion", "lost", "extra"))
        if kind == "confusion" and truth[position] in CONFUSED:
            text = truth[:position] + CONFUSED[truth[position]] + truth[position + 1 :]
        elif kind == "lost":
            text = truth[:position] + truth[position + 1 :]
        elif kind == "extra":
            text = truth[:position] + rng.choice("aeiorsn") + truth[position:]
        else:
            continue
        if text == truth or book.rank_of(text) >= 0:
            continue
        conf = [0.97] * len(text)
        conf[min(position, len(text) - 1)] = 0.45
        texts.append(text)
        truths.append(truth)
        chars.append(tuple(conf))
        details.append(_difference(text, truth))
    n = len(texts)
    x = np.full((n, len(signals.COLUMNS)), np.nan, dtype=np.float32)
    x[:, INDEX["in_dictionary"]] = 0.0
    x[:, INDEX["conf_mean"]] = [sum(c) / len(c) for c in chars]
    chars_array = np.empty(n, dtype=object)
    for i, c in enumerate(chars):
        chars_array[i] = c
    return SimpleNamespace(
        text=np.array(texts, dtype=object),
        truth=np.array(truths, dtype=object),
        y=np.ones(n, dtype=np.uint8),
        kind=np.array(["substitution"] * n, dtype=object),
        detail=np.array(details, dtype=object),
        x=x,
        chars=chars_array,
    )


class QueriesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_rows = simulated_rows(700, 1)
        cls.valid_rows = simulated_rows(300, 2)
        cls.train = fixer_train.build_queries(cls.train_rows, range(len(cls.train_rows.y)))
        cls.valid = fixer_train.build_queries(cls.valid_rows, range(len(cls.valid_rows.y)))
        cls.ranker = fixer_train.fit_ranker(cls.train)

    def test_G8_the_truth_is_among_the_candidates_for_nearly_every_simulated_misreading(self):
        report = fixer_train.candidate_recall(self.valid)
        self.assertGreater(report["candidate_recall"], 0.85)
        self.assertEqual(report["queries"], len(self.valid))

    def test_G8_candidate_recall_is_also_given_by_kind_of_error(self):
        report = fixer_train.candidate_recall(self.valid)
        self.assertTrue(any(key.startswith("by_") for key in report))

    def test_G8_the_ranker_beats_taking_the_most_frequent_candidate(self):
        report = fixer_train.accuracy(self.ranker, self.valid)
        present = [(f, k) for f, k in zip(self.valid.features, self.valid.truth_at, strict=True) if k >= 0]
        frequency_top = np.mean(
            [int(np.argmin(f[:, fixer_train.fixer.FEATURES.index("frequency_rank")])) == k for f, k in present]
        )
        self.assertGreater(report["top1_given_candidate"], frequency_top + 0.1)
        self.assertGreater(report["top1_all"], 0.5)

    def test_G8_top_choices_return_one_position_and_score_per_query(self):
        best, scores = fixer_train.top_choices(self.ranker, self.valid)
        self.assertEqual((len(best), len(scores)), (len(self.valid), len(self.valid)))
        self.assertTrue(all(0 <= b < len(c) for b, c in zip(best, self.valid.candidates, strict=True)))

    def test_G8_only_one_to_one_substitutions_of_plain_words_train_the_ranker(self):
        rows = simulated_rows(20, 3)
        rows.kind[0] = "merge"
        rows.text[1] = "ab1c"
        rows.y[2] = 0
        chosen = fixer_train.wrong_pairs(rows)
        self.assertNotIn(0, chosen)
        self.assertNotIn(1, chosen)
        self.assertNotIn(2, chosen)
        self.assertEqual(len(chosen), 17)

    def test_G8_a_ranker_cannot_be_trained_without_any_truth_among_the_candidates(self):
        rows = simulated_rows(10, 4)
        rows.truth[:] = "zzzzzzzzzzzz"
        with self.assertRaises(ValueError):
            fixer_train.fit_ranker(fixer_train.build_queries(rows, range(10)))


class StudyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train_rows = simulated_rows(700, 1)
        cls.valid_rows = simulated_rows(400, 2)
        doubtful = np.ones(len(cls.valid_rows.y), dtype=bool)
        cls.ranker, cls.meta, cls.report = fixer_train.study(cls.train_rows, cls.valid_rows, doubtful)

    def test_G8_the_study_reports_candidate_recall_accuracy_and_the_shown_suggestions(self):
        self.assertGreater(self.report["candidate_recall"]["candidate_recall"], 0.85)
        self.assertGreater(self.report["accuracy"]["top1_all"], 0.5)
        self.assertEqual(self.report["shown"]["doubtful_plain_words"], len(self.valid_rows.y))
        self.assertGreater(self.report["shown"]["suggested"], 0)

    def test_G8_the_threshold_gives_the_precision_floor_on_validation(self):
        self.assertGreaterEqual(self.report["shown"]["precision"], fixer_train.PRECISION_FLOOR)
        self.assertEqual(self.meta["t_suggest"], self.report["shown"]["threshold"])

    def test_G8_the_meta_carries_the_feature_fingerprint_and_a_calibration_table(self):
        self.assertEqual(self.meta["feature_fingerprint"], fixer_train.fixer.feature_fingerprint())
        self.assertTrue(np.all(np.diff(self.meta["calibration"]["probs"]) >= 0))

    def test_G8_words_that_are_right_count_against_the_suggestion_when_they_are_flagged(self):
        right = simulated_rows(300, 5)
        right.truth[:] = right.text
        right.y[:] = 0
        right.kind[:] = "ok"
        mixed_rows = type(self.valid_rows)(
            **{k: np.concatenate([getattr(self.valid_rows, k), getattr(right, k)]) for k in vars(self.valid_rows)}
        )
        doubtful = np.ones(len(mixed_rows.y), dtype=bool)
        _, _, report = fixer_train.study(self.train_rows, mixed_rows, doubtful)
        self.assertEqual(report["shown"]["right_words_among_them"], 300)
        self.assertLess(report["shown"]["coverage"], self.report["shown"]["coverage"])

    def test_G8_too_few_flagged_words_to_calibrate_means_no_suggestions_not_a_crash(self):
        flagged = np.zeros(len(self.valid_rows.y), dtype=bool)
        flagged[:50] = True
        ranker, meta, report = fixer_train.study(self.train_rows, self.valid_rows, flagged)
        self.assertIsNone(meta)
        self.assertIn("no suggestions", report["shown"]["note"])
        self.assertIsNotNone(ranker)

    def test_G7_tokens_with_digits_are_not_in_the_shown_population(self):
        rows = simulated_rows(10, 6)
        rows.text[0] = "ab1"
        rows.x[1, fixer_train._column("has_digit")] = 1.0
        shown = fixer_train.shown_population(rows, np.ones(10, dtype=bool))
        self.assertNotIn(0, shown)
        self.assertNotIn(1, shown)
        self.assertEqual(len(shown), 8)


class ThresholdTest(unittest.TestCase):
    def test_G8_the_threshold_keeps_precision_at_the_floor_and_reports_coverage(self):
        probability = np.array([0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])
        correct = np.array([1, 1, 1, 1, 0, 1, 0, 0, 0, 0])
        result = fixer_train.choose_threshold(probability, correct)
        self.assertEqual(result["suggested"], 4)
        self.assertEqual(result["precision"], 1.0)
        self.assertEqual(result["threshold"], 0.8)
        self.assertAlmostEqual(result["coverage"], 0.4)

    def test_G8_when_no_threshold_reaches_the_floor_nothing_is_suggested(self):
        result = fixer_train.choose_threshold(np.array([0.9, 0.8, 0.7]), np.array([0, 0, 1]))
        self.assertGreater(result["threshold"], 1.0)
        self.assertEqual(result["suggested"], 0)

    def test_G8_the_floor_can_be_given(self):
        result = fixer_train.choose_threshold(np.array([0.9, 0.8, 0.7]), np.array([1, 0, 1]), floor=0.6)
        self.assertEqual(result["suggested"], 3)

    def test_G8_a_calibration_on_too_few_queries_is_refused(self):
        with self.assertRaises(ValueError):
            fixer_train.calibrate_top_choice(np.arange(100.0), np.zeros(100))
        table = fixer_train.calibrate_top_choice(np.arange(300.0), (np.arange(300) > 150).astype(int))
        self.assertEqual(table["probs"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
