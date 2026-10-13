import unittest

from bench.corpus.labels import Label
from bench.guard import accounting
from bench.guard.accounting import Edit, components_of, final_texts, net_change, precision, tally


def label(kind, truth="", ids=("t0",), wrong=True):
    return Label("w", wrong, kind, None, truth, tuple(ids))


def pair(truth, right=False):
    return label("ok" if right else "substitution", truth, wrong=not right)


def run(texts, labels, edits):
    return tally(components_of(texts, labels), texts, edits)


class FinalTextsTest(unittest.TestCase):
    def test_M2_a_join_gives_the_first_word_the_text_and_absorbs_the_rest(self):
        self.assertEqual(final_texts(["ca", "sa", "x"], [Edit("join", (0, 1), "casa")]), ["casa", None, "x"])

    def test_M2_an_edit_that_touches_a_word_already_used_is_skipped(self):
        edits = [Edit("join", (0, 1), "casa"), Edit("text", (1,), "otra")]
        self.assertEqual(final_texts(["ca", "sa"], edits), ["casa", None])

    def test_M2_a_drop_removes_the_word_and_an_unknown_edit_is_refused(self):
        self.assertEqual(final_texts(["a", "b"], [Edit("drop", (1,))]), ["a", None])
        with self.assertRaises(ValueError):
            final_texts(["a"], [Edit("swap", (0,))])


class OutcomeTest(unittest.TestCase):
    def test_M2_fixed_broken_changed_still_wrong_and_left(self):
        texts = ["casx", "caza", "perro", "gato"]
        labels = [pair("casa"), pair("caza", right=True), pair("perra"), pair("gato", right=True)]
        edits = [Edit("text", (0,), "casa"), Edit("text", (1,), "cazo"), Edit("text", (2,), "pero")]
        weights = run(texts, labels, edits)["weights"]
        self.assertEqual(weights, {"fixed": 1, "broken": 1, "changed_still_wrong": 1, "left_right": 1})

    def test_M2_precision_counts_a_changed_word_that_stays_wrong_as_a_miss(self):
        texts = ["casx", "perro"]
        labels = [pair("casa"), pair("perra")]
        edits = [Edit("text", (0,), "casa"), Edit("text", (1,), "pero")]
        self.assertEqual(precision(run(texts, labels, edits)), 0.5)

    def test_M2_normalisation_is_applied_to_both_sides(self):
        texts = ["x"]
        self.assertEqual(run(texts, [pair("“x”")], [Edit("text", (0,), '"x"')])["weights"], {"fixed": 1})


class StructureTest(unittest.TestCase):
    def test_M2_a_join_of_the_halves_of_one_split_word_fixes_it(self):
        texts = ["ca", "sa"]
        labels = [label("split", "casa"), label("split", "casa")]
        tallied = run(texts, labels, [Edit("join", (0, 1), "casa")])
        self.assertEqual((tallied["weights"], tallied["components"]), ({"fixed": 1}, {"fixed": 1}))

    def test_M2_a_join_of_two_right_words_breaks_both(self):
        texts = ["de", "la"]
        labels = [pair("de", right=True), pair("la", right=True)]
        tallied = run(texts, labels, [Edit("join", (0, 1), "dela")])
        self.assertEqual(tallied["weights"], {"broken": 2})
        self.assertEqual(net_change(tallied), 1.0)

    def test_M2_a_segment_of_a_merged_word_into_the_truth_words_fixes_it_with_its_weight(self):
        labels = [label("merge", "de la casa")]
        tallied = run(["delacasa"], labels, [Edit("segment", (0,), "de la casa")])
        self.assertEqual(tallied["weights"], {"fixed": 3})
        self.assertEqual(tallied["components"], {"fixed": 1})

    def test_M2_dropping_an_invented_speck_fixes_it_and_dropping_a_right_word_breaks_it(self):
        texts = ["i", "casa"]
        labels = [label("invented", None, ids=()), pair("casa", right=True)]
        self.assertEqual(run(texts, labels, [Edit("drop", (0,))])["weights"], {"fixed": 1, "left_right": 1})
        self.assertEqual(run(texts, labels, [Edit("drop", (1,))])["weights"], {"left_wrong": 1, "broken": 1})


class DenominatorTest(unittest.TestCase):
    def test_M2_ungrounded_words_are_left_out_and_truth_words_are_the_weights(self):
        labels = [pair("casa"), Label("w", None, "ungrounded", None, None, ()), label("merge", "de la casa")]
        components = components_of(["casx", "logo", "delacasa"], labels)
        self.assertEqual(sum(c.weight for c in components), 4)
        self.assertEqual(len(components), 2)

    def test_M2_the_character_distance_is_summed_over_the_edited_components_only(self):
        texts = ["casx", "perrx"]
        labels = [pair("casa"), pair("perro")]
        tallied = run(texts, labels, [Edit("text", (0,), "casa")])
        self.assertEqual((tallied["distance_before"], tallied["distance_after"]), (1, 0))
        self.assertEqual(tallied["truth_characters"], 9)

    def test_M2_the_tallies_of_many_pages_add_up(self):
        one = run(["casx"], [pair("casa")], [Edit("text", (0,), "casa")])
        total = accounting.combine([one, one])
        self.assertEqual(total["weights"], {"fixed": 2})
        self.assertEqual(total["truth_words"], 2)
