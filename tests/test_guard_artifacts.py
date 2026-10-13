import inspect
import unittest
from pathlib import Path

from bench.corpus.labels import TruthWord
from bench.guard import accounting, artifacts, dataset, pairs, pool_recall, readback
from extract.guard.structure import generate_structure
from tests.guard_pages import labelled_page


def marked(tokens, truth):
    return labelled_page(tokens, truth, mark_artifacts=True)


class ReasonTest(unittest.TestCase):
    def test_G34_a_word_the_layer_wrote_twice_or_three_times_is_an_artifact(self):
        for copies in (2, 3):
            truth = (
                [("la", [0], None)] + [("Borrador", [1], (k, copies)) for k in range(copies)] + [("final", [2], None)]
            )
            p = marked(["la", "Borrador", "final"], truth)
            self.assertEqual(dict(p.artifacts), {"layer_duplicate": 1}, copies)
            self.assertIsNone(p.labels[1].wrong)
            self.assertEqual((p.labels[1].kind, p.labels[1].detail), ("artifact", "layer_duplicate"))

    def test_G34_the_copies_differing_in_case_only_are_still_the_same_word(self):
        component = accounting.Component("merge", (0,), "Borrador BORRADOR", 2, False)
        self.assertEqual(artifacts.reason(component, ["Borrdor"]), "layer_duplicate")

    def test_G34_a_heading_the_layer_spaced_out_is_an_artifact_but_two_letters_are_a_real_merge(self):
        spaced = marked(["ANEXO", "casa"], [(c, [0], (i, 5)) for i, c in enumerate("ANEXO")] + [("casa", [1], None)])
        self.assertEqual(dict(spaced.artifacts), {"layer_spaced": 1})
        punctuated = marked(["RESUELVO:"], [(c, [0], (i, 8)) for i, c in enumerate("RESUELV")] + [("O:", [0], (7, 8))])
        self.assertEqual(dict(punctuated.artifacts), {"layer_spaced": 1})
        two = marked(["yo", "casa"], [("y", [0], (0, 2)), ("o", [0], (1, 2)), ("casa", [1], None)])
        self.assertEqual(dict(two.artifacts), {})

    def test_G34_a_drop_cap_the_layer_set_apart_is_an_artifact(self):
        p = marked(["General", "casa"], [("G", [0], (0, 2)), ("eneral", [0], (1, 2)), ("casa", [1], None)])
        self.assertEqual(dict(p.artifacts), {"layer_dropcap": 1})

    def test_G34_two_words_the_layer_glued_are_an_artifact_whether_listed_or_capitalised_names(self):
        for tokens in (["los", "servicios"], ["Sergio", "Bracuetas"], ["de", "Gáldar,"]):
            glued = "".join(tokens)
            p = marked(tokens + ["de"], [(glued, [0, 1], None), ("de", [2], None)])
            self.assertEqual(dict(p.artifacts), {"layer_glue": 1}, tokens)
            self.assertTrue(all(label.wrong is None for label in p.labels[:2]))

    def test_G34_a_real_merge_a_real_split_and_a_split_with_a_misread_are_not_artifacts(self):
        merged = marked(["conla", "casa"], [("con", [0], (0, 2)), ("la", [0], (1, 2)), ("casa", [1], None)])
        split = marked(["orde", "nanza"], [("ordenanza", [0, 1], None)])
        misread = marked(["orde", "nansa"], [("ordenanza", [0, 1], None)])
        listed_join = marked(["con", "cierto"], [("concierto", [0, 1], None)])
        differing = marked(["los", "servicios"], [("losservicio", [0, 1], None)])
        lowercase_part = marked(["Vals", "equillo"], [("Valsequillo", [0, 1], None)])
        for p in (merged, split, misread, listed_join, differing, lowercase_part):
            self.assertEqual(dict(p.artifacts), {})
            self.assertTrue(all(label.wrong is not None for label in p.labels))

    def test_G34_an_artifact_leaves_the_components_the_labels_and_the_structure_sites(self):
        p = marked(
            ["la", "Borrador", "final"],
            [("la", [0], None), ("Borrador", [1], (0, 2)), ("Borrador", [1], (1, 2)), ("final", [2], None)],
        )
        self.assertEqual([c.kind for c in p.components], ["pair", "pair"])
        self.assertNotIn(1, p.grounded().tolist())
        found = generate_structure(p.context, "segment", (1,))
        self.assertIsNone(pairs.structure_labels(p, (1,), found))
        sites, _ = pairs.structure_sites(p)
        self.assertNotIn(("segment", (1,)), sites)

    def test_G34_without_the_marking_the_same_page_keeps_them_as_wrong_components(self):
        truth = [("la", [0], None), ("Borrador", [1], (0, 2)), ("Borrador", [1], (1, 2)), ("final", [2], None)]
        p = labelled_page(["la", "Borrador", "final"], truth, mark_artifacts=False)
        self.assertEqual(sorted(c.kind for c in p.components), ["merge", "pair", "pair"])

    def test_G34_the_rule_reads_only_the_layer_and_the_tokens_as_read(self):
        truth = [("la", [0], None), ("Borrador", [1], (0, 2)), ("Borrador", [1], (1, 2)), ("final", [2], None)]
        one = marked(["la", "Borrador", "final"], truth)
        other = marked(["la", "Borrxdor", "final"], truth)
        self.assertEqual(dict(one.artifacts), dict(other.artifacts))

    def test_G34_the_reasons_are_the_four_the_report_names(self):
        self.assertEqual(artifacts.REASONS, ("layer_duplicate", "layer_spaced", "layer_dropcap", "layer_glue"))

    def test_A2_a_part_of_two_capital_letters_is_no_name_so_the_pair_is_a_real_split(self):
        p = marked(["Qj", "Bracuetas"], [("QjBracuetas", [0, 1], None)])
        self.assertEqual(dict(p.artifacts), {})


class OneMarkingEverywhereTest(unittest.TestCase):
    """Every loader of labelled pages marks the artifacts the same way ."""

    def test_L6_the_defaults_mark_the_artifacts(self):
        for function in (pairs.load_pages, pairs.page_of, dataset.rows_of_page, artifacts.align):
            default = inspect.signature(function).parameters["mark_artifacts"].default
            self.assertIs(default, True, function.__name__)

    def test_L7_only_the_two_corpus_builders_align_without_the_marking_and_they_were_frozen_that_way(self):
        # build5.py (the bands of T5 and D) and pilot.py (the pilot pages) chose their pages before the rule of the
        # artifacts existed: the bands count a text-layer artifact as an error, and T5 is frozen. Nothing else may.
        root = Path(__file__).resolve().parent.parent
        offenders = {
            str(path.relative_to(root))
            for path in sorted((root / "bench").rglob("*.py"))
            if path.name not in ("artifacts.py", "labels.py") and "label_words(" in path.read_text(encoding="utf-8")
        }
        self.assertEqual(offenders, {"bench/corpus/build5.py", "bench/corpus/pilot.py"})

    def test_L7_align_marks_and_recounts_the_kinds_and_can_be_switched_off(self):
        read = [("p1w0", "Borrador", (0.0, 0.0, 40.0, 12.0))]
        truth = [
            TruthWord("t0", "Borrador", (0.0, 0.0, 20.0, 12.0)),
            TruthWord("t1", "Borrador", (20.0, 0.0, 40.0, 12.0)),
        ]
        marked_alignment, found = artifacts.align(read, truth)
        self.assertEqual(dict(found), {"layer_duplicate": 1})
        self.assertEqual(dict(marked_alignment.kinds), {"artifact": 1})
        plain, none = artifacts.align(read, truth, mark_artifacts=False)
        self.assertEqual(dict(none), {})
        self.assertNotIn("artifact", plain.kinds)

    def test_L7_the_loaders_label_through_align(self):
        for module in (pairs, dataset, pool_recall, readback):
            self.assertIn("artifacts.align(", inspect.getsource(module), module.__name__)


if __name__ == "__main__":
    unittest.main()
