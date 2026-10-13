import dataclasses
import random
import unittest
from unittest import mock

import numpy as np

from bench.corpus.labels import TruthWord, label_words
from bench.guard import pairs
from extract.guard import features, signals, structure
from extract.guard.candidates import PageContext
from extract.guard.features import INDEX, structure_rows
from extract.guard.forms import normalise
from extract.guard.select import Proposal, in_reading_order, select
from extract.guard.structure import generate_structure, join_spans
from tests.guard_pages import labelled_page as built
from tests.guard_words import Reading, page
from tests.words import WORDS

LANGUAGES = ("es", "en")


def site_labels(p, family, span):
    found = generate_structure(p.context, family, span)
    return found, pairs.structure_labels(p, span, found)


def candidate_texts(found):
    return [c.text for c in found]


class JoinCandidatesTest(unittest.TestCase):
    def test_G33_the_two_halves_of_a_split_word_are_joined(self):
        p = built(
            ["la", "orde", "nanza", "del", "plan"],
            [("la", [0], None), ("ordenanza", [1, 2], None), ("del", [3], None), ("plan", [4], None)],
        )
        self.assertIn((1, 2), join_spans(p.context))
        found, labels = site_labels(p, "join", (1, 2))
        self.assertEqual(found[0].text, "orde nanza")
        self.assertEqual((found[0].op, found[0].kind), ("keep", "keep"))
        self.assertIn("ordenanza", candidate_texts(found))
        self.assertEqual(labels[candidate_texts(found).index("ordenanza")], 1)
        self.assertEqual(int(labels.sum()), 1)
        self.assertEqual(labels[0], 0)

    def test_G33_a_word_split_in_three_is_joined_by_the_run_of_three(self):
        p = built(
            ["la", "orde", "nan", "za", "del"], [("la", [0], None), ("ordenanza", [1, 2, 3], None), ("del", [4], None)]
        )
        self.assertIn((1, 2, 3), join_spans(p.context))
        found, labels = site_labels(p, "join", (1, 2, 3))
        self.assertEqual(labels[candidate_texts(found).index("ordenanza")], 1)

    def test_G34_a_span_that_holds_only_part_of_a_split_is_left_out_as_open(self):
        p = built(
            ["la", "orde", "nan", "za", "del"], [("la", [0], None), ("ordenanza", [1, 2, 3], None), ("del", [4], None)]
        )
        found = generate_structure(p.context, "join", (1, 2))
        self.assertIsNone(pairs.structure_labels(p, (1, 2), found))
        sites, left_out = pairs.structure_sites(p)
        self.assertNotIn(("join", (1, 2)), sites)
        self.assertIn(("join", (1, 2, 3)), sites)
        self.assertGreaterEqual(left_out["join"], 1)

    def test_G33_two_right_words_of_two_components_are_not_joined_and_keep_wins(self):
        p = built(["la", "suelo", "obra"], [("la", [0], None), ("suelo", [1], None), ("obra", [2], None)])
        self.assertEqual(len(generate_structure(p.context, "join", (1, 2))), 1)
        with mock.patch.object(structure, "JOIN_AS_WRITTEN", "any"):
            found, labels = site_labels(p, "join", (1, 2))
        self.assertIn("sueloobra", candidate_texts(found))
        self.assertEqual(labels.tolist(), [1] + [0] * (len(found) - 1))

    def test_G33_two_listed_words_side_by_side_get_no_dictionary_edit(self):
        found = generate_structure(PageContext(page("la suelo obra", gap=4), LANGUAGES), "join", (1, 2))
        self.assertFalse([c for c in found if "join_edit" in c.sources])

    def test_G33_the_page_vocabulary_vouches_for_a_joined_name(self):
        ctx = PageContext(page("Valsequillo Valsequillo Vals equillo", gap=4), LANGUAGES)
        found = {c.text: c for c in generate_structure(ctx, "join", (2, 3))}
        self.assertEqual(found["Valsequillo"].sources[0], "joined")
        lonely = PageContext(page("Valsequillo Vals equillo", gap=4), LANGUAGES)
        self.assertNotIn("Valsequillo", candidate_texts(generate_structure(lonely, "join", (1, 2))))

    def test_G33_a_join_never_holds_a_digit_crosses_a_line_or_bridges_a_wide_gap(self):
        digit = PageContext(page("orde nan7a", gap=4), LANGUAGES)
        self.assertEqual(join_spans(digit), [])
        lines = PageContext(page("orde", "nanza", gap=4), LANGUAGES)
        self.assertEqual(join_spans(lines), [])
        wide = PageContext(page("orde nanza", gap=20), LANGUAGES)
        self.assertEqual(join_spans(wide), [])
        near = PageContext(page("orde nanza", gap=14), LANGUAGES)
        self.assertEqual(join_spans(near), [(0, 1)])
        mark = PageContext(page("orde, nanza", gap=4), LANGUAGES)
        self.assertEqual(join_spans(mark), [])
        protected = PageContext(page("orde@x nanza", gap=4), LANGUAGES)
        self.assertEqual(join_spans(protected), [])

    def test_G33_a_run_is_at_most_three_words_and_ends_are_kept_with_their_punctuation(self):
        spans = join_spans(PageContext(page("la orde nan za mas", gap=4), LANGUAGES))
        self.assertIn((1, 2, 3), spans)
        self.assertNotIn((1, 2, 3, 4), spans)
        self.assertEqual(max(len(s) for s in spans), 3)
        ctx = PageContext(page("(orde nan za)", gap=4), LANGUAGES)
        self.assertIn("(ordenanza)", candidate_texts(generate_structure(ctx, "join", (0, 1, 2))))

    def test_G33_a_reader_that_read_the_run_as_one_word_proposes_it_even_if_unlisted(self):
        reading = Reading("latin", "Qwxzy", group=("p1w0", "p1w1"))
        extras = {"0": {"readings": (reading,)}, "1": {"readings": (reading,)}}
        ctx = PageContext(page("Qwx zy", gap=4, **extras), LANGUAGES)
        found = generate_structure(ctx, "join", (0, 1))
        self.assertEqual([c.sources for c in found[1:]], [("reading:latin",)])

    def test_G33_a_run_is_made_of_consecutive_places_on_one_line(self):
        words = page("orde nanza", gap=4)
        other_line = [words[0], dataclasses.replace(words[1], line_id=1)]
        skipped_place = [words[0], dataclasses.replace(words[1], idx_in_line=2)]
        for run in (other_line, skipped_place):
            self.assertEqual(join_spans(PageContext(run, LANGUAGES)), [])

    def test_G33_a_protected_acronym_never_takes_part_in_a_run(self):
        ctx = PageContext(page("BOE BOE BOE nanza", gap=4), LANGUAGES)
        self.assertFalse(ctx.is_site(2))
        self.assertNotIn((2, 3), join_spans(ctx))

    def test_G33_two_listed_words_side_by_side_get_no_dictionary_edit_even_when_edits_exist(self):
        found = generate_structure(PageContext(page("de la", gap=4), LANGUAGES), "join", (0, 1))
        self.assertFalse([c for c in found if "join_edit" in c.sources])

    def test_G33_one_dictionary_edit_of_the_joined_letters_is_proposed(self):
        ctx = PageContext(page("la orde nanzs", gap=4), LANGUAGES)
        found = {c.text: c for c in generate_structure(ctx, "join", (1, 2))}
        self.assertEqual(found["ordenanza"].sources, ("join_edit",))
        self.assertNotIn("ordenanzs", found)


class SegmentAndDropTest(unittest.TestCase):
    def test_G33_a_merged_token_is_cut_into_dictionary_words(self):
        p = built(
            ["la", "delacasa", "grande"],
            [("la", [0], None), ("de", [1], (0, 3)), ("la", [1], (1, 3)), ("casa", [1], (2, 3)), ("grande", [2], None)],
        )
        found, labels = site_labels(p, "segment", (1,))
        self.assertIn("de la casa", candidate_texts(found))
        self.assertEqual(labels[candidate_texts(found).index("de la casa")], 1)
        self.assertEqual(int(labels.sum()), 1)
        self.assertLessEqual(len(found), 1 + structure.SEGMENT_CANDIDATES)

    def test_G33_capital_roman_numerals_and_unaccented_capitals_are_pieces(self):
        ctx = PageContext(page("CAPITULOI ORDENACIONDEL"), LANGUAGES)
        self.assertIn("CAPITULO I", candidate_texts(generate_structure(ctx, "segment", (0,))))
        self.assertIn("ORDENACION DEL", candidate_texts(generate_structure(ctx, "segment", (1,))))

    def test_G33_a_piece_may_need_one_dictionary_edit_but_not_three(self):
        ctx = PageContext(page("ordenansadel ordenansadelsuelp"), LANGUAGES)
        self.assertIn("ordenanza del", candidate_texts(generate_structure(ctx, "segment", (0,))))
        self.assertIn("ordenanza del suelo", candidate_texts(generate_structure(ctx, "segment", (1,))))
        with mock.patch.object(structure, "SEGMENT_EDITS", 1):
            self.assertNotIn("ordenanza del suelo", candidate_texts(generate_structure(ctx, "segment", (1,))))

    def test_G33_the_marks_attached_to_the_ends_of_a_token_can_be_set_apart(self):
        ctx = PageContext(page("\u00b7Planes Terreno, *(Reposici\u00f3n)"), LANGUAGES)
        self.assertEqual(generate_structure(ctx, "segment", (0,))[1].text, "\u00b7 Planes")
        self.assertEqual(generate_structure(ctx, "segment", (1,))[1].text, "Terreno ,")
        self.assertEqual(generate_structure(ctx, "segment", (2,))[1].text, "*( Reposici\u00f3n )")
        plain = PageContext(page("Terreno"), LANGUAGES)
        self.assertNotIn("Terreno ", [c.text for c in generate_structure(plain, "segment", (0,))])

    def test_G33_the_segment_limits_are_two_edits_and_six_candidates(self):
        self.assertEqual((structure.SEGMENT_EDITS, structure.SEGMENT_CANDIDATES), (2, 6))

    def test_G33_a_token_with_many_cuts_gives_at_most_six_candidates(self):
        ctx = PageContext(page("ordenanzadelsuelo"), LANGUAGES)
        found = generate_structure(ctx, "segment", (0,))
        self.assertEqual(len(found), 1 + structure.SEGMENT_CANDIDATES)
        self.assertEqual(structure.SEGMENT_CANDIDATES, 6)

    def test_G33_a_short_token_is_not_a_segment_site_and_one_letter_parts_are_only_ayoeu(self):
        ctx = PageContext(page("casa de la delacasa xcasa"), LANGUAGES)
        self.assertFalse(structure.segment_ok(ctx, 0))
        self.assertTrue(structure.segment_ok(ctx, 3))
        cuts = [c.text for c in generate_structure(ctx, "segment", (4,))[1:]]
        self.assertFalse([c for c in cuts if any(len(part) == 1 and part not in "ayoeu" for part in c.split())])

    def test_G33_a_speck_is_dropped_when_it_is_small_or_doubtful_whether_or_not_a_list_holds_it(self):
        words = page("la zq casa", gap=4)
        small = dataclasses.replace(words[1], box=(40.0, 4.0, 52.0, 8.0))
        doubtful = dataclasses.replace(words[1], conf=0.4)
        for variant in (small, doubtful):
            ctx = PageContext([words[0], variant, words[2]], LANGUAGES)
            self.assertTrue(structure.drop_ok(ctx, 1))
            found = generate_structure(ctx, "drop", (1,))
            self.assertEqual([(c.text, c.op) for c in found], [("zq", "keep"), ("", "drop")])
        self.assertFalse(
            structure.drop_ok(
                PageContext([words[0], dataclasses.replace(words[1], conf=0.995), words[2]], LANGUAGES), 1
            )
        )
        for text, conf, expected in (
            ("de", 0.995, False),
            ("de", 0.4, True),
            (".", 0.4, True),
            ("\u2022", 0.4, True),
            ("-", 0.4, False),
            ("\u2019", 0.4, False),
            ("a1", 0.4, False),
            (".", 0.995, False),
        ):
            word = dataclasses.replace(words[1], text=text, conf=conf)
            self.assertEqual(structure.drop_ok(PageContext([words[0], word, words[2]], LANGUAGES), 1), expected, text)
        self.assertFalse(
            structure.drop_ok(
                PageContext([words[0], dataclasses.replace(words[1], text="zqx", conf=0.4), words[2]], LANGUAGES), 1
            )
        )

    def test_G34_an_invented_speck_is_right_when_dropped(self):
        words = page("la zq casa", gap=4)
        words[1] = dataclasses.replace(words[1], conf=0.4)
        read = [(w.id, w.text, w.box) for w in words]
        truth = [TruthWord("t0", "la", words[0].box), TruthWord("t1", "casa", words[2].box)]
        p = pairs.Page(
            "k", "d", "light", tuple(words), signals.extract(words), label_words(read, truth).labels, frozenset()
        )
        found = generate_structure(p.context, "drop", (1,))
        self.assertEqual(pairs.structure_labels(p, (1,), found).tolist(), [0, 1])

    def test_G34_a_right_word_that_a_drop_would_remove_keeps_its_own_label(self):
        words = page("la zq casa", gap=4)
        words[1] = dataclasses.replace(words[1], conf=0.4)
        read = [(w.id, w.text, w.box) for w in words]
        truth = [TruthWord(f"t{i}", w.text, w.box) for i, w in enumerate(words)]
        p = pairs.Page(
            "k", "d", "light", tuple(words), signals.extract(words), label_words(read, truth).labels, frozenset()
        )
        found = generate_structure(p.context, "drop", (1,))
        self.assertEqual(pairs.structure_labels(p, (1,), found).tolist(), [1, 0])

    def test_G34_a_word_the_layer_does_not_ground_cannot_be_labelled(self):
        words = page("la zq casa", gap=4)
        words[1] = dataclasses.replace(words[1], conf=0.4)
        read = [(w.id, w.text, w.box) for w in words]
        truth = [TruthWord("t0", "la", words[0].box), TruthWord("t1", "casa", words[2].box)]
        labels = label_words(read, truth, ungrounded=[words[1].box]).labels
        p = pairs.Page("k", "d", "light", tuple(words), signals.extract(words), labels, frozenset())
        found = generate_structure(p.context, "drop", (1,))
        self.assertIsNone(pairs.structure_labels(p, (1,), found))


class ListShapeTest(unittest.TestCase):
    def pages(self):
        rng = random.Random(5)
        vocabulary = ["ordenanza", "plan", "casa", "de", "la", "del", "suelo", "urbano", "obra", "proyecto"]
        for _ in range(12):
            tokens, truth = [], []
            for _ in range(14):
                word = rng.choice(vocabulary)
                roll = rng.random()
                at = len(tokens)
                if roll < 0.2 and len(word) >= 5:
                    cut = rng.randrange(1, len(word))
                    tokens += [word[:cut], word[cut:]]
                    truth.append((word, [at, at + 1], None))
                elif roll < 0.35:
                    other = rng.choice(vocabulary)
                    tokens.append(word + other)
                    truth += [(word, [at], (0, 2)), (other, [at], (1, 2))]
                else:
                    tokens.append(word)
                    truth.append((word, [at], None))
            yield built(tokens, truth)

    def test_G33_lists_are_unique_by_final_text_start_with_keep_stay_under_the_cap_and_repeat(self):
        total = 0
        for p in self.pages():
            sites, _ = pairs.structure_sites(p)
            for family, span in sites:
                found = generate_structure(p.context, family, span)
                keys = [normalise(c.text) for c in found]
                self.assertEqual(len(keys), len(set(keys)))
                self.assertEqual(found[0].op, "keep")
                self.assertLessEqual(len(found), 16)
                self.assertEqual(found, generate_structure(PageContext(p.words, LANGUAGES), family, span))
                total += 1
        self.assertGreater(total, 50)

    def test_G34_the_labels_of_a_structure_site_are_exclusive_and_the_planted_errors_are_reached(self):
        planted = reached = 0
        for p in self.pages():
            sites, _ = pairs.structure_sites(p)
            for family, span in sites:
                found = generate_structure(p.context, family, span)
                labels = pairs.structure_labels(p, span, found)
                self.assertLessEqual(int(labels.sum()), 1)
            for component in p.components:
                if component.kind in ("split", "merge"):
                    planted += 1
                    family = "join" if component.kind == "split" else "segment"
                    span = tuple(sorted(component.words))
                    found = generate_structure(p.context, family, span)
                    labels = pairs.structure_labels(p, span, found)
                    if labels is not None and labels[1:].any():
                        reached += 1
        self.assertGreater(planted, 20)
        self.assertGreater(reached / planted, 0.8)


class StructureRowsTest(unittest.TestCase):
    def rows(self, family, span, line, **extras):
        words = page(line, gap=4, **extras)
        ctx = PageContext(words, LANGUAGES)
        found = generate_structure(ctx, family, span)
        return found, structure_rows(ctx, family, span, found, signals.extract(words)[span[0]])

    def test_G33_there_is_a_float32_row_per_candidate_with_the_op_and_the_structure_columns_filled(self):
        found, rows = self.rows("join", (1, 2), "la orde nanza del")
        self.assertEqual(rows.shape, (len(found), len(features.FEATURES)))
        self.assertEqual(rows.dtype, np.float32)
        self.assertTrue(np.all(rows[:, INDEX["op"]] == features.OPS.index("join")))
        self.assertTrue(np.all(rows[:, INDEX["n_tokens"]] == 2))
        self.assertEqual(rows[0, INDEX["kind"]], features.KINDS.index("keep"))
        self.assertTrue(np.all(rows[1:, INDEX["kind"]] == features.KINDS.index("structure")))
        self.assertTrue(np.all(np.isfinite(rows[:, INDEX["gap_ratio_max"]])))
        joined = rows[[c.text for c in found].index("ordenanza")]
        self.assertEqual(joined[INDEX["joined_in_dictionary"]], 1.0)
        self.assertTrue(np.isnan(rows[0, INDEX["joined_in_dictionary"]]))
        self.assertTrue(np.isnan(joined[INDEX["box_height_ratio"]]))

    def test_G33_a_segment_row_says_where_the_cut_falls_and_whether_the_parts_are_listed(self):
        found, rows = self.rows("segment", (1,), "la delacasa grande")
        row = rows[[c.text for c in found].index("de la casa")]
        self.assertEqual(row[INDEX["op"]], features.OPS.index("segment"))
        self.assertAlmostEqual(float(row[INDEX["split_position_ratio"]]), 0.25)
        self.assertEqual(row[INDEX["parts_in_dictionary_share"]], 1.0)
        self.assertEqual(rows[0, INDEX["parts_in_dictionary_share"]], 0.0)

    def test_G33_the_gap_columns_hold_the_mean_and_the_widest_gap_of_the_run(self):
        words = page("orde nan za", gap=4)
        words[2] = dataclasses.replace(words[2], box=(words[2].box[0] + 6, *words[2].box[1:]))
        ctx = PageContext(words, LANGUAGES)
        found = generate_structure(ctx, "join", (0, 1, 2))
        rows = structure_rows(ctx, "join", (0, 1, 2), found, signals.extract(words)[0])
        self.assertAlmostEqual(float(rows[0, INDEX["gap_ratio_mean"]]), (4 / 12 + 10 / 12) / 2, places=3)
        self.assertAlmostEqual(float(rows[0, INDEX["gap_ratio_max"]]), 10 / 12, places=3)

    def test_G33_a_drop_row_carries_the_size_of_the_box_against_the_line(self):
        words = page("la zq casa", gap=4)
        words[1] = dataclasses.replace(words[1], box=(40.0, 4.0, 52.0, 8.0), conf=0.4)
        ctx = PageContext(words, LANGUAGES)
        found = generate_structure(ctx, "drop", (1,))
        rows = structure_rows(ctx, "drop", (1,), found, signals.extract(words)[1])
        self.assertAlmostEqual(float(rows[1, INDEX["box_height_ratio"]]), 4 / 12, places=3)
        self.assertEqual(rows[1, INDEX["op"]], features.OPS.index("drop"))

    def test_G33_readers_that_read_the_run_as_one_word_are_counted_and_no_readings_is_not_zero(self):
        reading = Reading("latin", "ordenanza", group=("p1w1", "p1w2"))
        extras = {"1": {"readings": (reading,)}, "2": {"readings": (reading,)}}
        _, rows = self.rows("join", (1, 2), "la orde nanza del", **extras)
        self.assertTrue(np.all(rows[:, INDEX["reader_group_support"]] == 1))
        _, none = self.rows("join", (1, 2), "la orde nanza del")
        self.assertTrue(np.all(np.isnan(none[:, INDEX["reader_group_support"]])))

    def test_G14_text_rows_now_carry_the_text_op_and_no_structure_values(self):
        words = page("la ordenansa del")
        ctx = PageContext(words, LANGUAGES)
        from extract.guard.candidates import generate

        found = generate(ctx, 1)
        rows = features.feature_rows(ctx, 1, found, signals.extract(words)[1])
        self.assertTrue(np.all(rows[:, INDEX["op"]] == features.OPS.index("text")))
        self.assertTrue(np.all(np.isnan(rows[:, INDEX["gap_ratio_max"]])))
        self.assertIn("informacion_acc", WORDS)


class SelectionTest(unittest.TestCase):
    def test_G35_proposals_are_taken_by_gain_and_an_overlapping_one_is_only_a_suggestion(self):
        proposals = [
            Proposal((1, 2), "join", 0.4, "ab"),
            Proposal((2,), "text", 0.9, "x"),
            Proposal((0,), "text", 0.2, "y"),
            Proposal((2, 3), "join", 0.5, "cd"),
        ]
        applied, suggested = select(proposals)
        self.assertEqual([p.span for p in applied], [(0,), (2,)])
        self.assertEqual([p.span for p in suggested], [(2, 3), (1, 2)])

    def test_G35_the_result_does_not_depend_on_the_order_of_arrival(self):
        rng = random.Random(2)
        proposals = [
            Proposal(tuple(range(s, s + rng.choice((1, 2, 3)))), "join", round(rng.random(), 2)) for s in range(30)
        ]
        first = select(proposals)
        for _ in range(10):
            rng.shuffle(proposals)
            self.assertEqual(select(proposals), first)

    def test_G35_ties_are_broken_by_span_op_and_text(self):
        a, b = Proposal((3,), "text", 0.5, "z"), Proposal((1, 3), "join", 0.5, "a")
        self.assertEqual(select([a, b])[0], [b])
        self.assertEqual(select([b, a])[0], [b])

    def test_G35_the_reading_order_rule_of_the_app_skips_nothing_of_what_was_selected(self):
        rng = random.Random(4)
        for _ in range(20):
            proposals = [
                Proposal(tuple(range(s, s + rng.choice((1, 2, 3)))), rng.choice(("text", "join")), rng.random())
                for s in range(25)
            ]
            applied, _ = select(proposals)
            self.assertEqual(in_reading_order(applied), applied)

    def test_G35_equal_gains_and_spans_are_ordered_by_op_then_text(self):
        join, text = Proposal((1, 2), "join", 0.5, "ab"), Proposal((1, 2), "text", 0.5, "ab")
        other = Proposal((1, 2), "join", 0.5, "aa")
        self.assertEqual(select([text, join, other])[0], [other])
        self.assertEqual([p.text for p in select([text, join, other])[1]], ["ab", "ab"])
        self.assertEqual([p.op for p in select([text, join, other])[1]], ["join", "text"])

    def test_G35_the_two_orders_disagree_only_when_proposals_overlap_and_the_selection_is_the_better_one(self):
        proposals = [Proposal((0, 1), "join", 0.1), Proposal((1,), "text", 0.9)]
        self.assertEqual([p.op for p in in_reading_order(proposals)], ["join"])
        self.assertEqual([p.op for p in select(proposals)[0]], ["text"])


if __name__ == "__main__":
    unittest.main()
