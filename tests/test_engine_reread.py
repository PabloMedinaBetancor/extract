import dataclasses
import random
import unittest

from extract.engine import reread
from extract.engine.reread import Aligned
from extract.types import Reading, Word


def line_of(spec, line_id=0, first=0, y=0.0):
    """Words of one line. `spec` is a list of (text, char confidence of every character, word confidence)."""
    words = []
    for k, (text, char, conf) in enumerate(spec):
        x = 10.0 + 60 * k
        words.append(
            Word(f"p1w{first + k}", 1, text, (x, y, x + 50.0, y + 12.0), conf, (char,) * len(text), line_id, k)
        )
    return words


def sure(text):
    return (text, 0.99, 0.99)


def doubt(text, char=0.5):
    return (text, char, 0.99)


class SelectionTest(unittest.TestCase):
    def test_R4_a_word_is_suspect_by_a_low_character_or_a_low_word_confidence(self):
        self.assertTrue(reread.is_suspect(line_of([doubt("casa", 0.5)])[0]))
        self.assertTrue(reread.is_suspect(line_of([("casa", 0.99, 0.5)])[0]))
        self.assertFalse(reread.is_suspect(line_of([sure("casa")])[0]))

    def test_R4_a_word_without_confidences_is_never_suspect(self):
        native = Word("p1w0", 1, "casa", (0.0, 0.0, 1.0, 1.0))
        self.assertFalse(reread.is_suspect(native))

    def test_R4_lines_come_with_most_suspect_words_first_then_lowest_confidence(self):
        words = (
            line_of([doubt("a1", 0.8), sure("b1")], 0, 0)
            + line_of([doubt("a2", 0.6), doubt("b2", 0.7)], 1, 2, 20.0)
            + line_of([doubt("a3", 0.3), sure("b3")], 2, 4, 40.0)
        )
        units = reread.select_lines(words)
        self.assertEqual([u.line_id for u in units], [1, 2, 0])
        self.assertEqual([u.rank for u in units], [0, 1, 2])
        self.assertEqual(units[0].suspects, (2, 3))

    def test_R4_lines_are_capped_in_units_not_seconds(self):
        words = []
        for k in range(30):
            words += line_of([doubt(f"w{k}", 0.5 + k / 100)], k, k, 20.0 * k)
        self.assertEqual(len(reread.select_lines(words)), reread.MAX_REREAD_LINES)
        self.assertEqual(len(reread.select_lines(words, limit=3)), 3)

    def test_R4_a_zone_takes_one_neighbour_on_each_side(self):
        words = line_of([sure("a"), sure("b"), doubt("c"), sure("d"), sure("e")])
        (zone,) = reread.select_zones(words)
        self.assertEqual(zone.indices, (1, 2, 3))
        self.assertEqual(zone.suspects, (2,))

    def test_R4_a_zone_at_the_edge_of_the_line_has_one_neighbour(self):
        words = line_of([doubt("a"), sure("b"), sure("c")])
        (zone,) = reread.select_zones(words)
        self.assertEqual(zone.indices, (0, 1))

    def test_R4_zones_that_touch_or_overlap_merge(self):
        words = line_of(
            [sure("a"), doubt("b"), sure("c"), doubt("d"), sure("e"), sure("f"), sure("g"), doubt("h"), sure("i")]
        )
        zones = reread.select_zones(words)
        self.assertEqual([z.indices for z in zones], [(0, 1, 2, 3, 4), (6, 7, 8)])
        self.assertEqual(zones[0].suspects, (1, 3))

    def test_R4_zones_that_only_touch_merge_into_one(self):
        words = line_of([doubt("a"), sure("b"), sure("c"), doubt("d"), sure("e")])
        (zone,) = reread.select_zones(words)
        self.assertEqual(zone.indices, (0, 1, 2, 3, 4))

    def test_R4_a_zone_is_never_longer_than_five_words_and_keeps_every_suspect(self):
        for run in range(1, 12):
            words = line_of([sure("x")] + [doubt(f"d{k}") for k in range(run)] + [sure("y")])
            zones = reread.select_zones(words, limit=100)
            self.assertTrue(all(len(z.indices) <= reread.ZONE_WORDS for z in zones), run)
            self.assertEqual(sorted(i for z in zones for i in z.suspects), list(range(1, run + 1)))
            covered = [i for z in zones for i in z.indices]
            self.assertEqual(len(covered), len(set(covered)), "zones of one run must not share words")

    def test_R4_property_a_zone_has_at_most_five_words_whatever_the_suspects_and_none_is_shared(self):
        rng = random.Random(11)
        for _ in range(400):
            length = rng.randint(1, 16)
            doubtful = {k for k in range(length) if rng.random() < 0.4}
            words = line_of([doubt(f"d{k}") if k in doubtful else sure(f"s{k}") for k in range(length)])
            zones = reread.select_zones(words, limit=100)
            self.assertTrue(all(len(z.indices) <= 5 for z in zones), doubtful)
            self.assertEqual(sorted(i for z in zones for i in z.suspects), sorted(doubtful))
            covered = [i for z in zones for i in z.indices]
            self.assertEqual(len(covered), len(set(covered)), doubtful)
            for z in zones:
                self.assertTrue(set(z.suspects) <= set(z.indices))
                self.assertEqual(z.indices, tuple(range(z.indices[0], z.indices[-1] + 1)))

    def test_R4_suspects_two_apart_in_a_long_line_give_zones_of_at_most_five_words(self):
        words = line_of(
            [
                sure("a"),
                sure("b"),
                doubt("c"),
                sure("d"),
                sure("e"),
                doubt("f"),
                sure("g"),
                sure("h"),
                doubt("i"),
                sure("j"),
                sure("k"),
                sure("l"),
            ]
        )
        zones = reread.select_zones(words, limit=100)
        self.assertTrue(all(len(z.indices) <= 5 for z in zones))
        self.assertEqual(sorted(i for z in zones for i in z.suspects), [2, 5, 8])

    def test_R4_zones_never_cross_a_line(self):
        words = line_of([sure("a"), doubt("b")], 0, 0) + line_of([doubt("c"), sure("d")], 1, 2, 20.0)
        zones = reread.select_zones(words)
        self.assertEqual([z.indices for z in zones], [(0, 1), (2, 3)])

    def test_R4_zones_come_most_doubtful_first_and_are_capped(self):
        words = line_of([doubt("a", 0.8)], 0, 0) + line_of([doubt("b", 0.2)], 1, 1, 20.0)
        zones = reread.select_zones(words)
        self.assertEqual([z.indices for z in zones], [(1,), (0,)])
        self.assertEqual(len(reread.select_zones(words, limit=1)), 1)
        many = []
        for k in range(60):
            many += line_of([doubt(f"w{k}")], k, k, 20.0 * k)
        self.assertEqual(len(reread.select_zones(many)), reread.MAX_ZONES)

    def test_R4_units_depend_only_on_the_confidences_of_the_first_reading(self):
        words = line_of([sure("a"), doubt("b"), sure("c")])
        changed = [dataclasses.replace(w, text=w.text + "zz", box=(1.0, 1.0, 2.0, 2.0), id="q" + w.id) for w in words]
        for unit in ("line", "zone"):
            before = [(u.indices, u.suspects, u.rank) for u in reread.select_units(words, unit)]
            after = [(u.indices, u.suspects, u.rank) for u in reread.select_units(changed, unit)]
            self.assertEqual(before, after)

    def test_R4_a_page_without_suspect_words_has_no_units(self):
        words = line_of([sure("a"), sure("b")])
        self.assertEqual(reread.select_units(words, "line"), [])
        self.assertEqual(reread.select_units(words, "zone"), [])

    def test_R4_an_unknown_unit_is_an_error(self):
        with self.assertRaises(ValueError):
            reread.select_units([], "page")


def reading(source, text, rank):
    return Reading(source, text, 0.9, None, (), rank)


class ViewTest(unittest.TestCase):
    def words(self):
        words = line_of([sure("a"), doubt("b"), doubt("c")])
        return (
            words[0],
            dataclasses.replace(words[1], readings=(reading("latin", "b", 0), reading("bin", "b", 0))),
            dataclasses.replace(words[2], readings=(reading("latin", "c", 2), reading("bin", "k", 2))),
        )

    def test_G39_a_cap_hides_exactly_the_readings_of_units_with_rank_at_or_above_it(self):
        words = self.words()
        for cap in range(0, 5):
            seen = reread.view(words, cap)
            for before, after in zip(words, seen, strict=True):
                kept = [r for r in (before.readings or ()) if r.rank < cap]
                self.assertEqual(list(after.readings or ()), kept)

    def test_G39_a_reading_with_no_rank_belongs_to_no_unit_and_is_never_hidden(self):
        word = dataclasses.replace(line_of([sure("a")])[0], readings=(Reading("latin", "a", 0.9, None, (), None),))
        self.assertIs(reread.view((word,), 0)[0], word)

    def test_G39_a_word_that_loses_all_its_readings_has_none_not_an_empty_tuple(self):
        seen = reread.view(self.words(), 1)
        self.assertIsNone(seen[2].readings)
        self.assertIsNone(seen[0].readings)
        self.assertEqual(len(seen[1].readings), 2)

    def test_G39_words_that_lose_nothing_are_returned_as_they_are(self):
        words = self.words()
        seen = reread.view(words, 3)
        self.assertTrue(all(a is b for a, b in zip(words, seen, strict=True)))

    def test_G39_the_view_is_pure_and_repeatable(self):
        words = self.words()
        first, second = reread.view(words, 2), reread.view(words, 2)
        self.assertEqual(first, second)
        self.assertEqual(words, self.words())

    def test_G39_everything_but_the_readings_is_left_alone(self):
        words = self.words()
        for before, after in zip(words, reread.view(words, 0), strict=True):
            self.assertEqual(dataclasses.replace(before, readings=None), dataclasses.replace(after, readings=None))


def run(first, read):
    result = reread.align(first, read)
    return None if result is None else [(a.text, a.group) for a in result]


class AlignTest(unittest.TestCase):
    def test_R5_one_to_one_pairs_by_index(self):
        self.assertEqual(run(["la", "casa"], ["la", "caza"]), [("la", (0,)), ("caza", (1,))])

    def test_R5_two_words_read_as_one_are_a_join_for_both(self):
        self.assertEqual(
            run(["de", "la", "casa"], ["dela", "casa"]), [("dela", (0, 1)), ("dela", (0, 1)), ("casa", (2,))]
        )

    def test_R5_three_words_read_as_one(self):
        self.assertEqual(run(["en", "el", "mar"], ["enelmar"]), [("enelmar", (0, 1, 2))] * 3)

    def test_R5_a_word_cut_in_two_is_a_segment_with_its_own_group(self):
        self.assertEqual(run(["entrada", "casa"], ["en", "trada", "casa"]), [("en trada", (0,)), ("casa", (1,))])

    def test_R5_an_extra_space_inside_a_word(self):
        self.assertEqual(run(["calle", "mayor"], ["cal", "le", "mayor"]), [("cal le", (0,)), ("mayor", (1,))])

    def test_R5_a_word_the_reader_added_goes_to_no_one(self):
        self.assertEqual(run(["calle", "mayor"], ["calle", "de", "mayor"]), [("calle", (0,)), ("mayor", (1,))])

    def test_R5_a_word_the_reader_dropped_has_no_text(self):
        self.assertEqual(
            run(["calle", "de", "mayor"], ["calle", "mayor"]), [("calle", (0,)), (None, (1,)), ("mayor", (2,))]
        )

    def test_R5_an_empty_answer_aligns_nothing(self):
        self.assertIsNone(run(["calle", "mayor"], []))

    def test_R5_an_answer_with_more_than_n_plus_one_words_is_not_aligned(self):
        self.assertIsNone(run(["calle"], ["c", "a", "l"]))
        self.assertIsNotNone(run(["calle"], ["ca", "lle"]))

    def test_R5_half_the_characters_is_the_limit_of_the_cost(self):
        self.assertIsNotNone(run(["abcd"], ["abxy"]))
        self.assertIsNone(run(["abcd"], ["axyz"]))

    def test_R5_two_readers_are_aligned_on_their_own(self):
        first = ["la", "casa", "grande"]
        self.assertEqual(run(first, ["la", "casa", "grande"]), [("la", (0,)), ("casa", (1,)), ("grande", (2,))])
        self.assertEqual(run(first, ["lacasa", "grande"]), [("lacasa", (0, 1)), ("lacasa", (0, 1)), ("grande", (2,))])

    def test_R5_a_zone_of_one_word_is_the_one_to_one_case(self):
        self.assertEqual(run(["Pagina"], ["Pagina"]), [("Pagina", (0,))])
        self.assertIsNone(run(["Pagina"], []))

    def test_R5_a_zone_of_five_words(self):
        first = ["la", "casa", "de", "mi", "abuela"]
        self.assertEqual([t for t, _ in run(first, ["la", "casa", "de", "mi", "abuela"])], first)

    def test_R5_a_neighbour_that_was_not_suspect_gets_its_reading_too(self):
        result = reread.align(["la", "casa", "grande"], ["la", "caza", "grande"])
        self.assertEqual([a.text for a in result], ["la", "caza", "grande"])

    def test_R5_every_word_of_the_unit_is_answered_and_none_outside_it(self):
        rng = random.Random(7)
        vocabulary = ["casa", "de", "la", "calle", "mayor", "numero", "tres", "puerta", "a"]
        for _ in range(300):
            first = [rng.choice(vocabulary) for _ in range(rng.randint(1, 5))]
            read = [rng.choice(vocabulary) for _ in range(rng.randint(0, 6))]
            result = reread.align(first, read)
            if result is None:
                continue
            self.assertEqual(len(result), len(first))
            for k, item in enumerate(result):
                self.assertIsInstance(item, Aligned)
                self.assertTrue(all(0 <= g < len(first) for g in item.group))
                self.assertTrue(all(0 <= r < len(read) for r in item.read))
                self.assertIn(k, item.group if item.text is not None else (k,))

    def test_R5_identical_words_always_align_one_to_one(self):
        rng = random.Random(3)
        vocabulary = ["casa", "de", "la", "calle", "mayor", "numero"]
        for _ in range(100):
            words = [rng.choice(vocabulary) for _ in range(rng.randint(1, 5))]
            self.assertEqual(run(words, words), [(w, (k,)) for k, w in enumerate(words)])


if __name__ == "__main__":
    unittest.main()
