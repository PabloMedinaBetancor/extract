import ast
import unittest
from pathlib import Path

from extract.guard import candidates
from extract.guard.candidates import (
    MAX_CANDIDATES,
    PageContext,
    dictionary_texts,
    engine_alternative_texts,
    generate,
    page_texts,
    reading_texts,
)
from extract.guard.edits import weighted_distance
from extract.guard.forms import normalise
from tests.guard_words import Reading, page
from tests.words import WORDS

GUARD = Path(candidates.__file__).parent


def texts(words, index, **kwargs):
    ctx = PageContext(words, ("es", "en"), **kwargs)
    return [c.text for c in generate(ctx, index)]


def by_text(words, index):
    return {c.text: c for c in generate(PageContext(words, ("es", "en")), index)}


class ListShapeTest(unittest.TestCase):
    def test_G16_the_token_as_read_comes_first_and_alone(self):
        words = page("el plan de ordenansa")
        listed = generate(PageContext(words, ("es", "en")), 3)
        self.assertEqual((listed[0].text, listed[0].op, listed[0].kind), ("ordenansa", "keep", "keep"))
        self.assertEqual(sum(c.op == "keep" for c in listed), 1)

    def test_G16_candidates_are_unique_by_normalised_form_and_at_most_the_cap(self):
        for token in ("ordenansa", "Ordenansa", "ORDENANSA,", "a", "informacion"):
            listed = generate(PageContext(page(f"el {token} de"), ("es", "en")), 1)
            keys = [normalise(c.text) for c in listed]
            self.assertEqual(len(keys), len(set(keys)), token)
            self.assertLessEqual(len(listed), MAX_CANDIDATES, token)

    def test_G16_the_order_is_deterministic_and_does_not_depend_on_pages_seen_before(self):
        words = page("la ordenansa del plan", "el sueio urbano")
        first = texts(words, 2)
        texts(page("otra pagina con soelo y vivlenda"), 1)
        self.assertEqual(first, texts(words, 2))

    def test_G16_a_candidate_never_equals_the_token_in_another_case_of_the_same_normal_form(self):
        listed = generate(PageContext(page("el plan"), ("es", "en")), 1)
        self.assertNotIn("plan", [c.text for c in listed[1:]])


class WrittenFormTest(unittest.TestCase):
    def test_G16_a_title_case_token_gets_title_case_candidates(self):
        self.assertIn("Ordenanza", texts(page("la Ordenansa"), 1))

    def test_G16_an_upper_case_token_keeps_its_case_and_punctuation(self):
        self.assertIn("ORDENANZA,", texts(page("la ORDENANSA, del"), 1))

    def test_G16_a_lost_accent_is_restored_in_every_regular_case(self):
        words = page("la INFORMACION del")
        found = by_text(words, 1)
        accented = WORDS["informacion_acc"]
        self.assertIn(accented.upper(), found)
        self.assertEqual(found[accented.upper()].kind, "accent")
        self.assertIn(accented, found)
        self.assertEqual(found[accented].kind, "accent_case")

    def test_G16_a_comma_is_restored_when_the_line_ends_there(self):
        self.assertIn("plan,", texts(page("el gran plan", "otra linea"), 2))

    def test_G16_no_mark_is_added_in_the_middle_of_a_sentence(self):
        found = texts(page("el gran plan de obra"), 2)
        self.assertNotIn("plan,", found)
        self.assertNotIn("plan.", found)

    def test_G16_a_wrong_mark_is_replaced_or_removed(self):
        found = texts(page("el plan, de obra"), 1)
        self.assertIn("plan.", found)
        self.assertIn("plan", found)

    def test_G16_a_stray_mark_in_front_of_a_word_is_dropped(self):
        self.assertIn("plan", texts(page("el 'plan de"), 1))

    def test_G16_a_misread_letter_that_comes_with_a_misread_comma_is_one_candidate(self):
        accent = WORDS["urbanistico"]
        found = texts(page("el " + accent.replace("\u00ed", "i") + ". de"), 1)
        self.assertIn(accent + ",", found)

    def test_G16_a_stray_mark_and_a_misread_letter_are_one_candidate(self):
        self.assertIn("plan", texts(page("el 'pian de"), 1))

    def test_G16_the_letters_of_an_unlisted_token_are_offered_in_the_other_cases(self):
        self.assertIn("Valsequillo", texts(page("de valsequillo y"), 1))


class SourcesTest(unittest.TestCase):
    def test_G16_the_dictionary_proposes_a_nearby_listed_word(self):
        found = by_text(page("la ordenansa del"), 1)
        self.assertEqual(found["ordenanza"].sources[0], "dictionary")

    def test_G16_a_word_repeated_on_the_page_is_proposed_by_the_page_even_if_unlisted(self):
        found = by_text(page("Valsequillo y Valsequillo con Valsequiilo"), 4)
        self.assertEqual(found["Valsequillo"].sources, ("page",))

    def test_G16_a_word_seen_once_is_not_page_vocabulary(self):
        self.assertNotIn("Valsequillo", texts(page("Valsequillo y Valsequiilo"), 2))

    def test_G16_sources_that_agree_are_listed_together_in_a_fixed_order(self):
        found = by_text(page("la ordenansa del", "y la ordenanza ordenanza"), 1)
        self.assertEqual(found["ordenanza"].sources, ("dictionary", "page"))

    def test_G16_the_engine_alternatives_propose_the_next_best_class_of_the_doubtful_characters(self):
        alts = tuple(((c, 0.99),) for c in "cata")
        alts = (alts[0], alts[1], (("t", 0.4), ("s", 0.35)), alts[3])
        words = page("el cata", **{"1": {"char_alts": alts}})
        found = by_text(words, 1)
        self.assertIn("ctc_alt", found["casa"].sources)

    def test_G16_alternatives_that_do_not_fit_the_word_are_ignored(self):
        words = page("el cata", **{"1": {"char_alts": ((("c", 0.9),),)}})
        self.assertNotIn("ctc_alt", {s for c in generate(PageContext(words, ("es", "en")), 1) for s in c.sources})

    def test_G16_a_second_reading_that_differs_is_a_candidate_named_after_its_reader(self):
        reading = Reading("latin", "obra")
        words = page("la cbra", **{"1": {"readings": (reading,)}})
        found = by_text(words, 1)
        self.assertIn("reading:latin", found["obra"].sources)

    def test_G16_a_reading_aligned_one_to_one_has_the_words_own_id_as_its_group_and_still_proposes_its_text(self):
        # the aligner gives every reading a group, the word itself when the reader saw it alone (found on the pilot's real readings)
        words = page("la cbra", **{"1": {"readings": (Reading("latin", "obra", group=("p1w1",)),)}})
        self.assertIn("reading:latin", by_text(words, 1)["obra"].sources)
        self.assertFalse(candidates.joins_words(Reading("latin", "obra", group=("p1w1",))))
        self.assertFalse(candidates.joins_words(Reading("latin", None)))
        self.assertTrue(candidates.joins_words(Reading("latin", "obra", group=("p1w1", "p1w2"))))

    def test_G16_a_reading_that_agrees_with_the_token_proposes_nothing(self):
        words = page("la obra", **{"1": {"readings": (Reading("latin", "obra"),)}})
        listed = generate(PageContext(words, ("es", "en")), 1)
        self.assertNotIn("reading:latin", {s for c in listed for s in c.sources})

    def test_G16_an_unaligned_reading_and_a_cut_or_joined_one_propose_no_text(self):
        readings = (Reading("latin", None), Reading("bin", "ob ra"), Reading("clahe", "obra", group=("p1w1", "p1w2")))
        words = page("la cbra", **{"1": {"readings": readings}})
        sources = {s for c in generate(PageContext(words, ("es", "en")), 1) for s in c.sources}
        self.assertFalse({s for s in sources if s.startswith("reading:")})

    def test_G16_the_masked_decode_is_a_candidate_when_it_differs(self):
        words = page("la ob" + "丨" + "a", **{"1": {"masked": "obra"}})
        self.assertIn("masked", by_text(words, 1)["obra"].sources)

    def test_G16_candidates_with_a_digit_are_never_proposed(self):
        alts = ((("l", 0.5), ("1", 0.4)),) + tuple(((c, 0.99),) for c in "ibro")
        words = page("el libro", **{"1": {"char_alts": alts}})
        self.assertFalse([t for t in texts(words, 1) if any(c.isdigit() for c in t)])


class CapsAndPoolsTest(unittest.TestCase):
    def test_G16_a_token_with_more_than_sixteen_proposals_is_cut_at_sixteen_and_the_token_stays_first(self):
        ctx = PageContext(page("la ca de"), ("es", "en"))
        proposed = {t for texts_ in candidates.by_source(ctx, 1).values() for t in texts_}
        self.assertGreater(len({normalise(t) for t in proposed}), 16)
        listed = generate(ctx, 1)
        self.assertEqual(len(listed), 16)
        self.assertEqual(listed[0].text, "ca")

    def test_G16_the_dictionary_offers_twenty_words_and_the_case_variants_of_the_nearest_five(self):
        texts_ = dictionary_texts(PageContext(page("la el de"), ("es", "en")), 1)
        forms: dict[str, set[str]] = {}
        for text in texts_:
            forms.setdefault(text.casefold(), set()).add(text)
        self.assertEqual(len(forms), 20)
        self.assertEqual(sorted(len(found) for found in forms.values()).count(3), 5)
        self.assertTrue(all(len(found) in (1, 3) for found in forms.values()))

    def test_G16_the_engine_alternatives_use_only_the_three_least_certain_positions(self):
        doubtful = {1: (0.5, ("b", 0.3)), 3: (0.6, ("c", 0.3)), 5: (0.7, ("d", 0.3)), 7: (0.8, ("e", 0.3))}
        found = engine_alternative_texts(_word("aaaaaaaa", doubtful))
        self.assertEqual(len(found), 6)
        self.assertFalse([t for t in found if t[7] != "a"])
        self.assertTrue(all(sum(c != "a" for c in t) <= 2 for t in found))
        self.assertIn("abaaaaaa", found)
        self.assertIn("abacaaaa", found)

    def test_G16_the_engine_alternatives_are_at_most_twelve_and_ordered_by_likelihood(self):
        doubtful = {
            1: (0.5, ("b", 0.30), ("c", 0.20), ("d", 0.10)),
            3: (0.6, ("e", 0.29), ("f", 0.19), ("g", 0.09)),
            5: (0.7, ("h", 0.28), ("i", 0.18), ("j", 0.08)),
        }
        found = engine_alternative_texts(_word("aaaaaaaa", doubtful))
        self.assertEqual(len(found), 12)
        self.assertEqual(found[0], "abaaaaaa")

    def test_G16_two_alternatives_of_one_position_are_never_applied_together(self):
        doubtful = {0: (0.5, ("b", 0.3), ("c", 0.2)), 1: (0.6, ("d", 0.35))}
        self.assertEqual(engine_alternative_texts(_word("aaaa", doubtful)), ["adaa", "baaa", "caaa", "bdaa", "cdaa"])

    def test_G16_alternatives_longer_or_shorter_than_the_word_give_no_candidates(self):
        longer = _alts("cala", {0: (0.5, ("e", 0.3))}) + ((("x", 0.99),),)
        shorter = _alts("cala", {0: (0.5, ("e", 0.3))})[:3]
        for alts in (longer, shorter):
            word = page("cala", **{"0": {"char_alts": alts}})[0]
            self.assertEqual(engine_alternative_texts(word), [])

    def test_G16_candidates_with_the_same_sources_are_ordered_by_the_weighted_distance(self):
        alts = _alts("zcalzz", {0: (0.5, ("q", 0.3)), 1: (0.6, ("e", 0.35))})
        words = page("la zcalzz", **{"1": {"char_alts": alts}})
        listed = [c for c in generate(PageContext(words, ("es", "en")), 1) if c.sources == ("ctc_alt",)]
        self.assertEqual([c.text for c in listed][:2], ["zealzz", "qcalzz"])

    def test_G16_the_page_proposes_repeated_words_within_two_edits_and_not_three(self):
        words = page("Valsequillo Valsequillo Valsxquixlo Valsxquixxo")
        ctx = PageContext(words, ("es", "en"))
        self.assertIn("Valsequillo", page_texts(ctx, 2))
        self.assertNotIn("Valsequillo", page_texts(ctx, 3))

    def test_G16_a_candidate_that_differs_from_the_token_only_in_typography_is_not_listed(self):
        alts = _alts("obra\u2019", {4: (0.5, ("'", 0.3), (",", 0.2))})
        words = page("la obra\u2019 de", **{"1": {"char_alts": alts}})
        listed = [c.text for c in generate(PageContext(words, ("es", "en")), 1)]
        self.assertNotIn("obra'", listed)
        self.assertEqual(len({normalise(t) for t in listed}), len(listed))

    def test_G16_two_proposals_that_differ_only_in_typography_are_one_candidate(self):
        alts = _alts("obra,", {4: (0.5, ("\u2019", 0.3), ("'", 0.2))})
        words = page("la obra, de", **{"1": {"char_alts": alts}})
        listed = [c.text for c in generate(PageContext(words, ("es", "en")), 1)]
        self.assertEqual(len([t for t in listed if t in ("obra'", "obra\u2019")]), 1)

    def test_G16_a_masked_decode_with_a_digit_proposes_nothing(self):
        word = page("la ob" + "\u4e28" + "a", **{"1": {"masked": "ob1a"}})[1]
        self.assertEqual(reading_texts(word), {})

    def test_G16_a_misread_pair_of_letters_ranks_by_its_confusion_cost_not_by_plain_edits(self):
        self.assertEqual(weighted_distance("rnesa", "mesa"), 0.25)
        self.assertEqual(weighted_distance("rnesa", "resa"), 1.0)
        self.assertEqual(weighted_distance("m0sa", "mosa"), 0.3)
        ctx = PageContext(page("la rnesa de"), ("es", "en"))
        self.assertEqual(dictionary_texts(ctx, 1)[:2], ["mesa", "resa"])
        listed = [c.text for c in generate(ctx, 1)]
        self.assertLess(listed.index("mesa"), listed.index("resa") if "resa" in listed else len(listed))


def _alts(text, doubtful):
    """Per-character alternatives: certain (0.99) except where `doubtful` gives (top probability, *(char, probability))."""
    out = []
    for i, char in enumerate(text):
        if i in doubtful:
            top, *others = doubtful[i]
            out.append(((char, top), *others))
        else:
            out.append(((char, 0.99),))
    return tuple(out)


def _word(text, doubtful):
    return page(text, **{"0": {"char_alts": _alts(text, doubtful)}})[0]


class ProtectionTest(unittest.TestCase):
    def site(self, token, *, repeated=0, line="el X de"):
        words = page(line.replace("X", token), " ".join([token] * repeated) or "relleno")
        return PageContext(words, ("es", "en")).is_site(1)

    def test_G23_ordinary_words_are_sites(self):
        for token in ("obra", "Obra", "OBRA", "obra,", "(obra)"):
            self.assertTrue(self.site(token), token)

    def test_G23_tokens_with_digits_and_symbols_are_never_sites(self):
        for token in ("a1b", "1234", "a@b", "a/b", "foo_bar", "#obra", "a\\b", "10%", "casa-obra", "don't"):
            self.assertFalse(self.site(token), token)

    def test_G23_a_token_over_forty_characters_is_not_a_site(self):
        self.assertFalse(self.site("a" * 41))
        self.assertTrue(self.site("a" * 40))

    def test_G23_two_interior_capitals_protect_a_token_but_one_does_not(self):
        for token in ("GrafCan", "GrafCaN", "McDonald"):
            self.assertFalse(self.site(token), token)
        for token in ("hOla", "iPhone", "Grafcan"):
            self.assertTrue(self.site(token), token)

    def test_G23_a_hyphen_or_an_apostrophe_at_either_end_protects_the_token(self):
        marks = ("-", "\u2010", "\u2011", "\u2013", "\u2014", "\u2212", "\u00ad-", "'", "\u2019", "\u2018")
        for mark in marks:
            for token in (f"obra{mark}", f"{mark}obra", f"l{mark}", f"{mark}obra{mark}", f"({mark}obra)"):
                self.assertFalse(self.site(token), token)

    def test_G23_no_site_ever_has_a_hyphen_or_an_apostrophe_in_any_position(self):
        tokens = [f"{a}obra{b}" for a in ("", "(", "-", "'", "\u2019") for b in ("", ",", ".", "-", "'", "\u2019", ")")]
        ctx = PageContext(page(" ".join(tokens)), ("es", "en"))
        for index, token in enumerate(tokens):
            if ctx.is_site(index):
                self.assertFalse(set(token) & set("-'\u2019\u2010\u2013"), token)
        self.assertTrue(any(ctx.is_site(i) for i in range(len(tokens))))

    def test_G23_a_colon_closes_a_word_only_as_its_very_last_character(self):
        for token in ("Urbanisticos:", "obra:", "(obra):"):
            self.assertTrue(self.site(token), token)
        for token in ("a:b", ":a", ":obra", "http:", "https:", "HTTP:", "obra:)", "obra,:x", "ob:ra"):
            self.assertFalse(self.site(token), token)

    def test_G23_a_short_acronym_repeated_on_its_page_is_protected_and_a_single_one_is_not(self):
        self.assertFalse(self.site("BOE", repeated=3))
        self.assertTrue(self.site("BOE", repeated=1))
        self.assertTrue(self.site("ABCDEFG", repeated=3))

    def test_G23_a_line_end_hyphen_fragment_is_not_a_site(self):
        words = page("el plan de ordena-", "miento general")
        ctx = PageContext(words, ("es", "en"), fragments={3, 4})
        self.assertFalse(ctx.is_site(3))
        self.assertFalse(ctx.is_site(4))
        self.assertTrue(ctx.is_site(5))

    def test_G23_a_token_that_is_not_plain_has_no_parts(self):
        ctx = PageContext(page("a1 b"), ("es", "en"))
        self.assertIsNone(ctx.parts[0])
        self.assertFalse(ctx.is_site(0))


class ImportGraphTest(unittest.TestCase):
    def test_G14_the_guard_imports_nothing_from_the_benchmark(self):
        for path in GUARD.glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                names = [a.name for a in node.names] if isinstance(node, ast.Import) else []
                if isinstance(node, ast.ImportFrom) and node.module:
                    names.append(node.module)
                for name in names:
                    self.assertNotEqual(name.split(".")[0], "bench", f"{path.name} imports {name}")
