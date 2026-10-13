import json
import unittest
from collections import Counter
from pathlib import Path

from bench.corpus.degrade import BAND_SHARES, draw_band
from bench.corpus.split import assign_split, stable_rng, stable_unit

FRACTIONS = {"train": 0.5, "val": 0.2, "test": 0.3}


class SplitTest(unittest.TestCase):
    def test_G4_split_depends_only_on_id_and_seed(self):
        first = [assign_split(f"doc{i}", 7, FRACTIONS) for i in range(200)]
        reordered = dict(reversed(list(FRACTIONS.items())))
        self.assertEqual(first, [assign_split(f"doc{i}", 7, reordered) for i in range(200)])
        self.assertNotEqual(first, [assign_split(f"doc{i}", 8, FRACTIONS) for i in range(200)])

    def test_G4_split_shares_follow_the_fractions(self):
        counts = Counter(assign_split(f"doc{i}", 7, FRACTIONS) for i in range(6000))
        for name, share in FRACTIONS.items():
            self.assertAlmostEqual(counts[name] / 6000, share, delta=0.03)

    def test_G4_fractions_that_do_not_add_up_are_refused(self):
        with self.assertRaises(ValueError):
            assign_split("doc", 1, {"train": 0.5, "test": 0.4})
        with self.assertRaises(ValueError):
            assign_split("doc", 1, {})

    def test_G4_a_new_unit_never_changes_the_split_of_the_others(self):
        before = {f"doc{i}": assign_split(f"doc{i}", 3, FRACTIONS) for i in range(100)}
        for i in range(100, 300):
            assign_split(f"doc{i}", 3, FRACTIONS)
        self.assertEqual(before, {k: assign_split(k, 3, FRACTIONS) for k in before})


class PlanTest(unittest.TestCase):
    plan = json.loads((Path(__file__).parents[1] / "bench" / "corpus" / "split.json").read_text(encoding="utf-8"))

    def test_G4_no_document_is_in_two_sets_of_the_plan(self):
        sets_of_document = {}
        for page in self.plan["pages"]:
            if page["set"] in ("train", "val", "test", "unseen"):
                sets_of_document.setdefault(page["doc_id"], set()).add(page["set"])
        self.assertGreaterEqual(len(sets_of_document), 190)
        self.assertEqual([d for d, sets in sets_of_document.items() if len(sets) > 1], [])

    def test_G4_every_page_of_the_plan_follows_its_documents_hash(self):
        fractions = {"train": 0.465, "val": 0.186, "test": 0.279, "unseen": 0.07}
        for page in self.plan["pages"]:
            if page["set"] in fractions:
                self.assertEqual(assign_split(page["doc_id"], self.plan["seed"], fractions), page["set"])

    def test_G4_the_plan_has_no_page_listed_twice_and_its_fingerprint_matches(self):
        import hashlib

        keys = [(p["set"], p["doc_id"], p["page"]) for p in self.plan["pages"]]
        self.assertEqual(len(keys), len(set(keys)))
        entries = sorted(self.plan["pages"], key=lambda e: (e["set"], e["doc_id"], e["page"]))
        digest = hashlib.sha256(json.dumps(entries, sort_keys=True).encode()).hexdigest()
        self.assertEqual(digest, self.plan["fingerprint"])

    def test_G4_clean_and_pilot_pages_come_from_train_and_validation_documents_only(self):
        owner = {p["doc_id"]: p["set"] for p in self.plan["pages"] if p["set"] in ("train", "val", "test", "unseen")}
        extra = [p for p in self.plan["pages"] if p["set"] in ("clean", "pilot")]
        self.assertGreaterEqual(len(extra), 70)
        self.assertTrue(all(owner[p["doc_id"]] in ("train", "val") for p in extra))
        main_pages = {(p["doc_id"], p["page"]) for p in self.plan["pages"] if p["set"] in owner.values()}
        self.assertFalse(any((p["doc_id"], p["page"]) in main_pages for p in extra))


class BandAndStreamTest(unittest.TestCase):
    def test_G4_band_shares_follow_the_design(self):
        counts = Counter(draw_band(f"d/{i}", 3) for i in range(6000))
        for band, share in BAND_SHARES:
            self.assertAlmostEqual(counts[band] / 6000, share, delta=0.03)

    def test_G4_random_streams_are_reproducible_and_separate(self):
        self.assertEqual(stable_unit(1, "a"), stable_unit(1, "a"))
        self.assertNotEqual(stable_unit(1, "a"), stable_unit(1, "b"))
        self.assertEqual(stable_rng(5, "x").random(), stable_rng(5, "x").random())
        self.assertNotEqual(stable_unit(1, "ab", "c"), stable_unit(1, "a", "bc"))


if __name__ == "__main__":
    unittest.main()
