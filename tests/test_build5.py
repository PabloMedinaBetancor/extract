import csv
import json
import random
import sqlite3
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from reportlab.pdfgen import canvas

from bench.corpus import build5, invoices, survey5
from bench.corpus.degrade import draw_band
from bench.corpus.survey5 import DuplicateIndex, shingles
from tests.test_corpus_catalog import VOCABULARY, document, make_pdf

SPLIT_V5 = Path(__file__).parents[1] / "bench" / "corpus" / "split-v5.json"


def row(doc_id, pages):
    return {"doc_id": doc_id, "catalog_id": 1, "pages": [{"page": n, "words": 200} for n in pages]}


class SurveyTest(unittest.TestCase):
    def test_G17_a_page_that_repeats_another_has_jaccard_one_and_an_unrelated_one_zero(self):
        words = [f"palabra{i}" for i in range(60)]
        index = DuplicateIndex()
        index.add(shingles(words))
        self.assertEqual(index.max_jaccard(shingles(words)), 1.0)
        self.assertEqual(index.max_jaccard(shingles([f"otra{i}" for i in range(60)])), 0.0)

    def test_G17_the_same_words_in_another_order_are_not_a_republished_page(self):
        words = [f"palabra{i}" for i in range(100)]
        shuffled = words[:]
        random.Random(1).shuffle(shuffled)
        index = DuplicateIndex()
        index.add(shingles(words))
        self.assertLess(index.max_jaccard(shingles(shuffled)), 0.1)

    def test_G17_a_republished_page_with_the_second_half_changed_reaches_the_threshold(self):
        words = [f"palabra{i}" for i in range(100)]
        republished = words[:70] + [f"nueva{i}" for i in range(30)]
        index = DuplicateIndex()
        index.add(shingles(words))
        self.assertGreaterEqual(index.max_jaccard(shingles(republished)), survey5.MAX_JACCARD)

    def test_G17_a_page_sharing_a_third_of_its_text_stays_under_the_threshold(self):
        words = [f"palabra{i}" for i in range(100)]
        index = DuplicateIndex()
        index.add(shingles(words))
        mostly_new = words[:30] + [f"nueva{i}" for i in range(70)]
        self.assertLess(index.max_jaccard(shingles(mostly_new)), survey5.MAX_JACCARD)

    def test_G17_the_survey_drops_a_page_that_repeats_a_development_page_and_one_that_repeats_a_t5_page(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            make_pdf(tmp / "a.pdf", seed=1, pages=2)
            make_pdf(tmp / "b.pdf", seed=1, pages=2)
            make_pdf(tmp / "c.pdf", seed=2, pages=2)
            index, keys, rejected = DuplicateIndex(), set(), Counter()
            first = survey5.check_document(document(tmp / "a.pdf", "ca"), index, keys, rejected, tmp / "sel")
            repeat = survey5.check_document(document(tmp / "b.pdf", "cb"), index, keys, rejected, tmp / "sel")
            other = survey5.check_document(document(tmp / "c.pdf", "cc"), index, keys, rejected, tmp / "sel")
        self.assertGreaterEqual(len(first), 1)
        self.assertEqual(repeat, [])
        self.assertGreaterEqual(len(other), 1)
        self.assertGreaterEqual(rejected["repeated"] + rejected["near_duplicate"], 2)

    def test_G17_the_kept_truth_of_an_eligible_page_is_written_under_its_document(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            make_pdf(tmp / "a.pdf", seed=3, pages=3)
            found = survey5.check_document(document(tmp / "a.pdf", "cz"), DuplicateIndex(), set(), Counter(), tmp / "s")
            kept = json.loads((tmp / "s" / "survey5" / "cz" / f"{found[0]['page']}.json").read_text(encoding="utf-8"))
        self.assertEqual(len(kept["words"]), found[0]["words"])

    def test_G17_used_documents_come_from_sources_and_from_the_v4_plan(self):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            with open(tmp / "sources.csv", "w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["doc_id", "sha256"])
                writer.writerow(["c1", "h1"])
            (tmp / "split.json").write_text(json.dumps({"pages": [{"doc_id": "c2"}]}), encoding="utf-8")
            ids, hashes = survey5.used_documents(tmp / "sources.csv", tmp / "split.json")
        self.assertEqual((ids, hashes), ({"c1", "c2"}, {"h1"}))


def lines_of(seed, count=28):
    rng = random.Random(seed)
    return [" ".join(f"{rng.choice(VOCABULARY)}{rng.randrange(100)}" for _ in range(8)) for _ in range(count)]


def pdf_of(path, lines):
    sheet = canvas.Canvas(str(path), pagesize=(595, 842))
    sheet.setFont("Helvetica", 10)
    for row, line in enumerate(lines):
        sheet.drawString(40, 790 - 13 * row, line)
    sheet.save()


class EachFilterTest(unittest.TestCase):
    """The two republication filters of B1, each pinned on its own: the 500-character key and the shingle Jaccard."""

    def survey_pair(self, first, second):
        with tempfile.TemporaryDirectory() as folder:
            tmp = Path(folder)
            pdf_of(tmp / "a.pdf", first)
            pdf_of(tmp / "b.pdf", second)
            index, keys, rejected = DuplicateIndex(), set(), Counter()
            survey5.check_document(document(tmp / "a.pdf", "ca"), index, keys, rejected, tmp / "sel")
            kept = survey5.check_document(document(tmp / "b.pdf", "cb"), index, keys, rejected, tmp / "sel")
        return kept, rejected

    def test_G17_the_key_alone_drops_a_page_with_the_same_first_500_characters_and_a_different_rest(self):
        original = lines_of(1)
        other = original[:12] + lines_of(2, 100)
        kept, rejected = self.survey_pair(original, other)
        self.assertEqual(kept, [])
        self.assertEqual((rejected["repeated"], rejected["near_duplicate"]), (1, 0))
        self.assertLess(DuplicateIndex().max_jaccard(frozenset()), 0.5)

    def test_G17_the_jaccard_alone_drops_a_page_with_other_first_characters_and_most_of_the_text_repeated(self):
        original = lines_of(3)
        republished = lines_of(4, 8) + original[8:]
        kept, rejected = self.survey_pair(original, republished)
        self.assertEqual(kept, [])
        self.assertEqual((rejected["repeated"], rejected["near_duplicate"]), (0, 1))

    def test_G17_a_page_that_shares_only_a_fifth_of_its_text_is_kept(self):
        original = lines_of(5)
        mostly_new = lines_of(6, 23) + original[:5]
        kept, rejected = self.survey_pair(original, mostly_new)
        self.assertEqual(len(kept), 1)
        self.assertEqual(rejected["near_duplicate"] + rejected["repeated"], 0)

    def test_G17_the_development_index_has_the_pages_of_every_v4_set_and_not_the_survey_of_v5(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            for name in ("train", "val", "test", "unseen", "clean", "pilot", "survey5", "clean5"):
                target = data / "selection" / name / "cdoc" / "1.json"
                target.parent.mkdir(parents=True)
                words = [[i, f"{name}{i}", [0, 0, 1, 1]] for i in range(20)]
                target.write_text(json.dumps({"key": f"key-{name}", "words": words}), encoding="utf-8")
            index, keys = survey5.development_index(data)
        self.assertEqual(len(index.sizes), 6)
        self.assertEqual(keys, {f"key-{name}" for name in ("train", "val", "test", "unseen", "clean", "pilot")})

    def test_G17_a_document_is_left_out_by_its_hash_even_when_its_id_is_not_listed(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "originals").mkdir()
            (root / "originals" / "a.pdf").write_bytes(b"%PDF")
            db = sqlite3.connect(root / "catalog.db")
            db.execute(
                "CREATE TABLE sources (id INTEGER, url TEXT, original_path TEXT, original_sha256 TEXT, state TEXT, kind TEXT)"
            )
            db.execute(
                "INSERT INTO sources VALUES (1, 'https://www.boe.es/a.pdf', 'originals/a.pdf', ?, 'current', 'regulation')",
                ("ab" * 32,),
            )
            db.commit()
            db.close()
            kept, dropped = survey5.candidates(root, set(), {"ab" * 32})
            again, _ = survey5.candidates(root, set(), set())
        self.assertEqual((kept, dropped["used_by_v4"]), ([], 1))
        self.assertEqual(len(again), 1)


class PlanTest(unittest.TestCase):
    def rows(self, eligible, singles=0):
        rows = [row(f"e{i}", [1, 2, 3, 4]) for i in range(eligible)]
        return rows[:2] + [row(f"s{i}", [7]) for i in range(singles)] + rows[2:]

    def test_G17_three_pages_a_document_when_at_least_100_documents_qualify_and_four_when_fewer(self):
        many = build5.plan_t5(self.rows(130))
        few = build5.plan_t5(self.rows(90))
        self.assertEqual({len(d["pages"]) for d in many["documents"]}, {3})
        self.assertEqual({len(d["pages"]) for d in few["documents"]}, {4})
        self.assertEqual(many["pages_per_document"], 3)

    def test_G17_the_list_is_capped_in_survey_order_and_the_last_documents_carry_the_unseen_damage(self):
        plan = build5.plan_t5(self.rows(170))
        names = [d["doc_id"] for d in plan["documents"]]
        self.assertEqual(names, [f"e{i}" for i in range(150)])
        slices = [d["slice"] for d in plan["documents"]]
        self.assertEqual(slices, ["pages"] * 135 + ["unseen"] * 15)

    def test_G17_the_read_back_pages_are_spare_pages_of_standard_t5_documents_and_never_t5_pages(self):
        plan = build5.plan_t5(self.rows(130), readback=30)
        standard = {d["doc_id"]: d for d in plan["documents"] if d["slice"] == "pages"}
        self.assertEqual(len(plan["readback"]), 30)
        for item in plan["readback"]:
            self.assertIn(item["doc_id"], standard)
            self.assertNotIn(item["page"], standard[item["doc_id"]]["pages"])

    def test_G17_a_document_with_no_spare_page_has_no_read_back_page(self):
        rows = [row(f"e{i}", [1, 2, 3]) for i in range(120)]
        self.assertEqual(build5.plan_t5(rows)["readback"], [])

    def test_G17_a_document_with_one_page_is_not_in_t5(self):
        plan = build5.plan_t5(self.rows(120, singles=10))
        self.assertFalse({d["doc_id"] for d in plan["documents"]} & {f"s{i}" for i in range(10)})

    def test_G17_the_band_of_a_t5_page_comes_from_the_v5_seed_and_the_page_id(self):
        plan = build5.plan_t5(self.rows(120))
        for item in build5.t5_entries(plan):
            if item["set"] == "test5":
                self.assertEqual(item["band"], draw_band(f"{item['doc_id']}/{item['page']}", survey5.SEED))

    def test_G17_a_document_with_two_pages_keeps_both(self):
        rows = [row(f"e{i}", [1, 2]) for i in range(120)]
        plan = build5.plan_t5(rows)
        self.assertEqual({len(d["pages"]) for d in plan["documents"]}, {2})

    def test_G17_t5_entries_name_each_page_once_with_its_set_band_and_seed(self):
        plan = build5.plan_t5(self.rows(120))
        entries = build5.t5_entries(plan)
        keys = [(e["set"], e["doc_id"], e["page"]) for e in entries]
        self.assertEqual(len(keys), len(set(keys)))
        self.assertEqual({e["band"] for e in entries if e["set"] == "unseen5"}, {"unseen"})
        self.assertEqual({e["band"] for e in entries if e["set"] == "clean5"}, {"clean"})
        self.assertTrue(all(len(e["seed"]) == 8 for e in entries))


class FoldsTest(unittest.TestCase):
    def units(self):
        return {f"d{i}": {"light": 2, "medium": i % 3, "heavy": i % 2} for i in range(150)}

    def test_G17_every_unit_is_in_exactly_one_fold_and_the_assignment_is_a_function_of_the_ids(self):
        first = build5.assign_folds(self.units())
        second = build5.assign_folds(dict(reversed(list(self.units().items()))))
        self.assertEqual(first["assignment"], second["assignment"])
        self.assertEqual(set(first["assignment"].values()), {f"fold{k}" for k in range(5)})

    def test_G17_the_chosen_hash_seed_balances_the_bands_better_than_seed_zero(self):
        units = self.units()
        chosen = build5.assign_folds(units)
        single = build5.assign_folds(units, tries=1)
        self.assertLessEqual(chosen["worst_band_deviation"], single["worst_band_deviation"])
        self.assertLess(chosen["worst_band_deviation"], 0.25)


class RolesTest(unittest.TestCase):
    def pages(self):
        pages = []
        for set_name, docs in (("train", ["c1", "c2", "i1", "i2"]), ("val", ["c3"]), ("test", ["c4", "i3"])):
            for doc in docs:
                source = "invoice" if doc.startswith("i") else "catalog"
                pages.append(
                    {"set": set_name, "doc_id": doc, "page": 1, "band": "light", "seed": "00", "source": source}
                )
        pages.append({"set": "unseen", "doc_id": "c5", "page": 1, "band": "unseen", "seed": "00", "source": "catalog"})
        pages.append({"set": "clean", "doc_id": "c1", "page": 2, "band": "clean", "seed": "00", "source": "catalog"})
        return pages

    def sidecars(self):
        return {
            "i1": {"layout": "L1", "font": "dejavu-sans"},
            "i2": {"layout": "L3", "font": "dejavu-sans"},
            "i3": {"layout": "L5", "font": "ubuntu"},
        }

    def test_G17_invoices_of_the_held_out_layouts_and_fonts_are_set_aside_and_never_get_a_fold(self):
        entries, _ = build5.v5_entries(self.pages(), self.sidecars())
        role = {e["doc_id"]: e["role"] for e in entries if e["set"] != "clean"}
        self.assertEqual((role["i1"], role["i2"], role["i3"]), ("dev", "aside", "aside"))
        self.assertTrue(all("fold" not in e for e in entries if e["role"] != "dev"))
        self.assertTrue(all("fold" in e for e in entries if e["role"] == "dev"))

    def test_G17_the_v4_test_is_development_and_the_unseen_slice_a_shift_check_that_never_trains(self):
        entries, _ = build5.v5_entries(self.pages(), self.sidecars())
        role = {e["doc_id"]: e["role"] for e in entries}
        self.assertEqual((role["c4"], role["c5"]), ("dev", "shift"))
        self.assertEqual({e["role"] for e in entries if e["set"] == "clean"}, {"aux"})

    def test_G17_a_document_has_one_fold_for_all_its_pages(self):
        pages = self.pages() + [
            {"set": "train", "doc_id": "c1", "page": 9, "band": "heavy", "seed": "00", "source": "catalog"}
        ]
        entries, _ = build5.v5_entries(pages, self.sidecars())
        self.assertEqual(len({e["fold"] for e in entries if e["doc_id"] == "c1" and e["role"] == "dev"}), 1)

    def test_G17_the_held_out_rule_names_the_two_layouts_and_two_fonts(self):
        self.assertTrue(invoices.held_out_in_v5("L3", "dejavu-sans"))
        self.assertTrue(invoices.held_out_in_v5("L1", "liberation-mono"))
        self.assertFalse(invoices.held_out_in_v5("L1", "dejavu-sans"))


class InvoicePlanTest(unittest.TestCase):
    def test_G17_the_new_invoices_use_only_held_out_layouts_and_half_of_them_a_held_out_font(self):
        units = invoices.plan_v5()
        self.assertEqual(len(units), 60)
        self.assertEqual({u["layout"].key for u in units}, set(invoices.V5_HELD_LAYOUTS))
        held_fonts = [u for u in units if u["font"].key in invoices.V5_HELD_FONTS]
        self.assertEqual(len(held_fonts), 30)
        self.assertEqual({u["layout"].key for u in held_fonts}, set(invoices.V5_HELD_LAYOUTS))
        self.assertEqual(len({u["id"] for u in units}), 60)

    def test_G17_no_new_invoice_id_or_seed_is_one_of_v4(self):
        self.assertTrue(all(u["id"].startswith("inv-t5-") for u in invoices.plan_v5()))
        self.assertNotEqual(invoices.V5_SEED, 1)


class PilotTest(unittest.TestCase):
    def data(self):
        errors, bands = {}, {}
        for i in range(600):
            key = ("train", f"d{i // 6}", i % 6 + 1)
            bands[key] = ("heavy", "medium", "light")[i % 3]
            errors[key] = (i * 7) % 40
        return errors, bands

    def test_G17_the_pilot_has_25_heavy_25_medium_and_10_light_pages_each_with_errors(self):
        errors, bands = self.data()
        sample = build5.pilot_sample(errors, bands)
        self.assertEqual(Counter(e["band"] for e in sample), {"heavy": 25, "medium": 25, "light": 10})
        self.assertTrue(all(e["wrong"] > 0 for e in sample))
        self.assertEqual(len({(e["set"], e["doc_id"], e["page"]) for e in sample}), 60)

    def test_G17_the_pilot_spans_the_error_counts_and_takes_at_most_two_pages_of_a_document(self):
        errors, bands = self.data()
        sample = build5.pilot_sample(errors, bands)
        heavy = sorted(e["wrong"] for e in sample if e["band"] == "heavy")
        self.assertLess(heavy[0], 14)
        self.assertGreater(heavy[-1], 26)
        self.assertLessEqual(max(Counter(e["doc_id"] for e in sample).values()), 2)

    def test_G17_the_pilot_is_the_same_on_every_run(self):
        errors, bands = self.data()
        self.assertEqual(build5.pilot_sample(errors, bands), build5.pilot_sample(errors, bands))


class ManifestFileTest(unittest.TestCase):
    """The manifest the experiments read: written by `build5 manifest`, checked here."""

    def test_G17_split_v5_exists_and_its_fingerprint_matches_its_body(self):
        body = json.loads(SPLIT_V5.read_text(encoding="utf-8"))
        rest = {k: v for k, v in body.items() if k != "fingerprint"}
        self.assertEqual(build5.fingerprint_of(rest), body["fingerprint"])

    def test_M1_the_hashes_of_t5_are_in_the_manifest_and_equal_the_freeze_record(self):
        body = json.loads(SPLIT_V5.read_text(encoding="utf-8"))
        record = json.loads((SPLIT_V5.parents[1] / "results" / "test-freeze-v5.json").read_text(encoding="utf-8"))[-1]
        for part in ("images", "truth", "ungrounded"):
            self.assertEqual(body["t5"]["hashes"][part], record[part])
        self.assertEqual(body["t5"]["pages_hashed"], record["pages"])
        self.assertEqual(record["pages"], sum(1 for p in body["pages"] if p["role"] in ("test5", "unseen5")))

    def test_M1_the_read_back_of_30_undamaged_pages_of_t5_documents_reaches_97_percent(self):
        report = json.loads((SPLIT_V5.parents[1] / "results" / "readback-v5.json").read_text(encoding="utf-8"))
        self.assertEqual(report["pages"], 30)
        self.assertGreaterEqual(report["ok_share"], 0.97)
        self.assertTrue(report["meets_floor"])

    def test_G17_no_document_is_in_two_sets_so_development_and_t5_are_disjoint(self):
        group = {
            "dev": "development",
            "aux": "development",
            "aside": "aside",
            "shift": "shift",
            "test5": "t5",
            "unseen5": "t5-unseen",
            "readback": "t5",
            "test5en": "t5en",
        }
        groups = {}
        for page in json.loads(SPLIT_V5.read_text(encoding="utf-8"))["pages"]:
            # the English read-back pages are spare pages of English DEVELOPMENT documents (T5-EN is never opened)
            name = "development" if page["set"] == "clean5en" else group[page["role"]]
            groups.setdefault(page["doc_id"], set()).add(name)
        self.assertTrue(groups)
        self.assertEqual([doc for doc, found in groups.items() if len(found) != 1], [])

    def test_G17_every_development_document_is_in_one_fold_and_set_aside_ones_in_none(self):
        pages = json.loads(SPLIT_V5.read_text(encoding="utf-8"))["pages"]
        folds = {}
        for page in pages:
            if page["role"] == "dev":
                folds.setdefault(page["doc_id"], set()).add(page["fold"])
            else:
                self.assertNotIn("fold", page)
        self.assertTrue(all(len(found) == 1 for found in folds.values()))
        self.assertEqual({f for found in folds.values() for f in found}, {f"fold{k}" for k in range(5)})

    def test_G17_the_v4_test_pages_are_development_and_the_unseen_pages_are_shift(self):
        pages = json.loads(SPLIT_V5.read_text(encoding="utf-8"))["pages"]
        counts = Counter((p["set"], p["role"]) for p in pages)
        self.assertEqual(counts[("unseen", "shift")], 30)
        self.assertEqual(counts[("test", "dev")] + counts[("test", "aside")], 295)
        self.assertEqual(sum(v for (s, _), v in counts.items() if s in ("train", "val", "test")), 808)

    def test_M1_the_pilot_lists_60_development_pages_by_band(self):
        body = json.loads(SPLIT_V5.read_text(encoding="utf-8"))
        dev = {(p["set"], p["doc_id"], p["page"]) for p in body["pages"] if p["role"] == "dev"}
        self.assertEqual(Counter(e["band"] for e in body["pilot"]), {"heavy": 25, "medium": 25, "light": 10})
        self.assertTrue(all((e["set"], e["doc_id"], e["page"]) in dev for e in body["pilot"]))

    def test_M1_the_frozen_list_matches_the_t5_pages_and_t5_has_at_least_100_catalogue_documents(self):
        body = json.loads(SPLIT_V5.read_text(encoding="utf-8"))
        listed = {d["doc_id"] for d in body["t5"]["documents"] if d["slice"] != "invoices"}
        in_pages = {
            p["doc_id"] for p in body["pages"] if p["role"] in ("test5", "unseen5") and p["source"] == "catalog"
        }
        self.assertEqual(listed, in_pages)
        self.assertGreaterEqual(len(listed), 100)

    def test_M1_the_new_invoices_carry_only_held_out_layouts(self):
        pages = json.loads(SPLIT_V5.read_text(encoding="utf-8"))["pages"]
        layouts = {p["layout"] for p in pages if p["role"] == "test5" and p["source"] == "invoice"}
        self.assertEqual(layouts, set(invoices.V5_HELD_LAYOUTS))


if __name__ == "__main__":
    unittest.main()
