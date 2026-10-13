import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from bench import run_corpus
from bench.corpus import build5, build_en, discover_en, sealed, survey5

SPLIT_V5 = Path(__file__).parents[1] / "bench" / "corpus" / "split-v5.json"
# SHA-256 of the 1,426 entries of split-v5.json before the language was added, without any `language`.
SPANISH_ENTRIES = "dafe26ade4b29058aa3c633535ce468735b7609f6870c5e8aeaa44586e24765e"


def rows(count: int, pages: int = 4) -> list[dict]:
    return [
        {"doc_id": f"c{i:03d}", "catalog_id": i, "pages": [{"page": p + 1, "words": 400} for p in range(pages)]}
        for i in range(count)
    ]


class CandidatesTest(unittest.TestCase):
    def test_the_candidates_are_the_same_on_every_run_and_only_from_free_licence_publishers(self):
        first, second = discover_en.candidate_urls(), discover_en.candidate_urls()
        self.assertEqual(first, second)
        self.assertGreater(len(first), 300)
        self.assertEqual(len({url for url, _ in first}), len(first))
        for url, _ in first:
            self.assertIsNotNone(discover_en.licence_of(url), url)
        self.assertIsNone(discover_en.licence_of("https://example.com/a.pdf"))

    def test_the_rows_of_the_catalogue_come_from_the_log_once_per_file(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            (out / "catalog" / "originals").mkdir(parents=True)
            for name in ("cAAA", "cBBB"):
                (out / "catalog" / "originals" / f"{name}.pdf").write_bytes(b"%PDF")
            (out / "discover-en.log").write_text(
                "https://www.irs.gov/pub/irs-pdf/p15.pdf\tok\tcAAA\taaa\treport\n"
                "https://www.irs.gov/pub/irs-pdf/p17.pdf\thttp-404\n"
                "https://www.irs.gov/pub/irs-pdf/p15copy.pdf\tok\tcAAA\taaa\treport\n"
                "https://www.govinfo.gov/content/pkg/PLAW-118publ10/pdf/PLAW-118publ10.pdf\tok\tcBBB\tbbb\tlegislation\n",
                encoding="utf-8",
            )
            found = discover_en.rows_from_log(out)
        self.assertEqual([r["doc_id"] for r in found], ["cAAA", "cBBB"])
        self.assertEqual({r["language"] for r in found}, {"en"})
        self.assertEqual([r["unit_id"] for r in found], ["cAAA", "cBBB"])
        self.assertIn("public domain", found[0]["licence"])


class PlanTest(unittest.TestCase):
    def test_with_enough_documents_the_test_has_100_candidates_and_development_100_with_three_pages_each(self):
        frozen = build_en.plan(rows(260))
        self.assertEqual((len(frozen["test"]), len(frozen["development"])), (100, 100))
        self.assertEqual({len(d["pages"]) for d in frozen["test"] + frozen["development"]}, {3})
        self.assertEqual([d["position"] for d in frozen["test"]], list(range(100)))
        self.assertEqual({d["language"] for d in frozen["test"]}, {"en"})

    def test_no_document_is_on_both_sides_and_the_order_is_the_survey_order(self):
        data = rows(230)
        frozen = build_en.plan(data)
        test = [d["doc_id"] for d in frozen["test"]]
        development = [d["doc_id"] for d in frozen["development"]]
        self.assertFalse(set(test) & set(development))
        self.assertEqual(test + development, [r["doc_id"] for r in data][:200])

    def test_with_few_documents_the_test_keeps_its_floor_of_60_and_development_gets_the_rest(self):
        frozen = build_en.plan(rows(130))
        self.assertEqual((len(frozen["test"]), len(frozen["development"])), (60, 70))
        short = build_en.plan(rows(90))
        self.assertEqual((len(short["test"]), len(short["development"])), (60, 30))
        self.assertEqual(short["pages_per_document"], 4)

    def test_a_document_with_one_eligible_page_is_not_in_the_list(self):
        data = rows(220)
        data[3]["pages"] = data[3]["pages"][:1]
        frozen = build_en.plan(data)
        listed = {d["doc_id"] for d in frozen["test"] + frozen["development"]}
        self.assertNotIn("c003", listed)

    def test_the_folds_of_the_development_documents_balance_the_bands_and_every_page_carries_its_fold_and_language(
        self,
    ):
        frozen = build_en.plan(rows(220))
        folds = build5.assign_folds(build_en.development_units(frozen))
        made = build_en.entries(frozen, folds)
        development = [e for e in made if e["set"] == build_en.SET_DEV]
        self.assertEqual(len(development), 300)
        self.assertEqual({e["language"] for e in made}, {"en"})
        self.assertEqual({e["fold"] for e in development}, {f"fold{k}" for k in range(5)})
        self.assertTrue(all("fold" not in e for e in made if e["set"] == build_en.SET_TEST))
        self.assertEqual({e["role"] for e in made}, {"dev", "test5en", "readback"})
        readback = [e for e in made if e["role"] == "readback"]
        self.assertEqual(len(readback), 30)
        self.assertEqual({e["band"] for e in readback}, {"clean"})
        self.assertTrue({e["doc_id"] for e in readback} <= {d["doc_id"] for d in frozen["development"]})
        self.assertFalse({e["doc_id"] for e in readback} & {d["doc_id"] for d in frozen["test"]})
        by_document = {}
        for e in development:
            by_document.setdefault(e["doc_id"], set()).add(e["fold"])
        self.assertTrue(all(len(folds_) == 1 for folds_ in by_document.values()))


class ManifestTest(unittest.TestCase):
    def test_the_spanish_entries_did_not_move_when_the_language_was_added(self):
        body = json.loads(SPLIT_V5.read_text(encoding="utf-8"))
        pages = [p for p in body["pages"] if p["set"] not in build_en.ENGLISH_SETS]
        self.assertEqual(len(pages), 1426)
        self.assertEqual(build_en.spanish_fingerprint([dict(p, language="es") for p in pages]), SPANISH_ENTRIES)

    def test_a_catalogue_page_is_spanish_and_an_invoice_has_the_language_of_its_sidecar(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            (data / "invoices" / "train").mkdir(parents=True)
            (data / "invoices" / "train" / "inv-1.json").write_text(json.dumps({"language": "en"}), encoding="utf-8")
            (data / "invoices" / "train" / "inv-2.json").write_text(json.dumps({"language": "es"}), encoding="utf-8")
            pages = [
                {"set": "train", "doc_id": "cX", "page": 1, "source": "catalog"},
                {"set": "train", "doc_id": "inv-1", "page": 1, "source": "invoice"},
                {"set": "train", "doc_id": "inv-2", "page": 1, "source": "invoice"},
                {"set": "dev5en", "doc_id": "cE", "page": 1, "source": "catalog", "language": "en"},
            ]
            done = build_en.add_languages(pages, data)
        self.assertEqual([p["language"] for p in done], ["es", "en", "es", "en"])
        self.assertEqual(
            [{k: v for k, v in p.items() if k != "language"} for p in done],
            [{k: v for k, v in p.items() if k != "language"} for p in pages],
        )

    def test_every_page_of_the_published_manifest_says_its_language(self):
        body = json.loads(SPLIT_V5.read_text(encoding="utf-8"))
        if "t5en" not in body:
            self.skipTest("the English corpus is not built yet")
        self.assertEqual({p["language"] for p in body["pages"]}, {"es", "en"})
        self.assertEqual(
            body["t5"]["list_fingerprint"], "40630d08ca3d13781f7458d33cd0eedc2c5629d59652c8f2df6a29b5b1b11c26"
        )
        self.assertEqual(body["spanish_entries_fingerprint"], SPANISH_ENTRIES)
        self.assertEqual(body["t5en"]["list_fingerprint"], build5.fingerprint_of(body["t5en"]["documents"]))
        self.assertGreaterEqual(len(body["t5en"]["documents"]), build_en.MIN_TEST_DOCUMENTS)


class SealedTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        for name, doc, _position in (
            ("test5en", "eA", 0),
            ("test5en", "eB", 1),
            ("test5en", "eC", 2),
            ("dev5en", "eD", None),
        ):
            path = self.root / "pages" / name / doc / "1.json"
            path.parent.mkdir(parents=True)
            cv2.imwrite(str(path.with_suffix(".png")), np.full((10, 10, 3), 255, np.uint8))
            path.write_text(
                json.dumps({"set": name, "doc_id": doc, "page": 1, "truth": {"words": []}}), encoding="utf-8"
            )

    def test_the_english_test_is_refused_to_development_and_read_only_through_its_own_sets(self):
        with self.assertRaises(sealed.SealedTestError):
            sealed.refuse_sealed_set("test5en")
        sealed.refuse_sealed_set("dev5en")
        self.assertNotIn("test5en", sealed.SEALED_SETS)
        reads = self.root / "reads.json"
        pages = sealed.read_test_pages(self.root, "ocr", reads, sets=sealed.EN_TEST_SETS)
        self.assertEqual([r["doc_id"] for _, r in pages], ["eA", "eB", "eC"])
        self.assertEqual(sealed.load_reads(reads)[0]["pages"], 3)

    def test_run_corpus_reads_the_first_n_documents_of_the_english_test_and_the_development_set_whole(self):
        entries = [
            {"set": "test5en", "doc_id": d, "page": 1, "role": "test5en", "position": i}
            for i, d in enumerate(("eA", "eB", "eC"))
        ] + [{"set": "dev5en", "doc_id": "eD", "page": 1, "role": "dev"}]
        manifest = self.root / "split.json"
        manifest.write_text(json.dumps({"pages": entries}), encoding="utf-8")
        before = Path.cwd()
        os.chdir(self.root)
        try:
            limited = run_corpus.pages_of(manifest, ["test5en"], self.root, documents=2)
            development = run_corpus.pages_of(manifest, ["dev5en"], self.root)
            everything = run_corpus.pages_of(manifest, ["test5en"], self.root)
        finally:
            os.chdir(before)
        self.assertEqual(sorted(p.parent.name for _, p in limited), ["eA", "eB"])
        self.assertEqual([p.parent.name for _, p in development], ["eD"])
        self.assertEqual(len(everything), 3)
        self.assertTrue((self.root / "bench" / "results" / "test_reads_v5en.json").is_file())
        self.assertFalse((self.root / "bench" / "results" / "test_reads_v5.json").exists())


class SurveyTest(unittest.TestCase):
    def test_the_english_survey_keeps_its_own_folder_and_its_duplicate_index_holds_every_set(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            for name in ("train", "survey5", "survey5en"):
                kept = data / "selection" / name / "cA"
                kept.mkdir(parents=True)
                words = [[f"w{i}", f"word{name}{i}", [0, 0, 1, 1]] for i in range(30)]
                (kept / "1.json").write_text(json.dumps({"key": f"key-{name}", "words": words}), encoding="utf-8")
            spanish, keys = survey5.development_index(data)
            everything, all_keys = survey5.development_index(data, everything=True)
        self.assertEqual((len(spanish.sizes), keys), (1, {"key-train"}))
        self.assertEqual((len(everything.sizes), all_keys), (3, {"key-train", "key-survey5", "key-survey5en"}))


def words_of(seed: str, count: int = 80) -> list[str]:
    return [f"{seed}{i}" for i in range(count)]


class VerifyTest(unittest.TestCase):
    """`verify` finds what is planted in it: a repeated page, a near page, a document on both sides, a translation split
    across the sides, and says nothing when there is nothing."""

    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.pages: list[dict] = []
        self.units: dict[str, str] = {}

    def page(self, set_name: str, doc: str, words: list[str], number: int = 1, role: str = "dev") -> None:
        path = self.root / "pages" / set_name / doc / f"{number}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        record = {"doc_id": doc, "set": set_name, "source": "catalog", "page": number}
        record["truth"] = {"words": [{"id": f"t{i}", "text": w, "box": [0, 0, 1, 1]} for i, w in enumerate(words)]}
        path.write_text(json.dumps(record), encoding="utf-8")
        if set_name in ("dev5en", "test5en", "clean5en"):
            self.pages.append({"set": set_name, "doc_id": doc, "page": number, "role": role})

    def check(self) -> dict:
        split = self.root / "split.json"
        split.write_text(json.dumps({"pages": self.pages}), encoding="utf-8")
        sources = self.root / "sources.csv"
        with open(sources, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=discover_en.SOURCES_COLUMNS)
            writer.writeheader()
            docs = {p["doc_id"] for p in self.pages}
            for doc in sorted(docs):
                writer.writerow({"doc_id": doc, "unit_id": self.units.get(doc, doc)})
        return build_en.verify(self.root, split, sources)

    def clean_corpus(self) -> None:
        self.page("train", "cES", words_of("es"))
        self.page("test5", "cT5", words_of("ts"))
        self.page("dev5en", "cEd", words_of("dev"))
        self.page("test5en", "cEt", words_of("tst"), role="test5en")
        self.page("clean5en", "cEd", words_of("rb"), number=2, role="readback")

    def test_nothing_planted_gives_zeros(self):
        self.clean_corpus()
        found = self.check()
        for key in (
            "pages_repeating_another_page",
            "pages_near_a_page_of_another_set",
            "english_pages_near_an_earlier_english_page",
            "documents_in_both",
            "units_on_both_sides",
            "readback_documents_in_test",
        ):
            self.assertEqual(found[key], 0, key)
        self.assertEqual((found["english_pages_checked"], found["pages_checked_against"]), (3, 2))

    def test_an_english_page_copied_from_a_spanish_set_is_repeated_and_near(self):
        self.clean_corpus()
        self.page("dev5en", "cEx", words_of("es"), role="dev")
        found = self.check()
        self.assertEqual(found["pages_repeating_another_page"], 1)
        self.assertEqual(found["pages_near_a_page_of_another_set"], 1)
        self.assertEqual(found["largest_jaccard_against_other_sets"], 1.0)

    def test_a_page_that_differs_in_its_first_words_is_near_but_not_repeated(self):
        self.clean_corpus()
        copy = words_of("ts")
        copy[:3] = ["x1", "x2", "x3"]
        self.page("test5en", "cEy", copy, role="test5en")
        found = self.check()
        self.assertEqual(found["pages_repeating_another_page"], 0)
        self.assertEqual(found["pages_near_a_page_of_another_set"], 1)

    def test_an_english_page_near_an_earlier_english_page_is_found(self):
        self.clean_corpus()
        self.page("dev5en", "cEz", words_of("dev"), role="dev")
        found = self.check()
        self.assertGreaterEqual(found["english_pages_near_an_earlier_english_page"], 1)
        self.assertEqual(found["largest_jaccard_between_english_pages"], 1.0)

    def test_a_document_in_both_development_and_test_is_found(self):
        self.clean_corpus()
        self.page("test5en", "cEd", words_of("other"), number=3, role="test5en")
        self.assertEqual(self.check()["documents_in_both"], 1)

    def test_a_readback_page_of_a_test_document_is_found(self):
        self.clean_corpus()
        self.page("clean5en", "cEt", words_of("rbt"), role="readback")
        self.assertEqual(self.check()["readback_documents_in_test"], 1)

    def test_two_documents_of_one_unit_on_different_sides_are_found(self):
        self.clean_corpus()
        self.units = {"cEd": "unit-1", "cEt": "unit-1"}
        found = self.check()
        self.assertEqual(found["units_on_both_sides"], 1)
        self.assertEqual(found["documents_in_both"], 0)


if __name__ == "__main__":
    unittest.main()
