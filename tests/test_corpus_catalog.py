import csv
import hashlib
import json
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from bench.corpus import catalog, degrade
from bench.corpus.catalog import Document, build_page, check_page, licence_of, list_documents

GAZETTE = "public administration (state official gazette)"
VOCABULARY = json.loads((Path(__file__).parent / "data" / "spanish_words.json").read_text(encoding="utf-8"))[
    "vocabulary"
]


def make_pdf(
    path, seed=1, words=220, leaders=0, logo=False, hidden=False, cover_image=False, pages=1, rotation=0, cropbox=None
):
    rng = random.Random(seed)
    top = 550 if rotation in (90, 270) else 790
    sheet = canvas.Canvas(str(path), pagesize=(595, 842))
    for _ in range(pages):
        if cover_image:
            picture = path.with_suffix(".png")
            cv2.imwrite(str(picture), np.full((60, 60, 3), 200, np.uint8))
            sheet.drawImage(ImageReader(str(picture)), 0, 0, 595, 842)
        sheet.setFont("Helvetica", 10)
        if hidden:
            sheet.setFillColorRGB(1, 1, 1)
        for row in range(words // 8 + 1):
            line = " ".join(f"{rng.choice(VOCABULARY)}{rng.randrange(100)}" for _ in range(8))
            sheet.drawString(40, top - 13 * row, line + (" " + "." * 40 + "12" if row < leaders else ""))
        if logo:
            sheet.setFillColorRGB(0, 0, 0)
            sheet.rect(400, 80, 120, 60, fill=1, stroke=0)
        if rotation:
            sheet.setPageRotation(rotation)
        if cropbox:
            sheet.setCropBox(cropbox)
        sheet.showPage()
    sheet.save()


def document(path, doc_id="c0001"):
    return Document(doc_id, 1, "https://www.boe.es/x.pdf", path, "00" * 32, GAZETTE)


class ScratchCase(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tmp = Path(folder.name)


class SourcesTest(ScratchCase):
    def test_G4_only_public_administrations_have_a_licence_class(self):
        self.assertEqual(licence_of("https://www.boe.es/buscar/pdf/1994/x.pdf"), GAZETTE)
        self.assertEqual(
            licence_of("https://sede.gobiernodecanarias.org/a.pdf"), "public administration (regional government)"
        )
        self.assertIsNone(licence_of("https://arquitectosgrancanaria.es/doc.pdf"))
        self.assertIsNone(licence_of("https://evilboe.es/doc.pdf"))
        self.assertIsNone(licence_of(None))
        self.assertIsNone(licence_of(""))

    def test_G4_the_document_list_keeps_current_public_pdfs_that_exist(self):
        (self.tmp / "originals").mkdir()
        make_pdf(self.tmp / "originals" / "a.pdf")
        connection = sqlite3.connect(self.tmp / "catalog.db")
        connection.execute(
            "CREATE TABLE sources(id INTEGER, url TEXT, original_path TEXT, original_sha256 TEXT, state TEXT, kind TEXT)"
        )
        rows = [
            (1, "https://www.boe.es/a.pdf", "originals/a.pdf", "a" * 64, "current", "regulation"),
            (2, "https://arquitectosgrancanaria.es/b.pdf", "originals/a.pdf", "b" * 64, "current", "planning"),
            (3, "https://www.boe.es/c.html", "originals/c.html", "c" * 64, "current", "regulation"),
            (4, "https://www.boe.es/d.pdf", "originals/gone.pdf", "d" * 64, "current", "regulation"),
            (5, "https://www.boe.es/e.pdf", "originals/a.pdf", "e" * 64, "superseded", "regulation"),
            (6, "https://www.boe.es/f.pdf", "originals/a.pdf", "f" * 64, "current", "market"),
        ]
        connection.executemany("INSERT INTO sources VALUES (?,?,?,?,?,?)", rows)
        connection.commit()
        connection.close()
        documents, skipped = list_documents(self.tmp)
        self.assertEqual([d.catalog_id for d in documents], [1])
        self.assertEqual(documents[0].doc_id, "c" + "a" * 10)
        self.assertEqual(skipped, {"not_pdf": 1, "private_publisher": 1, "missing_file": 1})


class EligibilityTest(ScratchCase):
    def reason(self, name, **options):
        make_pdf(self.tmp / name, **options)
        return check_page(document(self.tmp / name), 1, set())

    def test_G3_a_faithful_text_page_is_eligible(self):
        page, reason = self.reason("ok.pdf")
        self.assertEqual(reason, "ok")
        self.assertGreaterEqual(len(page.words), 150)

    def test_G3_pages_with_few_words_are_rejected(self):
        self.assertEqual(self.reason("short.pdf", words=40)[1], "few_words")

    def test_G3_a_page_covered_by_an_image_is_rejected(self):
        self.assertEqual(self.reason("scan.pdf", cover_image=True)[1], "image_cover")

    def test_G3_a_contents_page_with_dot_leaders_is_rejected(self):
        self.assertEqual(self.reason("index.pdf", leaders=6)[1], "dot_leaders")
        self.assertEqual(self.reason("one.pdf", leaders=1)[1], "ok")

    def test_G3_text_that_cannot_be_seen_is_rejected(self):
        self.assertEqual(self.reason("ghost.pdf", hidden=True)[1], "no_ink")

    def test_G3_a_missing_poppler_tool_stops_the_run_instead_of_rejecting_every_page(self):
        make_pdf(self.tmp / "ok.pdf")
        missing = FileNotFoundError(2, "No such file or directory", "pdfimages")
        with mock.patch.object(catalog.subprocess, "run", side_effect=missing):
            with self.assertRaisesRegex(catalog.CorpusError, "poppler-utils"):
                check_page(document(self.tmp / "ok.pdf"), 1, set())

    def test_G4_a_repeated_page_is_rejected_the_second_time(self):
        make_pdf(self.tmp / "a.pdf", seed=5)
        make_pdf(self.tmp / "b.pdf", seed=5)
        seen = set()
        page, _ = check_page(document(self.tmp / "a.pdf"), 1, seen)
        seen.add(page.key)
        self.assertEqual(check_page(document(self.tmp / "b.pdf", "c0002"), 1, seen)[1], "repeated")

    def test_G3_an_unreadable_file_is_rejected_not_accepted(self):
        (self.tmp / "broken.pdf").write_bytes(b"not a pdf")
        page, reason = check_page(document(self.tmp / "broken.pdf"), 1, set())
        self.assertIsNone(page)
        self.assertNotEqual(reason, "ok")


class PageFilesTest(ScratchCase):
    def test_G3_truth_boxes_sit_on_the_ink_of_the_written_page(self):
        make_pdf(self.tmp / "p.pdf", seed=3)
        doc = document(self.tmp / "p.pdf")
        page, _ = check_page(doc, 1, set())
        for band in ("clean", "light", "heavy"):
            with self.subTest(band=band):
                build_page(doc, page, band, band, self.tmp / "data", seed=9)
                folder = self.tmp / "data" / "pages" / band / doc.doc_id
                record = json.loads((folder / "1.json").read_text(encoding="utf-8"))
                image = cv2.imread(str(folder / "1.png"), cv2.IMREAD_GRAYSCALE)
                self.assertIsNone(record["ocr"])
                self.assertEqual((record["image"]["width"], record["image"]["height"]), image.shape[::-1])
                inked = 0
                for entry in record["truth"]["words"]:
                    x0, y0, x1, y1 = (int(round(v)) for v in entry["box"])
                    region = image[max(0, y0) : y1 + 1, max(0, x0) : x1 + 1]
                    inked += bool(region.size and region.min() < 140)
                self.assertGreaterEqual(inked / len(record["truth"]["words"]), 0.95)

    def test_G3_truth_boxes_follow_a_rotated_page_and_an_offset_crop_box(self):
        cases = {
            "turned 90": {"rotation": 90},
            "turned 180": {"rotation": 180},
            "turned 270": {"rotation": 270},
            "cropped": {"cropbox": (30, 40, 560, 800)},
        }
        for name, options in cases.items():
            with self.subTest(case=name):
                make_pdf(self.tmp / "r.pdf", seed=6, **options)
                doc = document(self.tmp / "r.pdf")
                page, reason = check_page(doc, 1, set())
                self.assertEqual(reason, "ok")
                build_page(doc, page, "clean", "clean", self.tmp / name, seed=1)
                folder = self.tmp / name / "pages" / "clean" / doc.doc_id
                record = json.loads((folder / "1.json").read_text(encoding="utf-8"))
                image = cv2.imread(str(folder / "1.png"), cv2.IMREAD_GRAYSCALE)
                inked = sum(
                    bool(image[int(y0) : int(y1) + 1, int(x0) : int(x1) + 1].min() < 140)
                    for x0, y0, x1, y1 in (w["box"] for w in record["truth"]["words"])
                )
                self.assertGreaterEqual(inked / len(record["truth"]["words"]), 0.95)

    def test_G3_a_pdf_outside_the_catalogue_gets_the_same_page_files_with_its_own_source(self):
        make_pdf(self.tmp / "invoice.pdf", seed=4)
        band = catalog.build_pdf_page(self.tmp / "invoice.pdf", 1, "i0001", "train", self.tmp / "data")
        self.assertIn(band, ("light", "medium", "heavy"))
        folder = self.tmp / "data" / "pages" / "train" / "i0001"
        record = json.loads((folder / "1.json").read_text(encoding="utf-8"))
        self.assertEqual((record["source"], record["band"], record["ocr"]), ("invoice", band, None))
        self.assertGreater(len(record["truth"]["words"]), 100)
        self.assertTrue((folder / "1.png").is_file())

    def test_G3_ink_that_the_text_layer_does_not_describe_is_listed_as_ungrounded(self):
        make_pdf(self.tmp / "logo.pdf", seed=2, logo=True)
        doc = document(self.tmp / "logo.pdf")
        page, _ = check_page(doc, 1, set())
        build_page(doc, page, "clean", "clean", self.tmp / "data", seed=1)
        record = json.loads((self.tmp / "data" / "pages" / "clean" / doc.doc_id / "1.json").read_text(encoding="utf-8"))
        self.assertEqual(len(record["ungrounded"]), 1)
        x0, y0, x1, y1 = record["ungrounded"][0]
        scale = record["image"]["dpi"] / 72.0
        self.assertAlmostEqual(x0 / scale, 400, delta=8)
        self.assertAlmostEqual(y1 / scale, 842 - 80, delta=8)

    def test_G3_a_page_that_holds_an_ocr_reading_is_not_rendered_over_by_mistake(self):
        make_pdf(self.tmp / "p.pdf", seed=3)
        doc = document(self.tmp / "p.pdf")
        page, _ = check_page(doc, 1, set())
        build_page(doc, page, "train", "light", self.tmp / "data", seed=1)
        target = self.tmp / "data" / "pages" / "train" / doc.doc_id / "1.json"
        record = json.loads(target.read_text(encoding="utf-8"))
        target.write_text(json.dumps({**record, "ocr": {"words": []}}), encoding="utf-8")
        with self.assertRaises(catalog.CorpusError):
            build_page(doc, page, "train", "light", self.tmp / "data", seed=1)
        self.assertIsNotNone(json.loads(target.read_text(encoding="utf-8"))["ocr"])
        build_page(doc, page, "train", "light", self.tmp / "data", seed=1, overwrite=True)
        self.assertIsNone(json.loads(target.read_text(encoding="utf-8"))["ocr"])

    def test_G3_rendering_again_with_other_ranges_keeps_the_selected_truth(self):
        make_pdf(self.tmp / "p.pdf", seed=8)
        doc = document(self.tmp / "p.pdf")
        page, _ = check_page(doc, 1, set())
        gentle = {"light": {**degrade.BANDS["light"], "blur_sigma": (0.0, 0.0), "noise_sigma": (0.0, 0.0)}}
        harsh = {"light": {**degrade.BANDS["light"], "blur_sigma": (1.5, 1.5), "noise_sigma": (20.0, 20.0)}}
        texts, pixels = [], []
        for name, ranges in (("a", gentle), ("b", harsh)):
            build_page(doc, page, name, "light", self.tmp / "data", seed=2, ranges=ranges)
            folder = self.tmp / "data" / "pages" / name / doc.doc_id
            record = json.loads((folder / "1.json").read_text(encoding="utf-8"))
            texts.append([w["text"] for w in record["truth"]["words"]])
            pixels.append(cv2.imread(str(folder / "1.png"), cv2.IMREAD_GRAYSCALE))
        self.assertEqual(texts[0], texts[1])
        self.assertFalse(np.array_equal(pixels[0], pixels[1]))


class SmallBuildTest(ScratchCase):
    QUOTAS = {"train": (4, 2), "val": (2, 1), "test": (3, 1), "unseen": (2, 1)}
    FRACTIONS = {"train": 0.25, "val": 0.25, "test": 0.25, "unseen": 0.25}

    def make_catalogue(self):
        catalog_dir = self.tmp / "catalog"
        (catalog_dir / "originals").mkdir(parents=True)
        connection = sqlite3.connect(catalog_dir / "catalog.db")
        connection.execute(
            "CREATE TABLE sources(id INTEGER, url TEXT, original_path TEXT, original_sha256 TEXT, state TEXT, kind TEXT)"
        )
        for number in range(30):
            make_pdf(catalog_dir / "originals" / f"{number}.pdf", seed=100 + number, pages=3)
            sha = hashlib.sha256(str(number).encode()).hexdigest()
            connection.execute(
                "INSERT INTO sources VALUES (?,?,?,?,?,?)",
                (number, f"https://www.boe.es/{number}.pdf", f"originals/{number}.pdf", sha, "current", "regulation"),
            )
        connection.commit()
        connection.close()
        return catalog_dir

    def build(self, catalog_dir):
        data = self.tmp / "data"
        with (
            mock.patch.object(catalog, "QUOTAS", self.QUOTAS),
            mock.patch.object(catalog, "FRACTIONS", self.FRACTIONS),
            mock.patch.object(catalog, "CLEAN_PAGES", 3),
            mock.patch.object(catalog, "PILOT_PER_BAND", 1),
        ):
            result = catalog.select(data, catalog_dir)
            catalog.write_manifests(result, self.tmp / "SOURCES.csv", self.tmp / "split.json")
            return catalog.render_selected(data, self.tmp / "split.json", catalog_dir)

    def test_G4_a_small_build_never_puts_a_document_in_two_sets(self):
        data = self.tmp / "data"
        quotas = self.QUOTAS
        rendered = self.build(self.make_catalogue())
        plan = json.loads((self.tmp / "split.json").read_text(encoding="utf-8"))
        sets_of_document = {}
        for entry in plan["pages"]:
            if entry["set"] in quotas:
                sets_of_document.setdefault(entry["doc_id"], set()).add(entry["set"])
        self.assertTrue(sets_of_document)
        self.assertTrue(all(len(sets) == 1 for sets in sets_of_document.values()))
        self.assertEqual({e["set"] for e in plan["pages"]}, {"train", "val", "test", "unseen", "clean", "pilot"})
        self.assertEqual(len(plan["fingerprint"]), 64)
        with open(self.tmp / "SOURCES.csv", newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual({r["licence"] for r in rows}, {GAZETTE})
        self.assertEqual(len({r["doc_id"] for r in rows}), len(rows))
        self.assertEqual(rendered, len(plan["pages"]))
        for entry in plan["pages"]:
            folder = data / "pages" / entry["set"] / entry["doc_id"]
            self.assertTrue((folder / f"{entry['page']}.png").is_file())
            self.assertTrue((folder / f"{entry['page']}.json").is_file())

    def test_G4_growing_the_test_set_adds_documents_and_changes_nothing_that_was_selected(self):
        catalog_dir = self.make_catalogue()
        self.build(catalog_dir)
        data, split, sources = self.tmp / "data", self.tmp / "split.json", self.tmp / "SOURCES.csv"
        before = json.loads(split.read_text(encoding="utf-8"))
        old_files = {p: p.read_bytes() for p in (data / "pages").glob("*/*/*") if p.suffix in (".png", ".json")}
        wanted = len({e["doc_id"] for e in before["pages"] if e["set"] == "test"}) + 1
        with (
            mock.patch.object(catalog, "QUOTAS", self.QUOTAS),
            mock.patch.object(catalog, "FRACTIONS", self.FRACTIONS),
        ):
            outcome = catalog.extend_test(data, catalog_dir, split, sources, test_documents=wanted)
        after = json.loads(split.read_text(encoding="utf-8"))
        self.assertEqual(outcome["new_documents"], 1)
        self.assertEqual(outcome["test_documents"], wanted)
        self.assertNotEqual(after["fingerprint"], before["fingerprint"])
        self.assertEqual([e for e in after["pages"] if e in before["pages"]], before["pages"])
        added = [e for e in after["pages"] if e not in before["pages"]]
        self.assertEqual({e["set"] for e in added}, {"test"})
        self.assertFalse({e["doc_id"] for e in added} & {e["doc_id"] for e in before["pages"]})
        for path, content in old_files.items():
            self.assertEqual(path.read_bytes(), content)
        for entry in added:
            self.assertTrue((data / "pages" / entry["set"] / entry["doc_id"] / f"{entry['page']}.png").is_file())
        with open(sources, newline="", encoding="utf-8") as handle:
            self.assertEqual({r["doc_id"] for r in csv.DictReader(handle)} >= {e["doc_id"] for e in added}, True)


if __name__ == "__main__":
    unittest.main()
