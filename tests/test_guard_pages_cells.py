"""Loading the development pages with their second readings: only the `dev` pages of the manifest, and the readings of
the cells of pass 2 merged on the words as the product has them."""

import dataclasses
import json
import random
import tempfile
import unittest
from pathlib import Path

from bench.guard import pairs
from extract.engine.cache import reading_from_dict, rereads_to_dict
from extract.guard import readings
from tests.guard_pages import record
from tests.guard_words import Reading


class PagesWithCellsTest(unittest.TestCase):
    def build(self, folder: Path, cells_of: dict[str, tuple[str, ...]], documents: int = 3):
        """Three documents in `train`: d0 and d1 are `dev`, d2 is set aside; `cells_of` says which cells each has read."""
        rng = random.Random(4)
        entries = []
        for d in range(documents):
            rec = record(f"d{d}", 1, rng, errors=0.3, words=20)
            target = folder / "pages" / "train" / f"d{d}"
            target.mkdir(parents=True)
            (target / "1.json").write_text(json.dumps(rec), encoding="utf-8")
            entries.append(
                {
                    "set": "train",
                    "doc_id": f"d{d}",
                    "page": 1,
                    "role": "aside" if d == documents - 1 else "dev",
                    "fold": f"fold{d + 3}",
                }
            )
            first = reading_from_dict(rec["ocr"]["reading"])
            for cell in cells_of.get(f"d{d}", ()):
                source = "latin" if cell == "L" else "deep"
                words = tuple(
                    dataclasses.replace(w, readings=(Reading(source, w.text, rank=0, group=(w.id,)),))
                    if i % 3 == 0
                    else w
                    for i, w in enumerate(first.words)
                )
                page = dataclasses.replace(first, words=words, rereads={0: (source,)})
                (target / f"1.rr2-{cell}.json").write_text(json.dumps(rereads_to_dict(page, "k")), encoding="utf-8")
        manifest = folder / "split.json"
        manifest.write_text(json.dumps({"pages": entries}), encoding="utf-8")
        return manifest

    def test_only_the_development_pages_of_the_manifest_are_loaded(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = self.build(folder, {})
            everything = pairs.load_pages(folder, ["train"])
            dev = pairs.load_pages(folder, ["train"], manifest=manifest)
        self.assertEqual(sorted(p.doc for p in everything), ["d0", "d1", "d2"])
        self.assertEqual(sorted(p.doc for p in dev), ["d0", "d1"])

    def test_every_development_page_carries_the_fold_the_manifest_gives_it(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = self.build(folder, {})
            dev = {p.doc: p.fold for p in pairs.load_pages(folder, ["train"], manifest=manifest)}
            plain = {p.doc: p.fold for p in pairs.load_pages(folder, ["train"])}
        self.assertEqual(dev, {"d0": 3, "d1": 4})
        self.assertEqual(plain, {"d0": None, "d1": None, "d2": None})

    def test_the_runners_read_the_development_sets_and_the_manifest_unless_told_otherwise(self):
        import argparse

        parser = argparse.ArgumentParser()
        pairs.add_data_arguments(parser)
        args = parser.parse_args([])
        self.assertEqual(args.sets, ["train", "val", "test", "dev5en"])
        self.assertIsNone(args.cells)  # left out: the cell of the product, see `reading_cells`
        self.assertEqual(parser.parse_args(["--cells"]).cells, [])
        cells, spec = pairs.reading_cells(args)
        self.assertEqual((cells, spec), ([readings.product_spec().cell], readings.product_spec()))
        self.assertEqual(args.spec, spec)  # what a model saved by this run says it was trained on
        explored = parser.parse_args(["--cells", "L", "Z"])
        self.assertEqual(pairs.reading_cells(explored), (["L", "Z"], None))
        self.assertIsNone(explored.spec)
        self.assertEqual(pairs.reading_cells(parser.parse_args(["--cells", "L", "--cap", "4"]))[1].cap, 4)
        self.assertTrue(args.manifest.name == "split-v5.json" and args.manifest.exists())
        self.assertIsNone(parser.parse_args(["--no-manifest"]).manifest)

    def entries_of(self, manifest, extra):
        data = json.loads(manifest.read_text(encoding="utf-8"))
        data["pages"] += extra
        manifest.write_text(json.dumps(data), encoding="utf-8")

    def test_a_manifest_page_that_was_not_loaded_is_said_and_too_many_of_them_are_an_error(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = self.build(folder, {})
            self.entries_of(manifest, [{"set": "train", "doc_id": "gone", "page": 1, "role": "dev", "fold": "fold0"}])
            with self.assertRaises(ValueError) as caught:  # 1 of the 3 development pages is far over 1 %
                pairs.load_pages(folder, ["train"], manifest=manifest)
            self.assertIn("1 of the 3", str(caught.exception))

    def test_a_manifest_page_of_a_set_that_was_not_asked_for_is_not_counted(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = self.build(folder, {})
            self.entries_of(manifest, [{"set": "val", "doc_id": "other", "page": 1, "role": "dev", "fold": "fold0"}])
            with self.assertNoLogs("bench.guard.pairs", level="WARNING"):
                self.assertEqual(len(pairs.load_pages(folder, ["train"], manifest=manifest)), 2)

    def test_one_missing_page_in_two_hundred_is_a_warning_and_not_an_error(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = self.build(folder, {}, documents=200)
            self.entries_of(manifest, [{"set": "train", "doc_id": "gone", "page": 1, "role": "dev", "fold": "fold0"}])
            with self.assertLogs("bench.guard.pairs", level="WARNING") as logged:
                pages = pairs.load_pages(folder, ["train"], manifest=manifest)
            self.assertEqual(len(pages), 199)  # the last document is set aside
            self.assertIn("1 of the 200", logged.output[0])

    def test_the_readings_of_every_cell_are_on_the_words_in_the_order_of_the_cells(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = self.build(folder, {"d0": ("L", "Z"), "d1": ("Z",)})
            pages = {p.doc: p for p in pairs.load_pages(folder, ["train"], manifest=manifest, cells=["L", "Z"])}
        both = [w.readings for w in pages["d0"].words if w.readings]
        self.assertTrue(both)
        self.assertTrue(all([r.source for r in readings] == ["latin", "deep"] for readings in both))
        only_z = [w.readings for w in pages["d1"].words if w.readings]  # its L is missing: the Z readings stay
        self.assertTrue(only_z)
        self.assertTrue(all([r.source for r in readings] == ["deep"] for readings in only_z))

    def test_without_cells_the_words_carry_no_readings_and_a_page_a_cell_has_not_read_has_none_of_it(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = self.build(folder, {"d0": ("L",)})
            plain = pairs.load_pages(folder, ["train"], manifest=manifest)
            merged = {p.doc: p for p in pairs.load_pages(folder, ["train"], manifest=manifest, cells=["L"])}
        self.assertFalse(any(w.readings for p in plain for w in p.words))
        self.assertTrue(any(w.readings for w in merged["d0"].words))
        self.assertFalse(any(w.readings for w in merged["d1"].words))

    def test_the_labels_and_the_words_do_not_change_with_the_readings(self):
        with tempfile.TemporaryDirectory() as name:
            folder = Path(name)
            manifest = self.build(folder, {"d0": ("L", "Z"), "d1": ("L", "Z")})
            plain = pairs.load_pages(folder, ["train"], manifest=manifest)
            merged = pairs.load_pages(folder, ["train"], manifest=manifest, cells=["L", "Z"])
        for a, b in zip(plain, merged, strict=True):
            self.assertEqual([w.text for w in a.words], [w.text for w in b.words])
            self.assertEqual([x.wrong for x in a.labels], [x.wrong for x in b.labels])


if __name__ == "__main__":
    unittest.main()
