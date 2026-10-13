import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from bench import reader_bakeoff as bakeoff
from bench.corpus import sealed


def record(doc="d1", page=1, truth_text="garbage"):
    """A readable page: one line of five words, the third doubtful, and a second line of two sure words."""
    cols = [(10.0, 60.0), (70.0, 120.0), (130.0, 180.0), (190.0, 240.0), (250.0, 300.0)]
    words = [
        [f"p1w{i}", t, [a, 10.0, b, 30.0], conf, [cc] * len(t), 0, i]
        for i, ((a, b), t, conf, cc) in enumerate(
            zip(
                cols,
                ["la", "casa", "grnde", "de", "Juan"],
                [0.99, 0.99, 0.5, 0.99, 0.99],
                [0.99, 0.99, 0.4, 0.99, 0.99],
                strict=True,
            )
        )
    ]
    words += [["p1w5", "ok", [10.0, 50.0, 60.0, 70.0], 0.99, [0.99, 0.99], 1, 0]]
    return {
        "set": "train",
        "doc_id": doc,
        "page": page,
        "band": "light",
        "truth": {"words": [{"id": "t0", "text": truth_text, "box": [0, 0, 1, 1]}]},
        "ungrounded": [],
        "ocr": {
            "reading": {
                "number": page,
                "width_px": 400,
                "height_px": 100,
                "width_pt": 300.0,
                "height_pt": 75.0,
                "dpi": 100,
                "status": "ocr",
                "reason": None,
                "error": None,
                "mean_conf": 0.9,
                "unaligned_words": 0,
                "words": words,
            }
        },
    }


class ZonesTest(unittest.TestCase):
    def test_G37_the_zone_list_never_looks_at_the_truth(self):
        self.assertEqual(
            bakeoff.page_zones(record(truth_text="grande")), bakeoff.page_zones(record(truth_text="zzzz zz"))
        )

    def test_G37_a_zone_holds_the_doubtful_word_and_a_neighbour_on_each_side(self):
        (zone,) = bakeoff.page_zones(record())
        self.assertEqual(zone.words, ("casa", "grnde", "de"))
        self.assertEqual(zone.suspects, (2,))

    def test_G37_a_doubtful_word_with_a_digit_is_not_measured(self):
        page = record()
        page["ocr"]["reading"]["words"][2][1] = "gr2nde"
        self.assertEqual(bakeoff.page_zones(page), [])

    def test_G37_an_unreadable_page_has_no_zones(self):
        page = record()
        page["ocr"]["reading"]["status"] = "unreadable"
        self.assertEqual(bakeoff.page_zones(page), [])

    def test_G37_the_list_fingerprint_changes_with_any_zone(self):
        zones = bakeoff.page_zones(record())
        self.assertEqual(bakeoff.list_fingerprint(zones), bakeoff.list_fingerprint(bakeoff.page_zones(record())))
        self.assertNotEqual(bakeoff.list_fingerprint(zones), bakeoff.list_fingerprint(zones[:0]))

    def test_G37_the_crop_is_the_zone_slice_and_a_tilt_of_zero_does_not_move_pixels(self):
        image = np.arange(100 * 400 * 3, dtype=np.uint8).reshape(100, 400, 3)
        crop = bakeoff.cut(image, (200.0, 50.0, 100.0, 20.0, 0.0))
        self.assertEqual(crop.shape, (20, 100, 3))
        self.assertTrue(np.array_equal(crop, image[40:60, 150:250]))

    def test_G37_the_page_tilt_is_the_median_slope_of_the_lines(self):
        page = record()
        reading = page["ocr"]["reading"]["words"]
        for k, row in enumerate(reading[:5]):
            row[2] = [row[2][0], 10.0 + 3 * k, row[2][2], 30.0 + 3 * k]
        from extract.engine.cache import reading_from_dict

        words = reading_from_dict(page["ocr"]["reading"]).words
        self.assertAlmostEqual(bakeoff.page_angle(words), float(np.degrees(np.arctan(0.05))), places=1)


class DevelopmentTest(unittest.TestCase):
    def manifest(self, set_name):
        folder = tempfile.mkdtemp()
        path = Path(folder) / "m.json"
        path.write_text(json.dumps({"pages": [{"set": set_name, "doc_id": "d1", "page": 1}]}))
        return path

    def test_G37_a_train_page_passes(self):
        bakeoff.assert_development([(Path("x"), record())], [self.manifest("train")])

    def test_G37_a_page_listed_as_anything_but_train_is_refused(self):
        with self.assertRaises(ValueError):
            bakeoff.assert_development([(Path("x"), record())], [self.manifest("val")])

    def test_G37_a_page_that_is_listed_nowhere_is_refused(self):
        with self.assertRaises(ValueError):
            bakeoff.assert_development([(Path("x"), record(doc="other"))], [self.manifest("train")])

    def test_G37_a_sealed_set_is_refused(self):
        """The bake-off asks the sealed reader about every page's set before it opens anything."""
        page = record()
        with mock.patch.object(sealed, "refuse_sealed_set", side_effect=sealed.SealedTestError("sealed")) as refuse:
            with self.assertRaises(sealed.SealedTestError):
                bakeoff.assert_development([(Path("x"), page)], [self.manifest("train")])
        refuse.assert_called_once_with(page["set"])

    def test_G37_the_chosen_pages_cover_the_bands_and_repeat_a_document_last(self):
        records = []
        for doc in range(6):
            for page in range(2):
                item = record(doc=f"d{doc}", page=page + 1)
                item["band"] = ("light", "medium", "heavy")[doc % 3]
                records.append((Path(f"{doc}-{page}.json"), item))
        chosen = bakeoff.choose_pages(records, 1, count=6)
        self.assertEqual(len({r["doc_id"] for _, r in chosen}), 6)
        self.assertEqual({r["band"] for _, r in chosen}, {"light", "medium", "heavy"})
        self.assertEqual([p for p, _ in chosen], [p for p, _ in bakeoff.choose_pages(records, 1, count=6)])


def labels_for(*rows):
    """{(doc, page): {index: (truth, kind, beyond pool)}} for a zone of three words whose middle one is index 2."""
    return {("d1", 1): {index: info for index, info in rows}}


class ScoreTest(unittest.TestCase):
    zone = bakeoff.Zone(
        "d1", 1, "light", 1, 3, (2,), (0.0, 0.0, 1.0, 1.0, 0.0), ("casa", "grnde", "de"), ("a", "b", "c")
    )

    def test_G37_a_wrong_word_read_right_is_a_gain_counted_in_net100(self):
        scored = bakeoff.score([self.zone], labels_for((2, ("grande", "letters", False))), ["casa grande de"])
        self.assertEqual((scored["wrong_words"], scored["fixed"], scored["harmed"]), (1, 1, 0))
        self.assertEqual(scored["net100"], 100.0)
        self.assertEqual(scored["change_precision"], 1.0)

    def test_G37_a_right_word_read_wrong_is_a_harm_that_net100_pays_for(self):
        zone = bakeoff.Zone(
            "d1", 1, "light", 1, 3, (2, 3), (0.0, 0.0, 1.0, 1.0, 0.0), ("casa", "grnde", "de"), ("a", "b", "c")
        )
        info = {2: ("grande", "letters", False), 3: ("de", "right", False)}
        scored = bakeoff.score([zone], {("d1", 1): info}, ["casa grande dx"])
        self.assertEqual((scored["fixed"], scored["harmed"]), (1, 1))
        self.assertEqual(scored["net100"], 0.0)
        self.assertEqual(scored["reader_harm"], 1.0)
        self.assertEqual(scored["change_precision"], 0.5)

    def test_G37_a_word_the_reader_leaves_alone_is_neither(self):
        scored = bakeoff.score([self.zone], labels_for((2, ("grande", "letters", False))), ["casa grnde de"])
        self.assertEqual((scored["fixed"], scored["harmed"], scored["changed"]), (0, 0, 0))

    def test_G37_the_written_form_is_compared_after_the_normalisation(self):
        scored = bakeoff.score([self.zone], labels_for((2, ("it's", "punctuation", False))), ["casa it’s de"])
        self.assertEqual(scored["fixed"], 1)

    def test_G37_an_unaligned_answer_changes_nothing_and_is_counted(self):
        scored = bakeoff.score([self.zone], labels_for((2, ("grande", "letters", False))), ["z"])
        self.assertEqual((scored["fixed"], scored["zones_unaligned"]), (0, 1))

    def test_G37_a_cut_word_is_not_a_one_to_one_reading(self):
        scored = bakeoff.score([self.zone], labels_for((2, ("grande", "letters", False))), ["casa gran de de"])
        self.assertEqual(scored["fixed"], 0)

    def test_G37_recall_beyond_pool_counts_only_what_neither_pool_reaches(self):
        scored = bakeoff.score([self.zone], labels_for((2, ("grande", "letters", True))), ["casa grande de"])
        self.assertEqual((scored["wrong_beyond_pool"], scored["recall_beyond_pool"]), (1, 1.0))

    def test_G37_targets_of_other_kinds_are_not_measured(self):
        scored = bakeoff.score([self.zone], labels_for(), ["casa grande de"])
        self.assertEqual((scored["wrong_words"], scored["right_words"], scored["net100"]), (0, 0, None))

    def test_G37_reproduction_is_the_share_of_zones_given_back_exactly(self):
        self.assertEqual(bakeoff.reproduction([self.zone, self.zone], ["casa grnde de", "casa grande de"]), 0.5)


class RegistryTest(unittest.TestCase):
    def test_G37_every_candidate_has_an_origin_a_licence_and_a_sha256(self):
        for name, spec in bakeoff.CANDIDATES.items():
            self.assertTrue(spec["url"].startswith("https://"), name)
            self.assertEqual(len(spec["sha256"]), 64, name)
            self.assertTrue(spec["licence"], name)
            self.assertIn(spec["role"], ("control", "candidate"))

    def test_G37_the_french_vocabulary_of_onnxtr_cannot_write_the_spanish_accents(self):
        known = bakeoff._english() + bakeoff.FRENCH_EXTRA
        missing = "".join(c for c in bakeoff.SPANISH_CHARACTERS if c not in known)
        self.assertEqual(set(missing), set("áíóúñÁÍÓÚÑ¿¡"))

    def test_G37_a_generative_answer_is_reduced_to_one_line_of_words(self):
        self.assertEqual(bakeoff.reader_parse(" la\n casa \t grande "), "la casa grande")


if __name__ == "__main__":
    unittest.main()


RESULT = Path(__file__).resolve().parents[1] / "bench" / "results" / "reader-bakeoff.json"


class ResultFileTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads(RESULT.read_text(encoding="utf-8"))

    def test_G37_every_admissible_model_has_the_metrics_the_gate_asks_for(self):
        for name, model in self.result["models"].items():
            for key in ("url", "licence", "sha256", "size_bytes", "emits", "hand_test"):
                self.assertIn(key, model, name)
            if model["admissible"]:
                for key in ("net100", "reader_recall", "recall_beyond_pool", "change_precision", "reader_harm"):
                    self.assertIn(key, model["full"], name)
                for key in ("seconds_4_threads", "seconds_1_thread"):
                    self.assertEqual(set(model[key]), {"median", "p95"}, name)
                self.assertGreater(model["peak_rss_mb"], 0)

    def test_G37_every_model_read_the_same_crop_list(self):
        sizes = {m["full"]["zones"] for m in self.result["models"].values() if m["admissible"]}
        self.assertEqual(sizes, {self.result["crops"]})
        self.assertEqual(len(self.result["crop_fingerprint"]), 64)

    @pytest.mark.corpus
    def test_G37_the_pages_are_development_train_pages_only(self):
        if "EXTRACT_BENCH_DATA" not in bakeoff.os.environ:
            self.skipTest("EXTRACT_BENCH_DATA (the development corpus) is not set")
        data = Path(bakeoff.os.environ["EXTRACT_BENCH_DATA"])
        manifests = [
            Path(__file__).resolve().parents[1] / "bench" / "corpus" / "split.json",
            data / "invoices" / "pages.json",
        ]
        listed = {}
        for manifest in manifests:
            for entry in json.loads(manifest.read_text())["pages"]:
                listed[f"{entry['doc_id']}/{entry['page']}"] = entry["set"]
        self.assertEqual({listed.get(page) for page in self.result["pages"]}, {"train"})

    def test_G37_deep_is_the_admissible_candidate_with_the_highest_net100_and_models_that_cannot_write_spanish_are_out(
        self,
    ):
        candidates = {n: m for n, m in self.result["models"].items() if m["role"] == "candidate" and m["admissible"]}
        best = max(candidates, key=lambda n: candidates[n]["full"]["net100"])
        self.assertEqual(self.result["deep"], best)
        self.assertEqual(self.result["ranking_by_net100"][0], best)
        self.assertTrue(self.result["vocabulary_excluded"])
        for entry in self.result["vocabulary_excluded"].values():
            self.assertIn("ñ", entry["missing"])
