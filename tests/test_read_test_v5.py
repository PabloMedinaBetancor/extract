import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from bench import read_test_v5 as reader
from bench.corpus.labels import Label
from bench.corpus.sidecar_inputs import build, generated_truths
from extract.types import Fix, Word

ENGINE = "engine|v1"


def tally(fixed=0, broken=0, still=0, wrong=0, words=100, characters=500, distance=0):
    return {
        "weights": {"fixed": fixed, "broken": broken, "changed_still_wrong": still, "left_wrong": wrong},
        "components": {},
        "truth_words": words,
        "truth_characters": characters,
        "distance_before": 0,
        "distance_after": distance,
    }


def page(doc, language="es", slice_="pages", **series):
    tallies = {"none": tally(wrong=10)}
    for key in ("default|safe", "default|more", "thorough|safe", "thorough|more"):
        tallies[key] = series.get(key, tally(wrong=3))
    return {"doc": doc, "page": 1, "language": language, "slice": slice_, "band": "light", "tallies": tallies}


class StatisticsTest(unittest.TestCase):
    def test_N10_the_interval_is_a_ratio_of_sums_over_documents_and_is_reproducible(self):
        num, den = np.array([1.0, 3.0, 2.0]), np.array([10.0, 10.0, 20.0])
        first = reader.interval(num, den, 300)
        self.assertAlmostEqual(first["value"], 6 / 40)
        self.assertEqual(first, reader.interval(num, den, 300))
        self.assertLessEqual(first["low"], first["value"] + 1e-9)
        self.assertGreaterEqual(first["high"], first["value"] - 1e-9)
        self.assertEqual(first["documents"], 3)

    def test_N10_no_documents_gives_no_interval(self):
        self.assertIsNone(reader.interval(np.array([]), np.array([]))["value"])

    def test_N10_the_bound_is_the_rule_of_three_for_zero_failures_and_wilson_otherwise(self):
        self.assertEqual(reader.upper_bound_95(0, 60), 3 / 60)
        self.assertGreater(reader.upper_bound_95(2, 60), 2 / 60)
        self.assertIsNone(reader.upper_bound_95(0, 0))


class TableTest(unittest.TestCase):
    def test_M3_d_wer_of_a_mode_is_broken_minus_fixed_over_the_truth_words_and_none_is_the_raw_text(self):
        pages = [page(f"d{i}", **{"default|more": tally(fixed=4, broken=1, wrong=10)}) for i in range(5)]
        block = reader.table_for(pages, 200)
        more = block["models"]["default"]["more"]
        self.assertAlmostEqual(more["d_wer"]["value"], -3 / 100)
        self.assertEqual(more["fixed"], 20)
        self.assertEqual(more["broken"], 5)
        self.assertAlmostEqual(more["net_fixed_per_100_wrong"]["value"], 30.0)
        self.assertAlmostEqual(block["none"]["wer_as_read"]["value"], 0.10)
        self.assertEqual(block["models"]["default"]["safe"]["d_wer"]["value"], 0.0)

    def test_G19_the_paired_difference_is_thorough_minus_default_over_the_same_documents(self):
        pages = [
            page(
                f"d{i}",
                **{
                    "default|more": tally(fixed=2, broken=1, wrong=10),
                    "thorough|more": tally(fixed=5, broken=1, wrong=10),
                },
            )
            for i in range(6)
        ]
        paired = reader.table_for(pages, 200)["paired_thorough_minus_default"]["more"]
        self.assertAlmostEqual(paired["d_wer_difference"]["value"], -0.03)
        self.assertAlmostEqual(paired["net_fixed_per_100_wrong_difference"]["value"], 30.0)
        self.assertLess(paired["d_wer_difference"]["high"], 0)
        self.assertEqual(
            reader.table_for(pages, 200)["paired_thorough_minus_default"]["safe"]["d_wer_difference"]["value"], 0.0
        )

    def test_G44_an_unreadable_page_is_counted_apart_and_never_in_the_tallies(self):
        pages = [page("a"), {"doc": "b", "page": 1, "language": "es", "slice": "pages", "unreadable": True}]
        block = reader.table_for(pages, 100)
        self.assertEqual((block["pages"], block["unreadable_pages"], block["documents"]), (2, 1, 1))

    def test_G44_a_table_of_unreadable_pages_only_has_no_numbers_and_does_not_fail(self):
        block = reader.table_for([{"doc": "b", "page": 1, "language": "en", "slice": "pages", "unreadable": True}], 100)
        self.assertEqual(block["models"], {})


class PagesTest(unittest.TestCase):
    def test_G42_the_language_of_a_page_is_its_sidecars_for_an_invoice_and_its_sets_otherwise(self):
        with tempfile.TemporaryDirectory() as folder:
            data = Path(folder)
            (data / "invoices" / "test5").mkdir(parents=True)
            (data / "invoices" / "test5" / "inv-1.json").write_text(json.dumps({"language": "en"}), encoding="utf-8")
            invoice = {"set": "test5", "source": "invoice", "doc_id": "inv-1"}
            self.assertEqual(reader.language_of(invoice, data), "en")
            self.assertEqual(reader.language_of({"set": "test5en", "source": "catalog"}, data), "en")
            self.assertEqual(reader.language_of({"set": "test5", "source": "catalog"}, data), "es")
            self.assertEqual(reader.language_of({"set": "unseen5", "source": "catalog"}, data), "es")

    def test_M4_the_slice_of_a_page_is_unseen_invoices_or_pages(self):
        self.assertEqual(reader.slice_of({"set": "unseen5", "source": "catalog"}), "unseen")
        self.assertEqual(reader.slice_of({"set": "test5", "source": "invoice"}), "invoices")
        self.assertEqual(reader.slice_of({"set": "test5en", "source": "catalog"}), "pages")

    def test_M2_the_edits_are_the_applied_fixes_over_the_word_indexes_of_the_first_reading(self):
        words = [Word(f"w{i}", 1, t, (i, 0, i + 1, 1)) for i, t in enumerate(["ab", "cd", "ef", "gh"])]
        original = mock.Mock(words=tuple(words))
        fix = lambda op, ids, text: Fix(op, "x", text, ids, "s", ("s",), 0.9, 0.1, None, ("safe",), ())  # noqa: E731
        applied = list(words)
        applied[0] = dataclasses.replace(
            words[0], text="abcd", status="fixed", corrected_by="guard", fix=fix("join", ("w0", "w1"), "abcd")
        )
        applied[1] = dataclasses.replace(words[1], text="", status="absorbed", absorbed_by="w0", corrected_by="guard")
        applied[2] = dataclasses.replace(
            words[2], text="", status="absorbed", corrected_by="guard", fix=fix("drop", ("w2",), "")
        )
        applied[3] = dataclasses.replace(
            words[3], text="gx", status="fixed", corrected_by="guard", fix=fix("text", ("w3",), "gx")
        )
        edits = reader.edits_of(original, mock.Mock(words=tuple(applied)))
        self.assertEqual(
            [(e.op, e.words, e.text) for e in edits],
            [("join", (0, 1), "abcd"), ("drop", (2,), ""), ("text", (3,), "gx")],
        )


def label(text, wrong, kind="pair"):
    return Label("x", wrong, kind, None, text, ())


class ArithmeticSuggestionTest(unittest.TestCase):
    def broken_invoice(self):
        """An invoice with one digit of the total changed so that an arithmetic rule gives a verified suggestion."""
        for truth in (t for t in generated_truths() if len(t["vat"]) == 1):
            inputs = build(truth)
            total = truth["total"]
            word = next((w for w in inputs.words if w.text == total), None)
            if word is None:
                continue
            for position, char in enumerate(word.text):
                if not char.isdigit():
                    continue
                other = str((int(char) + 1) % 10)
                broken = inputs.with_text(word.id, word.text[:position] + other + word.text[position + 1 :])
                found = reader.suggestions_of(
                    broken.words, broken.fields, truth["language"], [label(w.text, False) for w in inputs.words]
                )
                if found:
                    return truth, inputs, broken, word
        self.fail("no invoice of the generator gave an arithmetic suggestion")

    def test_N10_a_wrong_digit_of_an_invoice_total_leaves_an_arithmetic_suggestion_that_is_the_true_text(self):
        truth, inputs, broken, word = self.broken_invoice()
        index = [w.id for w in inputs.words].index(word.id)
        labels = [label(w.text, w.id == word.id) for w in inputs.words]
        found = reader.suggestions_of(broken.words, broken.fields, truth["language"], labels)
        self.assertTrue(found)
        mine = [f for f in found if f["verdict"] == "true"]
        self.assertTrue(mine, found)
        self.assertIn(mine[0]["rule"], set(reader.ARITHMETIC.values()))
        # the same suggestion against a different truth is false, and with no one-to-one truth word it is unaligned
        wrong_truth = list(labels)
        wrong_truth[index] = label("0", True)
        self.assertIn(
            "false",
            [f["verdict"] for f in reader.suggestions_of(broken.words, broken.fields, truth["language"], wrong_truth)],
        )
        unaligned = list(labels)
        unaligned[index] = label(None, None, "invented")
        self.assertNotIn(
            "true",
            [f["verdict"] for f in reader.suggestions_of(broken.words, broken.fields, truth["language"], unaligned)],
        )

    def test_N10_only_verified_arithmetic_suggestions_count_not_check_digit_ones_nor_unverified_ones(self):
        from extract.types import DigitSuggestion, NumberResult

        words = [Word("w0", 1, "1234", (0, 0, 8, 10)), Word("w1", 1, "5678", (10, 0, 18, 10))]
        labels = [label("7234", True), label("5678", False)]

        def result(word_id, verified, rule):
            return NumberResult(
                (word_id,), "x", None, "amount", "mismatch", (), DigitSuggestion(0, "1", "7", verified, rule)
            )

        results = [
            result("w0", False, "checks.rule.total"),  # not verified
            result("w0", True, "checks.rule.dni"),  # a check digit: applied alone in safe, not a suggestion
            result("w0", True, "checks.rule.total"),
            result("w1", True, "checks.rule.vat_amount"),  # the word was right: the suggestion is false
        ]
        with mock.patch.object(reader.checks, "run", return_value=results):
            found = reader.suggestions_of(words, None, "es", labels)
        self.assertEqual(found, [{"rule": "total", "verdict": "true"}, {"rule": "vat_amount", "verdict": "false"}])

    def test_N10_with_no_suggestion_left_the_report_says_zero_of_them_and_the_invoices_it_looked_at(self):
        report = reader.n10_report([{**page("i1", slice_="invoices"), "n10": {"default": [], "thorough": []}}])
        for name in reader.MODELS:
            self.assertEqual(report[name]["invoice_pages"], 1)
            self.assertEqual(report[name]["by_rule"]["all"]["suggestions"], 0)
            self.assertIsNone(report[name]["by_rule"]["all"]["true_fraction_of_decided"])

    def test_N10_the_report_counts_by_rule_and_in_all_and_says_how_many_are_the_true_value(self):
        items = [
            {"rule": "total", "verdict": "true"},
            {"rule": "total", "verdict": "false"},
            {"rule": "lines_sum", "verdict": "true"},
        ]
        pages = [
            {**page("i1", slice_="invoices"), "n10": {"default": items, "thorough": items[:1]}},
            {**page("i2", slice_="invoices"), "n10": {"default": [], "thorough": []}},
        ]
        report = reader.n10_report(pages)
        block = report["default"]["by_rule"]
        self.assertEqual(block["all"]["suggestions"], 3)
        self.assertEqual(block["total"]["true"], 1)
        self.assertAlmostEqual(block["all"]["true_fraction_of_decided"], 2 / 3)
        self.assertEqual(report["default"]["invoice_pages"], 2)
        self.assertEqual(report["thorough"]["by_rule"]["all"]["false_upper_bound_95"], 3 / 1)


class SeriesTest(unittest.TestCase):
    """series_report on three documents with numbers worked out by hand."""

    def pages(self):
        # (fixed, broken, still, wrong as read, truth words, truth characters, distance change)
        rows = [(3, 1, 1, 6, 100, 400, -4), (0, 2, 0, 4, 50, 300, 3), (5, 0, 2, 9, 150, 900, -8)]
        out = []
        for i, (f, b, st, w, words, characters, distance) in enumerate(rows):
            out.append(
                page(
                    f"d{i}",
                    **{
                        "default|more": tally(f, b, st, 99, words, characters, distance),
                        "default|safe": tally(0, 0, 0, 77, words, characters, 0),
                    },
                )
            )
            out[-1]["tallies"]["none"] = tally(0, 0, 0, w, words, characters, 0)
        return out

    def test_M3_every_figure_of_a_series_is_the_ratio_of_sums_with_the_right_denominator(self):
        report = reader.series_report(self.pages(), "default|more", 400)
        self.assertEqual((report["fixed"], report["broken"], report["changed_still_wrong"]), (8, 3, 3))
        self.assertEqual(report["applied"], 14)  # fixed + broken + changed still wrong
        self.assertEqual(report["wrong_as_read"], 19)  # from the series `none`, not from this one
        self.assertAlmostEqual(report["precision"]["value"], 8 / 14)
        self.assertAlmostEqual(report["d_wer"]["value"], (3 - 8) / 300)
        self.assertAlmostEqual(report["d_cer"]["value"], -9 / 1600)  # characters, not words
        self.assertAlmostEqual(report["net_fixed_per_100_wrong"]["value"], 100 * (8 - 3) / 19)
        self.assertAlmostEqual(report["wer_as_read"]["value"], 19 / 300)
        self.assertEqual(report["documents"], 3)

    def test_M3_the_interval_is_the_95_percent_percentile_of_resampled_documents_with_the_fixed_seed(self):
        report = reader.series_report(self.pages(), "default|more", 400)
        num = np.array([-2.0, 2.0, -5.0])  # broken - fixed by document
        den = np.array([100.0, 50.0, 150.0])
        chosen = np.random.default_rng(1).integers(0, 3, size=(400, 3))
        stats = num[chosen].sum(axis=1) / den[chosen].sum(axis=1)
        low, high = np.percentile(stats, [2.5, 97.5])
        self.assertAlmostEqual(report["d_wer"]["low"], low)
        self.assertAlmostEqual(report["d_wer"]["high"], high)

    def test_G19_with_the_same_series_the_paired_interval_is_exactly_zero(self):
        pages = [page(f"d{i}", **{"default|more": tally(fixed=i + 1, wrong=10)}) for i in range(6)]
        for p in pages:
            p["tallies"]["thorough|more"] = p["tallies"]["default|more"]
        paired = reader.table_for(pages, 300)["paired_thorough_minus_default"]["more"]
        for name in ("d_wer_difference", "net_fixed_per_100_wrong_difference"):
            self.assertEqual((paired[name]["value"], paired[name]["low"], paired[name]["high"]), (0.0, 0.0, 0.0))

    def test_G19_the_paired_interval_shares_its_resamples_so_a_constant_gain_has_no_width(self):
        pages = [
            page(
                f"d{i}",
                **{
                    "default|more": tally(fixed=i + 1, wrong=10),
                    "thorough|more": tally(fixed=i + 3, wrong=10),  # two more fixed words in every document
                },
            )
            for i in range(8)
        ]
        paired = reader.table_for(pages, 300)["paired_thorough_minus_default"]["more"]["d_wer_difference"]
        self.assertAlmostEqual(paired["value"], -0.02)
        self.assertAlmostEqual(paired["low"], paired["value"])
        self.assertAlmostEqual(paired["high"], paired["value"])


class FreezeTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        for name in (
            "models/guard.model",
            "models/guard-thorough.model",
            "src/extract/fixes.py",
            "src/extract/pipeline.py",
            "src/extract/guard/data/dict_es.tsv.gz",
            "bench/read_test_v5.py",
            "bench/guard/pairs.py",
        ):
            (self.root / name).parent.mkdir(parents=True, exist_ok=True)
            (self.root / name).write_bytes(name.encode())

    def test_M8_a_freeze_lists_both_models_the_product_and_the_bench_code_and_a_check_passes_until_one_changes(self):
        self.assertEqual(reader.freeze(self.root, ENGINE), 7)
        self.assertEqual(reader.verify(self.root), ENGINE)
        for name in ("models/guard-thorough.model", "src/extract/pipeline.py", "bench/guard/pairs.py"):
            original = (self.root / name).read_bytes()
            (self.root / name).write_bytes(b"changed")
            with self.assertRaises(reader.FrozenError, msg=name):
                reader.verify(self.root)
            (self.root / name).write_bytes(original)
        self.assertEqual(reader.verify(self.root), ENGINE)

    def test_M8_a_new_python_file_under_src_or_the_bench_breaks_the_check(self):
        reader.freeze(self.root, ENGINE)
        for name in (
            "src/extract/engine/reread.py",
            "src/extract/guard/extra.py",
            "bench/guard/dataset.py",
            "bench/corpus/x.py",
        ):
            target = self.root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("x = 1\n", encoding="utf-8")
            with self.assertRaises(reader.FrozenError, msg=name):
                reader.verify(self.root)
            target.unlink()
        self.assertEqual(reader.verify(self.root), ENGINE)

    def test_M8_a_missing_listed_file_breaks_the_check(self):
        reader.freeze(self.root, ENGINE)
        (self.root / "models" / "guard.model").unlink()
        with self.assertRaises(reader.FrozenError):
            reader.verify(self.root)

    def test_M8_the_frozen_set_is_written_once_and_a_read_without_it_is_refused(self):
        with self.assertRaises(reader.FrozenError):
            reader.verify(self.root)
        reader.freeze(self.root, ENGINE)
        with self.assertRaises(reader.FrozenError):
            reader.freeze(self.root, ENGINE)

    def test_M8_the_read_opens_nothing_sealed_when_the_freeze_check_fails(self):
        with mock.patch.object(reader.sealed, "read_test_pages", side_effect=AssertionError("opened")):
            with self.assertRaises(reader.FrozenError):
                reader.sealed_read(self.root, self.root, (Path("a"), Path("b")), 10, {"es": 1, "en": 1})


class SealedTree:
    """A small sealed tree: two tests (`es`, `en`) of two catalogue documents of two pages and, for `es`, one invoice,
    with their cell files, manifest, freeze records, frozen files and engine, built the way the real ones are."""

    ENGINE = "engine|v1"

    def __init__(self, folder: Path):
        self.root = folder
        self.data = folder / "data"
        self.sources = []
        manifest = {}
        for language, sets, key in (("es", ("t5x",), "t5"), ("en", ("t5xen",), "t5en")):
            docs = []
            for d in range(2):
                docs.append({"doc_id": f"{language}{d}", "position": d, "slice": "pages", "pages": [1, 2]})
                self.add_doc(sets[0], f"{language}{d}", 2, "catalog")
            if language == "es":
                docs.append({"doc_id": "inv-1", "position": 2, "slice": "invoices", "pages": [1]})
                self.add_doc(sets[0], "inv-1", 1, "invoice")
            manifest[key] = {"documents": docs}
            reads = Path(f"bench/results/test_reads_{language}.json")
            freeze = Path(f"bench/results/test-freeze-{language}.json")
            self.sources.append(reader.Source(language, sets, reads, key, freeze))
        (folder / "bench" / "corpus").mkdir(parents=True, exist_ok=True)
        (folder / reader.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
        for name in ("models/guard.model", "models/guard-thorough.model", "src/extract/fixes.py"):
            (folder / name).parent.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(name.encode())
        for source in self.sources:
            (folder / source.reads).parent.mkdir(parents=True, exist_ok=True)
            reader.sealed.freeze_record(self.data, folder / source.freeze, "test", folder / source.reads, source.sets)
            (folder / source.reads).unlink()  # the hash-only reads of the freeze are not part of the story
        reader.freeze(folder, self.ENGINE)

    def page_path(self, set_name, doc, n):
        return self.data / "pages" / set_name / doc / f"{n}.json"

    def add_doc(self, set_name, doc, pages, source):
        for n in range(1, pages + 1):
            path = self.page_path(set_name, doc, n)
            path.parent.mkdir(parents=True, exist_ok=True)
            record = {
                "set": set_name,
                "doc_id": doc,
                "page": n,
                "source": source,
                "truth": {"words": []},
                "ocr": {"engine": self.ENGINE + "|u:0.9", "reading": {"words": [["w"]]}},
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            path.with_suffix(".png").write_bytes(f"{set_name}{doc}{n}".encode())
            for cell in reader.cell_files(path):
                cell.write_text("{}", encoding="utf-8")

    def lines(self, source):
        path = self.root / source.reads
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else []

    def metric_lines(self, source):
        return [line for line in self.lines(source) if line["purpose"] == "metrics"]

    def run(self, analyse=None, documents=None):
        analyse = analyse or (lambda *a, **k: {"pages_without_cells": 0, "pages": 7})
        with mock.patch.object(reader, "analyse", side_effect=analyse), mock.patch.object(reader, "load_guards"):
            return reader.sealed_read(
                self.root, self.data, (Path("a"), Path("b")), 10, documents or {"es": 2, "en": 2}, tuple(self.sources)
            )


class SealedReadTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tree = SealedTree(Path(folder.name))

    def metrics_total(self):
        return sum(len(self.tree.metric_lines(s)) for s in self.tree.sources)

    def test_M8_the_happy_path_registers_one_metrics_read_per_test_after_the_numbers_and_prints_what_it_wrote(self):
        seen = {}

        def analyse(records, data, guards, resamples):
            seen["metrics_before"] = self.metrics_total()
            seen["tmp_written"] = False
            return {"pages_without_cells": 0, "pages": len(records)}

        text = self.tree.run(analyse)
        self.assertEqual(seen["metrics_before"], 0)  # nothing is registered while the numbers are made
        for source in self.tree.sources:
            self.assertEqual(len(self.tree.metric_lines(source)), 1)
        written = (self.tree.root / reader.OUT).read_text(encoding="utf-8")
        self.assertEqual(text, written)
        self.assertEqual(json.loads(text)["read_pages"], {"es": 5, "en": 4})
        self.assertFalse((self.tree.root / reader.OUT).with_name(reader.OUT.name + ".tmp").exists())

    def test_M8_a_failing_analysis_spends_no_read_and_writes_nothing(self):
        def boom(*args, **kwargs):
            raise RuntimeError("the analysis failed")

        with self.assertRaises(RuntimeError):
            self.tree.run(boom)
        self.assertEqual(self.metrics_total(), 0)
        self.assertFalse((self.tree.root / reader.OUT).exists())

    def test_M8_a_failing_write_of_the_file_spends_no_read(self):
        with mock.patch.object(Path, "write_text", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.tree.run()
        self.assertEqual(self.metrics_total(), 0)

    def test_M8_a_failing_registration_leaves_no_results_file(self):
        real = reader.sealed.read_test_pages

        def fail_on_metrics(data, purpose, *args, **kwargs):
            if purpose == "metrics":
                raise OSError("cannot write the register")
            return real(data, purpose, *args, **kwargs)

        with mock.patch.object(reader.sealed, "read_test_pages", side_effect=fail_on_metrics):
            with self.assertRaises(OSError):
                self.tree.run()
        self.assertFalse((self.tree.root / reader.OUT).exists())

    def test_M8_a_second_read_for_metrics_is_refused_before_anything_is_opened(self):
        self.tree.run()
        with mock.patch.object(reader.sealed, "read_test_pages", side_effect=AssertionError("opened")):
            with self.assertRaises(reader.FrozenError):
                self.tree.run()

    def test_M8_a_page_changed_after_the_freeze_is_refused(self):
        self.tree.page_path("t5x", "es0", 1).with_suffix(".png").write_bytes(b"another image")
        with self.assertRaisesRegex(reader.FrozenError, "differ from the freeze record"):
            self.tree.run()
        self.assertEqual(self.metrics_total(), 0)
        path = self.tree.page_path("t5xen", "en0", 1)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["truth"] = {"words": [{"id": "x"}]}
        path.write_text(json.dumps(record), encoding="utf-8")
        self.tree.page_path("t5x", "es0", 1).with_suffix(".png").write_bytes(b"t5xes01")  # the image as frozen again
        with self.assertRaisesRegex(reader.FrozenError, "differ from the freeze record"):
            self.tree.run()  # but the truth of an English page changed

    def unread(self, set_name, doc, pages=(1, 2)):
        for n in pages:  # the OCR reading is not part of what the freeze record hashes
            path = self.tree.page_path(set_name, doc, n)
            record = json.loads(path.read_text(encoding="utf-8"))
            record["ocr"] = {}
            path.write_text(json.dumps(record), encoding="utf-8")

    def test_M7_the_documents_read_must_be_exactly_the_n_first_of_the_list(self):
        with self.assertRaisesRegex(reader.FrozenError, "first 1"):
            self.tree.run(documents={"es": 1, "en": 2})  # both documents were read, n says one
        with self.assertRaisesRegex(reader.FrozenError, "asked"):
            self.tree.run(documents={"es": 2, "en": 3})  # more than the list holds
        self.unread("t5x", "es1")
        with self.assertRaisesRegex(reader.FrozenError, "first 2"):
            self.tree.run()  # n says two, one was read
        self.assertEqual(self.metrics_total(), 0)

    def test_M7_a_document_skipped_in_the_middle_or_read_in_part_or_an_invoice_missing_is_refused(self):
        self.unread("t5xen", "en0")
        with self.assertRaisesRegex(reader.FrozenError, "first 1"):
            self.tree.run(documents={"es": 2, "en": 1})  # the second one was read, not the first
        self.setUp()
        self.unread("t5x", "es0", pages=(2,))
        with self.assertRaisesRegex(reader.FrozenError, "in part"):
            self.tree.run()
        self.setUp()
        self.unread("t5x", "inv-1", pages=(1,))
        with self.assertRaisesRegex(reader.FrozenError, "invoices"):
            self.tree.run()

    def test_M7_a_page_without_its_cell_files_is_refused(self):
        reader.cell_files(self.tree.page_path("t5xen", "en1", 2))[1].unlink()
        with self.assertRaisesRegex(reader.FrozenError, "cells"):
            self.tree.run()
        self.assertEqual(self.metrics_total(), 0)

    def test_M7_an_unreadable_page_has_no_cell_files_and_does_not_stop_the_read_but_an_ocr_page_without_them_does(self):
        path = self.tree.page_path("t5xen", "en1", 2)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["ocr"]["reading"] = {"words": [["w"]], "status": "unreadable"}
        path.write_text(json.dumps(record), encoding="utf-8")
        for cell in reader.cell_files(path):
            cell.unlink()
        self.assertTrue(self.tree.run())  # pass 2 does not run on an unreadable page
        self.setUp()
        path = self.tree.page_path("t5xen", "en1", 2)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["ocr"]["reading"] = {"words": [["w"]], "status": "ocr"}
        path.write_text(json.dumps(record), encoding="utf-8")
        reader.cell_files(path)[0].unlink()
        with self.assertRaisesRegex(reader.FrozenError, "cells"):
            self.tree.run()

    def test_M8_pages_judged_without_readings_refuse_the_read_and_spend_none(self):
        with self.assertRaisesRegex(reader.FrozenError, "no readings"):
            self.tree.run(lambda *a, **k: {"pages_without_cells": 3, "pages": 9})
        self.assertEqual(self.metrics_total(), 0)
        self.assertFalse((self.tree.root / reader.OUT).exists())

    def test_M8_pages_read_by_another_engine_are_refused(self):
        path = self.tree.page_path("t5x", "es0", 1)
        record = json.loads(path.read_text(encoding="utf-8"))
        record["ocr"]["engine"] = "other-engine|v2"
        path.write_text(json.dumps(record), encoding="utf-8")
        # an OCR reading is not part of what the freeze record hashes, so the images and the truth still agree
        with self.assertRaisesRegex(reader.FrozenError, "frozen engine"):
            self.tree.run()
        self.assertEqual(self.metrics_total(), 0)

    def test_M8_a_frozen_file_changed_after_the_freeze_is_refused_before_anything_sealed_is_opened(self):
        (self.tree.root / "models" / "guard-thorough.model").write_bytes(b"retrained")
        with mock.patch.object(reader.sealed, "read_test_pages", side_effect=AssertionError("opened")):
            with self.assertRaises(reader.FrozenError):
                self.tree.run()

    def test_M8_the_command_prints_what_the_read_returns_and_needs_the_n_of_each_test(self):
        import contextlib
        import io

        with mock.patch.object(Path, "cwd", return_value=self.tree.root):
            with self.assertRaises(reader.FrozenError):
                reader.main(["read", "--data", str(self.tree.data)])
            buffer = io.StringIO()
            with (
                contextlib.redirect_stdout(buffer),
                mock.patch.object(reader, "sealed_read", return_value="THE NUMBERS\n") as read,
            ):
                reader.main(["read", "--data", str(self.tree.data), "--documents-es", "100", "--documents-en", "60"])
            self.assertEqual(buffer.getvalue(), "THE NUMBERS\n")
            self.assertEqual(read.call_args.args[4], {"es": 100, "en": 60})


class ProductCallTest(unittest.TestCase):
    """judge_page on a made-up page, with a fake guard that fixes a word only when it is shown readings."""

    def build(self, folder: Path, source="catalog"):
        from extract.engine.cache import reading_to_dict
        from extract.types import PageReading

        texts = ["cazu"] + [f"perro{chr(97 + i)}" for i in range(29)]  # 30 words: one fix is under the 8 % brake
        words = tuple(
            Word(f"p1w{i}", 1, t, (10.0 * i, 0.0, 10.0 * i + 8.0, 10.0), conf=0.5, line_id=0, idx_in_line=i)
            for i, t in enumerate(texts)
        )
        reading = PageReading(1, 100, 100, 100.0, 100.0, 100, words, mean_conf=0.5)
        truth = [
            {"id": f"t{i}", "text": t, "box": list(w.box)}
            for i, (t, w) in enumerate(zip(["caza"] + texts[1:], words, strict=True))
        ]
        path = folder / "1.json"
        record = {
            "set": "x",
            "doc_id": "d",
            "page": 1,
            "source": source,
            "band": "light",
            "truth": {"words": truth},
            "ungrounded": [],
            "ocr": {"engine": "e", "reading": reading_to_dict(reading)},
        }
        path.write_text(json.dumps(record), encoding="utf-8")
        return path, record

    def write_cell(self, path, cell):
        readings = [[["latin", "caza", 0.9, None, [], 0, False]]] + [None] * 29
        reader.cell_files(path)[reader.CELLS.index(cell)].write_text(
            json.dumps({"key": "k", "readings": readings, "rereads": {"0": ["latin"]}, "report": {}}), encoding="utf-8"
        )

    def guards(self, calls):
        from extract.types import Scored

        class Fake:
            info = {"model_id": "fake"}
            last_warnings: list = []

            def __init__(self, name, needs_readings):
                self.name, self.needs = name, needs_readings

            def score_page(self, page, *, languages, locale, budget):
                calls.append((self.name, tuple(languages), budget, [bool(w.readings) for w in page.words]))
                out = []
                for i, w in enumerate(page.words):
                    fix = None
                    if i == 0 and (not self.needs or w.readings):
                        fix = Fix("text", "x", "caza", (w.id,), "s", ("s",), 0.99, 0.01, None, ("safe", "more"), ())
                    out.append(
                        Scored(w.id, 0.9 if i == 0 else 0.01, "doubtful" if i == 0 else "ok", (), None, None, fix)
                    )
                return out

        return {"default": Fake("default", False), "thorough": Fake("thorough", True)}

    def test_G32_the_thorough_model_sees_the_readings_of_L_and_Z_and_the_default_one_does_not(self):
        with tempfile.TemporaryDirectory() as folder:
            path, record = self.build(Path(folder))
            self.write_cell(path, "L")
            calls = []
            out = reader.judge_page(path, record, self.guards(calls), Path(folder))
            self.assertEqual([c[0] for c in calls], ["default", "thorough"])
            self.assertEqual(calls[0][3], [False] * 30)
            self.assertEqual(calls[1][3], [True] + [False] * 29)
            self.assertFalse(out["without_cells"])
            # the fake thorough model fixes the word only when it has readings: that is the gain over the default
            self.assertEqual(out["tallies"]["thorough|more"]["weights"].get("fixed"), 1)
            self.assertEqual(out["tallies"]["default|more"]["weights"].get("fixed"), 1)  # the fake default fixes it too

    def test_G32_a_page_whose_cell_files_are_missing_is_flagged_and_the_thorough_model_gets_no_readings(self):
        with tempfile.TemporaryDirectory() as folder:
            path, record = self.build(Path(folder))
            calls = []
            out = reader.judge_page(path, record, self.guards(calls), Path(folder))
            self.assertTrue(out["without_cells"])
            self.assertEqual(calls[1][3], [False] * 30)
            self.assertEqual(out["tallies"]["thorough|safe"]["weights"].get("fixed", 0), 0)
            self.assertEqual(out["tallies"]["default|safe"]["weights"].get("fixed"), 1)

    def write_empty_cells(self, path, report):
        for cell in reader.cell_files(path):
            cell.write_text(
                json.dumps({"key": "k", "readings": [None] * 30, "rereads": {}, "report": report}), encoding="utf-8"
            )

    def judge_empty(self, report):
        with tempfile.TemporaryDirectory() as folder:
            path, record = self.build(Path(folder))
            self.write_empty_cells(path, report)
            return reader.judge_page(path, record, self.guards([]), Path(folder))

    def test_B49_a_page_with_nothing_doubtful_has_no_readings_by_construction_and_is_counted_apart(self):
        out = self.judge_empty({"units_line": 0, "units_zone": 0, "units_read": 0})
        self.assertFalse(out["without_cells"])
        self.assertTrue(out["without_doubtful_units"])

    def test_B49_a_page_with_doubtful_lines_but_no_readings_is_still_refused(self):
        out = self.judge_empty({"units_line": 3, "units_read": 0})
        self.assertTrue(out["without_cells"])
        self.assertFalse(out["without_doubtful_units"])

    def test_B49_a_cell_whose_reader_failed_is_refused_even_with_no_doubtful_lines(self):
        out = self.judge_empty({"units_line": 0, "reader_failed": 1})
        self.assertTrue(out["without_cells"])
        self.assertFalse(out["without_doubtful_units"])

    def test_B49_a_page_with_readings_is_in_neither_count(self):
        with tempfile.TemporaryDirectory() as folder:
            path, record = self.build(Path(folder))
            self.write_cell(path, "L")
            out = reader.judge_page(path, record, self.guards([]), Path(folder))
        self.assertFalse(out["without_cells"])
        self.assertFalse(out["without_doubtful_units"])

    def test_G28_the_languages_and_the_review_budget_are_the_products_not_literals(self):
        from extract import pipeline
        from extract.options import Options

        with tempfile.TemporaryDirectory() as folder:
            path, record = self.build(Path(folder))
            self.write_cell(path, "L")
            calls = []
            reader.judge_page(path, record, self.guards(calls), Path(folder))
            self.assertEqual(calls[0][1], Options().languages)
            self.assertEqual(calls[0][2], pipeline.REVIEW_BUDGETS[Options().review])
            calls.clear()
            with mock.patch.dict(pipeline.REVIEW_BUDGETS, {"balanced": 0.5}):
                reader.judge_page(path, record, self.guards(calls), Path(folder))
            self.assertEqual({c[2] for c in calls}, {0.5})

    def test_N10_the_invoice_suggestions_are_counted_on_the_page_that_safe_leaves_not_on_the_one_more_leaves(self):
        seen = []
        with tempfile.TemporaryDirectory() as folder:
            path, record = self.build(Path(folder), source="invoice")
            self.write_cell(path, "L")
            data = Path(folder) / "data"
            (data / "invoices" / "x").mkdir(parents=True)
            (data / "invoices" / "x" / "d.json").write_text(json.dumps({"language": "es"}), encoding="utf-8")
            only_more = self.guards([])
            for guard in only_more.values():
                original = guard.score_page

                def score(page, *, languages, locale, budget, _original=original):
                    out = _original(page, languages=languages, locale=locale, budget=budget)
                    return [
                        dataclasses.replace(s, fix=dataclasses.replace(s.fix, modes=("more",))) if s.fix else s
                        for s in out
                    ]

                guard.score_page = score
            with mock.patch.object(
                reader,
                "arithmetic_suggestions",
                side_effect=lambda applied, *a: seen.append(applied.words[0].text) or [],
            ):
                reader.judge_page(path, record, only_more, data)
        self.assertEqual(seen, ["cazu", "cazu"])  # safe leaves the word as read; `more` would have made it "caza"
