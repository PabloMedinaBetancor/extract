import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from bench import read_test_v5 as reader
from bench import read_test_v5_tables as tables
from bench.guard import accounting
from extract.types import Fix
from tests.test_read_test_v5 import ProductCallTest, SealedTree, tally


def component(kind, words, truth, right_before, weight=1):
    return accounting.Component(kind, tuple(words), truth, weight, right_before)


def fix(op, kind="k", source="s"):
    return Fix(op, kind, "x", ("a",), source, (source,), 0.9, 0.1, None, ("safe", "more"), ())


class AttributeTest(unittest.TestCase):
    def test_every_touched_component_is_credited_to_the_op_that_touched_it(self):
        texts = ["caza", "peno", "ab", "cd", "ghost"]
        comps = [
            component("pair", [0], "caza", True),  # right, edit breaks it
            component("pair", [1], "perro", False),  # wrong, edit fixes it
            component("split", [2, 3], "abcd", False, 1),  # a join fixes it
            component("invented", [4], "", False),  # a drop fixes it
        ]
        pairs_ = [
            (accounting.Edit("text", (0,), "cazo"), fix("text", "k1", "s1")),
            (accounting.Edit("text", (1,), "perro"), fix("text", "k2", "s2")),
            (accounting.Edit("join", (2, 3), "abcd"), fix("join", "k3", "s3")),
            (accounting.Edit("drop", (4,)), fix("drop", "k4", "s4")),
        ]
        rows = tables.attribute(comps, texts, pairs_)
        self.assertEqual(
            [(r["op"], r["outcome"], r["weight"]) for r in rows],
            [("text", "broken", 1), ("text", "fixed", 1), ("join", "fixed", 1), ("drop", "fixed", 1)],
        )
        self.assertEqual([r["source"] for r in rows], ["s1", "s2", "s3", "s4"])
        self.assertEqual([r["kind"] for r in rows], ["k1", "k2", "k3", "k4"])

    def test_an_edit_that_overlaps_an_earlier_one_is_skipped_and_the_first_gets_the_credit(self):
        texts = ["ab", "cd"]
        comps = [component("split", [0, 1], "abcd", False)]
        pairs_ = [
            (accounting.Edit("join", (0, 1), "abcd"), fix("join")),
            (accounting.Edit("text", (1,), "zz"), fix("text")),
        ]
        rows = tables.attribute(comps, texts, pairs_)
        self.assertEqual([(r["op"], r["outcome"]) for r in rows], [("join", "fixed")])

    def test_a_changed_still_wrong_component_is_counted_and_an_untouched_one_is_not(self):
        comps = [component("pair", [0], "a", False), component("pair", [1], "b", False)]
        rows = tables.attribute(comps, ["x", "y"], [(accounting.Edit("text", (0,), "z"), fix("text"))])
        self.assertEqual([r["outcome"] for r in rows], ["changed_still_wrong"])

    def test_a_single_word_component_on_the_shared_word_is_credited_to_the_edit_that_was_applied(self):
        comps = [component("pair", [0], "ab", True), component("pair", [1], "cd", True)]
        pairs_ = [
            (accounting.Edit("join", (0, 1), "abcd"), fix("join")),
            (accounting.Edit("text", (1,), "zz"), fix("text")),
        ]
        rows = tables.attribute(comps, ["ab", "cd"], pairs_)
        self.assertEqual([(r["op"], r["outcome"]) for r in rows], [("join", "broken"), ("join", "broken")])

    def test_a_component_touched_by_two_separate_edits_is_credited_to_the_first_one_in_the_list(self):
        texts = ["ab", "cd"]
        comps = [component("split", [0, 1], "abcd", False)]
        for order in ((("text", "s1"), ("segment", "s2")), (("segment", "s2"), ("text", "s1"))):
            pairs_ = [
                (accounting.Edit(op, (i,), "x" if i == 0 else "y"), fix(op, "k-" + src, src))
                for i, (op, src) in enumerate(order)
            ]
            rows = tables.attribute(comps, texts, pairs_)
            self.assertEqual(
                [(r["op"], r["source"], r["kind"]) for r in rows], [(order[0][0], order[0][1], "k-" + order[0][1])]
            )

    def test_the_attribution_adds_up_to_the_pages_tally_and_a_wrong_sum_stops(self):
        comps = [component("pair", [0], "caza", True), component("pair", [1], "perro", False)]
        texts = ["caza", "peno"]
        edits = [accounting.Edit("text", (0,), "cazo"), accounting.Edit("text", (1,), "perro")]
        tallied = reader.tally_dict(accounting.tally(comps, texts, edits))
        rows = tables.attribute(comps, texts, [(e, fix("text")) for e in edits])
        kinds = {"pair": tallied, **{k: reader.tally_dict(accounting.tally([], texts, edits)) for k in ("merge",)}}
        record = {"doc_id": "d", "page": 1}
        tables._check_page(tallied, rows, kinds, "default|safe", record)
        with self.assertRaises(tables.TablesError):
            tables._check_page(tallied, rows[:1], kinds, "default|safe", record)  # an edit lost
        bad = dict(tallied, truth_words=tallied["truth_words"] + 1)
        with self.assertRaises(tables.TablesError):
            tables._check_page(tallied, rows, {"pair": bad}, "default|safe", record)  # a kind that does not add up


class KnownAnswerTest(unittest.TestCase):
    """The published tables on two hand-made documents (different bands, sources and kinds), with every number worked
    out by hand: a wrong grouping, credit, denominator or document index changes one of them."""

    KEYS = ("default|safe", "default|more", "thorough|safe", "thorough|more")

    @staticmethod
    def page(doc, band, source, slice_, none, kinds, rows):
        """`kinds`: kind -> (fixed, broken, still, wrong, words) of default|more; the other series are empty."""
        by_kind = {"none": {k: tally(wrong=w[3], words=w[4]) for k, w in kinds.items()}}
        for key in KnownAnswerTest.KEYS:
            by_kind[key] = {
                k: tally(w[0], w[1], w[2], w[3], w[4]) if key == "default|more" else tally(wrong=w[3], words=w[4])
                for k, w in kinds.items()
            }
        for kind in tables.KINDS:
            for series in by_kind.values():
                series.setdefault(kind, tally(words=0))
        total = {
            f: sum(w[i] for w in kinds.values()) for i, f in enumerate(("fixed", "broken", "still", "wrong", "words"))
        }
        tallies = {"none": tally(wrong=none, words=total["words"])}
        for key in KnownAnswerTest.KEYS:
            tallies[key] = (
                tally(total["fixed"], total["broken"], total["still"], total["wrong"], total["words"])
                if key == "default|more"
                else tally(wrong=total["wrong"], words=total["words"])
            )
        judged = {
            "doc": doc,
            "page": 1,
            "language": "es",
            "slice": slice_,
            "band": band,
            "page_source": source,
            "tallies": tallies,
        }
        extra = {"by_kind": by_kind, "rows": {k: (rows if k == "default|more" else []) for k in KnownAnswerTest.KEYS}}
        return judged, extra

    def setUp(self):
        row = lambda op, outcome, weight, kind="k", source="s": {  # noqa: E731
            "op": op,
            "kind": kind,
            "source": source,
            "outcome": outcome,
            "weight": weight,
        }
        # d1: light, catalogue. pair 6 words: fixed 2, broken 1; split 2 words: fixed 1; wrong as read 10 on 100 words
        d1 = self.page(
            "d1",
            "light",
            "catalog",
            "pages",
            10,
            {"pair": (2, 1, 0, 7, 80), "split": (1, 0, 0, 3, 20)},
            [
                row("text", "fixed", 2),
                row("text", "broken", 1),
                row("text", "left_right", 4),
                row("join", "fixed", 1, "j", "other"),
            ],
        )
        # d2: heavy, invoice. pair: fixed 3, still wrong 2; invented: fixed 1, wrong as read 20 on 200 words
        d2 = self.page(
            "d2",
            "heavy",
            "invoice",
            "invoices",
            20,
            {"pair": (3, 0, 2, 15, 150), "invented": (1, 0, 0, 5, 50)},
            [row("text", "fixed", 3), row("text", "changed_still_wrong", 2), row("drop", "fixed", 1, "d", "drop")],
        )
        self.judged = [d1[0], d2[0]]
        self.extras = [d1[1], d2[1]]
        self.out = tables.tables_for("es", self.judged, self.extras, 50)

    def more(self, table, model="default"):
        return table["models"][model]["more"]

    def test_by_band_has_each_band_with_its_own_documents_and_numbers(self):
        bands = self.out["by_band"]["all"]
        self.assertEqual(set(bands), {"light", "heavy"})
        self.assertEqual((self.more(bands["light"])["fixed"], self.more(bands["light"])["broken"]), (3, 1))
        self.assertEqual((self.more(bands["heavy"])["fixed"], self.more(bands["heavy"])["changed_still_wrong"]), (4, 2))
        self.assertEqual(bands["light"]["documents"], 1)
        self.assertEqual(set(self.out["by_band"]["pages"]), {"light"})
        self.assertEqual(set(self.out["by_band"]["invoices"]), {"heavy"})

    def test_by_page_source_has_catalogue_and_invoice_apart(self):
        sources = self.out["by_page_source"]["all"]
        self.assertEqual(set(sources), {"catalog", "invoice"})
        self.assertEqual(self.more(sources["catalog"])["fixed"], 3)
        self.assertEqual(self.more(sources["invoice"])["fixed"], 4)

    def test_by_component_kind_gives_each_kind_its_own_components(self):
        kinds = self.out["by_component_kind"]["all"]
        self.assertEqual(set(kinds), set(tables.KINDS))
        self.assertEqual(self.more(kinds["pair"])["fixed"], 5)
        self.assertEqual(self.more(kinds["pair"])["broken"], 1)
        self.assertEqual(self.more(kinds["split"])["fixed"], 1)
        self.assertEqual(self.more(kinds["invented"])["fixed"], 1)
        self.assertEqual((kinds["merge"]["pages"], kinds["merge"]["documents"]), (0, 0))
        self.assertEqual(kinds["pair"]["documents"], 2)
        self.assertEqual(kinds["split"]["documents"], 1)  # only d1 has split components

    def test_by_fix_numbers_and_their_denominators(self):
        text = self.out["by_fix"]["default|more"]["op=text"]
        self.assertEqual((text["fixed"], text["broken"], text["changed_still_wrong"]), (5, 1, 2))
        # per document: d1 fixed 2 broken 1; d2 fixed 3 still 2; wrong as read 10 and 20; truth words 100 and 200
        fixed, broken, still = np.array([2.0, 3.0]), np.array([1.0, 0.0]), np.array([0.0, 2.0])
        wrong, words = np.array([10.0, 20.0]), np.array([100.0, 200.0])
        iv = lambda n, d: reader.interval(n, d, 50)  # noqa: E731
        self.assertEqual(text["precision"], iv(fixed, fixed + broken + still))  # 5 / 8, no left_right
        self.assertAlmostEqual(text["precision"]["value"], 5 / 8)
        self.assertEqual(text["broken_share_of_fixed"], iv(broken, fixed))  # 1 / 5
        self.assertAlmostEqual(text["broken_share_of_fixed"]["value"], 0.2)
        self.assertEqual(text["net_fixed_per_100_wrong"], reader._per_100(iv(fixed - broken, wrong)))  # 4 / 30
        self.assertAlmostEqual(text["net_fixed_per_100_wrong"]["value"], 400 / 30)
        self.assertEqual(text["d_wer"], iv(broken - fixed, words))  # -4 / 300
        self.assertAlmostEqual(text["d_wer"]["value"], -4 / 300)
        self.assertEqual(text["documents"], 2)
        self.assertEqual(text["left_right"], 4)  # reported apart: not fixed, not in the precision denominator
        drop = self.out["by_fix"]["default|more"]["op=drop"]  # only d2 has it, the interval is over both documents
        self.assertEqual(drop["fixed"], 1)
        self.assertEqual(drop["d_wer"], iv(np.array([0.0, -1.0]), words))

    def test_by_fix_groups_by_kind_and_by_source_too(self):
        groups = self.out["by_fix"]["default|more"]
        self.assertEqual(groups["source=other"]["fixed"], 1)
        self.assertEqual(groups["kind=j"]["fixed"], 1)
        self.assertEqual(groups["source=s"]["fixed"], 5)
        self.assertEqual(groups["kind=k"]["changed_still_wrong"], 2)

    def test_the_families_add_up_to_the_first_reads_table_and_a_difference_stops(self):
        first = {
            "all": {"models": {m: {"safe": {"fixed": 0, "broken": 0, "changed_still_wrong": 0}} for m in reader.MODELS}}
        }
        for m in reader.MODELS:
            first["all"]["models"][m]["more"] = {"fixed": 0, "broken": 0, "changed_still_wrong": 0}
        first["all"]["models"]["default"]["more"] = {"fixed": 7, "broken": 1, "changed_still_wrong": 2}
        tables.check_fix_totals(first, self.out["by_fix"], "es")
        first["all"]["models"]["default"]["more"]["fixed"] = 8
        with self.assertRaises(tables.TablesError):
            tables.check_fix_totals(first, self.out["by_fix"], "es")


class GatesTest(unittest.TestCase):
    """Every integrity stop of the tool fires in a test (second review)."""

    record = {"doc_id": "d", "page": 1}

    def test_the_kinds_must_add_up_in_every_weight_and_in_the_none_series(self):
        whole = tally(fixed=1, broken=1, still=1, wrong=4, words=7)
        whole["weights"]["left_right"] = 3
        parts = {"pair": tally(fixed=1, broken=1, still=1, wrong=4, words=7)}
        parts["pair"]["weights"]["left_right"] = 3
        tables._check_kinds(whole, parts, "none", self.record)
        for outcome in ("left_right", "left_wrong", "fixed", "broken", "changed_still_wrong"):
            with self.subTest(outcome):
                bad = json.loads(json.dumps(parts))
                bad["pair"]["weights"][outcome] += 1
                with self.assertRaises(tables.TablesError):
                    tables._check_kinds(whole, bad, "none", self.record)
        for field in ("truth_words", "truth_characters", "distance_before", "distance_after"):
            with self.subTest(field):
                bad = json.loads(json.dumps(parts))
                bad["pair"][field] += 1
                with self.assertRaises(tables.TablesError):
                    tables._check_kinds(whole, bad, "none", self.record)

    def groups(self):
        def group(f, b, c):
            return {"fixed": f, "broken": b, "changed_still_wrong": c}

        by_fix = {f"{n}|{m}": {} for n in reader.MODELS for m in reader.MODES}
        for key in by_fix:
            by_fix[key] = {
                "op=text": group(2, 1, 0),
                "op=join": group(1, 0, 3),
                "kind=a": group(3, 1, 3),
                "source=s": group(3, 1, 3),
            }
        first = {"all": {"models": {n: {m: group(3, 1, 3) for m in reader.MODES} for n in reader.MODELS}}}
        return first, by_fix

    def test_check_fix_totals_fires_for_every_family_field_model_and_mode(self):
        first, by_fix = self.groups()
        tables.check_fix_totals(first, by_fix, "es")
        for key in by_fix:
            for family, name in (("op", "op=text"), ("kind", "kind=a"), ("source", "source=s")):
                for field in ("fixed", "broken", "changed_still_wrong"):
                    with self.subTest(key=key, family=family, field=field):
                        bad = json.loads(json.dumps(by_fix))
                        bad[key][name][field] += 1
                        with self.assertRaises(tables.TablesError):
                            tables.check_fix_totals(first, bad, "es")

    def test_a_dropped_group_stops_analyse(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(
                tables, "families_report", return_value={f"{n}|{m}": {} for n in reader.MODELS for m in reader.MODES}
            ):
                with self.assertRaises(tables.TablesError):
                    OnPageTest("test_the_tables_are_added_and_add_up_to_the_first_read").run_analyse(folder)

    def test_a_dropped_edit_stops_the_page(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(tables, "edits_with_fixes", return_value=[]):
                with self.assertRaisesRegex(tables.TablesError, "edits differ"):
                    OnPageTest("test_the_tables_are_added_and_add_up_to_the_first_read").run_analyse(folder)

    def test_a_none_series_that_does_not_add_up_stops_the_page(self):
        case = OnPageTest("test_the_tables_are_added_and_add_up_to_the_first_read")
        with tempfile.TemporaryDirectory() as folder:
            path, record = case.setup_page(folder)
            guards = case.guards([])
            judged = reader.judge_page(path, record, guards, Path(folder))
            tables.attribute_page(path, record, guards, judged)
            judged["tallies"]["none"]["truth_words"] += 1
            with self.assertRaisesRegex(tables.TablesError, "none"):
                tables.attribute_page(path, record, guards, judged)


class DetectorTest(unittest.TestCase):
    """Detector-comparison plumbing: the groups handed to the comparison and what is added to its report."""

    def test_the_comparison_is_per_language_and_per_language_and_slice_and_never_pooled(self):
        from bench.guard import detector_compare

        seen = []

        def rows_by_group(records, key):
            seen.append(sorted({key(r) for _, r in records}))
            return {"x": 1, "all": 2}

        guards = {"default": mock.Mock(model="m1"), "thorough": mock.Mock(model="m2")}
        records = [
            (Path("a"), {"set": "val", "source": "catalog", "doc_id": "a"}),
            (Path("b"), {"set": "dev5en", "source": "catalog", "doc_id": "b"}),
            (Path("c"), {"set": "val", "source": "invoice", "doc_id": "c"}),
        ]
        compared = []
        with (
            mock.patch.object(detector_compare, "rows_by_group", rows_by_group),
            mock.patch.object(
                detector_compare,
                "compare",
                lambda groups, cands, resamples: compared.append((groups, cands, resamples)) or {"g": 1},
            ),
            mock.patch.object(
                reader, "language_of", lambda record, data: "en" if record["set"].endswith("en") else "es"
            ),
        ):
            out = tables.detector_tables(records, Path("."), guards, 77)
        self.assertEqual(seen, [["en", "es"], ["en|pages", "es|invoices", "es|pages"]])
        self.assertEqual(set(out), {"by_language", "by_language_and_slice", "note"})
        self.assertTrue(all("all" not in groups for groups, _, _ in compared))
        self.assertEqual(compared[0][1], {"default": "m1", "thorough": "m2"})
        self.assertEqual(compared[0][2], 77)

    def test_an_english_invoice_in_the_spanish_set_is_grouped_as_english_by_its_sidecar(self):
        from bench.guard import detector_compare

        keys = []
        with tempfile.TemporaryDirectory() as folder:
            sidecar = Path(folder) / "invoices" / "val" / "inv1.json"
            sidecar.parent.mkdir(parents=True)
            sidecar.write_text(json.dumps({"language": "en"}), encoding="utf-8")
            record = {"set": "val", "source": "invoice", "doc_id": "inv1"}
            with (
                mock.patch.object(
                    detector_compare, "rows_by_group", lambda recs, key: keys.append(key(record)) or {"all": 1}
                ),
                mock.patch.object(detector_compare, "compare", lambda *a, **k: {}),
            ):
                out = tables.detector_tables([(Path("x"), record)], Path(folder), {}, 5)
        self.assertEqual(keys, ["en", "en|invoices"])
        self.assertIn("v5 labelling", out["note"])

    def test_analyse_hands_the_detector_the_records_the_data_the_guards_and_the_resamples(self):
        seen = []

        def detector(records, data, guards, resamples):
            seen.append((len(records), data, sorted(guards), resamples))
            return {}

        with tempfile.TemporaryDirectory() as folder:
            OnPageTest("test_the_tables_are_added_and_add_up_to_the_first_read").run_analyse(folder, detector=detector)
            self.assertEqual(seen, [(1, Path(folder), ["default", "thorough"], 50)])

    def test_the_raw_ocr_wer_of_the_same_pages_is_added_from_the_first_read(self):
        with tempfile.TemporaryDirectory() as folder:
            out = OnPageTest("test_the_tables_are_added_and_add_up_to_the_first_read").run_analyse(
                folder, detector=lambda records, data, guards, resamples: {"x": 1}
            )
        self.assertEqual(out["g31_m9"]["x"], 1)
        self.assertEqual(out["g31_m9"]["raw_ocr_wer_as_read"]["es"]["documents"], 1)
        self.assertIn("value", out["g31_m9"]["raw_ocr_wer_as_read"]["es"])


class UnknownSourceTest(unittest.TestCase):
    def test_a_page_with_no_source_is_its_own_group_not_dropped(self):
        judged, extra = KnownAnswerTest.page("d", "light", None, "pages", 1, {"pair": (0, 0, 0, 1, 10)}, [])
        out = tables.tables_for("es", [judged], [extra], 20)
        self.assertEqual(list(out["by_page_source"]["all"]), ["unknown"])
        self.assertEqual(out["by_page_source"]["all"]["unknown"]["documents"], 1)


class AssertionTest(unittest.TestCase):
    STORED = {"engines": ["e"], "pages": 2, "by_language": {"es": {"all": {"d_wer": {"value": 0.25, "low": -0.5}}}}}

    def fresh(self, mutate=None):
        report = json.loads(json.dumps(self.STORED))
        if mutate:
            mutate(report)
        return report

    def test_the_same_report_passes_and_read_pages_is_the_only_key_that_may_be_added(self):
        stored = dict(self.STORED, read_pages={"es": 2})
        tables.assert_first_read(stored, self.fresh())

    def test_M_A29_a_changed_number_stops_the_run(self):
        for mutate in (
            lambda r: r["by_language"]["es"]["all"]["d_wer"].update(value=0.2500001),
            lambda r: r.update(pages=3),
            lambda r: r["by_language"]["es"]["all"]["d_wer"].pop("low"),
            lambda r: r["by_language"]["es"].update(extra={}),
            lambda r: r.update(engines=["e", "f"]),
        ):
            with self.subTest():
                with self.assertRaises(tables.TablesError):
                    tables.assert_first_read(self.STORED, self.fresh(mutate))

    def test_M_A29_an_integer_that_became_a_float_is_a_difference_too(self):
        with self.assertRaises(tables.TablesError):
            tables.assert_first_read({"pages": 2}, {"pages": 2.0})


class OnPageTest(ProductCallTest):
    """The whole path on a made-up page with the fake guards of the first read's tests."""

    def setup_page(self, folder):
        path, record = self.build(Path(folder))
        self.write_cell(path, "L")
        return path, record

    def run_analyse(self, folder, stored_edit=None, detector=None):
        path, record = self.setup_page(folder)
        guards = self.guards([])
        for guard in guards.values():
            guard.model = mock.Mock(model_id=lambda: "fake")
        records = [(path, record)]
        first = reader.analyse(records, Path(folder), guards, 50)
        stored = json.loads(reader.render(first))
        if stored_edit:
            stored_edit(stored)
        return tables.analyse(records, Path(folder), guards, 50, detector, stored)

    def test_the_tables_are_added_and_add_up_to_the_first_read(self):
        with tempfile.TemporaryDirectory() as folder:
            out = self.run_analyse(folder, detector=lambda records, data, guards, resamples: {"rows": len(records)})
        self.assertTrue(out["first_read_reproduced"])
        self.assertEqual(out["g31_m9"]["rows"], 1)
        block = out["by_language"]["es"]
        self.assertEqual(list(block["by_band"]["pages"]), ["light"])
        self.assertEqual(list(block["by_page_source"]["pages"]), ["catalog"])
        self.assertEqual(set(block["by_component_kind"]["all"]), set(tables.KINDS))
        pair = block["by_component_kind"]["all"]["pair"]["models"]["default"]["more"]
        self.assertEqual(pair["fixed"], 1)
        fixes_ = block["by_fix"]["thorough|more"]
        self.assertEqual(fixes_["op=text"]["fixed"], 1)
        self.assertEqual(fixes_["op=text"]["broken"], 0)
        self.assertEqual(fixes_["source=s"]["precision"]["value"], 1.0)
        self.assertEqual(fixes_["kind=x"]["documents"], 1)

    def test_M_A29_the_run_stops_when_the_first_read_is_not_reproduced(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(tables.TablesError):
                self.run_analyse(
                    folder,
                    stored_edit=lambda s: s["by_language"]["es"]["all"]["models"]["default"]["more"].update(fixed=7),
                )

    def test_M_A29_a_by_edit_credit_that_does_not_match_the_page_stops_the_run(self):
        with tempfile.TemporaryDirectory() as folder:
            with mock.patch.object(tables, "attribute", return_value=[]):
                with self.assertRaises(tables.TablesError):
                    self.run_analyse(folder)


class SecondReadTest(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.tree = SealedTree(Path(folder.name))
        self.first = Path("bench/results/first.json")
        (self.tree.root / self.first).parent.mkdir(parents=True, exist_ok=True)
        (self.tree.root / self.first).write_text(json.dumps({"pages": 9}), encoding="utf-8")

    def first_read_done(self, reason=None):
        for source in self.tree.sources:
            reader.sealed.read_test_pages(
                self.tree.data, "metrics", self.tree.root / source.reads, reason=reason, sets=source.sets
            )

    def go(self, analyse=None):
        analyse = analyse or (lambda *a, **k: {"pages": 1, "by_language": {}})
        with mock.patch.object(tables, "analyse", side_effect=analyse), mock.patch.object(reader, "load_guards"):
            return tables.sealed_read_tables(
                self.tree.root,
                self.tree.data,
                (Path("a"), Path("b")),
                10,
                {"es": 2, "en": 2},
                lambda r, d, g, n: {},
                tuple(self.tree.sources),
                first=self.first,
            )

    def metrics(self):
        return [m for s in self.tree.sources for m in self.tree.metric_lines(s)]

    def test_it_refuses_to_read_without_the_detector_comparison_and_spends_nothing(self):
        self.first_read_done()
        with mock.patch.object(reader, "load_guards"):
            with self.assertRaises(tables.TablesError):
                tables.sealed_read_tables(
                    self.tree.root,
                    self.tree.data,
                    (Path("a"), Path("b")),
                    10,
                    {"es": 2, "en": 2},
                    None,
                    tuple(self.tree.sources),
                    first=self.first,
                )
        self.assertEqual(len(self.metrics()), 2)
        self.assertFalse((self.tree.root / tables.OUT).exists())

    def test_it_needs_exactly_one_earlier_metrics_read_of_each_test(self):
        with self.assertRaises(tables.TablesError):
            self.go()  # none yet
        self.first_read_done()
        self.first_read_done("another")
        with self.assertRaises(tables.TablesError):
            self.go()  # two already: a third is not this one
        self.assertFalse((self.tree.root / tables.OUT).exists())

    def test_the_second_read_is_registered_with_its_reason_after_the_tables_and_the_file_has_what_it_printed(self):
        self.first_read_done()
        seen = {}

        def analyse(records, data, guards, resamples, detector, stored):
            seen["before"] = len(self.metrics())
            seen["stored"] = stored
            return {"pages": len(records), "by_language": {}}

        text = self.go(analyse)
        self.assertEqual(seen["before"], 2)  # the first reads only, while the tables are made
        self.assertEqual(seen["stored"], {"pages": 9})
        lines = self.metrics()
        self.assertEqual(len(lines), 4)
        self.assertEqual([line["reason"] for line in lines], [None, tables.REASON, None, tables.REASON])
        self.assertEqual((self.tree.root / tables.OUT).read_text(encoding="utf-8"), text)
        self.assertEqual(json.loads(text)["reason"], "second-read-tables")
        self.assertEqual(json.loads(text)["first_read_sha256"], reader.digest(self.tree.root / self.first))

    def test_a_failed_assertion_registers_nothing_and_writes_nothing(self):
        self.first_read_done()

        def analyse(*args, **kwargs):
            raise tables.TablesError("the first read is not reproduced")

        with self.assertRaises(tables.TablesError):
            self.go(analyse)
        self.assertEqual(len(self.metrics()), 2)
        self.assertFalse((self.tree.root / tables.OUT).exists())

    def test_a_changed_frozen_file_stops_before_any_page_is_opened(self):
        self.first_read_done()
        (self.tree.root / "src/extract/fixes.py").write_bytes(b"changed")
        with mock.patch.object(reader.sealed, "read_test_pages", side_effect=AssertionError("opened")):
            with self.assertRaises(reader.FrozenError):
                self.go()


if __name__ == "__main__":
    unittest.main()
