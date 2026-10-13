"""The tables of the second read of T5 in the README: rendered from the file's structure, every number tied to the
file. Tested on a development dry run of `bench.read_test_v5_tables` (`tests/data/read-test-v5-tables-dry.json`, made with
`dry-run --sets val dev5en --documents 4 --resamples 200`); the real file `bench/results/read-test-v5-tables.json` has
the same structure and regenerates the README with the same code."""

import importlib.util
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "data" / "read-test-v5-tables-dry.json"
NAME = "read-test-v5-tables.json"
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def load_script():
    spec = importlib.util.spec_from_file_location("api_docs_tables", ROOT / "scripts" / "api_docs.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def leaves(node):
    if isinstance(node, dict):
        for value in node.values():
            yield from leaves(value)
    elif isinstance(node, list):
        for value in node:
            yield from leaves(value)
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield node


def allowed(data: dict) -> set[str]:
    out = set()
    for value in leaves(data):
        for scale in (1, 100):
            scaled = value * scale
            for digits in range(4):
                for text in (f"{scaled:.{digits}f}", f"{abs(scaled):.{digits}f}"):
                    out.add(text)
            out.add(f"{round(abs(scaled)):,}")
    # a zero written with its digits; the 95 % level; "per 100"; the broken-share floor (7.5 %) and the rows' minimum (50 fixed words)
    for block in data.get(
        "by_language", {}
    ).values():  # the band table's documents, added (a document is in several bands)
        out.add(str(sum(table["documents"] for table in block["by_band"]["all"].values())))
    out |= {"0.00", "0.0", "95", "100", "7.5", "50"}
    return out


def numbers_of(lines):
    return {t.strip(",").lstrip("+-") for line in lines for t in NUMBER.findall(line)}


class TablesTest(unittest.TestCase):
    def setUp(self):
        self.script = load_script()
        self.tmp = Path(tempfile.mkdtemp())
        shutil.copy(FIXTURE, self.tmp / NAME)
        self.data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        self.lines = self.script.t5_tables_lines(self.tmp)

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def rows(self, title: str, language: str) -> list[str]:
        """The table rows that follow the header line starting with `title`, in the language's part."""
        part = "\n".join(self.lines).split(f"\n{language}, `more`")[1].split("\n\n")
        for chunk in part:
            if chunk.lstrip().startswith(f"| {title}"):
                return chunk.strip().splitlines()[2:]
        self.fail(f"no table {title!r} for {language}")

    def test_without_the_file_nothing_is_rendered(self):
        self.assertEqual(self.script.t5_tables_lines(self.tmp / "nowhere"), [])

    def test_the_two_reads_sentence_says_what_the_file_says(self):
        self.assertIn("added the tables below and nothing else", self.lines[0])
        self.assertIn("not compared with a stored one", self.lines[0])  # the dry run has no first read
        real = json.loads(json.dumps(self.data))
        real["first_read_reproduced"] = True
        (self.tmp / NAME).write_text(json.dumps(real), encoding="utf-8")
        first = self.script.t5_tables_lines(self.tmp)[0]
        self.assertIn("recomputed and compared number by number", first)
        self.assertIn("the headline numbers above are the first read's", first)

    def test_both_languages_apart_each_with_its_tables(self):
        text = "\n".join(self.lines)
        self.assertLess(text.index("Spanish, `more`"), text.index("English, `more`"))
        for language in ("Spanish", "English"):
            for title in ("damage band", "page source", "component kind", "default, `more`", "thorough, `more`"):
                self.rows(title, language)

    def test_a_row_for_every_group_of_the_file_in_its_order(self):
        for code, language in (("es", "Spanish"), ("en", "English")):
            block = self.data["by_language"][code]
            for title, key in (
                ("damage band", "by_band"),
                ("page source", "by_page_source"),
                ("component kind", "by_component_kind"),
            ):
                labels = [row.split("|")[1].strip() for row in self.rows(title, language)]
                expected = list(block[key]["all"])
                if title == "page source":  # the rows say public-administration pages, not the file's key
                    expected = [load_script().PAGE_SOURCE_NAMES.get(name, name) for name in expected]
                self.assertEqual(labels, expected, (language, title))
            for model in ("default", "thorough"):
                labels = [row.split("|")[1].strip() for row in self.rows(f"{model}, `more`", language)]
                self.assertEqual(labels, list(block["by_fix"][f"{model}|more"]), (language, model))

    def test_every_number_is_in_the_file(self):
        known = allowed(self.data)
        self.assertEqual(sorted(n for n in numbers_of(self.lines) if n not in known), [])

    def test_each_cell_is_the_value_of_its_place(self):
        for code, language in (("es", "Spanish"), ("en", "English")):
            block = self.data["by_language"][code]
            for model in ("default", "thorough"):
                rows = self.rows(f"{model}, `more`", language)
                for row, (group, cell) in zip(rows, block["by_fix"][f"{model}|more"].items(), strict=True):
                    cells = [c.strip() for c in row.strip("|").split("|")]
                    self.assertEqual(cells[0], group)
                    self.assertEqual(
                        cells[1:4], [f"{cell['fixed']:,}", f"{cell['broken']:,}", f"{cell['changed_still_wrong']:,}"]
                    )
                    if cell["fixed"] < self.script.MIN_FIXED_FOR_SHARES:
                        self.assertEqual(cells[4:7], ["-", "-", "-"], group)  # a small row: its counts and nothing else
                        continue
                    self.assertTrue(cells[4].startswith(f"{cell['precision']['value'] * 100:.1f} %"), (group, cells[4]))
                    self.assertTrue(cells[5].startswith(f"{cell['broken_share_of_fixed']['value'] * 100:.1f} %"))
                    self.assertTrue(cells[6].startswith(f"{cell['net_fixed_per_100_wrong']['value']:.1f} ("))
            band = block["by_band"]["all"]
            for row, (label, table) in zip(self.rows("damage band", language), band.items(), strict=True):
                cells = [c.strip() for c in row.strip("|").split("|")]
                self.assertEqual((cells[0], cells[1]), (label, str(table["documents"])))
                if not table["none"]:
                    continue
                self.assertTrue(
                    cells[2].startswith(f"{table['none']['wer_as_read']['value'] * 100:.2f} %"), (label, cells[2])
                )
                for name, (d_wer_at, net_at) in (("default", (3, 4)), ("thorough", (5, 6))):
                    more = table["models"][name]["more"]
                    self.assertTrue(
                        cells[d_wer_at].startswith(f"{more['d_wer']['value'] * 100:.2f} ("), (label, name, cells)
                    )
                    self.assertTrue(
                        cells[net_at].startswith(f"{more['net_fixed_per_100_wrong']['value']:.1f} ("),
                        (label, name, cells),
                    )
                paired = table["paired_thorough_minus_default"]["more"]["net_fixed_per_100_wrong_difference"]
                self.assertTrue(cells[7].startswith(f"{paired['value']:.1f} ("), (label, cells[7]))

    def test_the_tables_are_the_all_slice_and_not_another(self):
        changed = json.loads(json.dumps(self.data))
        for key in ("by_band", "by_page_source", "by_component_kind"):
            for table in changed["by_language"]["es"][key]["pages"].values():
                table["documents"] = 987
        (self.tmp / NAME).write_text(json.dumps(changed), encoding="utf-8")
        self.assertEqual(self.script.t5_tables_lines(self.tmp), self.lines)

    def test_a_group_with_nothing_in_it_is_dashes_and_not_a_number(self):
        changed = json.loads(json.dumps(self.data))
        kind = next(iter(changed["by_language"]["es"]["by_component_kind"]["all"]))
        changed["by_language"]["es"]["by_component_kind"]["all"][kind] = {
            "pages": 0,
            "unreadable_pages": 0,
            "documents": 0,
            "none": None,
            "models": {},
            "paired_thorough_minus_default": {},
        }
        (self.tmp / NAME).write_text(json.dumps(changed), encoding="utf-8")
        script = load_script()
        part = "\n".join(script.t5_tables_lines(self.tmp)).split("\nSpanish, `more`")[1]
        self.assertIn(f"| {kind} | 0 | - | - | - | - | - | - |", part)

    def test_the_detector_table_names_the_v4_detector_and_both_models_and_says_met_or_not(self):
        for code, language in (("es", "Spanish"), ("en", "English")):
            group = self.data["g31_m9"]["by_language"]["groups"][code]
            rows = self.rows(f"{language}, review budget", language)
            self.assertTrue(rows[0].startswith("| v4 detector"))
            share = self.data["g31_m9"]["by_language"]["share"]
            self.assertIn(f"review budget {share * 100:.0f} % of the words", "\n".join(self.lines))
            for row, name in zip(rows[1:3], ("default", "thorough"), strict=True):
                self.assertIn(f"{name} model's detector", row)
                g31 = group[name]["g31"]
                gap = g31["loss"] * 100
                side = "higher" if gap <= 0 else "lower"
                verdict = "met" if g31["met"] else "not met"
                self.assertIn(
                    f"{verdict} (v5 {abs(gap):.1f} points {side} than v4; allowed loss {g31['tolerance'] * 100:.1f})",
                    row,
                )
        flipped = json.loads(json.dumps(self.data))
        for name in ("default", "thorough"):
            flipped["g31_m9"]["by_language"]["groups"]["es"][name]["g31"]["met"] = not flipped["g31_m9"]["by_language"][
                "groups"
            ]["es"][name]["g31"]["met"]
        (self.tmp / NAME).write_text(json.dumps(flipped), encoding="utf-8")
        self.assertNotEqual(self.script.t5_tables_lines(self.tmp), self.lines)

    def part(self, language: str) -> str:
        return "\n".join(self.lines).split(f"\n{language}, `more`")[1].split("\nEnglish, `more`")[0]

    def test_the_detector_verdict_is_worse_better_or_not_worse_by_the_file_and_never_better_when_the_interval_includes_zero(
        self,
    ):
        for code, language in (("es", "Spanish"), ("en", "English")):
            group = self.data["g31_m9"]["by_language"]["groups"][code]["default"]
            line = next(line for line in self.part(language).splitlines() if line.startswith(f"{language} detector:"))
            if not group["g31"]["met"]:
                self.assertIn("worse than the v4 detector (the criterion is not met)", line)
            elif group["minus_v4_recall"]["low"] > 0:
                self.assertIn("better than the v4 detector", line)
            else:
                self.assertIn("not worse than the v4 detector", line)
                self.assertNotIn("better than", line)
        changed = json.loads(json.dumps(self.data))
        spanish = changed["g31_m9"]["by_language"]["groups"]["es"]
        for name in ("default", "thorough"):
            spanish[name]["minus_v4_recall"]["low"] = 0.01  # now the interval excludes zero
            spanish[name]["g31"]["met"] = True
        (self.tmp / NAME).write_text(json.dumps(changed), encoding="utf-8")
        self.assertIn("Spanish detector: better than the v4 detector", "\n".join(self.script.t5_tables_lines(self.tmp)))
        for name in ("default", "thorough"):
            spanish[name]["g31"]["met"] = False
        (self.tmp / NAME).write_text(json.dumps(changed), encoding="utf-8")
        self.assertIn(
            "Spanish detector: worse than the v4 detector (the criterion is not met)",
            "\n".join(self.script.t5_tables_lines(self.tmp)),
        )

    def test_one_detector_is_said_only_if_the_two_are_identical(self):
        for language in ("Spanish", "English"):
            self.assertIn("use the same detector", self.part(language))
        changed = json.loads(json.dumps(self.data))
        changed["g31_m9"]["by_language"]["groups"]["es"]["thorough"]["recall"]["value"] += 0.01
        (self.tmp / NAME).write_text(json.dumps(changed), encoding="utf-8")
        text = "\n".join(self.script.t5_tables_lines(self.tmp))
        self.assertNotIn(
            "Spanish detector: not worse than the v4 detector (the criterion is met; the interval of the difference includes zero, so not better either). The default",
            text,
        )

    def test_the_slices_where_the_detector_is_not_met_are_named_and_never_hidden(self):
        groups = self.data["g31_m9"]["by_language_and_slice"]["groups"]
        missed = {key for key, part in groups.items() if "default" in part and not part["default"]["g31"]["met"]}
        for key, part in groups.items():
            language, _, slice_ = key.partition("|")
            text = self.part({"es": "Spanish", "en": "English"}[language])
            name = self.script.SLICE_NAMES.get(slice_, slice_)
            if key in missed:
                self.assertIn(f"{name} ({part['wrong']:,} wrong words)", text, key)
        for language in ("Spanish", "English"):
            code = "es" if language == "Spanish" else "en"
            mine = [k for k in missed if k.startswith(code + "|")]
            self.assertEqual("the criterion is not met on" in self.part(language), bool(mine), language)
            met = {k: p["default"]["g31"] for k, p in groups.items() if k.startswith(code + "|") and k not in missed}
            clears = [
                self.script.SLICE_NAMES.get(k.partition("|")[2], k) for k, g in met.items() if g["interval_clears"]
            ]
            short = [
                self.script.SLICE_NAMES.get(k.partition("|")[2], k) for k, g in met.items() if not g["interval_clears"]
            ]
            if not met:
                continue  # every slice missed (or there is none): nothing to say about the ones that were met
            line = next(line for line in self.part(language).splitlines() if line.startswith("By slice"))
            self.assertIn(f"clears the allowed loss on {', '.join(clears) or 'none of them'}", line)
            if short:
                self.assertIn(f"and not on {', '.join(short)}", line)
            else:
                self.assertNotIn("and not on", line)

    def test_the_invoices_carry_no_claim_and_the_kinds_over_the_floor_are_listed(self):
        for code, language in (("es", "Spanish"), ("en", "English")):
            block = self.data["by_language"][code]
            invoice = block["by_page_source"]["all"].get("invoice")
            if invoice:
                wrong = invoice["none"]["wrong_as_read"]
                self.assertIn(
                    f"{language} invoice pages hold {wrong:,} wrong words: too few to tell", self.part(language)
                )
            else:
                self.assertNotIn("invoice pages hold", self.part(language))
            over = []
            for model in ("default", "thorough"):
                for group, cell in block["by_fix"][f"{model}|more"].items():
                    if group.startswith("kind=") and cell["fixed"] >= 50 and cell["broken"] / cell["fixed"] > 0.075:
                        over.append(
                            f"{model} {group} {cell['broken'] / cell['fixed'] * 100:.1f} % ({cell['broken']:,} of {cell['fixed']:,})"
                        )
            line = next(
                line for line in self.part(language).splitlines() if line.startswith(f"{language}, kinds of edit")
            )
            self.assertIn("; ".join(over) if over else "none", line)

    def test_what_structure_fixing_reaches_is_the_files_counts(self):
        for code, language in (("es", "Spanish"), ("en", "English")):
            line = next(
                line
                for line in self.part(language).splitlines()
                if line.startswith(f"{language}, what structure fixing reaches")
            )
            for kind, table in self.data["by_language"][code]["by_component_kind"]["all"].items():
                if kind == "pair" or not table.get("none"):
                    continue  # `pair` is the ordinary word: the line is about the structure kinds
                d, t = (table["models"][m]["more"]["fixed"] for m in ("default", "thorough"))
                self.assertIn(
                    f"{kind}: {d:,} fixed with the default model and {t:,} with the thorough mode of {table['none']['wrong_as_read']:,} wrong words",
                    line,
                )

    def test_the_footnote_on_bands_and_on_the_100_percent_is_said_when_the_bands_add_up_to_more_documents(self):
        for code, language in (("es", "Spanish"), ("en", "English")):
            block = self.data["by_language"][code]
            total = next(cell["documents"] for groups in block["by_fix"].values() for cell in groups.values())
            bands = sum(table["documents"] for table in block["by_band"]["all"].values())
            text = self.part(language)
            if bands > total:
                self.assertIn(f"the band table adds up to {bands} for the {total} documents of the language", text)
                self.assertIn("`WER as read` is 100 % for merge, split, mixed and invented by definition", text)
            else:
                self.assertNotIn("the band table adds up to", text)
        changed = json.loads(json.dumps(self.data))
        bands = changed["by_language"]["es"]["by_band"]["all"]
        total = next(
            cell["documents"] for groups in changed["by_language"]["es"]["by_fix"].values() for cell in groups.values()
        )
        first = next(iter(bands))
        for label in bands:
            bands[label]["documents"] = 0
        bands[first]["documents"] = total  # one band holds every document: nothing to add up
        (self.tmp / NAME).write_text(json.dumps(changed), encoding="utf-8")
        self.assertNotIn(
            "the band table adds up to", "\n".join(self.script.t5_tables_lines(self.tmp)).split("English, `more`")[0]
        )

    def test_why_there_was_a_second_read_and_where_both_are_registered(self):
        first = self.lines[0]
        self.assertNotIn("Why a second read", first)  # no registers in this folder
        for name in ("test_reads_v5.json", "test_reads_v5en.json"):
            register = [
                {"purpose": "ocr"},
                {"purpose": "metrics"},
                {"purpose": "ocr"},
                {"purpose": "metrics", "reason": "second-read-tables"},
            ]
            if name.endswith("en.json"):
                register.append({"purpose": "metrics", "reason": "second-read-tables"})
            (self.tmp / name).write_text(json.dumps(register), encoding="utf-8")
        first = self.script.t5_tables_lines(self.tmp)[0]
        self.assertIn("Why a second read: the tool of the first read did not keep the tables fixed in advance", first)
        self.assertIn("no number was chosen after seeing one", first)
        self.assertIn(
            "`test_reads_v5.json` and `test_reads_v5en.json`: 2 and 3 metrics reads, the second with the reason second-read-tables",
            first,
        )

    def test_the_raw_ocr_wer_line_is_the_files(self):
        text = "\n".join(self.lines)
        for code, language in (("es", "Spanish"), ("en", "English")):
            raw = self.data["g31_m9"]["raw_ocr_wer_as_read"][code]
            self.assertIn(
                f"{language} pages: {raw['value'] * 100:.2f} % ({raw['low'] * 100:.2f} to {raw['high'] * 100:.2f})",
                text,
            )

    def test_a_number_changed_in_the_file_changes_exactly_that_cell_and_a_hand_edit_is_found(self):
        changed = json.loads(json.dumps(self.data))
        first = next(iter(changed["by_language"]["es"]["by_fix"]["thorough|more"].values()))
        first["fixed"] += 1000
        (self.tmp / NAME).write_text(json.dumps(changed), encoding="utf-8")
        new = self.script.t5_tables_lines(self.tmp)
        self.assertEqual(sum(a != b for a, b in zip(new, self.lines, strict=True)), 1)
        self.assertEqual(len(new), len(self.lines))
        self.assertIn(f"{first['fixed']:,}", "\n".join(new))
        known = allowed(self.data)  # the original file does not know the edited number
        self.assertIn(
            f"{first['fixed']:,}".replace(",", ""), {n.replace(",", "") for n in numbers_of(new) if n not in known}
        )


REAL = ROOT / "bench" / "results" / NAME


@unittest.skipUnless(REAL.is_file(), "the second read of T5 is not in this tree yet")
class RealTablesTest(TablesTest):
    """The same checks on the real file, once it is there (and the README must carry what it renders)."""

    def setUp(self):
        global FIXTURE
        self.fixture = FIXTURE
        FIXTURE = REAL
        try:
            super().setUp()
        finally:
            FIXTURE = self.fixture

    def test_the_two_reads_sentence_says_what_the_file_says(self):
        self.assertIn("added the tables below and nothing else", self.lines[0])
        self.assertTrue(self.data["first_read_reproduced"])
        self.assertIn("recomputed and compared number by number", self.lines[0])

    def test_the_measured_file_carries_the_tables(self):
        readme = (ROOT / "docs" / "MEASURED.md").read_text(encoding="utf-8")
        for line in self.script.t5_tables_lines():
            self.assertIn(line, readme)


if __name__ == "__main__":
    unittest.main()
