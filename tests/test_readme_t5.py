"""The sealed-test block of the README: every number in it comes from a results file or from the packaged cost, and a
number changed by hand (in the block or in a file) is found."""

import importlib.util
import json
import re
import shutil
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results"
SOURCES = (
    "read-test-v5.json",
    "cells-paired.json",
    "cells-en-paired.json",
    "portability.json",
    "thorough-run-nocap.json",
    "thorough-run.json",
    "thorough-run-cap1700-loaded.json",
)
COST = ROOT / "src" / "extract" / "data" / "second_reading.json"
# numbers that are part of the sentences, not measurements: the 95 % level, "per 100", the 8 % page brake, the 0.0 of
# an interval, the 60 pilot pages of the cost run
PROSE = {
    "95",
    "100",
    "8",
    "60",
    "0",
    "0.0",
    "16",
    "40",
    "1,600",
}  # 16 lines, 40 zones, the 1,600 MB cap (a constant of the code)
NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?")


def load_script():
    spec = importlib.util.spec_from_file_location("api_docs_t5", ROOT / "scripts" / "api_docs.py")
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


def allowed(results: Path, cost_file: Path) -> set[str]:
    """Every way the block writes a number of the files: as is, as a percentage, signed, with thousands, to 0-3 digits."""
    values = []
    for name in SOURCES:
        values += list(leaves(json.loads((results / name).read_text(encoding="utf-8"))))
    values += list(leaves(json.loads(cost_file.read_text(encoding="utf-8"))["thorough"]))
    out = set(PROSE)
    install = json.loads((results / "portability.json").read_text(encoding="utf-8"))["install"]["seconds"]
    out |= {str(int(install // 60)), str(round(install % 60))}  # written as `N min S s`
    from bench.guard import operating

    out.add(f"{(operating.MORE_BROKEN_SHARE + 0.025) * 100:.1f}")  # the broken-share floor on T5
    data = json.loads((results / "read-test-v5.json").read_text(encoding="utf-8"))
    for block in data["by_language"].values():  # broken per fixed word of each slice, model and mode
        for part in block.values():
            for model in part.get("models", {}).values() if isinstance(part, dict) else ():
                for cell in model.values():
                    if cell["fixed"]:
                        out.add(f"{cell['broken'] / cell['fixed'] * 100:.2f}")
    for value in values:
        for scale in (1, 100):
            scaled = value * scale
            for digits in range(4):
                for text in (f"{scaled:.{digits}f}", f"{scaled:+.{digits}f}", f"{abs(scaled):.{digits}f}"):
                    out.add(text)
            out.add(f"{round(abs(scaled)):,}")
            out.add(f"{round(scaled):,}")
    return out


def numbers_of(lines):
    return {t.strip(",") for line in lines for t in NUMBER.findall(line)}


class T5BlockTest(unittest.TestCase):
    def setUp(self):
        self.script = load_script()
        self.lines = self.script.t5_lines()

    def test_the_block_exists_and_covers_both_languages_apart(self):
        text = "\n".join(self.lines)
        self.assertGreater(len(self.lines), 10)
        for needle in (
            "Spanish, public-administration pages",
            "English, public-administration pages",
            "thorough minus default",
            "`safe` applied 0",
        ):
            self.assertIn(needle, text)
        self.assertNotIn("Spanish and English together", text)

    def test_every_number_of_the_block_is_in_a_results_file_or_the_packaged_cost(self):
        known = allowed(RESULTS, COST)
        unknown = sorted(n for n in numbers_of(self.lines) if n not in known and n.lstrip("+-") not in known)
        self.assertEqual(unknown, [])

    def test_a_number_edited_by_hand_in_the_block_is_found(self):
        edited = [self.lines[1].replace("995", "996")] + self.lines[2:]
        self.assertNotEqual(edited[0], self.lines[1])
        known = allowed(RESULTS, COST)
        self.assertIn("996", sorted(n for n in numbers_of(edited) if n not in known))

    def test_a_number_changed_in_a_results_file_changes_the_block_and_the_old_block_no_longer_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp)
            for name in SOURCES:
                shutil.copy(RESULTS / name, copy / name)
            data = json.loads((copy / "read-test-v5.json").read_text(encoding="utf-8"))
            data["by_language"]["es"]["pages"]["models"]["default"]["more"]["fixed"] += 1
            data["by_language"]["en"]["pages"]["models"]["thorough"]["more"]["broken"] += 1
            data["by_language"]["es"]["n10"]["default"]["by_rule"]["all"]["suggestions"] = 3
            (copy / "read-test-v5.json").write_text(json.dumps(data), encoding="utf-8")
            changed = self.script.t5_lines(copy, COST)
        self.assertNotEqual(changed, self.lines)
        self.assertEqual(sum(a != b for a, b in zip(changed, self.lines, strict=True)), 3)
        self.assertIn("made 3 suggestions", "\n".join(changed))
        self.assertIn("996", "\n".join(changed))

    def test_each_number_stands_where_the_file_has_it_for_that_language_slice_and_model(self):
        """The numbers exist in the files; here each one is also in the right place: `fixed`, `broken` and `applied` of
        each model, the paired net, and the safe and invoice counts, read again by their own path in the file."""
        data = json.loads((RESULTS / "read-test-v5.json").read_text(encoding="utf-8"))
        text = {
            line.split(":")[0]: line
            for line in self.lines
            if line.startswith("- Spanish,") or line.startswith("- English,")
        }
        for language, name in (("es", "Spanish"), ("en", "English")):
            block = data["by_language"][language]
            for key, label in (
                ("pages", "public-administration pages"),
                ("all", "all pages of the test"),
                ("invoices", "invoice pages"),
                ("unseen", "unseen-damage pages"),
            ):
                if key not in block:
                    continue
                line = text[f"- {name}, {label}"]
                head, _, thorough = line.partition("thorough mode:")
                for part, model in ((head, "default"), (thorough, "thorough")):
                    more = block[key]["models"][model]["more"]
                    self.assertIn(
                        f"fixed {more['fixed']:,}, broken {more['broken']:,}, applied {more['applied']:,}", part
                    )
                paired = block[key]["paired_thorough_minus_default"]["more"]["net_fixed_per_100_wrong_difference"]
                self.assertIn(f"net {paired['value']:+.2f} ({paired['low']:+.2f} to {paired['high']:+.2f})", line)
                self.assertIn(f"{block[key]['documents']} documents, {block[key]['pages']} pages", line)
            net = block["pages"]["models"]["thorough"]["more"]["net_fixed_per_100_wrong"]
            half = (net["high"] - net["low"]) / 2
            self.assertIn(
                f"about ±{half / net['value'] * 100:.0f} % of its value",
                text[f"- {name}"] if f"- {name}" in text else "\n".join(self.lines),
            )
            never = text[f"- {name}, what never changed"]
            n6 = block["extraction"]["n6"]
            self.assertIn(
                f"{n6['checked']} numbers of {n6['invoices']} invoices came out `checked` and {n6['false_checked']}",
                never,
            )

    def test_the_floor_of_more_is_g22_and_says_plainly_who_misses_it(self):
        """The `more` rule on T5: the WER interval of `more` below zero and the broken words at most 7.5 % of the fixed ones (the
        5 % of `bench.guard.operating` plus the 2.5 points for the shift); the line says meets or misses by that rule."""
        from bench.guard import operating

        floor = operating.MORE_BROKEN_SHARE + 0.025
        self.assertAlmostEqual(floor, 0.075)
        data = json.loads((RESULTS / "read-test-v5.json").read_text(encoding="utf-8"))
        for language, name in (("es", "Spanish"), ("en", "English")):
            line = next(line for line in self.lines if line.startswith(f"- {name}, floor of `more`"))
            for model, label in (("default", "default model"), ("thorough", "thorough mode")):
                more = data["by_language"][language]["all"]["models"][model]["more"]
                share = more["broken"] / more["fixed"]
                verdict = "meets" if share <= floor and more["d_wer"]["high"] < 0 else "misses"
                self.assertIn(f"{label} {share * 100:.2f} % ({more['broken']} of {more['fixed']}) {verdict} it", line)
        english = next(line for line in self.lines if line.startswith("- English, floor of"))
        self.assertIn("default model 7.62 % (38 of 499) misses it", english)  # the verdict of the English table
        self.assertIn("thorough mode 6.14 % (36 of 586) meets it", english)

    def test_a_wer_interval_that_reaches_zero_makes_the_floor_missed_even_with_few_broken_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp)
            for name in SOURCES:
                shutil.copy(RESULTS / name, copy / name)
            data = json.loads((copy / "read-test-v5.json").read_text(encoding="utf-8"))
            data["by_language"]["es"]["all"]["models"]["default"]["more"]["d_wer"]["high"] = 0.001
            (copy / "read-test-v5.json").write_text(json.dumps(data), encoding="utf-8")
            changed = self.script.t5_lines(copy, COST)
        line = next(line for line in changed if line.startswith("- Spanish, floor of"))
        self.assertIn("default model 4.99 % (57 of 1143) misses it; thorough mode 5.16 % (64 of 1241) meets it", line)

    def test_the_portability_line_is_the_file_and_says_identical_only_if_every_page_is(self):
        data = json.loads((RESULTS / "portability.json").read_text(encoding="utf-8"))
        line = self.script.portability_lines()[0]
        seconds = data["install"]["seconds"]
        self.assertIn(f"took {int(seconds // 60)} min {round(seconds % 60)} s", line)
        self.assertIn(f"first reading of {len(data['parity']['pages'])} development pages", line)
        self.assertIn("(identical;" if data["parity"]["all_identical"] else "(NOT identical;", line)
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp)
            for name in SOURCES:
                shutil.copy(RESULTS / name, copy / name)
            data["parity"]["pages"][0]["identical"] = False
            data["parity"]["all_identical"] = False
            (copy / "portability.json").write_text(json.dumps(data), encoding="utf-8")
            changed = self.script.portability_lines(copy)[0]
        self.assertIn("(NOT identical;", changed)
        self.assertIn(f"stored on {len(data['parity']['pages']) - 1} of {len(data['parity']['pages'])} pages", changed)

    def test_the_pages_are_counted_by_language_and_the_unreadable_ones_are_said_where_they_are(self):
        """The Spanish set also holds the English invoices: by language it is the `all` slice of each, not `read_pages`."""
        data = json.loads((RESULTS / "read-test-v5.json").read_text(encoding="utf-8"))
        es, en = (data["by_language"][lang]["all"]["pages"] for lang in ("es", "en"))
        self.assertNotEqual(
            (es, en), (data["read_pages"]["es"], data["read_pages"]["en"])
        )  # the sets are not the languages
        line = next(line for line in self.lines if line.startswith("- T5 read"))
        self.assertIn(f"T5 read {es + en} pages ({es} Spanish, {en} English;", line)
        unreadable = data["by_language"]["en"]["pages"]["unreadable_pages"]
        slice_line = next(line for line in self.lines if line.startswith("- English, public-administration pages"))
        self.assertIn(
            f"{data['by_language']['en']['pages']['pages']} pages ({unreadable} unreadable, counted apart)", slice_line
        )
        self.assertNotIn(
            "unreadable", next(line for line in self.lines if line.startswith("- Spanish, public-administration pages"))
        )

    def test_the_sums_over_slices_and_models_are_the_sums_of_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            copy = Path(tmp)
            for name in SOURCES:
                shutil.copy(RESULTS / name, copy / name)
            data = json.loads((copy / "read-test-v5.json").read_text(encoding="utf-8"))
            es = data["by_language"]["es"]
            es["pages"]["models"]["default"]["safe"]["applied"] = 2
            es["invoices"]["models"]["thorough"]["safe"]["applied"] = 5
            es["all"]["models"]["default"]["safe"]["applied"] = 11  # `all` already holds the other slices
            es["all"]["models"]["thorough"]["safe"]["applied"] = 4
            es["n10"]["default"]["by_rule"]["all"]["suggestions"] = 3
            es["n10"]["thorough"]["by_rule"]["all"]["suggestions"] = 4
            (copy / "read-test-v5.json").write_text(json.dumps(data), encoding="utf-8")
            changed = self.script.t5_lines(copy, COST)
        never = next(line for line in changed if line.startswith("- Spanish, what never changed"))
        self.assertIn("`safe` applied 15 fixes", never)
        self.assertIn("made 7 suggestions", never)

    def test_the_cost_in_the_block_is_the_packaged_one(self):
        block = json.loads(COST.read_text(encoding="utf-8"))["thorough"]
        text = "\n".join(self.lines)
        self.assertIn(f"{block['seconds_per_page']} s a page", text)
        self.assertIn(f"{block['reader_ram_mb']:,} MB", text)

    def test_the_measured_file_carries_the_block(self):
        readme = (ROOT / "docs" / "MEASURED.md").read_text(encoding="utf-8")
        for line in self.lines:
            self.assertIn(line, readme)


if __name__ == "__main__":
    unittest.main()
