"""bench/portability.py: the install time and the byte-for-byte comparison of a reading made on another machine."""

import json
import tempfile
import unittest
from pathlib import Path

from bench import portability

READING = {
    "number": 6,
    "status": "ocr",
    "words": [["p6w0", "Form", [1.0, 2.0, 3.0, 4.0], 0.99], ["p6w1", "990", [5.0, 6.0, 7.0, 8.0], 0.93]],
}


class InstallLogTest(unittest.TestCase):
    def test_the_real_time_of_the_time_command_with_a_comma_or_a_point(self):
        self.assertAlmostEqual(portability.install_seconds("x\nreal\t2m17,824s\nuser\t1m16,737s\n"), 137.824)
        self.assertAlmostEqual(portability.install_seconds("real 0m5.500s\n"), 5.5)
        self.assertAlmostEqual(portability.install_seconds("real 1m0,000s\nreal 3m1,000s\n"), 181.0)  # the last one
        self.assertIsNone(portability.install_seconds("Installed.\n"))


class CompareTest(unittest.TestCase):
    def setUp(self):
        self.data = Path(tempfile.mkdtemp())
        page = self.data / "pages" / "train" / "c0b715d2790"
        page.mkdir(parents=True)
        (page / "6.json").write_text(json.dumps({"ocr": {"engine": "e", "reading": READING}}))

    def other(self, reading):
        return {"engine": "e", "backend": "openvino", "pages": {"c0b715d2790-6": {"seconds": 11.7, "reading": reading}}}

    def test_the_same_reading_is_identical_whatever_the_order_of_its_keys(self):
        shuffled = dict(reversed(list(READING.items())))
        got = portability.compare(self.other(shuffled), self.data)
        self.assertTrue(got["all_identical"])
        page = got["pages"][0]
        self.assertEqual((page["set"], page["words"], page["seconds_other_machine"]), ("train", 2, 11.7))
        self.assertEqual(page["sha256_other_machine"], page["sha256_development"])
        self.assertEqual(len(page["sha256_other_machine"]), 64)

    def test_the_bookkeeping_of_the_second_reading_is_not_part_of_the_reading_and_nothing_else_is_ignored(self):
        with_report = {**READING, "reread_report": {"units_line": 9}}
        self.assertTrue(portability.compare(self.other(with_report), self.data)["all_identical"])
        extra = {**READING, "mean_conf": 0.5}
        self.assertFalse(portability.compare(self.other(extra), self.data)["all_identical"])

    def test_an_integer_and_the_same_float_are_one_value_and_a_last_digit_is_a_difference(self):
        ints = json.loads(json.dumps(READING).replace("1.0", "1"))
        self.assertTrue(portability.compare(self.other(ints), self.data)["all_identical"])
        nudged = json.loads(json.dumps(READING))
        nudged["words"][0][3] = 0.9900000000000001
        self.assertFalse(portability.compare(self.other(nudged), self.data)["all_identical"])

    def test_one_changed_digit_a_moved_box_or_a_missing_page_is_not_identical(self):
        for change in (
            lambda r: r["words"][1].__setitem__(3, 0.931),
            lambda r: r["words"][0].__setitem__(2, [1.0, 2.0, 3.0, 4.5]),
            lambda r: r["words"].pop(),
            lambda r: r.__setitem__("status", "native"),
        ):
            changed = json.loads(json.dumps(READING))
            change(changed)
            got = portability.compare(self.other(changed), self.data)
            self.assertFalse(got["all_identical"])
            self.assertNotEqual(got["pages"][0]["sha256_other_machine"], got["pages"][0]["sha256_development"])
        other = self.other(READING)
        other["pages"]["nope-1"] = other["pages"].pop("c0b715d2790-6")
        got = portability.compare(other, self.data)
        self.assertFalse(got["all_identical"])
        self.assertIsNone(got["pages"][0]["set"])

    def test_one_page_of_two_that_differs_makes_the_whole_comparison_fail(self):
        (self.data / "pages" / "train" / "c0b715d2790" / "7.json").write_text(
            json.dumps({"ocr": {"engine": "e", "reading": READING}})
        )
        changed = json.loads(json.dumps(READING))
        changed["words"][0][1] = "Farm"
        other = self.other(READING)
        other["pages"]["c0b715d2790-7"] = {"seconds": 1.0, "reading": changed}
        got = portability.compare(other, self.data)
        self.assertEqual([p["identical"] for p in got["pages"]], [True, False])
        self.assertFalse(got["all_identical"])

    def test_no_pages_is_not_a_pass(self):
        self.assertFalse(portability.compare({"engine": "e", "backend": "b", "pages": {}}, self.data)["all_identical"])

    def test_the_command_writes_the_file_and_its_exit_code_is_the_verdict(self):
        log = self.data / "time.log"
        log.write_text("real\t2m17,824s\n")
        reading = self.data / "other.json"
        reading.write_text(json.dumps(self.other(READING)))
        out = self.data / "portability.json"
        args = [
            "--install-log",
            str(log),
            "--other-machine-reading",
            str(reading),
            "--data",
            str(self.data),
            "--out",
            str(out),
        ]
        self.assertEqual(portability.main(args), 0)
        written = json.loads(out.read_text())
        self.assertAlmostEqual(written["install"]["seconds"], 137.824)
        self.assertTrue(written["parity"]["all_identical"])
        log.write_text("Installed.\n")
        self.assertEqual(portability.main(args), 1)


if __name__ == "__main__":
    unittest.main()
