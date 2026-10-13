"""Joint against single-language training: the rule that decides which chooser ships, and the number of each language."""

import unittest

from bench.guard import joint


def number(value, se, metric="more_net"):
    return {"metric": metric, "value": value, "se": se}


class ChooseTest(unittest.TestCase):
    def test_G45_the_joint_chooser_ships_unless_the_spanish_only_one_beats_it_by_more_than_one_standard_error(self):
        self.assertEqual(joint.choose_chooser(number(2700, 100), number(2650, 100))["ships"], "joint")
        self.assertEqual(joint.choose_chooser(number(2700, 100), number(2600, 100))["ships"], "joint")  # exactly one SE
        self.assertEqual(joint.choose_chooser(number(2700, 100), number(2599, 100))["ships"], "split")
        self.assertEqual(
            joint.choose_chooser(number(2500, 100), number(2700, 100))["ships"], "joint"
        )  # joint is better

    def test_G45_the_standard_error_is_the_one_of_the_spanish_only_number(self):
        self.assertEqual(joint.choose_chooser(number(2700, 10), number(2650, 500))["ships"], "split")
        self.assertEqual(joint.choose_chooser(number(2700, 500), number(2400, 10))["ships"], "joint")

    def test_G45_numbers_of_different_metrics_or_missing_are_not_compared(self):
        mixed = joint.choose_chooser(number(300, 20, "safe_fixed"), number(2600, 100))
        self.assertEqual(mixed["ships"], "joint")
        self.assertIn("different metrics", mixed["reason"])
        self.assertEqual(joint.choose_chooser(number(None, None), number(1, 1))["ships"], "joint")


def report(language, safe_applied, safe_fixed, more_net):
    return {
        "by_language": {
            language: {
                "safe": {"total": {"applied": safe_applied, "intervals": {"fixed": {"value": safe_fixed, "se": 5.0}}}},
                "more": {"total": {"applied": 900, "intervals": {"net": {"value": more_net, "se": 40.0}}}},
            }
        }
    }


class MetricTest(unittest.TestCase):
    def test_G45_a_language_is_judged_by_safe_when_it_applies_any_else_by_the_net_of_more(self):
        self.assertEqual(joint.metric_of(report("es", 301, 300, 2600), "es")["metric"], "safe_fixed")
        self.assertEqual(joint.metric_of(report("es", 301, 300, 2600), "es")["value"], 300)
        out = joint.metric_of(report("es", 0, 0, 2600), "es")
        self.assertEqual((out["metric"], out["value"], out["se"]), ("more_net", 2600, 40.0))

    def test_G45_a_language_with_no_block_has_no_number(self):
        self.assertEqual(joint.metric_of(report("es", 0, 0, 1), "en")["value"], None)


if __name__ == "__main__":
    unittest.main()
