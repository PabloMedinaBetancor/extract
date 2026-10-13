import unittest
from types import SimpleNamespace

import numpy as np

from bench.guard import grid
from tests.test_guard_operating_ops import raw_of, table_of


def results(**spec):
    """`spec` maps 'LxT' to (fixed, se, logloss)."""
    out = {}
    for key, (fixed, se, loss) in spec.items():
        leaves, trees = key.split("x")
        out[(int(leaves), int(trees))] = {"fixed": fixed, "se": se, "logloss": loss}
    return out


class MetricTest(unittest.TestCase):
    def cells(self, safe, net):
        return {
            (15, 100): {"fixed": safe[0], "se": 5, "logloss": 0.3, "more_net": net[0], "more_net_se": 50},
            (63, 400): {"fixed": safe[1], "se": 5, "logloss": 0.2, "more_net": net[1], "more_net_se": 60},
        }

    def test_V5B33_one_cell_of_nine_that_applies_something_at_safe_is_luck_and_does_not_make_safe_the_metric(self):
        nine = {
            (15 * (1 + i % 3), 100 * (1 + i // 3)): {
                "fixed": 0,
                "se": 0,
                "logloss": 0.2,
                "more_net": 2500 + i,
                "more_net_se": 280,
            }
            for i in range(9)
        }
        nine[(15, 200)]["fixed"] = 300
        self.assertEqual(grid.by_metric(nine)[1], "more_net")
        for cell in list(nine)[:6]:
            nine[cell]["fixed"] = 100
        self.assertEqual(grid.by_metric(nine)[1], "safe_fixed")

    def test_V5B33_the_correct_fixes_at_safe_are_compared_when_any_cell_applies_any(self):
        compared, metric = grid.by_metric(self.cells((10, 20), (2500, 2600)))
        self.assertEqual(metric, "safe_fixed")
        self.assertEqual(compared[(15, 100)]["fixed"], 10)

    def test_V5B33_with_safe_off_in_every_cell_the_net_of_more_is_compared_with_its_own_se(self):
        compared, metric = grid.by_metric(self.cells((0, 0), (2500, 2600)))
        self.assertEqual(metric, "more_net")
        self.assertEqual((compared[(63, 400)]["fixed"], compared[(63, 400)]["se"]), (2600, 60))
        self.assertEqual(grid.choose_cell(compared)["cell"], (63, 400))  # 2500 >= 2600 - 60 is false: the best wins
        compared, _ = grid.by_metric(self.cells((0, 0), (2570, 2600)))
        self.assertEqual(grid.choose_cell(compared)["cell"], (15, 100))


class ChooseCellTest(unittest.TestCase):
    def test_B4_the_cheapest_cell_within_one_standard_error_of_the_best_wins(self):
        r = results(**{"15x100": (940, 20, 0.30), "31x200": (950, 20, 0.28), "63x400": (960, 20, 0.27)})
        self.assertEqual(grid.choose_cell(r)["cell"], (15, 100))  # 940 >= 960 - 20

    def test_B4_a_cell_just_outside_the_standard_error_does_not_qualify(self):
        r = results(**{"15x100": (939, 20, 0.30), "31x200": (950, 20, 0.28), "63x400": (960, 20, 0.27)})
        self.assertEqual(grid.choose_cell(r)["cell"], (31, 200))

    def test_B4_the_se_that_sets_the_bar_is_the_one_of_the_best_cell(self):
        r = results(**{"15x100": (900, 1, 0.3), "63x400": (960, 80, 0.27)})
        self.assertEqual(grid.choose_cell(r)["cell"], (15, 100))
        r = results(**{"15x100": (900, 1, 0.3), "63x400": (960, 10, 0.27)})
        self.assertEqual(grid.choose_cell(r)["cell"], (63, 400))

    def test_B4_a_cell_over_the_cost_cap_is_not_eligible_even_if_it_is_the_best(self):
        r = results(**{"15x100": (900, 10, 0.3), "63x400": (999, 10, 0.2), "31x200": (950, 10, 0.25)})
        out = grid.choose_cell(r, eligible={(15, 100), (31, 200)})
        self.assertEqual((out["cell"], out["best"]), ((31, 200), (31, 200)))
        self.assertEqual(grid.choose_cell(r, eligible=set())["cell"], None)

    def test_B4_cost_is_leaves_times_trees_so_a_wide_small_forest_can_be_cheaper_than_a_deep_one(self):
        self.assertLess(grid.cost_of((15, 400)), grid.cost_of((63, 100)))
        r = results(**{"15x400": (950, 20, 0.3), "63x100": (950, 20, 0.3)})
        self.assertEqual(grid.choose_cell(r)["cell"], (15, 400))

    def test_B4_the_best_of_two_equal_cells_is_the_cheaper_and_a_tie_in_fixes_does_not_break_the_choice(self):
        r = results(**{"31x200": (950, 0, 0.3), "63x200": (950, 0, 0.3)})
        out = grid.choose_cell(r)
        self.assertEqual((out["best"], out["cell"]), ((31, 200), (31, 200)))

    def test_B4_the_log_loss_breaks_a_tie_in_cost(self):
        # two cells of the same cost cannot exist in the grid, so the tie is built by hand
        r = {(10, 20): {"fixed": 900, "se": 5, "logloss": 0.4}, (20, 10): {"fixed": 900, "se": 5, "logloss": 0.3}}
        self.assertEqual(grid.choose_cell(r)["cell"], (20, 10))

    def test_B4_the_grid_is_the_nine_cells_of_the_spec(self):
        self.assertEqual(len(grid.CELLS), 9)
        self.assertEqual(set(grid.LEAVES), {15, 31, 63})
        self.assertEqual(set(grid.TREES), {100, 200, 400})

    def test_B4_the_log_loss_is_the_mean_of_the_two_terms(self):
        class Table:
            y = np.array([1, 0])

        self.assertAlmostEqual(grid.logloss(Table, np.array([0.5, 0.5])), float(np.log(2)))


class EvaluateTest(unittest.TestCase):
    def test_B4_every_cell_is_evaluated_from_its_own_scores_over_the_same_table(self):
        rng = np.random.default_rng(1)
        sites = [("text", int(rng.random() < 0.7), i // 3, (i,), f"d{i % 30}") for i in range(900)]
        table = table_of(sites)
        good = raw_of(table, rng, 0.02)
        poor = raw_of(table, rng, 2.0)
        run = SimpleNamespace(table=table, raw={grid.name_of(c): good if c == (31, 200) else poor for c in grid.CELLS})
        out = grid.evaluate(run, resamples=10)
        self.assertEqual(set(out), set(grid.CELLS))
        self.assertGreater(out[(31, 200)]["fixed"], out[(15, 100)]["fixed"])
        self.assertLess(out[(31, 200)]["logloss"], out[(15, 100)]["logloss"])
        self.assertEqual(grid.choose_cell(out)["cell"], (31, 200))


if __name__ == "__main__":
    unittest.main()
