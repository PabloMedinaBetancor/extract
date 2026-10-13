import unittest

from bench.endurance import evaluate


def log(per_page, workers=4, pages=480, rss=1200, backend="openvino"):
    """A job log of `pages` pages over `workers` workers; `per_page(n)` gives the seconds of the n-th page."""
    entries = [{"code": "worker.ready", "pid": pid, "backend": backend} for pid in range(workers)]
    for n in range(pages):
        entries.append({"code": "pool.page", "pid": n // 120, "seconds": per_page(n), "rss_mb": rss})
    entries += [{"code": "worker.recycled", "pid": pid} for pid in range(workers - 1)]
    return entries


class EnduranceTest(unittest.TestCase):
    def test_O9_a_steady_run_passes(self):
        result = evaluate(log(lambda n: 5.0))
        self.assertTrue(result["passed"])
        self.assertEqual(result["slowdown"], 1.0)

    def test_O9_a_run_that_slows_down_fails(self):
        result = evaluate(log(lambda n: 5.0 + n / 40))
        self.assertGreater(result["slowdown"], 1.15)
        self.assertFalse(result["passed"])

    def test_O9_a_worker_over_its_memory_limit_fails(self):
        self.assertFalse(evaluate(log(lambda n: 5.0, rss=1700))["passed"])
        self.assertTrue(evaluate(log(lambda n: 5.0, rss=850, backend="onnxruntime"))["passed"])
        self.assertFalse(evaluate(log(lambda n: 5.0, rss=950, backend="onnxruntime"))["passed"])

    def test_O9_a_short_run_is_not_enough(self):
        self.assertFalse(evaluate(log(lambda n: 5.0, pages=300, workers=3))["passed"])

    def test_O9_the_warm_up_pages_of_each_worker_are_left_out(self):
        result = evaluate(log(lambda n: 30.0 if n % 120 < 5 else 5.0))
        self.assertEqual(result["slowdown"], 1.0)

    def test_O9_fewer_than_three_recycles_is_not_enough(self):
        entries = [e for e in log(lambda n: 5.0) if e["code"] != "worker.recycled"]
        self.assertFalse(evaluate(entries)["passed"])


if __name__ == "__main__":
    unittest.main()
