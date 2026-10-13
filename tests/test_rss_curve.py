import unittest

from bench import rss_curve


def page(pid, rss):
    return {"code": "pool.page", "pid": pid, "rss_mb": rss, "seconds": 4.0}


class CurveTest(unittest.TestCase):
    def test_each_worker_gets_its_load_time_and_the_mean_and_maximum_of_every_ten_pages(self):
        entries = [{"code": "worker.ready", "pid": 1, "backend": "openvino", "load_seconds": 30.5}]
        entries += [page(1, 1000 + i) for i in range(12)]
        entries += [{"code": "worker.recycled", "pid": 1, "reason": "pages"}]
        entries += [{"code": "worker.ready", "pid": 2, "backend": "openvino", "load_seconds": 1.5}, page(2, 900)]
        first, second = rss_curve.curve(entries)["workers"]
        self.assertEqual((first["load_seconds"], first["pages"], first["left_because"]), (30.5, 12, "pages"))
        self.assertEqual(first["by_ten_pages"][0], {"pages": "1-10", "mean_mb": 1004, "max_mb": 1009})
        self.assertEqual(first["by_ten_pages"][1], {"pages": "11-12", "mean_mb": 1010, "max_mb": 1011})
        self.assertEqual((second["load_seconds"], second["left_because"]), (1.5, None))

    def test_workers_come_out_in_the_order_they_started(self):
        entries = [
            {"code": "worker.ready", "pid": 9, "backend": "openvino", "load_seconds": 1.0},
            {"code": "worker.ready", "pid": 3, "backend": "openvino", "load_seconds": 2.0},
        ]
        self.assertEqual([w["pid"] for w in rss_curve.curve(entries)["workers"]], [9, 3])


if __name__ == "__main__":
    unittest.main()
