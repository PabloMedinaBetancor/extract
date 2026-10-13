"""The OpenVINO per-shape cache is bounded (worker memory, R7) and the bound never reaches the results."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from extract.engine import ocr


class RuntimeCacheTest(unittest.TestCase):
    def config(self) -> dict:
        from rapidocr.inference_engine.openvino.device_config import CPUConfig

        ocr._patch_runtime_cache()
        return CPUConfig(OmegaConf.create({"inference_num_threads": 2, "num_streams": 1})).get_config()

    def test_capacity_is_set_and_keeps_the_other_settings(self) -> None:
        config = self.config()
        self.assertEqual(config["CPU_RUNTIME_CACHE_CAPACITY"], str(ocr.CPU_RUNTIME_CACHE))
        self.assertEqual(config["NUM_STREAMS"], "1")
        self.assertEqual(config["INFERENCE_NUM_THREADS"], "2")

    def test_default_is_small_but_not_zero(self) -> None:
        self.assertTrue(1 <= ocr.CPU_RUNTIME_CACHE <= 64)

    def test_zero_turns_it_off(self) -> None:
        saved = ocr.CPU_RUNTIME_CACHE
        ocr.CPU_RUNTIME_CACHE = None
        try:
            self.assertNotIn("CPU_RUNTIME_CACHE_CAPACITY", self.config())
        finally:
            ocr.CPU_RUNTIME_CACHE = saved

    def test_measured_worker_stays_under_the_recycle_limit_and_does_not_grow(self) -> None:
        results = json.loads((Path(__file__).parents[1] / "bench/results/worker-memory.json").read_text("utf-8"))
        bounded = results["study+cache32"]
        self.assertGreaterEqual(bounded["pages_read"], 120)
        self.assertLess(bounded["peak_rss_mb"], 1400)
        self.assertLess(abs(bounded["growth_mb_per_page_second_half"]), 0.5)
        self.assertGreater(
            results["study"]["peak_rss_mb"], 1400
        )  # the unbounded cache, same pages, the reason for the cap


if __name__ == "__main__":
    unittest.main()
