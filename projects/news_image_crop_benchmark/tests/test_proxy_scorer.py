import math
import unittest

import numpy as np
from PIL import Image

from news_crop_benchmark.geometry import BBox
from news_crop_benchmark.proxy_scorer import (
    compute_visual_proxy_metrics,
    crop_area_score,
    crop_image,
    relative_title_relevance,
)


class ProxyScorerTests(unittest.TestCase):
    def test_crop_image_uses_bbox(self):
        image = Image.new("RGB", (100, 80), color="white")
        crop = crop_image(image, BBox(10, 20, 70, 60))

        self.assertEqual(crop.size, (60, 40))

    def test_visual_metrics_are_normalized(self):
        gradient = np.tile(np.arange(100, dtype=np.uint8), (80, 1))
        image = Image.fromarray(gradient, mode="L").convert("RGB")
        metrics = compute_visual_proxy_metrics(image, BBox(10, 10, 90, 70), maximum_side=100)

        for value in (metrics.saliency, metrics.composition, metrics.integrity, metrics.area):
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 1.0)
        self.assertTrue(math.isclose(metrics.area_fraction, 0.6))

    def test_relative_title_relevance_uses_original_as_baseline(self):
        self.assertAlmostEqual(relative_title_relevance(0.2, 0.2), 0.5)
        self.assertGreater(relative_title_relevance(0.2, 0.3), 0.5)
        self.assertLess(relative_title_relevance(0.3, 0.2), 0.5)

    def test_area_score_penalizes_tiny_and_full_image_crops(self):
        self.assertEqual(crop_area_score(0.01), 0.0)
        self.assertEqual(crop_area_score(0.40), 1.0)
        self.assertEqual(crop_area_score(1.0), 0.0)


if __name__ == "__main__":
    unittest.main()
