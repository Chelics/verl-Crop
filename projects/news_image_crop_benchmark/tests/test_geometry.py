import math
import unittest

from news_crop_benchmark.geometry import BBox, CropAction, action_to_bbox, bbox_iou, nearest_target_ratio


class CropGeometryTests(unittest.TestCase):
    def test_nearest_target_ratio_matches_served_crop(self):
        self.assertEqual(nearest_target_ratio(902, 566), 1.59)

    def test_action_produces_exact_ratio_inside_image(self):
        bbox = action_to_bbox(CropAction(center_x=500, center_y=500, area=500), 2900, 2900, 1.59)

        self.assertTrue(math.isclose(bbox.aspect_ratio, 1.59, rel_tol=1e-9))
        self.assertGreaterEqual(bbox.x1, 0)
        self.assertGreaterEqual(bbox.y1, 0)
        self.assertLessEqual(bbox.x2, 2900)
        self.assertLessEqual(bbox.y2, 2900)

    def test_action_near_edge_is_shifted_inside_image(self):
        bbox = action_to_bbox(CropAction(center_x=0, center_y=1000, area=300), 1600, 900, 1.91)

        self.assertTrue(math.isclose(bbox.x1, 0.0, abs_tol=1e-9))
        self.assertTrue(math.isclose(bbox.y2, 900.0, abs_tol=1e-9))
        self.assertTrue(math.isclose(bbox.aspect_ratio, 1.91, rel_tol=1e-9))

    def test_iou(self):
        left = BBox(0, 0, 100, 100)
        right = BBox(50, 50, 150, 150)

        self.assertTrue(math.isclose(bbox_iou(left, right), 2500 / 17500))

    def test_invalid_area_is_rejected(self):
        with self.assertRaises(ValueError):
            action_to_bbox(CropAction(center_x=500, center_y=500, area=0), 100, 100, 1.0)


if __name__ == "__main__":
    unittest.main()