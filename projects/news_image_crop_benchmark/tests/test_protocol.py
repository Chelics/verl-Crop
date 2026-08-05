import unittest

from news_crop_benchmark.protocol import parse_crop_action


class CropProtocolTests(unittest.TestCase):
    def test_parses_crop_json_inside_response(self):
        action = parse_crop_action('Result: <crop>{"cx": 500, "cy": 625, "area": 420}</crop>')

        self.assertEqual(action.center_x, 500)
        self.assertEqual(action.center_y, 625)
        self.assertEqual(action.area, 420)

    def test_rejects_extra_fields(self):
        with self.assertRaises(ValueError):
            parse_crop_action('<crop>{"cx": 500, "cy": 500, "area": 500, "ratio": 1.59}</crop>')

    def test_rejects_out_of_range_action(self):
        with self.assertRaises(ValueError):
            parse_crop_action('<crop>{"cx": -1, "cy": 500, "area": 500}</crop>')


if __name__ == "__main__":
    unittest.main()
