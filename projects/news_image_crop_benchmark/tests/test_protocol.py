import unittest

from news_crop_benchmark.protocol import parse_crop_action, parse_crop_action_with_format


class CropProtocolTests(unittest.TestCase):
    def test_parses_crop_json_inside_response(self):
        action = parse_crop_action('Result: <crop>{"cx": 500, "cy": 625, "area": 420}</crop>')

        self.assertEqual(action.center_x, 500)
        self.assertEqual(action.center_y, 625)
        self.assertEqual(action.area, 420)

    def test_recovers_complete_json_missing_only_closing_tag(self):
        result = parse_crop_action_with_format('<crop>{"cx": 453, "cy": 120, "area": 292}')

        self.assertEqual(result.action.center_x, 453)
        self.assertFalse(result.strict_format)
        with self.assertRaises(ValueError):
            parse_crop_action('<crop>{"cx": 453, "cy": 120, "area": 292}')

    def test_does_not_recover_missing_closing_tag_with_trailing_text(self):
        with self.assertRaises(ValueError):
            parse_crop_action_with_format('<crop>{"cx": 453, "cy": 120, "area": 292} explanation')

    def test_rejects_extra_fields(self):
        with self.assertRaises(ValueError):
            parse_crop_action('<crop>{"cx": 500, "cy": 500, "area": 500, "ratio": 1.59}</crop>')

    def test_rejects_out_of_range_action(self):
        with self.assertRaises(ValueError):
            parse_crop_action('<crop>{"cx": -1, "cy": 500, "area": 500}</crop>')


if __name__ == "__main__":
    unittest.main()
