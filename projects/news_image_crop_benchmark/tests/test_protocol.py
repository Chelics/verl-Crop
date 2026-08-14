import unittest

from news_crop_benchmark.protocol import (
    parse_crop_action,
    parse_crop_action_with_format,
    parse_layout_action,
    parse_mode_decision,
    parse_percent_crop_action,
)


class CropProtocolTests(unittest.TestCase):
    def test_parses_exact_mode_decision(self):
        self.assertEqual(parse_mode_decision('{"mode":"crop"}').mode, "crop")
        self.assertEqual(parse_mode_decision('\n{"mode":"pad"}\n').mode, "pad")

    def test_rejects_non_strict_or_invalid_mode_decision(self):
        invalid_responses = (
            'Result: {"mode":"crop"}',
            '```json\n{"mode":"pad"}\n```',
            '{"mode":"fit"}',
            '{"mode":"crop","confidence":"high"}',
            '[{"mode":"crop"}]',
        )
        for response in invalid_responses:
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_mode_decision(response)

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

    def test_rejects_non_integer_action(self):
        with self.assertRaisesRegex(ValueError, "must be integers"):
            parse_crop_action('<crop>{"cx": 500.1, "cy": 500, "area": 500}</crop>')

    def test_parses_exact_percentage_json_and_converts_to_internal_units(self):
        result = parse_percent_crop_action('{"cx_pct": 45, "cy_pct": 60, "area_pct": 25}')

        self.assertEqual(result.action.center_x, 450)
        self.assertEqual(result.action.center_y, 600)
        self.assertEqual(result.action.area, 250)
        self.assertTrue(result.strict_format)

    def test_rejects_percentage_json_with_extra_text_or_fields(self):
        with self.assertRaises(ValueError):
            parse_percent_crop_action('Result: {"cx_pct": 45, "cy_pct": 60, "area_pct": 25}')
        with self.assertRaises(ValueError):
            parse_percent_crop_action(
                '{"cx_pct": 45, "cy_pct": 60, "area_pct": 25, "ratio": 1.59}'
            )

    def test_recovers_single_json_code_block_without_extra_text(self):
        result = parse_percent_crop_action(
            '```json\n{"cx_pct": 45, "cy_pct": 60, "area_pct": 25}\n```'
        )

        self.assertEqual(result.action.center_x, 450)
        self.assertEqual(result.action.center_y, 600)
        self.assertEqual(result.action.area, 250)
        self.assertFalse(result.strict_format)

    def test_rejects_json_code_block_with_surrounding_explanation(self):
        with self.assertRaises(ValueError):
            parse_percent_crop_action(
                'Here is the crop:\n```json\n'
                '{"cx_pct": 45, "cy_pct": 60, "area_pct": 25}\n```'
            )

    def test_rejects_out_of_range_or_non_integer_percentage_values(self):
        with self.assertRaisesRegex(ValueError, "must be in"):
            parse_percent_crop_action('{"cx_pct": 101, "cy_pct": 60, "area_pct": 25}')
        with self.assertRaisesRegex(ValueError, "must be integers"):
            parse_percent_crop_action('{"cx_pct": 45.5, "cy_pct": 60, "area_pct": 25}')

    def test_parses_unified_layout_actions(self):
        result = parse_layout_action(
            '{"operation":"crop_pad","x1_pct":3,"y1_pct":12,"x2_pct":97,"y2_pct":88}'
        )

        self.assertEqual(result.action.operation, "crop_pad")
        self.assertEqual(result.action.x1_pct, 3)
        self.assertTrue(result.strict_format)

    def test_layout_action_recovers_one_json_code_block(self):
        result = parse_layout_action(
            '```json\n{"operation":"crop","x1_pct":0,"y1_pct":10,"x2_pct":100,"y2_pct":90}\n```'
        )

        self.assertEqual(result.action.operation, "crop")
        self.assertFalse(result.strict_format)

    def test_rejects_invalid_layout_actions(self):
        invalid_responses = (
            '{"operation":"pad","x1_pct":0,"y1_pct":1,"x2_pct":100,"y2_pct":100}',
            '{"operation":"crop_pad","x1_pct":30,"y1_pct":0,"x2_pct":20,"y2_pct":100}',
            '{"operation":"fit","x1_pct":0,"y1_pct":0,"x2_pct":100,"y2_pct":100}',
            '{"operation":"crop","x1_pct":0.5,"y1_pct":0,"x2_pct":100,"y2_pct":100}',
        )
        for response in invalid_responses:
            with self.subTest(response=response), self.assertRaises(ValueError):
                parse_layout_action(response)


if __name__ == "__main__":
    unittest.main()
