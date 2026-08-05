import json
import unittest

from news_crop_benchmark.data import assign_group_split, build_verl_row, sample_id, training_sample_id


class DatasetConstructionTests(unittest.TestCase):
    def test_trace_id_produces_stable_sample_id(self):
        self.assertEqual(sample_id("trace-123"), sample_id("trace-123"))
        self.assertNotEqual(sample_id("trace-123"), sample_id("trace-124"))

    def test_group_split_is_stable_for_duplicate_image(self):
        first = assign_group_split("https://example.test/original.webp", seed=42)
        second = assign_group_split("https://example.test/original.webp", seed=42)

        self.assertEqual(first, second)

    def test_training_sample_id_changes_with_ratio(self):
        square = training_sample_id("https://example.test/original.webp", "Title", 1.0)
        landscape = training_sample_id("https://example.test/original.webp", "Title", 1.91)

        self.assertNotEqual(square, landscape)

    def test_split_rejects_invalid_fractions(self):
        with self.assertRaises(ValueError):
            assign_group_split("image", fractions=(0.8, 0.2, 0.2))

    def test_builds_verl_multimodal_row_without_prompt_leakage(self):
        row = build_verl_row(
            sample_identifier="sample-1",
            source_index=7,
            split="train",
            title="A news title",
            original_image_path="/shared/images/original.webp",
            image_width=2900,
            image_height=2900,
            target_ratio=1.59,
        )

        prompt = row["prompt"][0]["content"]
        self.assertIn("<image>", prompt)
        self.assertIn("Target aspect ratio (width / height): 1.59", prompt)
        self.assertIn("0 is the left edge and 1000 is the right edge", prompt)
        self.assertIn("0 is the top edge and 1000 is the bottom edge", prompt)
        self.assertIn("1000 is the full image", prompt)
        self.assertIn("Do not include explanations", prompt)
        self.assertEqual(row["images"], ["/shared/images/original.webp"])
        self.assertNotIn("known_bad_bbox", row["extra_info"])
        self.assertNotIn("known_bad_crop_path", row["extra_info"])
        self.assertNotIn("reason", row["extra_info"])
        self.assertEqual(json.loads(row["reward_model"]["ground_truth"])["target_ratio"], 1.59)

    def test_rejects_relative_image_path(self):
        with self.assertRaises(ValueError):
            build_verl_row(
                sample_identifier="sample-1",
                source_index=0,
                split="train",
                title="Title",
                original_image_path="images/original.webp",
                image_width=100,
                image_height=100,
                target_ratio=1.0,
            )


if __name__ == "__main__":
    unittest.main()