import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image


class _FakeClipScorer:
    def score(self, title, original, candidate):
        return 0.20, 0.30


class _FakeVLMScorer:
    def score(self, original, candidate, caption, headline, log_context=None):
        assert original.size == (100, 80)
        assert candidate.size == (56, 56)
        assert caption == "A caption"
        assert headline == "A title"
        assert log_context == {
            "evaluation_id": None,
            "sample_id": None,
            "target_ratio": 1.0,
            "action": {"cx": 500, "cy": 500, "area": 400},
        }
        return 0.8, 1.0


class VerlRewardTests(unittest.TestCase):
    def setUp(self):
        from importlib.util import module_from_spec, spec_from_file_location

        reward_path = Path(__file__).parents[1] / "rewards" / "crop_reward.py"
        spec = spec_from_file_location("test_crop_reward", reward_path)
        self.reward_module = module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(self.reward_module)

    def test_smoke_mode_accepts_valid_action_without_reading_image(self):
        result = self.reward_module.compute_score(
            data_source="news_image_crop",
            solution_str='<crop>{"cx":500,"cy":500,"area":400}</crop>',
            ground_truth=json.dumps({"image_width": 100, "image_height": 80, "target_ratio": 1.0}),
            extra_info={},
            reward_mode="smoke",
        )

        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["format_reward"], 1.0)
        self.assertEqual(result["proxy_enabled"], 0.0)

    def test_invalid_output_gets_negative_reward(self):
        result = self.reward_module.compute_score(
            data_source="news_image_crop",
            solution_str="not a crop",
            ground_truth=json.dumps({"image_width": 100, "image_height": 80, "target_ratio": 1.0}),
            reward_mode="smoke",
        )

        self.assertEqual(result["score"], -1.0)
        self.assertEqual(result["format_reward"], 0.0)

    def test_missing_closing_tag_is_scored_with_format_penalty(self):
        result = self.reward_module.compute_score(
            data_source="news_image_crop",
            solution_str='<crop>{"cx":500,"cy":500,"area":400}',
            ground_truth=json.dumps({"image_width": 100, "image_height": 80, "target_ratio": 1.0}),
            reward_mode="smoke",
        )

        self.assertGreater(result["score"], -1.0)
        self.assertLess(result["format_reward"], 1.0)

    def test_proxy_mode_scores_real_image_with_injected_clip_scorer(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (100, 80), color="white").save(image_path)
            with patch.object(self.reward_module, "get_clip_title_scorer", return_value=_FakeClipScorer()):
                result = self.reward_module.compute_score(
                    data_source="news_image_crop",
                    solution_str='<crop>{"cx":500,"cy":500,"area":400}</crop>',
                    ground_truth=json.dumps({"image_width": 100, "image_height": 80, "target_ratio": 1.0}),
                    extra_info={"title": "A title", "original_image_path": str(image_path)},
                    reward_mode="proxy",
                    clip_model_path="/fake/clip",
                )

        self.assertGreater(result["score"], 0.0)
        self.assertEqual(result["proxy_enabled"], 1.0)
        self.assertAlmostEqual(result["clip_similarity_delta"], 0.1)

    def test_proxy_mode_scores_recovered_action_before_format_penalty(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (100, 80), color="white").save(image_path)
            with patch.object(self.reward_module, "get_clip_title_scorer", return_value=_FakeClipScorer()):
                result = self.reward_module.compute_score(
                    data_source="news_image_crop",
                    solution_str='<crop>{"cx":500,"cy":500,"area":400}',
                    ground_truth=json.dumps({"image_width": 100, "image_height": 80, "target_ratio": 1.0}),
                    extra_info={"title": "A title", "original_image_path": str(image_path)},
                    reward_mode="proxy",
                    clip_model_path="/fake/clip",
                )

        self.assertGreater(result["score"], 0.0)
        self.assertEqual(result["format_reward"], 0.5)
        self.assertEqual(result["strict_format"], 0.0)
        self.assertEqual(result["proxy_enabled"], 1.0)

    def test_proxy_model_failure_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (100, 80), color="white").save(image_path)
            with patch.object(self.reward_module, "get_clip_title_scorer", side_effect=RuntimeError("load failed")):
                with self.assertRaises(RuntimeError):
                    self.reward_module.compute_score(
                        data_source="news_image_crop",
                        solution_str='<crop>{"cx":500,"cy":500,"area":400}</crop>',
                        ground_truth=json.dumps({"image_width": 100, "image_height": 80, "target_ratio": 1.0}),
                        extra_info={"title": "A title", "original_image_path": str(image_path)},
                        reward_mode="proxy",
                        clip_model_path="/fake/clip",
                    )

    def test_vlm_mode_scores_rendered_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (100, 80), color="white").save(image_path)
            with patch.object(self.reward_module, "get_crop_vlm_scorer", return_value=_FakeVLMScorer()):
                result = self.reward_module.compute_score(
                    data_source="news_image_crop",
                    solution_str='<crop>{"cx":500,"cy":500,"area":400}</crop>',
                    ground_truth=json.dumps({"image_width": 100, "image_height": 80, "target_ratio": 1.0}),
                    extra_info={
                        "title": "A title",
                        "caption": "A caption",
                        "original_image_path": str(image_path),
                    },
                    reward_mode="vlm",
                    vlm_prompt_path="/fake/prompt.txt",
                )

        self.assertAlmostEqual(result["score"], 0.8)
        self.assertEqual(result["vlm_label"], 1.0)
        self.assertEqual(result["vlm_enabled"], 1.0)
        self.assertEqual(result["proxy_enabled"], 0.0)


if __name__ == "__main__":
    unittest.main()
