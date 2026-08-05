import unittest

from news_crop_benchmark.reward import RewardComponents, combine_proxy_reward


class ProxyRewardTests(unittest.TestCase):
    def test_invalid_output_gets_fixed_penalty(self):
        self.assertEqual(combine_proxy_reward(RewardComponents(valid=False)), -1.0)

    def test_combines_positive_metrics(self):
        components = RewardComponents(
            valid=True,
            title_relevance=1.0,
            saliency=1.0,
            composition=1.0,
            integrity=1.0,
            area=1.0,
        )

        self.assertAlmostEqual(combine_proxy_reward(components), 1.0)

    def test_rejects_unnormalized_metric(self):
        with self.assertRaises(ValueError):
            combine_proxy_reward(RewardComponents(valid=True, title_relevance=1.1))


if __name__ == "__main__":
    unittest.main()