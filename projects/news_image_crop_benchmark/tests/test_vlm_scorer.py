from types import SimpleNamespace

from PIL import Image

from news_crop_benchmark.vlm_scorer import CropVLMScorer, parse_label


class _FakeResponses:
    def __init__(self, output_text: str | None = None, error: Exception | None = None) -> None:
        self.output_text = output_text
        self.error = error
        self.requests = []

    def stream(self, **kwargs):
        self.requests.append(kwargs)
        return _FakeResponseStream(self.output_text, self.error)


class _FakeResponseStream:
    def __init__(self, output_text: str | None, error: Exception | None) -> None:
        self.output_text = output_text
        self.error = error

    def __enter__(self):
        if self.error is not None:
            raise self.error
        return self

    def __exit__(self, *_args) -> None:
        return None

    def get_final_response(self):
        return SimpleNamespace(id="response-123", output_text=self.output_text)


def _make_scorer(responses: _FakeResponses) -> CropVLMScorer:
    scorer = CropVLMScorer.__new__(CropVLMScorer)
    scorer.client = SimpleNamespace(responses=responses)
    scorer.model = "test-deployment"
    scorer.request_timeout = 1.0
    scorer.max_retries = 0
    scorer.retry_backoff = 1.0
    scorer.fallback_label = 5.0
    scorer.parse_fallback_label = 2.5
    scorer.eval_image_size = 32
    scorer.image_format = "JPEG"
    scorer.jpeg_quality = 70
    scorer.max_output_tokens = 128
    scorer.reasoning_effort = "low"
    scorer.output_verbosity = "low"
    scorer.response_log_path = None
    scorer.visual_log_dir = None
    scorer.visual_log_every = 0
    scorer._visual_log_counter = 0
    scorer.preprocess_mode = "letterbox"
    scorer.background_color = (255, 255, 255)
    scorer.rule_prompt = "Evaluate the crop."
    return scorer


def test_parse_label_prefers_json_schema_and_clamps_range():
    assert parse_label('{"evaluation":{"label":"1"}}') == 1.0
    assert parse_label('{"evaluation":{"label":8}}') == 5.0


def test_parse_label_recovers_label_from_malformed_json():
    assert parse_label('result: {"label": "3", trailing') == 3.0
    assert parse_label("no label", fallback_label=4.0) == 4.0


def test_score_maps_tier_to_reward_and_sends_two_images():
    responses = _FakeResponses('{"evaluation":{"label":"1"}}')
    scorer = _make_scorer(responses)

    reward, label = scorer.score(
        Image.new("RGB", (80, 60), color="white"),
        Image.new("RGB", (40, 30), color="black"),
        "A caption",
        "A headline",
    )

    assert reward == 0.8
    assert label == 1.0
    content = responses.requests[0]["input"][0]["content"]
    assert [item["type"] for item in content].count("input_image") == 2
    assert "A caption" in content[-1]["text"]
    assert "A headline" in content[-1]["text"]
    assert responses.requests[0]["reasoning"] == {"effort": "low"}
    assert responses.requests[0]["text"] == {"verbosity": "low"}


def test_score_returns_configured_fallback_after_request_failure():
    scorer = _make_scorer(_FakeResponses(error=RuntimeError("service unavailable")))

    reward, label = scorer.score(
        Image.new("RGB", (80, 60), color="white"),
        Image.new("RGB", (40, 30), color="black"),
        "",
        "A headline",
    )

    assert reward == 0.0
    assert label == 5.0


def test_score_logs_complete_response_without_image_payloads(tmp_path):
    output_text = '{"evaluation":{"label":"1","visual_reasoning":"Complete explanation"}}'
    scorer = _make_scorer(_FakeResponses(output_text))
    scorer.response_log_path = tmp_path / "responses.jsonl"

    scorer.score(
        Image.new("RGB", (80, 60), color="white"),
        Image.new("RGB", (40, 30), color="black"),
        "A caption",
        "A headline",
        log_context={"sample_id": "sample-1", "candidate": 2},
    )

    record = __import__("json").loads(scorer.response_log_path.read_text())
    assert record["model"] == "test-deployment"
    assert record["response_id"] == "response-123"
    assert record["output_text"] == output_text
    assert record["context"] == {"sample_id": "sample-1", "candidate": 2}
    assert record["label"] == 1.0
    assert record["reward"] == 0.8
    assert "data:image" not in scorer.response_log_path.read_text()


def test_score_logs_sampled_judge_images_and_latency(tmp_path):
    responses = _FakeResponses('{"evaluation":{"label":"1"}}')
    scorer = _make_scorer(responses)
    scorer.response_log_path = tmp_path / "responses.jsonl"
    scorer.visual_log_dir = tmp_path / "vlm_visuals"
    scorer.visual_log_every = 1

    scorer.score(
        Image.new("RGB", (80, 60), color="white"),
        Image.new("RGB", (40, 30), color="black"),
        "",
        "A headline",
        log_context={"sample_id": "sample-1"},
    )

    record = __import__("json").loads(scorer.response_log_path.read_text())
    assert record["attempt"] == 1
    assert record["request_latency_ms"] >= 0.0
    assert record["latency_ms"] >= record["request_latency_ms"]
    assert record["visual_artifacts"]["original"].startswith("vlm_visuals/sample-1_")
    assert record["visual_artifacts"]["candidate"].startswith("vlm_visuals/sample-1_")
    assert record["visual_artifacts"]["original"].endswith("_original.jpg")
    assert record["visual_artifacts"]["candidate"].endswith("_candidate.jpg")
    assert (tmp_path / record["visual_artifacts"]["original"]).is_file()
    assert (tmp_path / record["visual_artifacts"]["candidate"]).is_file()