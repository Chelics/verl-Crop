import importlib.util
import json
import sys
import types
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def load_evaluator_module():
    script_path = Path(__file__).parents[1] / "scripts" / "evaluate_image_once_vlm.py"
    spec = importlib.util.spec_from_file_location("test_evaluate_image_once_vlm", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_webp_payload(color: str = "white") -> bytes:
    output = BytesIO()
    Image.new("RGB", (80, 60), color=color).save(output, format="WEBP", lossless=True)
    return output.getvalue()


def make_args(max_attempts: int = 3):
    return SimpleNamespace(
        image_max_pixels=1048576,
        image_min_pixels=65536,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        max_model_len=2176,
        max_num_seqs=8,
        max_attempts=max_attempts,
        prompt_batch_size=8,
        temperature=0.7,
        top_p=0.95,
        max_tokens=128,
        seed=42,
    )


def test_raw_row_expands_to_four_ratio_tasks_without_using_reference_crop(tmp_path):
    module = load_evaluator_module()
    payload = make_webp_payload()
    with Image.open(BytesIO(payload)) as image:
        image_id = module.normalized_pixel_hash(image)
    data_path = tmp_path / "image_once_test.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": image_id,
                    "original_image": payload,
                    "cropped_image": b"ignored",
                    "title": "  A   news title  ",
                    "ImageCaption": "A caption",
                    "Reason": "must not be used",
                }
            ]
        ),
        data_path,
    )

    policy_prompt_template = (
        "<image>\nHeadline: {title}\nRatio: {target_ratio}\n"
        'Return <crop>{"cx": CX, "cy": CY, "area": AREA}</crop>.'
    )
    tasks, manifest = module.load_and_materialize_tasks(
        data_path,
        tmp_path / "output",
        policy_prompt_template=policy_prompt_template,
    )

    assert len(manifest) == 1
    assert [task["target_ratio"] for task in tasks] == list(module.TARGET_RATIOS)
    assert all(task["title"] == "A news title" for task in tasks)
    assert tasks[0]["prompt"].startswith("<image>\nHeadline: A news title\nRatio: 1")
    assert all("must not be used" not in json.dumps(task) for task in tasks)
    assert all(Path(task["image_path"]).is_file() for task in tasks)


def test_generation_retries_invalid_output_with_new_seed_and_keeps_history(tmp_path):
    module = load_evaluator_module()
    calls = []

    class FakeProcessor:
        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"][1]["text"]

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(_model_path, local_files_only):
            assert local_files_only
            return FakeProcessor()

    class FakeSamplingParams:
        def __init__(self, **kwargs):
            self.seed = kwargs["seed"]

    class FakeLLM:
        def __init__(self, **_kwargs):
            pass

        def generate(self, requests, sampling_params):
            calls.append(sampling_params.seed)
            response = "invalid" if len(calls) == 1 else '<crop>{"cx":500,"cy":500,"area":400}</crop>'
            return [SimpleNamespace(outputs=[SimpleNamespace(text=response)]) for _ in requests]

    image_path = tmp_path / "image.webp"
    image_path.write_bytes(make_webp_payload())
    task = {
        "task_id": "image__ratio_1",
        "prompt": "<image>\nCrop this image",
        "image_path": str(image_path),
    }
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams

    with patch.dict(sys.modules, {"transformers": fake_transformers, "vllm": fake_vllm}):
        module.run_generation_worker(
            rank=0,
            gpu_devices=["0"],
            tasks=[task],
            output_dir=tmp_path,
            model_path=tmp_path / "model",
            args=make_args(),
        )

    progress = json.loads(module.generation_progress_path(tmp_path, task["task_id"]).read_text())
    assert calls == [42, 43]
    assert progress["status"] == "valid"
    assert [attempt["valid"] for attempt in progress["attempts"]] == [False, True]
    assert progress["attempts"][0]["response"] == "invalid"


def test_generation_marks_retry_exhausted_after_configured_attempt_limit(tmp_path):
    module = load_evaluator_module()

    class FakeProcessor:
        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"][1]["text"]

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(_model_path, local_files_only):
            return FakeProcessor()

    class FakeSamplingParams:
        def __init__(self, **_kwargs):
            pass

    class FakeLLM:
        def __init__(self, **_kwargs):
            pass

        def generate(self, requests, sampling_params):
            return [SimpleNamespace(outputs=[SimpleNamespace(text="invalid")]) for _ in requests]

    image_path = tmp_path / "image.webp"
    image_path.write_bytes(make_webp_payload())
    task = {
        "task_id": "image__ratio_1",
        "prompt": "<image>\nCrop this image",
        "image_path": str(image_path),
    }
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams

    with patch.dict(sys.modules, {"transformers": fake_transformers, "vllm": fake_vllm}):
        module.run_generation_worker(
            rank=0,
            gpu_devices=["0"],
            tasks=[task],
            output_dir=tmp_path,
            model_path=tmp_path / "model",
            args=make_args(max_attempts=2),
        )

    progress = json.loads(module.generation_progress_path(tmp_path, task["task_id"]).read_text())
    assert progress["status"] == "retry_exhausted"
    assert len(progress["attempts"]) == 2
    assert not any(attempt["valid"] for attempt in progress["attempts"])


def test_summary_does_not_count_request_failure_fallback_as_tier_five():
    module = load_evaluator_module()

    def detail(judge_status, label, generation_status="valid"):
        return {
            "generation_status": generation_status,
            "had_invalid_output": False,
            "invalid_attempt_count": 0,
            "total_attempt_count": 1,
            "strict_format": generation_status == "valid",
            "judge_status": judge_status,
            "judge_label": label,
            "judge_reward": 1.0 if label == 0 else 0.0,
            "judge_rules": ["T0.1"] if judge_status == "completed" else [],
            "judge_latency_ms": 10.0,
            "action_cx": 500.0,
            "action_cy": 500.0,
            "action_area": 400.0,
        }

    summary = module.summarize_subset(
        [
            detail("completed", 0.0),
            detail("failed", 5.0),
            detail("parse_fallback", 2.5),
            detail("not_run", None, generation_status="retry_exhausted"),
        ]
    )

    assert summary["tier_counts"] == {"0": 1}
    assert summary["judge_completed_count"] == 1
    assert summary["judge_failed_count"] == 1
    assert summary["judge_parse_fallback_count"] == 1
    assert summary["retry_exhausted_count"] == 1


def test_judge_pipeline_persists_crop_scoring_tables_and_report(tmp_path):
    module = load_evaluator_module()
    image_path = tmp_path / "source.webp"
    image_path.write_bytes(make_webp_payload())
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Judge this crop", encoding="utf-8")
    task = {
        "task_id": "image__ratio_1.59",
        "source_index": 0,
        "image_id": "image",
        "title": "A title",
        "caption": "A caption",
        "target_ratio": 1.59,
        "image_width": 80,
        "image_height": 60,
        "image_path": str(image_path),
        "prompt": "<image>\nCrop this image",
        "original_render_path": "renders/originals/image.jpg",
    }
    write_progress = {
        "task_id": task["task_id"],
        "rank": 0,
        "status": "valid",
        "attempts": [
            {
                "attempt": 1,
                "seed": 42,
                "response": '<crop>{"cx":500,"cy":500,"area":400}</crop>',
                "valid": True,
                "strict_format": True,
                "parse_error": None,
            }
        ],
        "action": {"cx": 500.0, "cy": 500.0, "area": 400.0},
    }
    module.write_json_atomic(
        module.generation_progress_path(tmp_path, task["task_id"]),
        write_progress,
    )

    class FakeScorer:
        def __init__(self, path):
            assert path == str(prompt_path)

        def score_detailed(self, original, candidate, caption, headline, log_context):
            assert original.size == (80, 60)
            assert round(candidate.width / candidate.height, 1) == 1.6
            assert caption == "A caption"
            assert headline == "A title"
            assert log_context["target_ratio"] == 1.59
            return SimpleNamespace(
                reward=0.8,
                label=1.0,
                status="completed",
                output_text=(
                    '{"evaluation":{"label":"1","tier_name":"Negligible",'
                    '"rules":["T1.2"],"confidence_score":"high"}}'
                ),
                response_id="response-1",
                attempt_count=1,
                latency_ms=12.5,
                error_type=None,
            )

    with patch.object(module, "CropVLMScorer", FakeScorer):
        module.run_judge([task], tmp_path, prompt_path, judge_workers=1)

    details = module.build_details([task], tmp_path)
    summary = module.summarize(details)
    module.write_generation_attempts([task], tmp_path)
    module.write_result_tables(details, summary, tmp_path)
    module.render_html_report(details, summary, tmp_path)

    assert details[0]["judge_status"] == "completed"
    assert details[0]["judge_rules"] == ["T1.2"]
    assert summary["overall"]["tier_counts"] == {"1": 1}
    assert (tmp_path / details[0]["candidate_path"]).is_file()
    assert (tmp_path / "generation_attempts.jsonl").is_file()
    assert (tmp_path / "details.jsonl").is_file()
    assert (tmp_path / "details.parquet").is_file()
    assert (tmp_path / "summary.csv").is_file()
    assert (tmp_path / "report.html").is_file()