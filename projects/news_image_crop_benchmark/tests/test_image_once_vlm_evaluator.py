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
        model_family="qwen35",
        image_max_pixels=1048576,
        image_min_pixels=65536,
        internvl_max_dynamic_patch=4,
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
        canonicalize_bare_json=False,
        action_protocol="legacy-crop-json",
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


def test_policy_model_adapters_build_native_prompts_and_engine_options():
    module = load_evaluator_module()
    image = Image.new("RGB", (80, 60), color="white")
    prompt = "<image>\nCrop this image"

    class QwenRenderer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert messages[0]["content"][1]["text"] == "Crop this image"
            return "qwen-rendered"

    class InternRenderer:
        def apply_chat_template(self, messages, **kwargs):
            assert "enable_thinking" not in kwargs
            assert messages == [{"role": "user", "content": prompt}]
            return "intern-rendered"

        def convert_tokens_to_ids(self, token):
            return {"<|im_start|>": 10, "<|im_end|>": 11}.get(token, -1)

    qwen = module.PolicyModelAdapter.create("qwen35")
    intern = module.PolicyModelAdapter.create("internvl2")
    molmo = module.PolicyModelAdapter.create("molmo")
    qwen_request = qwen.build_request(
        QwenRenderer(), prompt, image, image_max_pixels=100, image_min_pixels=10
    )
    intern_request = intern.build_request(
        InternRenderer(), prompt, image, image_max_pixels=100, image_min_pixels=10
    )
    molmo_request = molmo.build_request(
        None, prompt, image, image_max_pixels=100, image_min_pixels=10
    )

    assert qwen_request["prompt"] == "qwen-rendered"
    assert qwen_request["mm_processor_kwargs"]["size"]["longest_edge"] == 100
    assert intern_request["prompt"] == "intern-rendered"
    assert intern.llm_kwargs(internvl_max_dynamic_patch=4) == {
        "trust_remote_code": True,
        "mm_processor_kwargs": {"max_dynamic_patch": 4},
    }
    assert intern.sampling_kwargs(InternRenderer()) == {"stop_token_ids": [10, 11]}
    assert molmo_request["prompt"] == "Crop this image"
    assert molmo.llm_kwargs(internvl_max_dynamic_patch=4) == {"trust_remote_code": True}
    image.close()


def test_policy_model_adapter_canonicalizes_only_exact_bare_crop_json():
    adapter = load_evaluator_module().PolicyModelAdapter.create("internvl2")

    canonical, normalized = adapter.canonicalize_response('{"cx": 453, "cy": 120, "area": 292}')
    assert canonical == '<crop>{"cx": 453, "cy": 120, "area": 292}</crop>'
    assert normalized

    for response in (
        'Result: {"cx": 453, "cy": 120, "area": 292}',
        '{"cx": 453, "cy": 120, "area": 292, "ratio": 1.59}',
        '<crop>{"cx": 453, "cy": 120, "area": 292}</crop>',
    ):
        canonical, normalized = adapter.canonicalize_response(response)
        assert canonical == response
        assert not normalized


def test_retry_prompt_includes_previous_validation_error():
    module = load_evaluator_module()
    base_prompt = "<image>\nReturn a crop."
    prompt = module.build_attempt_prompt(
        base_prompt,
        [{"parse_error": "area must be in (0, 1000]", "response": "bad"}],
    )

    assert prompt.startswith(base_prompt)
    assert "area must be in (0, 1000]" in prompt
    assert "area is a normalized integer in [1, 1000]" in prompt


def test_percentage_retry_prompt_uses_public_percentage_fields():
    module = load_evaluator_module()
    prompt = module.build_attempt_prompt(
        "<image>\nReturn a crop.",
        [{"parse_error": "area_pct must be in [1, 100]"}],
        "percent-json-v1",
    )

    assert "cx_pct and cy_pct" in prompt
    assert "area_pct is an integer in [1, 100]" in prompt
    assert "<crop>" not in prompt


def test_layout_retry_prompt_uses_unified_fields():
    module = load_evaluator_module()
    prompt = module.build_attempt_prompt(
        "<image>\nReturn a layout.",
        [{"parse_error": "invalid operation"}],
        "layout-json-v1",
    )

    assert "operation is crop, crop_pad, or pad" in prompt
    assert "four percentage coordinates" in prompt


def test_unified_layout_candidate_renders_crop_then_pad(tmp_path):
    module = load_evaluator_module()
    image_path = tmp_path / "source.webp"
    image_path.write_bytes(make_webp_payload())
    task = {
        "task_id": "image__ratio_1.91",
        "image_path": str(image_path),
        "target_ratio": 1.91,
    }
    action = {
        "operation": "crop_pad",
        "x1_pct": 10,
        "y1_pct": 10,
        "x2_pct": 90,
        "y2_pct": 90,
    }

    path, metadata = module.render_unified_layout_candidate(task, action, tmp_path)

    with Image.open(path) as candidate:
        assert abs(candidate.width / candidate.height - 1.91) <= 1 / candidate.height
    assert metadata["selected_operation"] == "crop_pad"
    assert metadata["padding_fraction"] > 0
    assert metadata["background_hex"].startswith("#")


def test_judge_metadata_skips_unrelated_braces_before_valid_json():
    module = load_evaluator_module()
    metadata = module.parse_judge_metadata(
        'analysis {not json} then {"evaluation":{"tier_name":"Suboptimal",'
        '"rules":["T2.3"],"confidence_score":"high"}} trailing {noise}'
    )

    assert metadata == {
        "rules": ["T2.3"],
        "confidence_score": "high",
        "tier_name": "Suboptimal",
    }


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


def test_generation_loads_lora_adapter_for_vllm(tmp_path):
    module = load_evaluator_module()
    observed = {}

    class FakeProcessor:
        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"][1]["text"]

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(_model_path, local_files_only):
            assert local_files_only
            return FakeProcessor()

    class FakeSamplingParams:
        def __init__(self, **_kwargs):
            pass

    class FakeLoRARequest:
        def __init__(self, name, adapter_id, path):
            observed["request"] = (name, adapter_id, path)

    class FakeLLM:
        def __init__(self, **kwargs):
            observed["llm_kwargs"] = kwargs

        def generate(self, requests, sampling_params, lora_request=None):
            observed["lora_request"] = lora_request
            response = '<crop>{"cx":500,"cy":500,"area":400}</crop>'
            return [SimpleNamespace(outputs=[SimpleNamespace(text=response)]) for _ in requests]

    image_path = tmp_path / "image.webp"
    image_path.write_bytes(make_webp_payload())
    adapter_path = tmp_path / "adapter"
    adapter_path.mkdir()
    task = {"task_id": "image__ratio_1", "prompt": "<image>\nCrop", "image_path": str(image_path)}
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    fake_lora = types.ModuleType("vllm.lora")
    fake_lora_request = types.ModuleType("vllm.lora.request")
    fake_lora_request.LoRARequest = FakeLoRARequest
    args = make_args()
    args.lora_adapter_path = adapter_path
    args.lora_rank = 32

    fake_modules = {
        "transformers": fake_transformers,
        "vllm": fake_vllm,
        "vllm.lora": fake_lora,
        "vllm.lora.request": fake_lora_request,
    }
    with patch.dict(sys.modules, fake_modules):
        module.run_generation_worker(
            rank=0,
            gpu_devices=["0"],
            tasks=[task],
            output_dir=tmp_path,
            model_path=tmp_path / "model",
            args=args,
        )

    assert observed["llm_kwargs"]["enable_lora"] is True
    assert observed["llm_kwargs"]["max_lora_rank"] == 32
    assert observed["request"] == ("news-crop-sft", 1, str(adapter_path))
    assert observed["lora_request"] is not None


def test_generation_canonicalizes_bare_json_and_preserves_raw_response(tmp_path):
    module = load_evaluator_module()

    class FakeProcessor:
        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"][1]["text"]

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(_model_path, local_files_only):
            assert local_files_only
            return FakeProcessor()

    class FakeSamplingParams:
        def __init__(self, **_kwargs):
            pass

    class FakeLLM:
        def __init__(self, **_kwargs):
            pass

        def generate(self, requests, sampling_params):
            response = '{"cx":453,"cy":120,"area":292}'
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
    args = make_args()
    args.canonicalize_bare_json = True

    with patch.dict(sys.modules, {"transformers": fake_transformers, "vllm": fake_vllm}):
        module.run_generation_worker(
            rank=0,
            gpu_devices=["0"],
            tasks=[task],
            output_dir=tmp_path,
            model_path=tmp_path / "model",
            args=args,
        )

    progress = json.loads(module.generation_progress_path(tmp_path, task["task_id"]).read_text())
    attempt = progress["attempts"][0]
    assert progress["status"] == "valid"
    assert attempt["response"] == '{"cx":453,"cy":120,"area":292}'
    assert attempt["canonical_response"] == '<crop>{"cx":453,"cy":120,"area":292}</crop>'
    assert attempt["response_normalized"]
    assert attempt["canonical_format"]
    assert not attempt["strict_format"]


def test_generation_persists_unified_layout_action(tmp_path):
    module = load_evaluator_module()

    class FakeProcessor:
        def apply_chat_template(self, messages, **_kwargs):
            return messages[0]["content"][1]["text"]

    class FakeAutoProcessor:
        @staticmethod
        def from_pretrained(_model_path, local_files_only):
            assert local_files_only
            return FakeProcessor()

    class FakeSamplingParams:
        def __init__(self, **_kwargs):
            pass

    class FakeLLM:
        def __init__(self, **_kwargs):
            pass

        def generate(self, requests, sampling_params):
            response = (
                '{"operation":"crop_pad","x1_pct":5,"y1_pct":10,'
                '"x2_pct":95,"y2_pct":90}'
            )
            return [SimpleNamespace(outputs=[SimpleNamespace(text=response)]) for _ in requests]

    image_path = tmp_path / "image.webp"
    image_path.write_bytes(make_webp_payload())
    task = {
        "task_id": "image__ratio_1.91",
        "prompt": "<image>\nCreate a layout",
        "image_path": str(image_path),
    }
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    args = make_args()
    args.action_protocol = "layout-json-v1"

    with patch.dict(sys.modules, {"transformers": fake_transformers, "vllm": fake_vllm}):
        module.run_generation_worker(
            rank=0,
            gpu_devices=["0"],
            tasks=[task],
            output_dir=tmp_path,
            model_path=tmp_path / "model",
            args=args,
        )

    progress = json.loads(module.generation_progress_path(tmp_path, task["task_id"]).read_text())
    assert progress["status"] == "valid"
    assert progress["action"] == {
        "operation": "crop_pad",
        "x1_pct": 5,
        "y1_pct": 10,
        "x2_pct": 95,
        "y2_pct": 90,
    }


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
    module.write_json_atomic(
        module.judge_progress_path(tmp_path, task["task_id"]),
        {"task_id": task["task_id"], "status": "failed", "error_type": "TimeoutError"},
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
    summary["model_name"] = "Test Model"
    module.write_generation_attempts([task], tmp_path)
    module.write_result_tables(details, summary, tmp_path)
    module.render_markdown_report(details, summary, tmp_path)

    assert details[0]["judge_status"] == "completed"
    assert details[0]["judge_rules"] == ["T1.2"]
    assert summary["overall"]["tier_counts"] == {"1": 1}
    assert (tmp_path / details[0]["candidate_path"]).is_file()
    assert (tmp_path / "generation_attempts.jsonl").is_file()
    assert (tmp_path / "details.jsonl").is_file()
    assert (tmp_path / "details.parquet").is_file()
    assert (tmp_path / "summary.csv").is_file()
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "# Test Model Four-Ratio Crop Evaluation" in report
    assert "## Tier Distribution" in report
    assert "## Visual Results" in report
    assert "![Original](renders/originals/image.jpg)" in report
    assert "![Ratio 1.59](renders/candidates/image__ratio_1.59.jpg)" in report
    assert "`T1.2`" in report