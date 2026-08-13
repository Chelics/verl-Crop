import importlib.util
import json
import sys
import types
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def load_evaluator_module():
    script_path = Path(__file__).parents[1] / "scripts" / "evaluate_image_once_mode.py"
    spec = importlib.util.spec_from_file_location("test_evaluate_image_once_mode", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_webp_payload(color: str = "white") -> bytes:
    output = BytesIO()
    Image.new("RGB", (80, 60), color=color).save(output, format="WEBP", lossless=True)
    return output.getvalue()


def make_args(max_attempts: int = 2):
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
        temperature=0.0,
        top_p=1.0,
        max_tokens=32,
        seed=42,
    )


def test_gpu_device_resolution_ignores_empty_entries():
    module = load_evaluator_module()

    assert module.resolve_gpu_devices(2, "0, ,1,") == ["0", "1"]


def test_materializes_four_ratio_mode_tasks_without_reference_crop(tmp_path):
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
    prompt = '<image>\nTitle: {title}\nRatio: {target_ratio}\n{"mode":"crop"} or {"mode":"pad"}'

    tasks, manifest = module.load_and_materialize_tasks(
        data_path,
        tmp_path / "output",
        mode_prompt_template=prompt,
    )

    assert len(manifest) == 1
    assert len(tasks) == 4
    assert [task["target_ratio"] for task in tasks] == list(module.TARGET_RATIOS)
    assert all(task["title"] == "A news title" for task in tasks)
    assert all("must not be used" not in json.dumps(task) for task in tasks)
    assert all(Path(task["image_path"]).is_file() for task in tasks)


def test_generation_retries_invalid_mode_and_persists_valid_decision(tmp_path, monkeypatch):
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
            calls.append((sampling_params.seed, requests[0]["prompt"]))
            response = "invalid" if len(calls) == 1 else '{"mode":"pad"}'
            return [SimpleNamespace(outputs=[SimpleNamespace(text=response)]) for _ in requests]

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    image_path = tmp_path / "image.webp"
    image_path.write_bytes(make_webp_payload())
    task = {
        "task_id": "image__ratio_1",
        "prompt": "<image>\nChoose crop or pad.",
        "image_path": str(image_path),
    }

    module.run_generation_worker(
        rank=0,
        gpu_devices=["0"],
        tasks=[task],
        output_dir=tmp_path,
        model_path=tmp_path,
        args=make_args(),
    )

    progress = json.loads(module.generation_progress_path(tmp_path, task["task_id"]).read_text())
    assert progress["status"] == "valid"
    assert progress["mode"] == "pad"
    assert [attempt["valid"] for attempt in progress["attempts"]] == [False, True]
    assert calls[0][0] == 42
    assert calls[1][0] == 43
    assert "previous output was rejected" in calls[1][1]


def test_writes_unlabeled_reports_and_review_template(tmp_path):
    module = load_evaluator_module()
    details = [
        {
            "task_id": "image__ratio_1",
            "source_index": 0,
            "image_id": "image",
            "title": "Example title",
            "caption": "Example caption",
            "target_ratio": 1.0,
            "image_width": 80,
            "image_height": 60,
            "original_render_path": "renders/originals/image.jpg",
            "generation_status": "valid",
            "predicted_mode": "pad",
            "strict_format": True,
            "had_invalid_output": False,
            "invalid_attempt_count": 0,
            "total_attempt_count": 1,
            "final_response": '{"mode":"pad"}',
            "final_parse_error": None,
        }
    ]
    summary = {
        "run_id": "test",
        "model_name": "Qwen test",
        "model_family": "qwen35",
        "images": 1,
        **module.summarize(details),
    }

    module.write_result_tables(details, summary, tmp_path)
    module.render_html_report(details, summary, tmp_path)
    module.render_markdown_report(details, summary, tmp_path)

    assert (tmp_path / "report.html").is_file()
    assert (tmp_path / "report.md").is_file()
    assert "No crop rendering" in (tmp_path / "report.html").read_text(encoding="utf-8")
    review_rows = (tmp_path / "review_template.csv").read_text(encoding="utf-8-sig").splitlines()
    assert "human_label" in review_rows[0]
    assert "pad" in review_rows[1]