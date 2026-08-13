import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


def load_evaluator_module():
    script_path = Path(__file__).parents[1] / "scripts" / "evaluate_image_once_layout.py"
    spec = importlib.util.spec_from_file_location("test_evaluate_image_once_layout", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_task(tmp_path, task_id, ratio):
    image_path = tmp_path / "image.webp"
    if not image_path.exists():
        image = Image.new("RGB", (80, 60), color=(230, 235, 240))
        image.paste((30, 80, 130), (8, 8, 72, 52))
        image.save(image_path, format="WEBP", lossless=True)
    return {
        "task_id": task_id,
        "source_index": 0,
        "image_id": "image",
        "title": "Example title",
        "caption": "Example caption",
        "target_ratio": ratio,
        "image_width": 80,
        "image_height": 60,
        "image_path": str(image_path),
        "crop_prompt": "<image>\nReturn crop JSON",
        "original_render_path": "renders/originals/image.jpg",
    }


def make_mode_record(task, mode):
    return {
        "task_id": task["task_id"],
        "generation_status": "valid",
        "predicted_mode": mode,
        "final_response": json.dumps({"mode": mode}, separators=(",", ":")),
        "total_attempt_count": 1,
    }


def make_args():
    return SimpleNamespace(
        model_family="qwen35",
        image_max_pixels=1048576,
        image_min_pixels=65536,
        internvl_max_dynamic_patch=4,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.6,
        max_model_len=2176,
        max_num_seqs=8,
        crop_max_attempts=2,
        prompt_batch_size=8,
        crop_temperature=0.0,
        crop_top_p=1.0,
        crop_max_tokens=32,
        seed=42,
    )


def test_loads_complete_mode_results_for_layout_tasks(tmp_path):
    module = load_evaluator_module()
    tasks = [make_task(tmp_path, "image__ratio_1", 1.0)]
    mode_dir = tmp_path / "mode"
    mode_dir.mkdir()
    (mode_dir / "_MODE_EVAL_COMPLETE.json").write_text(
        json.dumps({"tasks": 1}), encoding="utf-8"
    )
    record = make_mode_record(tasks[0], "pad")
    (mode_dir / "details.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    loaded = module.load_mode_results(mode_dir, tasks)

    assert loaded[tasks[0]["task_id"]]["predicted_mode"] == "pad"


def test_crop_worker_retries_and_persists_v1_percentage_action(tmp_path, monkeypatch):
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
            response = "invalid" if len(calls) == 1 else '{"cx_pct":45,"cy_pct":60,"area_pct":70}'
            return [SimpleNamespace(outputs=[SimpleNamespace(text=response)]) for _ in requests]

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoProcessor = FakeAutoProcessor
    fake_vllm = types.ModuleType("vllm")
    fake_vllm.LLM = FakeLLM
    fake_vllm.SamplingParams = FakeSamplingParams
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    task = make_task(tmp_path, "image__ratio_1", 1.0)

    module.run_crop_worker(0, ["0"], [task], tmp_path, tmp_path, make_args())

    progress = json.loads(module.crop_progress_path(tmp_path, task["task_id"]).read_text())
    assert progress["status"] == "valid"
    assert progress["action"]["cx_pct"] == 45
    assert progress["action"]["cy_pct"] == 60
    assert progress["action"]["area_pct"] == 70
    assert [attempt["valid"] for attempt in progress["attempts"]] == [False, True]
    assert "previous output was rejected" in calls[1][1]


def test_renders_crop_and_edge_pad_into_one_report(tmp_path):
    module = load_evaluator_module()
    pad_task = make_task(tmp_path, "image__ratio_1", 1.0)
    crop_task = make_task(tmp_path, "image__ratio_1.91", 1.91)
    tasks = [pad_task, crop_task]
    modes = {
        pad_task["task_id"]: make_mode_record(pad_task, "pad"),
        crop_task["task_id"]: make_mode_record(crop_task, "crop"),
    }
    module.write_json_atomic(
        module.crop_progress_path(tmp_path, crop_task["task_id"]),
        {
            "task_id": crop_task["task_id"],
            "status": "valid",
            "attempts": [
                {
                    "response": '{"cx_pct":50,"cy_pct":50,"area_pct":70}',
                    "valid": True,
                    "strict_format": True,
                }
            ],
            "action": {
                "cx": 500.0,
                "cy": 500.0,
                "area": 700.0,
                "cx_pct": 50,
                "cy_pct": 50,
                "area_pct": 70,
            },
        },
    )

    rendered = module.render_layouts(tasks, modes, tmp_path, edge_fraction=0.05)
    details = module.build_details(tasks, modes, rendered, tmp_path)
    summary = {
        "model_name": "Qwen test",
        "overall": module.summarize_subset(details),
        "by_ratio": module.summarize(details)["by_ratio"],
    }
    module.write_results(details, summary, tmp_path)
    module.render_html_report(details, summary, tmp_path)
    module.render_markdown_report(details, summary, tmp_path)

    pad_detail = next(detail for detail in details if detail["predicted_mode"] == "pad")
    crop_detail = next(detail for detail in details if detail["predicted_mode"] == "crop")
    assert pad_detail["background_hex"] == "#E6EBF0"
    assert pad_detail["render_width"] == 80
    assert pad_detail["render_height"] == 80
    assert crop_detail["area_pct"] == 70
    assert Path(tmp_path, pad_detail["candidate_path"]).is_file()
    assert Path(tmp_path, crop_detail["candidate_path"]).is_file()
    assert (tmp_path / "report.html").is_file()
    assert (tmp_path / "report.md").is_file()
    assert (tmp_path / "review_template.csv").is_file()