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


def load_module():
    script = Path(__file__).parents[1] / "scripts" / "evaluate_crop_fill_action.py"
    spec = importlib.util.spec_from_file_location("test_evaluate_crop_fill_action", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_materializes_four_no_judge_tasks_with_caption(tmp_path):
    module = load_module()
    output = BytesIO()
    Image.new("RGB", (80, 60), color="white").save(output, format="WEBP", lossless=True)
    payload = output.getvalue()
    with Image.open(BytesIO(payload)) as image:
        image_id = module.normalized_pixel_hash(image)
    data_path = tmp_path / "test.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": image_id,
                    "original_image": payload,
                    "title": "A headline",
                    "ImageCaption": "A visible caption",
                }
            ]
        ),
        data_path,
    )
    template = "<image>\nTitle: {title}\nCaption: {caption}\nRatio: {target_ratio}"

    tasks, manifest = module.load_and_materialize_tasks(data_path, tmp_path / "output", template)

    assert len(tasks) == 4
    assert len(manifest) == 1
    assert tasks[0]["prompt"] == "<image>\nTitle: A headline\nCaption: A visible caption\nRatio: 1"
    assert Path(tasks[0]["image_path"]).is_file()


def test_summarizes_format_render_and_ratio_metrics():
    module = load_module()
    details = [
        {
            "generation_status": "valid",
            "first_attempt_valid": True,
            "attempt_count": 1,
            "operation": "crop",
            "render_status": "rendered",
            "ratio_compliant": True,
            "crop_box": [0.0, 0.0, 1.0, 0.5],
            "padding_fraction": 0.0,
            "target_ratio": 1.0,
        },
        {
            "generation_status": "retry_exhausted",
            "first_attempt_valid": False,
            "attempt_count": 3,
            "operation": None,
            "render_status": "not_rendered",
            "ratio_compliant": False,
            "crop_box": None,
            "padding_fraction": None,
            "target_ratio": 1.0,
        },
    ]

    metrics = module.summarize_subset(details)

    assert metrics["first_attempt_valid_rate"] == 0.5
    assert metrics["eventual_valid_rate"] == 0.5
    assert metrics["render_success_rate"] == 0.5
    assert metrics["ratio_compliance_rate"] == 1.0
    assert metrics["operation_counts"] == {"crop": 1, "invalid": 1}


def test_transformers_backend_loads_peft_and_writes_valid_progress(tmp_path):
    module = load_module()
    observed = {}

    class FakeTensor:
        def __init__(self, values):
            self.values = values

        @property
        def shape(self):
            return (len(self.values), len(self.values[0]))

        def to(self, _device):
            return self

        def __getitem__(self, item):
            rows, columns = item
            assert rows == slice(None)
            return FakeTensor([row[columns] for row in self.values])

        def tolist(self):
            return self.values

    class FakeInferenceMode:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    class FakeInputs(dict):
        pass

    class FakeProcessor:
        @classmethod
        def from_pretrained(cls, path, local_files_only):
            observed["processor_path"] = path
            assert local_files_only
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            assert messages[0]["content"][1]["text"] == "Choose an action."
            return FakeInputs(input_ids=FakeTensor([[1, 2, 3]]), pixel_values=FakeTensor([[1]]))

        def batch_decode(self, token_ids, **_kwargs):
            assert token_ids.tolist() == [[4, 5]]
            return [
                '{"target_ratio":1.0,"is_cropped":false,"is_filled":false,'
                '"crop_box":null,"fill_color":null}'
            ]

    class FakeModel:
        device = "cpu"

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            observed["model_path"] = path
            observed["model_kwargs"] = kwargs
            return cls()

        def eval(self):
            return self

        def generate(self, **kwargs):
            observed["generate_kwargs"] = kwargs
            return FakeTensor([[1, 2, 3, 4, 5]])

    class FakePeftModel:
        @staticmethod
        def from_pretrained(model, adapter_path, is_trainable):
            observed["adapter_path"] = adapter_path
            assert not is_trainable
            return SimpleNamespace(
                base_model=SimpleNamespace(model=model),
                device=model.device,
                eval=model.eval,
                generate=model.generate,
            )

    fake_transformers = types.ModuleType("transformers")
    fake_transformers.AutoModelForImageTextToText = FakeModel
    fake_transformers.AutoProcessor = FakeProcessor
    fake_peft = types.ModuleType("peft")
    fake_peft.PeftModel = FakePeftModel
    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "test"
    fake_torch.bfloat16 = "bfloat16"
    fake_torch.inference_mode = FakeInferenceMode
    image_path = tmp_path / "image.webp"
    Image.new("RGB", (10, 10), color="white").save(image_path, format="WEBP", lossless=True)
    args = SimpleNamespace(
        max_attempts=1,
        max_tokens=16,
        temperature=0.0,
        top_p=1.0,
        seed=42,
        lora_rank=32,
        response_protocol="action-v4",
    )
    task = {
        "task_id": "image__ratio_1",
        "target_ratio": 1.0,
        "image_path": str(image_path),
        "prompt": "<image>\nChoose an action.",
    }

    with patch.dict(sys.modules, {"torch": fake_torch, "transformers": fake_transformers, "peft": fake_peft}):
        module.generate_actions_transformers(
            [task],
            tmp_path / "output",
            tmp_path / "model",
            tmp_path / "adapter",
            args,
        )

    progress = json.loads(module.progress_path(tmp_path / "output", task["task_id"]).read_text())
    assert progress["status"] == "valid"
    assert progress["action"]["operation"] == "keep"
    assert observed["model_kwargs"]["attn_implementation"] == "sdpa"
    assert observed["adapter_path"] == tmp_path / "adapter"


def test_loads_exact_swift_message_prompt_and_detail_response(tmp_path):
    module = load_module()
    image_path = tmp_path / "image.webp"
    Image.new("RGB", (80, 60), color="white").save(image_path, format="WEBP", lossless=True)
    with Image.open(image_path) as image:
        image_id = module.normalized_pixel_hash(image)
    prompt = (
        "<image>\nNews headline: Exact headline\nImage caption: Exact caption\n"
        "Target aspect ratio (width/height): 1"
    )
    rows = [
        {
            "messages": [{"role": "user", "content": prompt.replace(": 1", f": {ratio:g}")}],
            "images": [str(image_path)],
            "image_id": image_id,
            "source_index": 0,
            "target_ratio": ratio,
        }
        for ratio in module.TARGET_RATIOS
    ]
    data_path = tmp_path / "swift_test.parquet"
    pq.write_table(pa.Table.from_pylist(rows), data_path)

    tasks, manifest = module.load_swift_message_tasks(data_path, tmp_path / "output")

    assert len(tasks) == 4
    assert len(manifest) == 1
    assert tasks[0]["prompt"] == rows[0]["messages"][0]["content"]
    response = (
        '{"target_ratio":1.0,"is_cropped":false,"is_filled":false,'
        '"crop_box":null,"fill_color":null,"description":"Keep the complete image."}'
    )
    assert module.parse_response(response, "detail-v4").action.operation == "keep"