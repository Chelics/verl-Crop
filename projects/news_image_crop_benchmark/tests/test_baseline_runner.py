import importlib.util
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image


def load_baseline_module():
    script_path = Path(__file__).parents[1] / "scripts" / "evaluate_vllm_baseline.py"
    spec = importlib.util.spec_from_file_location("test_evaluate_vllm_baseline", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


class BaselineRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load_baseline_module()

    def test_batches_requests_with_fixed_upper_bound(self):
        batches = list(self.module.batched(list(range(7)), 3))
        self.assertEqual(batches, [[0, 1, 2], [3, 4, 5], [6]])

    def test_resolves_visible_gpu_devices(self):
        devices = self.module.resolve_gpu_devices(3, "4, 6,7, 9")
        self.assertEqual(devices, ["4", "6", "7"])

    def test_rejects_too_few_visible_gpu_devices(self):
        with self.assertRaisesRegex(ValueError, "exposes only 2"):
            self.module.resolve_gpu_devices(3, "0,1")

    def test_empty_visible_gpu_devices_exposes_no_gpus(self):
        with self.assertRaisesRegex(ValueError, "exposes only 0"):
            self.module.resolve_gpu_devices(1, "")

    def test_minus_one_visible_gpu_device_terminates_device_list(self):
        self.assertEqual(self.module.resolve_gpu_devices(2, "4,6,-1,7"), ["4", "6"])
        with self.assertRaisesRegex(ValueError, "exposes only 0"):
            self.module.resolve_gpu_devices(1, "-1")

    def test_partitions_rows_across_data_parallel_workers(self):
        rows = [{"id": index} for index in range(7)]
        partitions = self.module.partition_rows(rows, 3)
        self.assertEqual([[row["id"] for row in partition] for partition in partitions], [[0, 3, 6], [1, 4], [2, 5]])

    def test_dataset_shards_are_disjoint_and_complete(self):
        rows = list(range(19))
        shards = [rows[index::4] for index in range(4)]
        self.assertEqual(sorted(item for shard in shards for item in shard), rows)
        self.assertEqual(sum(len(shard) for shard in shards), len(set(item for shard in shards for item in shard)))

    def test_fingerprint_detects_in_place_file_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text('{"version": 1}', encoding="utf-8")
            first = self.module.fingerprint_path(path)
            path.write_text('{"version": 2}', encoding="utf-8")
            second = self.module.fingerprint_path(path)
            self.assertNotEqual(first, second)

    def test_selected_row_fingerprint_covers_ground_truth(self):
        row = {
            "images": ["/shared/image.jpg"],
            "extra_info": {"sample_id": "sample-1", "title": "Title", "target_ratio": 1.0},
            "reward_model": {"ground_truth": '{"image_width": 100}'},
        }
        first = self.module.fingerprint_selected_rows([row])
        row["reward_model"]["ground_truth"] = '{"image_width": 200}'
        second = self.module.fingerprint_selected_rows([row])
        self.assertNotEqual(first, second)

    def test_selected_image_fingerprint_detects_replaced_image(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "image.png"
            Image.new("RGB", (8, 8), color="white").save(image_path)
            rows = [{"images": [str(image_path)]}]
            first = self.module.fingerprint_selected_images(rows)
            Image.new("RGB", (8, 8), color="black").save(image_path)
            second = self.module.fingerprint_selected_images(rows)
            self.assertNotEqual(first, second)

    def test_remaps_dataset_and_reward_image_paths(self):
        row = {
            "images": ["/mnt/blob_output/assets/image.jpg"],
            "extra_info": {"original_image_path": "/mnt/blob_output/assets/image.jpg"},
        }
        self.module.remap_dataset_paths([row], "/mnt/blob_output", "/mnt/default")
        self.assertEqual(row["images"][0], "/mnt/default/assets/image.jpg")
        self.assertEqual(row["extra_info"]["original_image_path"], "/mnt/default/assets/image.jpg")

    def test_generation_worker_writes_resumable_raw_output(self):
        captured = {}

        class FakeProcessor:
            def apply_chat_template(self, messages, **kwargs):
                return messages[0]["content"][1]["text"]

        class FakeAutoProcessor:
            @staticmethod
            def from_pretrained(model_path, local_files_only):
                return FakeProcessor()

        class FakeSamplingParams:
            def __init__(self, **kwargs):
                self.n = kwargs["n"]

        class FakeLLM:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def generate(self, requests, sampling_params):
                return [
                    SimpleNamespace(
                        outputs=[
                            SimpleNamespace(text=f'<crop>{{"cx":500,"cy":500,"area":{400 + index}}}</crop>')
                            for index in range(sampling_params.n)
                        ]
                    )
                    for _ in requests
                ]

        fake_transformers = types.ModuleType("transformers")
        fake_transformers.AutoProcessor = FakeAutoProcessor
        fake_vllm = types.ModuleType("vllm")
        fake_vllm.LLM = FakeLLM
        fake_vllm.SamplingParams = FakeSamplingParams

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_path = root / "image.jpg"
            Image.new("RGB", (32, 24), color="white").save(image_path)
            raw_dir = root / "raw"
            row = {
                "images": [str(image_path)],
                "prompt": [{"content": "<image>\nCrop this image"}],
                "extra_info": {"sample_id": "sample-1"},
            }
            with patch.dict(sys.modules, {"transformers": fake_transformers, "vllm": fake_vllm}):
                self.module.run_generation_worker(
                    rank=0,
                    gpu_devices=["7"],
                    rows=[row],
                    raw_dir=raw_dir,
                    model_path=root / "model",
                    tensor_parallel_size=1,
                    prompt_batch_size=4,
                    max_num_seqs=16,
                    gpu_memory_utilization=0.6,
                    max_model_len=2176,
                    image_max_pixels=1048576,
                    image_min_pixels=65536,
                    temperature=0.7,
                    top_p=0.95,
                    n=2,
                    max_tokens=128,
                    seed=42,
                )

            result = json.loads((raw_dir / "sample-1.json").read_text(encoding="utf-8"))
            self.assertEqual(len(result["responses"]), 2)
            self.assertEqual(captured["tensor_parallel_size"], 1)
            self.assertEqual(captured["max_num_seqs"], 16)

    def test_resume_requires_matching_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            config = {"data_parallel_size": 8, "tensor_parallel_size": 1}
            self.module.prepare_output_directory(output_dir, config, resume=False)
            self.module.prepare_output_directory(output_dir, config, resume=True)

            with self.assertRaisesRegex(ValueError, "does not match"):
                self.module.prepare_output_directory(
                    output_dir,
                    {"data_parallel_size": 4, "tensor_parallel_size": 1},
                    resume=True,
                )

    def test_nonempty_output_requires_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            (output_dir / "existing.txt").write_text("result", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "--resume"):
                self.module.prepare_output_directory(output_dir, {"data_parallel_size": 8}, resume=False)


if __name__ == "__main__":
    unittest.main()
