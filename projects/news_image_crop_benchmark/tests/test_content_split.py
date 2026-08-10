import tempfile
import unittest
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from news_crop_benchmark.data import build_verl_row
from projects.news_image_crop_benchmark.scripts.convert_to_verl import convert_dataset
from projects.news_image_crop_benchmark.scripts.resplit_by_image_content import resplit_dataset


class ContentSplitTests(unittest.TestCase):
    def test_full_conversion_uses_image_content_identity(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload_buffer = BytesIO()
            Image.new("RGB", (16, 8), color="red").save(payload_buffer, format="WEBP")
            payload = payload_buffer.getvalue()
            source_path = root / "source.parquet"
            policy_prompt_path = root / "policy_prompt.txt"
            policy_prompt_path.write_text(
                "<image>\nHeadline: {title}\nRatio: {target_ratio}",
                encoding="utf-8",
            )
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {
                            "TraceId": "trace-1",
                            "GemTitle": "Shared title",
                            "OriginalImageUrl": "https://example.test/first.webp",
                            "OriginalImageBytes": payload,
                        },
                        {
                            "TraceId": "trace-2",
                            "GemTitle": "Shared title",
                            "OriginalImageUrl": "https://example.test/alias.webp",
                            "OriginalImageBytes": payload,
                        },
                        {
                            "TraceId": "trace-3",
                            "GemTitle": "Other title",
                            "OriginalImageUrl": "https://example.test/alias.webp",
                            "OriginalImageBytes": payload,
                        },
                    ]
                ),
                source_path,
            )

            report = convert_dataset(
                source_path=source_path,
                output_dir=root / "output",
                prefix="content",
                batch_size=2,
                seed=42,
                limit=None,
                policy_prompt_path=policy_prompt_path,
            )

            self.assertEqual(report["unique_original_assets"], 2)
            self.assertEqual(report["unique_original_image_contents"], 1)
            self.assertEqual(report["unique_trusted_title_image_pairs"], 2)
            self.assertEqual(report["duplicate_title_image_pairs"], 1)
            self.assertEqual(report["expanded_rows"], 8)
            self.assertEqual(sum(count > 0 for count in report["split_rows"].values()), 1)
            self.assertEqual(report["policy_prompt_path"], str(policy_prompt_path.resolve()))
            self.assertEqual(len(report["policy_prompt_sha256"]), 64)
            output_path = next(Path(path) for path in report["outputs"].values() if pq.read_table(path).num_rows)
            prompt = pq.read_table(output_path).to_pylist()[0]["prompt"][0]["content"]
            self.assertIn("Headline: Shared title", prompt)
            self.assertIn("Ratio: 1", prompt)
            asset_files = list((root / "output" / "content_assets" / "original").rglob("*.webp"))
            self.assertEqual(len(asset_files), 1)

    def test_deduplicates_and_groups_by_image_checksum(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first_path = str(root / "first.webp")
            alias_path = str(root / "alias.webp")
            other_path = str(root / "other.webp")
            rows = [
                build_verl_row(
                    sample_identifier="old-1",
                    source_index=0,
                    split="train",
                    title="Shared title",
                    original_image_path=first_path,
                    image_width=100,
                    image_height=100,
                    target_ratio=1.0,
                ),
                build_verl_row(
                    sample_identifier="old-2",
                    source_index=1,
                    split="test",
                    title="Shared title",
                    original_image_path=alias_path,
                    image_width=100,
                    image_height=100,
                    target_ratio=1.0,
                ),
                build_verl_row(
                    sample_identifier="old-3",
                    source_index=2,
                    split="validation",
                    title="Other title",
                    original_image_path=other_path,
                    image_width=200,
                    image_height=100,
                    target_ratio=1.0,
                ),
            ]
            input_path = root / "input.parquet"
            pq.write_table(pa.Table.from_pylist(rows), input_path)
            manifest_path = root / "assets.parquet"
            pq.write_table(
                pa.Table.from_pylist(
                    [
                        {"path": first_path, "checksum": "same-checksum"},
                        {"path": alias_path, "checksum": "same-checksum"},
                        {"path": other_path, "checksum": "other-checksum"},
                    ]
                ),
                manifest_path,
            )

            report = resplit_dataset(
                input_paths=[input_path],
                asset_manifest_path=manifest_path,
                output_dir=root / "output",
                prefix="content",
                seed=42,
            )

            self.assertEqual(report["input_rows"], 3)
            self.assertEqual(report["output_rows"], 2)
            self.assertEqual(report["duplicate_content_tasks_removed"], 1)
            self.assertEqual(sum(report["split_image_contents"].values()), 2)


if __name__ == "__main__":
    unittest.main()