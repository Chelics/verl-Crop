import json

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from news_crop_benchmark.cropped_override_merge import merge_cropped_overrides


def base_row(trace_id, ratio):
    return {
        "source_index": 1,
        "trace_id": trace_id,
        "headline": "Title",
        "caption": "Caption",
        "original_image_url": "https://example.com/original.jpg",
        "original_width": 120,
        "original_height": 100,
        "aspect_ratio": f"{ratio:g}:1",
        "target_ratio": ratio,
        "crop_required": True,
        "feasible": True,
        "render_mode": "cropped",
        "was_cropped": True,
        "was_padded": False,
        "bbox_normalized": [0.0, 0.0, 1.0, 1.0],
        "bbox_pixels": [0, 0, 120, 100],
        "edge_artifact_trim_pixels": [1, 2, 3, 4],
        "background_color_rgb": [10, 20, 30],
        "padding_color_rgb": None,
        "output_width": 120,
        "output_height": 100,
        "reason": "baseline",
        "confidence": "medium",
        "cropped_image_format": "JPEG",
        "cropped_image": b"baseline-image",
        "error": None,
    }


def test_merges_rendered_override_with_exact_bbox(tmp_path):
    base_path = tmp_path / "base.parquet"
    base_rows = [base_row("image-a", 1.0), base_row("image-b", 1.0)]
    base_rows[1]["padding_color_rgb"] = [0, 0, 0]
    pq.write_table(pa.Table.from_pylist(base_rows), base_path)
    override_dir = tmp_path / "overrides"
    images = override_dir / "images"
    images.mkdir(parents=True)
    image_path = images / "image-a.jpg"
    Image.new("RGB", (100, 100), "blue").save(image_path, format="JPEG")
    record = {
        "trace_id": "image-a",
        "aspect_ratio": "1:1",
        "operation": "crop_pad",
        "reason": "keep subject",
        "source_box_pixels": [10, 5, 110, 95],
        "background_color_rgb": [1, 2, 3],
        "output_width": 100,
        "output_height": 100,
        "image_path": "images/image-a.jpg",
    }
    (override_dir / "overrides.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    output_path = tmp_path / "v3.parquet"
    report = merge_cropped_overrides(base_path, override_dir, output_path, row_group_size=1)
    output = pq.read_table(output_path).to_pylist()

    assert report["override_rows"] == 1
    assert report["untouched_rows_verified"] == 1
    assert report["non_null_bbox_rows"] == 1
    assert report["reason_source"] == "local_gpt_layout_v2_manifest"
    assert output[1] == base_rows[1]
    replacement = output[0]
    assert replacement["bbox_pixels"] == [10, 5, 110, 95]
    assert replacement["bbox_normalized"] == [10 / 120, 5 / 100, 110 / 120, 95 / 100]
    assert replacement["render_mode"] == "padded"
    assert replacement["was_cropped"] is True
    assert replacement["was_padded"] is True
    assert replacement["padding_color_rgb"] == [1, 2, 3]
    assert replacement["reason"] == "keep subject"
    assert replacement["cropped_image"] == image_path.read_bytes()