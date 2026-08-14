import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from news_crop_benchmark.cropped_overrides import load_override_specs, render_overrides


def test_renders_crop_and_pad_overrides(tmp_path):
    payload = io.BytesIO()
    Image.new("RGB", (120, 100), (20, 40, 60)).save(payload, format="JPEG")
    train_path = tmp_path / "train.parquet"
    pq.write_table(pa.Table.from_pylist([{"image_id": "image-a", "original_image": payload.getvalue()}]), train_path)
    manifest_path = tmp_path / "manifest.jsonl"
    specs = [
        {
            "trace_id": "image-a",
            "aspect_ratio": "1:1",
            "operation": "crop",
            "source_box_normalized": [0.0, 0.0, 1.0, 1.0],
            "reason": "crop test",
        },
        {
            "trace_id": "image-a",
            "aspect_ratio": "1.91:1",
            "operation": "crop_pad",
            "source_box_normalized": [0.1, 0.1, 0.9, 0.9],
            "background_color_rgb": [1, 2, 3],
            "reason": "pad test",
        },
    ]
    manifest_path.write_text("".join(json.dumps(spec) + "\n" for spec in specs), encoding="utf-8")

    records = render_overrides(train_path, manifest_path, tmp_path / "output")

    assert len(records) == 2
    assert records[0]["output_width"] == records[0]["output_height"]
    assert abs(records[1]["output_width"] / records[1]["output_height"] - 1.91) < 0.02
    assert records[1]["padding_fraction"] > 0
    assert all((tmp_path / "output" / record["image_path"]).is_file() for record in records)
    assert len(load_override_specs(manifest_path)) == 2