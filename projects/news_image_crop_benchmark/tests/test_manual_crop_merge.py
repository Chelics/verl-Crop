import io

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from news_crop_benchmark.manual_crop_merge import merge_manual_crops


def image_bytes(size, color, image_format):
    output = io.BytesIO()
    Image.new("RGBA" if image_format == "PNG" else "RGB", size, color).save(output, format=image_format)
    return output.getvalue()


def base_row(trace_id, ratio):
    return {
        "source_index": 1,
        "trace_id": trace_id,
        "headline": f"Title {trace_id}",
        "caption": f"Caption {trace_id}",
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
        "output_height": round(120 / ratio),
        "reason": "baseline",
        "confidence": "medium",
        "cropped_image_format": "JPEG",
        "cropped_image": image_bytes((120, round(120 / ratio)), "red", "JPEG"),
        "error": None,
    }


def test_merges_latest_valid_manual_rows_and_preserves_others(tmp_path):
    base_path = tmp_path / "base.parquet"
    base_rows = [base_row("image-a", 1.0), base_row("image-a", 1.91), base_row("image-b", 1.0)]
    base_rows[0]["padding_color_rgb"] = [0, 0, 0]
    base_table = pa.Table.from_pylist(base_rows)
    pq.write_table(base_table, base_path)
    manual_path = tmp_path / "manual.parquet"
    manual_rows = [
        {
            "save_id": "invalid-old",
            "image_id": "image-a",
            "manual_crop": image_bytes((100, 100), "blue", "PNG"),
            "ratio": 1.91,
            "theme_color": None,
            "fill_color_code": None,
            "is_filled": False,
            "is_cropped": False,
            "crop_reason": "",
            "title": "Title image-a",
            "ImageCaption": "Caption image-a",
            "saved_at": "2026-08-14T01:00:00Z",
        },
        {
            "save_id": "valid-new",
            "image_id": "image-a",
            "manual_crop": image_bytes((191, 100), (0, 0, 255, 128), "PNG"),
            "ratio": 1.91,
            "theme_color": "#010203",
            "fill_color_code": "#040506",
            "is_filled": True,
            "is_cropped": True,
            "crop_reason": "manual fix",
            "title": "Title image-a",
            "ImageCaption": "Caption image-a",
            "saved_at": "2026-08-14T02:00:00Z",
        },
    ]
    manual_schema = pa.schema(
        [
            ("save_id", pa.string()),
            ("image_id", pa.string()),
            ("manual_crop", pa.binary()),
            ("ratio", pa.float64()),
            ("theme_color", pa.string()),
            ("fill_color_code", pa.string()),
            ("is_filled", pa.bool_()),
            ("is_cropped", pa.bool_()),
            ("crop_reason", pa.string()),
            ("title", pa.string()),
            ("ImageCaption", pa.string()),
            ("saved_at", pa.timestamp("us", tz="UTC")),
        ]
    )
    for row in manual_rows:
        row["saved_at"] = __import__("datetime").datetime.fromisoformat(row["saved_at"].replace("Z", "+00:00"))
    pq.write_table(pa.Table.from_pylist(manual_rows, schema=manual_schema), manual_path)

    output_path = tmp_path / "v2.parquet"
    report = merge_manual_crops(base_path, manual_path, output_path, row_group_size=2)
    output = pq.read_table(output_path).to_pylist()

    assert report["manual_input_rows"] == 2
    assert report["manual_selected_rows"] == 1
    assert report["replacement_rows_verified"] == 1
    assert report["untouched_rows_verified"] == 2
    assert len(report["rejected_rows"]) == 1
    assert output[0] == base_rows[0]
    assert output[2] == base_rows[2]
    replacement = output[1]
    assert replacement["render_mode"] == "padded"
    assert replacement["was_cropped"] is True
    assert replacement["was_padded"] is True
    assert replacement["bbox_normalized"] is None
    assert replacement["padding_color_rgb"] == [4, 5, 6]
    assert replacement["cropped_image_format"] == "PNG"
    assert replacement["cropped_image"] == manual_rows[1]["manual_crop"]