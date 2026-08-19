import importlib.util
import json
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image


def load_converter():
    script = Path(__file__).parents[1] / "scripts" / "convert_cropped_v3_to_detail_sft.py"
    spec = importlib.util.spec_from_file_location("test_convert_cropped_v3_to_detail_sft", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def image_record(module, color, title):
    output = BytesIO()
    image = Image.new("RGB", (80, 60), color=color)
    image.save(output, format="WEBP", lossless=True)
    payload = output.getvalue()
    with Image.open(BytesIO(payload)) as decoded:
        image_id = module.normalized_pixel_hash(decoded)
    return {
        "image_id": image_id,
        "original_image": payload,
        "title": title,
        "ImageCaption": f"Caption for {title}",
    }


def annotation(image, ratio, operation, box, *, trim=None, color=None, source_index=0):
    return {
        "source_index": source_index,
        "trace_id": image["image_id"],
        "headline": image["title"],
        "caption": image["ImageCaption"],
        "target_ratio": ratio,
        "original_width": 80,
        "original_height": 60,
        "was_cropped": operation in {"crop", "crop_fill"},
        "was_padded": operation in {"crop_fill", "fill"},
        "bbox_pixels": box,
        "edge_artifact_trim_pixels": trim or [0, 0, 0, 0],
        "padding_color_rgb": color,
        "reason": f"Detailed reason for {operation} at {ratio:g}.",
        "error": None,
    }


def test_converts_all_actions_to_detail_sft_without_losing_trim(tmp_path):
    module = load_converter()
    images = [
        image_record(module, "red", "Crop title"),
        image_record(module, "blue", "Crop fill title"),
        image_record(module, "green", "Fill title"),
        image_record(module, "yellow", "Keep title"),
    ]
    raw_path = tmp_path / "image_once_train.parquet"
    pq.write_table(pa.Table.from_pylist(images), raw_path)
    annotations = [
        annotation(images[0], 1.0, "crop", [10, 0, 70, 60], source_index=0),
        annotation(
            images[1],
            1.59,
            "crop_fill",
            [1, 2, 79, 60],
            trim=[1, 2, 1, 0],
            color=[12, 34, 56],
            source_index=1,
        ),
        annotation(images[2], 1.77, "fill", [0, 0, 80, 60], color=[255, 255, 255], source_index=2),
        annotation(images[3], 1.91, "keep", [0, 0, 80, 60], source_index=3),
    ]
    annotations_path = tmp_path / "cropped_v3.parquet"
    pq.write_table(pa.Table.from_pylist(annotations), annotations_path)

    report = module.convert_cropped_v3_to_detail_sft(
        annotations_path=annotations_path,
        raw_train_path=raw_path,
        output_dir=tmp_path / "output",
        validation_fraction=0.5,
        serialized_asset_root="/mnt/blob_output/detail/assets/original",
    )

    assert report["selected_rows"] == 4
    assert report["nonzero_trim_rows_encoded_in_crop_box"] == 1
    assert report["operation_counts"] == {"crop": 1, "crop_fill": 1, "fill": 1, "keep": 1}
    rows = []
    for split in ("train", "validation"):
        rows.extend(pq.read_table(report["output_paths"][split]).to_pylist())
    assert set(rows[0]) == {
        "messages",
        "images",
        "image_id",
        "source_index",
        "target_ratio",
        "is_cropped",
        "is_filled",
    }

    by_title = {row["messages"][0]["content"].splitlines()[2]: row for row in rows}
    crop_fill = by_title["News headline: Crop fill title"]
    target = json.loads(crop_fill["messages"][1]["content"])
    assert list(target) == list(module.TARGET_FIELDS)
    assert target == {
        "target_ratio": 1.59,
        "is_cropped": True,
        "is_filled": True,
        "crop_box": [0.0125, 0.033333, 0.9875, 1.0],
        "fill_color": [12, 34, 56],
        "description": "Detailed reason for crop_fill at 1.59.",
    }
    assert crop_fill["images"] == [
        f"/mnt/blob_output/detail/assets/original/{crop_fill['image_id'][:2]}/{crop_fill['image_id']}.webp"
    ]
    physical_asset = tmp_path / "output" / "assets" / "original" / crop_fill["image_id"][:2]
    assert (physical_asset / f"{crop_fill['image_id']}.webp").is_file()
    assert "Image caption: Caption for Crop fill title" in crop_fill["messages"][0]["content"]

    fill = json.loads(by_title["News headline: Fill title"]["messages"][1]["content"])
    assert fill["crop_box"] is None
    assert fill["fill_color"] == [255, 255, 255]
    keep = json.loads(by_title["News headline: Keep title"]["messages"][1]["content"])
    assert keep["crop_box"] is None
    assert keep["fill_color"] is None


def test_rejects_nonzero_trim_without_cropped_box():
    module = load_converter()
    row = {
        "trace_id": "image",
        "target_ratio": 1.0,
        "original_width": 100,
        "original_height": 100,
        "was_cropped": False,
        "was_padded": True,
        "bbox_pixels": [0, 0, 100, 100],
        "edge_artifact_trim_pixels": [0, 1, 0, 0],
        "padding_color_rgb": [0, 0, 0],
        "reason": "Pad while preserving all content.",
        "error": None,
    }

    with pytest.raises(ValueError, match="nonzero edge trim"):
        module.build_detail_target(row)


def test_rejects_crop_only_box_with_wrong_target_ratio():
    module = load_converter()
    row = {
        "trace_id": "image",
        "target_ratio": 1.0,
        "original_width": 200,
        "original_height": 100,
        "was_cropped": True,
        "was_padded": False,
        "bbox_pixels": [0, 0, 200, 100],
        "edge_artifact_trim_pixels": [0, 0, 0, 0],
        "padding_color_rgb": None,
        "reason": "Crop to the requested square layout.",
        "error": None,
    }

    with pytest.raises(ValueError, match="does not match target ratio"):
        module.build_detail_target(row)