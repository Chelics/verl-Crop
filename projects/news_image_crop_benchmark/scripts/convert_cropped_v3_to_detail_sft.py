#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from io import BytesIO
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps

from news_crop_benchmark.data import assign_group_split


EdgeTrim = tuple[int, int, int, int]


ANNOTATION_COLUMNS = (
    "source_index",
    "trace_id",
    "headline",
    "caption",
    "target_ratio",
    "original_width",
    "original_height",
    "was_cropped",
    "was_padded",
    "bbox_pixels",
    "edge_artifact_trim_pixels",
    "padding_color_rgb",
    "reason",
    "error",
)
RAW_COLUMNS = ("image_id", "original_image", "title", "ImageCaption")
TARGET_FIELDS = ("target_ratio", "is_cropped", "is_filled", "crop_box", "fill_color", "description")


def normalized_pixel_hash(image: Image.Image) -> str:
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    payload = normalized.tobytes() + f"{normalized.width}x{normalized.height}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _clean_text(value: Any) -> str:
    return " ".join(str(value).split())


def _write_lossless_webp_atomic(path: Path, image: Image.Image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            image.save(output, format="WEBP", lossless=True, method=6)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _pixel_box(value: Sequence[int], width: int, height: int) -> tuple[int, int, int, int]:
    if value is None or len(value) != 4:
        raise ValueError(f"bbox_pixels must contain four values: {value}")
    if any(isinstance(item, bool) or int(item) != item for item in value):
        raise ValueError(f"bbox_pixels must contain integers: {value}")
    box = tuple(int(item) for item in value)
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError(f"bbox_pixels is outside the original image: box={box}, size=({width}, {height})")
    return box


def _edge_trim(annotation: dict[str, Any]) -> EdgeTrim:
    values = annotation["edge_artifact_trim_pixels"]
    if values is None or len(values) != 4:
        raise ValueError(f"edge_artifact_trim_pixels must contain four values: {values}")
    trim = tuple(int(value) for value in values)
    if any(isinstance(value, bool) or int(value) != value for value in values):
        raise ValueError(f"edge_artifact_trim_pixels must contain integers: {values}")
    width = int(annotation["original_width"])
    height = int(annotation["original_height"])
    if any(value < 0 for value in trim):
        raise ValueError(f"edge_artifact_trim_pixels must be non-negative: {values}")
    if trim[0] + trim[2] >= width or trim[1] + trim[3] >= height:
        raise ValueError(f"edge trim removes the complete image: trim={trim}, size=({width}, {height})")
    return trim


def _fill_color(value: Sequence[int] | None, is_filled: bool) -> list[int] | None:
    if not is_filled:
        if value is not None:
            raise ValueError("non-filled rows must not contain padding_color_rgb")
        return None
    if value is None or len(value) != 3:
        raise ValueError("filled rows must contain three padding_color_rgb values")
    color = [int(channel) for channel in value]
    if any(
        isinstance(channel, bool) or int(channel) != channel or not 0 <= int(channel) <= 255
        for channel in value
    ):
        raise ValueError(f"padding_color_rgb must contain integer RGB values: {value}")
    return color


def _normalized_box(box: Sequence[int], width: int, height: int) -> list[float]:
    return [
        round(int(box[0]) / width, 6),
        round(int(box[1]) / height, 6),
        round(int(box[2]) / width, 6),
        round(int(box[3]) / height, 6),
    ]


def build_detail_target(annotation: dict[str, Any]) -> dict[str, Any]:
    if annotation["error"] not in (None, ""):
        raise ValueError(f"annotation contains an error for {annotation['trace_id']}: {annotation['error']}")
    width = int(annotation["original_width"])
    height = int(annotation["original_height"])
    box = _pixel_box(annotation["bbox_pixels"], width, height)
    trim = _edge_trim(annotation)
    is_cropped = annotation["was_cropped"]
    is_filled = annotation["was_padded"]
    if not isinstance(is_cropped, bool) or not isinstance(is_filled, bool):
        raise ValueError("was_cropped and was_padded must be booleans")
    full_box = (0, 0, width, height)
    if not is_cropped and box != full_box:
        raise ValueError("non-cropped rows must retain the full original-image box")
    if any(trim) and not is_cropped:
        raise ValueError("nonzero edge trim must be represented by a cropped source box")
    if is_cropped and not (
        box[0] + 1 >= trim[0]
        and box[1] + 1 >= trim[1]
        and box[2] <= width - trim[2] + 1
        and box[3] <= height - trim[3] + 1
    ):
        raise ValueError(f"bbox_pixels does not encode edge trim within one-pixel tolerance: box={box}, trim={trim}")
    if is_cropped and not is_filled:
        actual_ratio = (box[2] - box[0]) / (box[3] - box[1])
        target_ratio = float(annotation["target_ratio"])
        if abs(actual_ratio - target_ratio) / target_ratio > 0.002:
            raise ValueError(
                f"crop-only bbox does not match target ratio: actual={actual_ratio}, target={target_ratio}"
            )
    description = str(annotation["reason"] or "").strip()
    if not description:
        raise ValueError("detail SFT rows must contain a non-empty reason")

    target = {
        "target_ratio": float(annotation["target_ratio"]),
        "is_cropped": is_cropped,
        "is_filled": is_filled,
        "crop_box": _normalized_box(box, width, height) if is_cropped else None,
        "fill_color": _fill_color(annotation["padding_color_rgb"], is_filled),
        "description": description,
    }
    if tuple(target) != TARGET_FIELDS:
        raise AssertionError("detail target field order changed")
    return target


def build_detail_prompt(title: str, caption: str, target_ratio: float) -> str:
    clean_title = _clean_text(title)
    clean_caption = _clean_text(caption)
    if not clean_title or not clean_caption:
        raise ValueError("title and caption must be non-empty")
    return (
        "<image>\n"
        "Create a crop and color-fill plan for this news image based on its content.\n"
        f"News headline: {clean_title}\n"
        f"Image caption: {clean_caption}\n"
        f"Target aspect ratio (width/height): {target_ratio:g}\n"
        "Return only one JSON object with these fields: target_ratio, is_cropped, is_filled, crop_box, "
        "fill_color, and description. Use normalized [left, top, right, bottom] coordinates for crop_box. "
        "crop_box is applied directly to the original image. When is_filled is false, crop_box itself must "
        "have the target aspect ratio. The description must explain in detail which visual elements are "
        "preserved or removed, why the crop is safe, and why color fill is needed when applicable."
    )


def _select_image_ids(
    annotation_rows: Sequence[dict[str, Any]],
    seed: int,
    validation_fraction: float,
    max_train_images: int | None,
    max_validation_images: int | None,
) -> tuple[dict[str, str], dict[str, set[str]]]:
    ordered_ids = list(dict.fromkeys(str(row["trace_id"]) for row in annotation_rows))
    split_ids: dict[str, list[str]] = {"train": [], "validation": []}
    for image_id in ordered_ids:
        split = assign_group_split(
            image_id,
            seed=seed,
            fractions=(1.0 - validation_fraction, validation_fraction, 0.0),
        )
        split_ids[split].append(image_id)
    if max_train_images is not None:
        split_ids["train"] = split_ids["train"][:max_train_images]
    if max_validation_images is not None:
        split_ids["validation"] = split_ids["validation"][:max_validation_images]
    if not split_ids["train"] or not split_ids["validation"]:
        raise ValueError("both train and validation splits must contain at least one image")
    selected = {split: set(values) for split, values in split_ids.items()}
    image_to_split = {image_id: split for split, values in selected.items() for image_id in values}
    return image_to_split, selected


def _manifest_hash(rows: Sequence[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def convert_cropped_v3_to_detail_sft(
    *,
    annotations_path: Path,
    raw_train_path: Path,
    output_dir: Path,
    seed: int = 42,
    validation_fraction: float = 0.1,
    max_train_images: int | None = None,
    max_validation_images: int | None = None,
    serialized_asset_root: str | None = None,
) -> dict[str, Any]:
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1)")
    for value, name in (
        (max_train_images, "max_train_images"),
        (max_validation_images, "max_validation_images"),
    ):
        if value is not None and value <= 0:
            raise ValueError(f"{name} must be positive")
    if serialized_asset_root is not None and not PurePosixPath(serialized_asset_root).is_absolute():
        raise ValueError("serialized_asset_root must be an absolute POSIX path")

    annotation_file = pq.ParquetFile(annotations_path)
    missing_annotations = sorted(set(ANNOTATION_COLUMNS) - set(annotation_file.schema_arrow.names))
    if missing_annotations:
        raise ValueError(f"annotations are missing required columns: {missing_annotations}")
    annotation_rows = pq.read_table(
        annotations_path,
        columns=list(ANNOTATION_COLUMNS),
        pre_buffer=False,
    ).to_pylist()
    if not annotation_rows:
        raise ValueError("annotations are empty")
    annotation_keys = [(str(row["trace_id"]), float(row["target_ratio"])) for row in annotation_rows]
    if len(annotation_keys) != len(set(annotation_keys)):
        raise ValueError("annotations contain duplicate trace_id and target_ratio keys")

    image_to_split, selected_ids = _select_image_ids(
        annotation_rows,
        seed,
        validation_fraction,
        max_train_images,
        max_validation_images,
    )
    selected_annotations = [row for row in annotation_rows if str(row["trace_id"]) in image_to_split]
    selected_image_ids = set(image_to_split)
    dimensions_by_image: dict[str, tuple[int, int]] = {}
    for annotation in selected_annotations:
        image_id = str(annotation["trace_id"])
        dimensions = (int(annotation["original_width"]), int(annotation["original_height"]))
        if image_id in dimensions_by_image and dimensions_by_image[image_id] != dimensions:
            raise ValueError(f"source dimensions differ across target ratios for {image_id}")
        dimensions_by_image[image_id] = dimensions

    output_dir = output_dir.resolve()
    asset_root = output_dir / "assets" / "original"
    raw_file = pq.ParquetFile(raw_train_path)
    missing_raw = sorted(set(RAW_COLUMNS) - set(raw_file.schema_arrow.names))
    if missing_raw:
        raise ValueError(f"raw train data is missing required columns: {missing_raw}")
    raw_by_id: dict[str, dict[str, Any]] = {}
    for batch in raw_file.iter_batches(batch_size=16, columns=list(RAW_COLUMNS), use_threads=False):
        for row in pa.Table.from_batches([batch]).to_pylist():
            image_id = str(row["image_id"])
            if image_id not in selected_image_ids:
                continue
            if image_id in raw_by_id:
                raise ValueError(f"raw train data contains duplicate image_id: {image_id}")
            with Image.open(BytesIO(bytes(row["original_image"]))) as encoded_image:
                encoded_image.load()
                if normalized_pixel_hash(encoded_image) != image_id:
                    raise ValueError(f"normalized pixel hash does not match image_id: {image_id}")
                original = ImageOps.exif_transpose(encoded_image).convert("RGB")
            try:
                if original.size != dimensions_by_image[image_id]:
                    raise ValueError(
                        f"annotation dimensions differ from source image for {image_id}: "
                        f"annotation={dimensions_by_image[image_id]}, source={original.size}"
                    )
                asset_path = (asset_root / image_id[:2] / f"{image_id}.webp").resolve()
                if asset_path.exists():
                    with Image.open(asset_path) as existing:
                        if normalized_pixel_hash(existing) != image_id:
                            raise ValueError(f"existing original asset differs from source: {asset_path}")
                else:
                    _write_lossless_webp_atomic(asset_path, original)
            finally:
                original.close()
            serialized_asset_path = (
                str(PurePosixPath(serialized_asset_root) / image_id[:2] / f"{image_id}.webp")
                if serialized_asset_root is not None
                else str(asset_path)
            )
            raw_by_id[image_id] = {
                "title": row["title"],
                "caption": row["ImageCaption"],
                "asset_path": serialized_asset_path,
            }
    missing_images = sorted(selected_image_ids - set(raw_by_id))
    if missing_images:
        raise ValueError(f"annotations cannot be joined to raw images: {missing_images[:5]}")

    output_rows: dict[str, list[dict[str, Any]]] = {"train": [], "validation": []}
    operation_counts: Counter[str] = Counter()
    reason_lengths: list[int] = []
    nonzero_trim_rows = 0
    for annotation in selected_annotations:
        image_id = str(annotation["trace_id"])
        raw = raw_by_id[image_id]
        title = _clean_text(raw["title"])
        caption = _clean_text(raw["caption"])
        if title != _clean_text(annotation["headline"]) or caption != _clean_text(annotation["caption"]):
            raise ValueError(f"annotation title or caption does not match raw image: {image_id}")
        target = build_detail_target(annotation)
        operation = {
            (True, False): "crop",
            (True, True): "crop_fill",
            (False, True): "fill",
            (False, False): "keep",
        }[(target["is_cropped"], target["is_filled"])]
        operation_counts[operation] += 1
        nonzero_trim_rows += int(any(annotation["edge_artifact_trim_pixels"]))
        reason_lengths.append(len(target["description"]))
        split = image_to_split[image_id]
        output_rows[split].append(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": build_detail_prompt(title, caption, target["target_ratio"]),
                    },
                    {
                        "role": "assistant",
                        "content": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
                    },
                ],
                "images": [raw["asset_path"]],
                "image_id": image_id,
                "source_index": int(annotation["source_index"]),
                "target_ratio": target["target_ratio"],
                "is_cropped": target["is_cropped"],
                "is_filled": target["is_filled"],
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = {}
    for split, rows in output_rows.items():
        output_path = output_dir / f"{split}.parquet"
        temporary_path = output_path.with_suffix(".parquet.tmp")
        pq.write_table(pa.Table.from_pylist(rows), temporary_path, compression="zstd", row_group_size=256)
        temporary_path.replace(output_path)
        output_paths[split] = str(output_path.resolve())

    manifest_path = output_dir / "split_manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps({"image_id": image_id, "split": split}, sort_keys=True) + "\n"
            for split in ("train", "validation")
            for image_id in sorted(selected_ids[split])
        ),
        encoding="utf-8",
    )
    report = {
        "protocol": "cropped-v3-detail-sft-v1",
        "annotations_path": os.path.abspath(os.path.expanduser(str(annotations_path))),
        "annotation_manifest_sha256": _manifest_hash(annotation_rows),
        "raw_train_path": os.path.abspath(os.path.expanduser(str(raw_train_path))),
        "output_dir": str(output_dir),
        "output_paths": output_paths,
        "serialized_asset_root": serialized_asset_root or str(asset_root),
        "seed": seed,
        "validation_fraction": validation_fraction,
        "source_rows": len(annotation_rows),
        "source_images": len({str(row["trace_id"]) for row in annotation_rows}),
        "selected_rows": sum(len(rows) for rows in output_rows.values()),
        "selected_images": len(selected_image_ids),
        "split_rows": {split: len(rows) for split, rows in output_rows.items()},
        "split_images": {split: len(values) for split, values in selected_ids.items()},
        "operation_counts": dict(sorted(operation_counts.items())),
        "nonzero_trim_rows_encoded_in_crop_box": nonzero_trim_rows,
        "target_fields": list(TARGET_FIELDS),
        "bbox_coordinate_space": "normalized_original_image",
        "reason_in_target": True,
        "reason_length_chars": {
            "min": min(reason_lengths),
            "mean": mean(reason_lengths),
            "max": max(reason_lengths),
        },
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert cropped_v3 into detailed Swift/verl multimodal SFT data.")
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--raw-train", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--max-train-images", type=int)
    parser.add_argument("--max-validation-images", type=int)
    parser.add_argument("--serialized-asset-root")
    args = parser.parse_args()
    report = convert_cropped_v3_to_detail_sft(
        annotations_path=args.annotations.expanduser(),
        raw_train_path=args.raw_train.expanduser(),
        output_dir=args.output_dir.expanduser(),
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        max_train_images=args.max_train_images,
        max_validation_images=args.max_validation_images,
        serialized_asset_root=args.serialized_asset_root,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()