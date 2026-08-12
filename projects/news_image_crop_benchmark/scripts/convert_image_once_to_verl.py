#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps

from news_crop_benchmark.data import build_verl_row, load_policy_prompt_template, training_sample_id
from news_crop_benchmark.geometry import TARGET_RATIOS

SOURCE_COLUMNS = ("image_id", "original_image", "title", "ImageCaption")


def normalized_pixel_hash(image: Image.Image) -> str:
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    payload = normalized.tobytes() + f"{normalized.width}x{normalized.height}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
        temporary_path.replace(path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def convert_split(
    *,
    source_path: Path,
    split: str,
    output_dir: Path,
    asset_root: Path,
    policy_prompt_template: str,
    limit: int | None = None,
) -> dict[str, Any]:
    if split not in {"train", "test"}:
        raise ValueError("split must be 'train' or 'test'")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    parquet = pq.ParquetFile(source_path)
    missing = sorted(set(SOURCE_COLUMNS) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"source dataset is missing required columns: {missing}")

    source_rows = pq.read_table(source_path, columns=list(SOURCE_COLUMNS)).to_pylist()
    if limit is not None:
        source_rows = source_rows[:limit]

    output_rows: list[dict[str, Any]] = []
    seen_image_ids: set[str] = set()
    for source_index, source_row in enumerate(source_rows):
        image_id = str(source_row["image_id"])
        title = " ".join(str(source_row["title"]).split())
        caption = " ".join(str(source_row["ImageCaption"]).split())
        payload = bytes(source_row["original_image"])
        if not image_id or image_id in seen_image_ids:
            raise ValueError(f"image_id must be non-empty and unique within {split}: {image_id!r}")
        if not title or not payload:
            raise ValueError(f"title and original_image must be non-empty for {image_id}")

        with Image.open(BytesIO(payload)) as encoded_image:
            encoded_image.load()
            if normalized_pixel_hash(encoded_image) != image_id:
                raise ValueError(f"normalized pixel hash does not match image_id: {image_id}")
            normalized = ImageOps.exif_transpose(encoded_image).convert("RGB")
            image_width, image_height = normalized.size
            image_format = (encoded_image.format or "WEBP").lower()
        extension = "jpg" if image_format == "jpeg" else image_format
        asset_path = (asset_root / image_id[:2] / f"{image_id}.{extension}").resolve()
        if asset_path.exists():
            if asset_path.read_bytes() != payload:
                raise ValueError(f"existing asset differs from source payload: {asset_path}")
        else:
            write_bytes_atomic(asset_path, payload)

        for target_ratio in TARGET_RATIOS:
            identifier = training_sample_id(image_id, title, target_ratio)
            row = build_verl_row(
                sample_identifier=identifier,
                source_index=len(output_rows),
                split=split,
                title=title,
                caption=caption,
                original_image_path=str(asset_path),
                image_width=image_width,
                image_height=image_height,
                target_ratio=target_ratio,
                policy_prompt_template=policy_prompt_template,
            )
            output_rows.append(row)
        seen_image_ids.add(image_id)

    output_path = output_dir / f"image_once_{split}_verl.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(output_rows), temporary_path, compression="zstd", row_group_size=1024)
    temporary_path.replace(output_path)
    return {
        "split": split,
        "source_path": str(source_path.resolve()),
        "source_rows": len(source_rows),
        "output_rows": len(output_rows),
        "unique_images": len(seen_image_ids),
        "output_path": str(output_path.resolve()),
    }


def convert_image_once_datasets(
    *,
    train_path: Path,
    test_path: Path,
    output_dir: Path,
    policy_prompt_path: Path,
    limit: int | None = None,
) -> dict[str, Any]:
    policy_prompt_template = load_policy_prompt_template(policy_prompt_path)
    output_dir = output_dir.resolve()
    asset_root = output_dir / "assets" / "original"
    train = convert_split(
        source_path=train_path,
        split="train",
        output_dir=output_dir,
        asset_root=asset_root,
        policy_prompt_template=policy_prompt_template,
        limit=limit,
    )
    test = convert_split(
        source_path=test_path,
        split="test",
        output_dir=output_dir,
        asset_root=asset_root,
        policy_prompt_template=policy_prompt_template,
        limit=limit,
    )
    report = {
        "train": train,
        "test": test,
        "asset_root": str(asset_root.resolve()),
        "policy_prompt_path": str(policy_prompt_path.resolve()),
        "policy_prompt_sha256": hashlib.sha256(policy_prompt_template.encode("utf-8")).hexdigest(),
        "target_ratios": list(TARGET_RATIOS),
    }
    (output_dir / "conversion_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert raw image_once train/test data to four-ratio verl Parquets.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--policy-prompt-path", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    report = convert_image_once_datasets(
        train_path=args.train,
        test_path=args.test,
        output_dir=args.output_dir,
        policy_prompt_path=args.policy_prompt_path,
        limit=args.limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()