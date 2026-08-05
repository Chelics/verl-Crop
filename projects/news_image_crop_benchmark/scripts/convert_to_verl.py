#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from news_crop_benchmark.data import (
    SPLIT_NAMES,
    assign_group_split,
    asset_id,
    build_verl_row,
    training_sample_id,
)
from news_crop_benchmark.geometry import TARGET_RATIOS

SOURCE_COLUMNS = ("TraceId", "GemTitle", "OriginalImageUrl", "OriginalImageBytes")
IMAGE_EXTENSIONS = {"GIF": "gif", "JPEG": "jpg", "PNG": "png", "WEBP": "webp"}

VERL_SCHEMA = pa.schema(
    [
        pa.field("data_source", pa.string(), nullable=False),
        pa.field(
            "prompt",
            pa.list_(
                pa.struct(
                    [
                        pa.field("role", pa.string(), nullable=False),
                        pa.field("content", pa.string(), nullable=False),
                    ]
                )
            ),
            nullable=False,
        ),
        pa.field("images", pa.list_(pa.string()), nullable=False),
        pa.field("ability", pa.string(), nullable=False),
        pa.field(
            "reward_model",
            pa.struct(
                [
                    pa.field("style", pa.string(), nullable=False),
                    pa.field("ground_truth", pa.string(), nullable=False),
                ]
            ),
            nullable=False,
        ),
        pa.field(
            "extra_info",
            pa.struct(
                [
                    pa.field("index", pa.int64(), nullable=False),
                    pa.field("sample_id", pa.string(), nullable=False),
                    pa.field("split", pa.string(), nullable=False),
                    pa.field("title", pa.string(), nullable=False),
                    pa.field("target_ratio", pa.float64(), nullable=False),
                    pa.field("image_width", pa.int64(), nullable=False),
                    pa.field("image_height", pa.int64(), nullable=False),
                    pa.field("original_image_path", pa.string(), nullable=False),
                ]
            ),
            nullable=False,
        ),
    ]
)


def normalize_title(title: str | None) -> str:
    return " ".join((title or "").split())


def inspect_image(payload: bytes) -> tuple[str, int, int, str]:
    checksum = hashlib.sha256(payload).hexdigest()
    with Image.open(BytesIO(payload)) as image:
        image_format = image.format
        width, height = image.size
        image.verify()
    if image_format not in IMAGE_EXTENSIONS:
        raise ValueError(f"unsupported image format: {image_format}")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    return IMAGE_EXTENSIONS[image_format], width, height, checksum


def atomic_write_bytes(path: Path, payload: bytes) -> None:
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


def materialize_asset(
    *,
    payload: bytes,
    original_url: str,
    asset_root: Path,
) -> dict[str, Any]:
    extension, width, height, checksum = inspect_image(payload)
    identifier = asset_id(original_url)
    path = (asset_root / identifier[:2] / f"{identifier}.{extension}").resolve()

    if path.exists():
        if path.stat().st_size != len(payload):
            raise ValueError(f"existing asset size mismatch: {path}")
    else:
        atomic_write_bytes(path, payload)

    return {
        "asset_id": identifier,
        "path": str(path),
        "width": width,
        "height": height,
        "checksum": checksum,
        "extension": extension,
    }


def write_parquet_atomic(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    table = pa.Table.from_pylist(rows, schema=VERL_SCHEMA)
    pq.write_table(table, temporary_path, compression="zstd", row_group_size=1024)
    temporary_path.replace(output_path)


def convert_dataset(
    *,
    source_path: Path,
    output_dir: Path,
    prefix: str,
    batch_size: int,
    seed: int,
    limit: int | None,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    output_dir = output_dir.resolve()
    asset_root = output_dir / f"{prefix}_assets" / "original"
    parquet_file = pq.ParquetFile(source_path)
    source_rows = 0
    empty_titles = 0
    invalid_images = 0
    duplicate_title_image_pairs = 0
    conflicting_asset_payloads = 0
    assets: dict[str, dict[str, Any]] = {}
    trusted_pairs: dict[tuple[str, str], dict[str, Any]] = {}
    rejected_asset_urls: set[str] = set()

    stop = False
    for batch in parquet_file.iter_batches(batch_size=batch_size, columns=list(SOURCE_COLUMNS), use_threads=False):
        for row in batch.to_pylist():
            if limit is not None and source_rows >= limit:
                stop = True
                break
            source_row_index = source_rows
            source_rows += 1

            original_url = row["OriginalImageUrl"]
            title = normalize_title(row["GemTitle"])
            payload = row["OriginalImageBytes"]
            if not original_url or not payload:
                invalid_images += 1
                continue
            if not title:
                empty_titles += 1
                continue

            previous_asset = assets.get(original_url)
            if previous_asset is not None:
                if hashlib.sha256(payload).hexdigest() != previous_asset["checksum"]:
                    conflicting_asset_payloads += 1
                    rejected_asset_urls.add(original_url)
                    assets.pop(original_url, None)
                    continue
                current_asset = previous_asset
            else:
                try:
                    current_asset = materialize_asset(payload=payload, original_url=original_url, asset_root=asset_root)
                except (OSError, ValueError):
                    invalid_images += 1
                    rejected_asset_urls.add(original_url)
                    assets.pop(original_url, None)
                    continue
                if original_url not in rejected_asset_urls:
                    assets[original_url] = current_asset

            pair_key = (original_url, title)
            if pair_key in trusted_pairs:
                duplicate_title_image_pairs += 1
                continue
            trusted_pairs[pair_key] = {
                "source_row_index": source_row_index,
                "trace_id": row["TraceId"],
                "original_url": original_url,
                "title": title,
            }
        if stop:
            break

    rows_by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    ratio_counts: Counter[str] = Counter()
    skipped_pairs_missing_asset = 0
    expanded_index = 0
    seen_training_ids: set[str] = set()

    for (original_url, title), pair in trusted_pairs.items():
        asset = assets.get(original_url)
        if asset is None or original_url in rejected_asset_urls:
            skipped_pairs_missing_asset += 1
            continue
        split = assign_group_split(original_url, seed=seed)
        for target_ratio in TARGET_RATIOS:
            identifier = training_sample_id(original_url, title, target_ratio)
            if identifier in seen_training_ids:
                raise ValueError(f"duplicate training sample ID: {identifier}")
            seen_training_ids.add(identifier)
            row = build_verl_row(
                sample_identifier=identifier,
                source_index=expanded_index,
                split=split,
                title=title,
                original_image_path=asset["path"],
                image_width=asset["width"],
                image_height=asset["height"],
                target_ratio=target_ratio,
            )
            rows_by_split[split].append(row)
            ratio_counts[f"{target_ratio:g}"] += 1
            expanded_index += 1

    output_paths = {}
    for split, rows in rows_by_split.items():
        output_path = output_dir / f"{prefix}_{split}.parquet"
        write_parquet_atomic(rows, output_path)
        output_paths[split] = str(output_path)

    manifest_rows = [
        {
            "original_url": original_url,
            **asset,
        }
        for original_url, asset in sorted(assets.items())
        if original_url not in rejected_asset_urls
    ]
    manifest_path = output_dir / f"{prefix}_assets.parquet"
    manifest_temporary_path = manifest_path.with_suffix(f"{manifest_path.suffix}.tmp")
    pq.write_table(pa.Table.from_pylist(manifest_rows), manifest_temporary_path, compression="zstd")
    manifest_temporary_path.replace(manifest_path)

    report = {
        "source_path": str(source_path.resolve()),
        "output_dir": str(output_dir),
        "prefix": prefix,
        "seed": seed,
        "limit": limit,
        "source_rows": source_rows,
        "unique_original_assets": len(manifest_rows),
        "unique_trusted_title_image_pairs": len(trusted_pairs) - skipped_pairs_missing_asset,
        "duplicate_title_image_pairs": duplicate_title_image_pairs,
        "expanded_rows": sum(len(rows) for rows in rows_by_split.values()),
        "split_rows": {split: len(rows) for split, rows in rows_by_split.items()},
        "ratio_rows": dict(sorted(ratio_counts.items())),
        "rejections": {
            "empty_title_rows": empty_titles,
            "invalid_image_rows": invalid_images,
            "conflicting_asset_payloads": conflicting_asset_payloads,
            "trusted_pairs_missing_asset": skipped_pairs_missing_asset,
        },
        "outputs": output_paths,
        "asset_manifest": str(manifest_path),
    }
    report_path = output_dir / f"{prefix}_conversion_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert trusted title/original-image pairs into verl Parquets.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="news_image_crop")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    report = convert_dataset(
        source_path=args.input.expanduser(),
        output_dir=args.output_dir.expanduser(),
        prefix=args.prefix,
        batch_size=args.batch_size,
        seed=args.seed,
        limit=args.limit,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()