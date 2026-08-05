#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from news_crop_benchmark.data import SPLIT_NAMES, assign_group_split, training_sample_id


def write_parquet_atomic(rows: list[dict[str, Any]], schema: pa.Schema, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(table, temporary_path, compression="zstd", row_group_size=1024)
    temporary_path.replace(output_path)


def resplit_dataset(
    *,
    input_paths: list[Path],
    asset_manifest_path: Path,
    output_dir: Path,
    prefix: str,
    seed: int,
) -> dict[str, Any]:
    if not input_paths:
        raise ValueError("at least one input Parquet is required")

    manifest_rows = pq.read_table(asset_manifest_path).to_pylist()
    path_to_checksum: dict[str, str] = {}
    for row in manifest_rows:
        previous_checksum = path_to_checksum.setdefault(row["path"], row["checksum"])
        if previous_checksum != row["checksum"]:
            raise ValueError(f"asset path maps to conflicting checksums: {row['path']}")

    input_rows: list[dict[str, Any]] = []
    schema: pa.Schema | None = None
    for input_path in input_paths:
        table = pq.read_table(input_path)
        if schema is None:
            schema = table.schema
        elif table.schema != schema:
            raise ValueError(f"input schema mismatch: {input_path}")
        input_rows.extend(table.to_pylist())
    assert schema is not None

    input_rows.sort(key=lambda row: row["extra_info"]["index"])
    unique_rows: dict[tuple[str, str, float], dict[str, Any]] = {}
    missing_manifest_paths: set[str] = set()
    duplicate_content_tasks = 0
    for row in input_rows:
        images = row["images"]
        if len(images) != 1:
            raise ValueError("each row must contain exactly one image path")
        image_path = images[0]
        checksum = path_to_checksum.get(image_path)
        if checksum is None:
            missing_manifest_paths.add(image_path)
            continue
        title = " ".join(row["extra_info"]["title"].split())
        target_ratio = float(row["extra_info"]["target_ratio"])
        key = (checksum, title, target_ratio)
        if key in unique_rows:
            duplicate_content_tasks += 1
            continue
        unique_rows[key] = row

    if missing_manifest_paths:
        examples = sorted(missing_manifest_paths)[:3]
        raise ValueError(f"image paths missing from asset manifest: {examples}")

    rows_by_split: dict[str, list[dict[str, Any]]] = {name: [] for name in SPLIT_NAMES}
    split_checksums: dict[str, set[str]] = {name: set() for name in SPLIT_NAMES}
    ratio_counts: Counter[str] = Counter()
    sample_ids: set[str] = set()
    for new_index, ((checksum, title, target_ratio), row) in enumerate(unique_rows.items()):
        split = assign_group_split(checksum, seed=seed)
        sample_identifier = training_sample_id(checksum, title, target_ratio)
        if sample_identifier in sample_ids:
            raise ValueError(f"duplicate training sample ID: {sample_identifier}")
        sample_ids.add(sample_identifier)

        row["extra_info"]["index"] = new_index
        row["extra_info"]["sample_id"] = sample_identifier
        row["extra_info"]["split"] = split
        rows_by_split[split].append(row)
        split_checksums[split].add(checksum)
        ratio_counts[f"{target_ratio:g}"] += 1

    for first_index, first in enumerate(SPLIT_NAMES):
        for second in SPLIT_NAMES[first_index + 1 :]:
            overlap = split_checksums[first] & split_checksums[second]
            if overlap:
                raise AssertionError(f"image checksum leakage between {first} and {second}: {len(overlap)}")

    output_dir = output_dir.resolve()
    output_paths = {}
    for split, rows in rows_by_split.items():
        output_path = output_dir / f"{prefix}_{split}.parquet"
        write_parquet_atomic(rows, schema, output_path)
        output_paths[split] = str(output_path)

    report = {
        "input_paths": [str(path.resolve()) for path in input_paths],
        "asset_manifest": str(asset_manifest_path.resolve()),
        "output_dir": str(output_dir),
        "prefix": prefix,
        "seed": seed,
        "split_group_key": "image_checksum",
        "deduplication_key": ["image_checksum", "normalized_title", "target_ratio"],
        "input_rows": len(input_rows),
        "output_rows": sum(len(rows) for rows in rows_by_split.values()),
        "duplicate_content_tasks_removed": duplicate_content_tasks,
        "unique_image_contents": len(set(path_to_checksum.values())),
        "split_rows": {split: len(rows) for split, rows in rows_by_split.items()},
        "split_image_contents": {split: len(checksums) for split, checksums in split_checksums.items()},
        "ratio_rows": dict(sorted(ratio_counts.items())),
        "outputs": output_paths,
    }
    report_path = output_dir / f"{prefix}_content_split_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Deduplicate and split existing verl data by image content checksum.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--asset-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="news_image_crop")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report = resplit_dataset(
        input_paths=[path.expanduser() for path in args.input],
        asset_manifest_path=args.asset_manifest.expanduser(),
        output_dir=args.output_dir.expanduser(),
        prefix=args.prefix,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()