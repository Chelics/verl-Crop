#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from news_crop_benchmark.data import assign_group_split, asset_id, sample_id

SOURCE_COLUMNS = (
    "RequestId",
    "TraceId",
    "GemId",
    "GemSnapshotId",
    "Scenario",
    "GemTitle",
    "ImageCaption",
    "OriginalImageUrl",
    "CroppedImageUrl",
    "IsCroppedImage",
    "Reason",
)


def build_source_index(source_path: Path, output_path: Path, seed: int, limit: int | None) -> dict:
    source_table = pq.read_table(source_path, columns=list(SOURCE_COLUMNS))
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        source_table = source_table.slice(0, limit)

    rows = source_table.to_pylist()
    trace_ids = [row["TraceId"] for row in rows]
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("TraceId must be unique within the selected source rows")

    index_rows = []
    group_splits: dict[str, str] = {}
    for source_row_index, row in enumerate(rows):
        original_url = row["OriginalImageUrl"]
        cropped_url = row["CroppedImageUrl"]
        split = assign_group_split(original_url, seed=seed)
        previous_split = group_splits.setdefault(original_url, split)
        if previous_split != split:
            raise AssertionError(f"group split changed for {original_url}")

        index_rows.append(
            {
                "source_row_index": source_row_index,
                "sample_id": sample_id(row["TraceId"]),
                "original_asset_id": asset_id(original_url),
                "split": split,
                "request_id": row["RequestId"],
                "trace_id": row["TraceId"],
                "gem_id": row["GemId"],
                "gem_snapshot_id": row["GemSnapshotId"],
                "scenario": row["Scenario"],
                "title": row["GemTitle"],
                "image_caption": row["ImageCaption"],
                "original_url": original_url,
                # Neither field below is reliably paired with this title. Retain them only for source auditing.
                "unpaired_cropped_url": cropped_url,
                "source_is_cropped_image": row["IsCroppedImage"],
                "unpaired_reason": row["Reason"],
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    pq.write_table(pa.Table.from_pylist(index_rows), temporary_path, compression="zstd")
    temporary_path.replace(output_path)

    split_counts = Counter(row["split"] for row in index_rows)
    report = {
        "source_path": str(source_path),
        "output_path": str(output_path),
        "seed": seed,
        "limit": limit,
        "rows": len(index_rows),
        "unique_original_assets": len({row["original_asset_id"] for row in index_rows}),
        "unique_unpaired_cropped_urls": len({row["unpaired_cropped_url"] for row in index_rows}),
        "split_rows": dict(sorted(split_counts.items())),
        "split_groups": dict(sorted(Counter(group_splits.values()).items())),
    }
    report_path = output_path.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a lightweight, deterministic index from the source Parquet.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    report = build_source_index(args.input.expanduser(), args.output.expanduser(), args.seed, args.limit)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
