#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq

LIGHTWEIGHT_COLUMNS = (
    "GemId",
    "GemSnapshotId",
    "GemTitle",
    "OriginalImageUrl",
    "CroppedImageUrl",
    "IsCroppedImage",
    "Reason",
)
REQUIRED_COLUMNS = (*LIGHTWEIGHT_COLUMNS, "OriginalImageBytes")


def inspect_source(path: Path) -> dict:
    metadata = pq.read_metadata(path)
    available_columns = set(metadata.schema.names)
    missing_columns = sorted(set(REQUIRED_COLUMNS) - available_columns)
    if missing_columns:
        raise ValueError(f"source parquet is missing columns: {missing_columns}")

    table = pq.read_table(path, columns=list(LIGHTWEIGHT_COLUMNS))
    null_counts = {name: table[name].null_count for name in LIGHTWEIGHT_COLUMNS}
    unique_counts = {
        name: pc.count_distinct(table[name]).as_py()
        for name in ("GemId", "GemSnapshotId", "OriginalImageUrl", "CroppedImageUrl", "Reason")
    }
    cropped_values = [
        {"value": item["values"], "count": item["counts"]}
        for item in pc.value_counts(table["IsCroppedImage"]).to_pylist()
    ]

    return {
        "path": str(path),
        "file_size_bytes": path.stat().st_size,
        "rows": metadata.num_rows,
        "row_groups": metadata.num_row_groups,
        "columns": metadata.schema.names,
        "null_counts": null_counts,
        "unique_counts": unique_counts,
        "is_cropped_image_values": cropped_values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the raw news image crop Parquet without reading image bytes.")
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    print(json.dumps(inspect_source(args.input.expanduser()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()