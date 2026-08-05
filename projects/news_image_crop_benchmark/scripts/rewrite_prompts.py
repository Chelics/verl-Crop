#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from news_crop_benchmark.data import build_prompt


def rewrite_prompts(path: Path) -> dict:
    table = pq.read_table(path)
    rows = table.to_pylist()
    changed = 0
    for row in rows:
        extra_info = row["extra_info"]
        prompt = build_prompt(extra_info["title"], float(extra_info["target_ratio"]))
        if row["prompt"] != [{"role": "user", "content": prompt}]:
            row["prompt"] = [{"role": "user", "content": prompt}]
            changed += 1

    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    rewritten = pa.Table.from_pylist(rows, schema=table.schema)
    pq.write_table(rewritten, temporary_path, compression="zstd", row_group_size=1024)
    temporary_path.replace(path)
    return {"path": str(path.resolve()), "rows": len(rows), "changed": changed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite converted Parquet prompts using the current prompt contract.")
    parser.add_argument("--data", type=Path, action="append", required=True)
    args = parser.parse_args()

    reports = [rewrite_prompts(path.expanduser()) for path in args.data]
    print(json.dumps(reports, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
