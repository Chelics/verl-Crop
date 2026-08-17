#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from news_crop_benchmark.cropped_override_merge import merge_cropped_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge rendered crop overrides into a new versioned Parquet.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--override-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--row-group-size", type=int, default=64)
    args = parser.parse_args()
    report = merge_cropped_overrides(
        args.base,
        args.override_dir,
        args.output,
        report_path=args.report,
        row_group_size=args.row_group_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()