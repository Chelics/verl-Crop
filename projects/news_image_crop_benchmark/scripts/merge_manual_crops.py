#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from news_crop_benchmark.manual_crop_merge import merge_manual_crops


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge reviewed manual crops into a new baseline Parquet.")
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--row-group-size", type=int, default=64)
    args = parser.parse_args()
    report = merge_manual_crops(
        args.base,
        args.manual,
        args.output,
        report_path=args.report,
        row_group_size=args.row_group_size,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()