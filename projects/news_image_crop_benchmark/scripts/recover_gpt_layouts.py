#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from news_crop_benchmark.gpt_bbox_recovery import recover_gpt_layouts


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover executable layout geometry from original/manual image pairs.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--audited-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-gpt", type=int)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()
    records = recover_gpt_layouts(
        args.train,
        args.manual,
        args.audited_manifest,
        args.output_dir,
        max_gpt=args.max_gpt,
        max_attempts=args.max_attempts,
    )
    print(f"Recovered {len(records)} layout actions in {args.output_dir}")


if __name__ == "__main__":
    main()