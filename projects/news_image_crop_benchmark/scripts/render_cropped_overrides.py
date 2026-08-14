#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from news_crop_benchmark.cropped_overrides import render_overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Render reviewed crop overrides from image_once train images.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quality", type=int, default=95)
    args = parser.parse_args()
    records = render_overrides(args.train, args.manifest, args.output_dir, quality=args.quality)
    print(f"Rendered {len(records)} overrides to {args.output_dir}")


if __name__ == "__main__":
    main()