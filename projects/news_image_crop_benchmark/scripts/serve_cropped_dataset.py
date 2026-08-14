#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from news_crop_benchmark.cropped_dataset_viewer import (
    DATASET_VIEWER_CSS,
    CroppedDataset,
    build_cropped_dataset_app,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse rendered images stored in cropped.parquet.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--override-dir", type=Path)
    parser.add_argument("--reason-prefix")
    parser.add_argument("--preview-max-side", type=int, default=1400)
    parser.add_argument("--preview-quality", type=int, default=90)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7865)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.server_port <= 0:
        raise ValueError("server-port must be positive")
    dataset = CroppedDataset(
        args.data,
        cache_dir=args.cache_dir,
        override_dir=args.override_dir,
        reason_prefix=args.reason_prefix,
    )
    print(f"Preparing {len(dataset.rows)} previews in {dataset.cache_dir}", flush=True)
    dataset.prepare_previews(
        maximum_side=args.preview_max_side,
        quality=args.preview_quality,
        progress=lambda completed, total: print(f"Prepared {completed}/{total}", flush=True),
    )
    app = build_cropped_dataset_app(dataset)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        css=DATASET_VIEWER_CSS,
    )


if __name__ == "__main__":
    main()