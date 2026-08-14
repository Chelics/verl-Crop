#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from news_crop_benchmark.manual_crops_viewer import MANUAL_VIEWER_CSS, ManualCropsDataset, build_manual_crops_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse a manual_crops Parquet file.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7866)
    args = parser.parse_args()
    if args.server_port <= 0:
        raise ValueError("server-port must be positive")
    dataset = ManualCropsDataset(args.data)
    app = build_manual_crops_app(dataset)
    app.launch(server_name=args.server_name, server_port=args.server_port, css=MANUAL_VIEWER_CSS)


if __name__ == "__main__":
    main()