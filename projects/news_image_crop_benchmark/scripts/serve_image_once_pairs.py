#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from news_crop_benchmark.image_once_pair_viewer import PAIR_VIEWER_CSS, ImageOncePairDataset, build_image_once_pair_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse original/reference pairs in an image-once Parquet.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7868)
    args = parser.parse_args()
    if args.server_port <= 0:
        raise ValueError("server-port must be positive")
    dataset = ImageOncePairDataset(args.data)
    app = build_image_once_pair_app(dataset)
    app.launch(server_name=args.server_name, server_port=args.server_port, css=PAIR_VIEWER_CSS)


if __name__ == "__main__":
    main()