#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from news_crop_benchmark.swift_sft_viewer import SWIFT_SFT_VIEWER_CSS, SwiftSFTDataset, build_swift_sft_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Browse Swift VLM-style crop SFT Parquet data.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7867)
    args = parser.parse_args()
    if args.server_port <= 0:
        raise ValueError("server-port must be positive")
    dataset = SwiftSFTDataset(args.data, cache_dir=args.cache_dir)
    print(f"Preparing {len(dataset.trace_ids)} source previews in {dataset.cache_dir}", flush=True)
    dataset.prepare_previews(
        progress=lambda completed, total: print(f"Prepared {completed}/{total}", flush=True)
    )
    app = build_swift_sft_app(dataset)
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        css=SWIFT_SFT_VIEWER_CSS,
        allowed_paths=[str(dataset.cache_dir.resolve())],
    )


if __name__ == "__main__":
    main()