#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from news_crop_benchmark.result_viewer import VIEWER_CSS, ResultCollection, ResultDataset, build_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Browse image crop evaluation results with Gradio.")
    parser.add_argument(
        "--result-dir",
        type=Path,
        required=True,
        help="One result directory or a root containing result directories.",
    )
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument(
        "--single-result",
        action="store_true",
        help="Expose only --result-dir and do not discover sibling experiments.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.server_port <= 0:
        raise ValueError("server-port must be positive")
    source = ResultDataset(args.result_dir) if args.single_result else ResultCollection(args.result_dir)
    app = build_app(source)

    username = os.getenv("GRADIO_AUTH_USERNAME")
    password = os.getenv("GRADIO_AUTH_PASSWORD")
    if bool(username) != bool(password):
        raise ValueError("GRADIO_AUTH_USERNAME and GRADIO_AUTH_PASSWORD must be set together")
    auth = (username, password) if username and password else None
    app.launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share,
        auth=auth,
        css=VIEWER_CSS,
    )


if __name__ == "__main__":
    main()