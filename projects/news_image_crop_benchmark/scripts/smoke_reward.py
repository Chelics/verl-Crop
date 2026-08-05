#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq


def load_reward_function(path: Path):
    spec = importlib.util.spec_from_file_location("news_crop_smoke_reward", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reward file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def main() -> None:
    parser = argparse.ArgumentParser(description="Run crop reward directly on converted Parquet rows.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reward-file", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "proxy"), default="smoke")
    parser.add_argument("--clip-model-path", type=str)
    parser.add_argument("--clip-device", default="cpu")
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--action", default='<crop>{"cx":500,"cy":500,"area":400}</crop>')
    args = parser.parse_args()
    if args.count <= 0:
        raise ValueError("count must be positive")
    if args.mode == "proxy" and not args.clip_model_path:
        parser.error("--clip-model-path is required in proxy mode")

    compute_score = load_reward_function(args.reward_file.resolve())
    table = pq.read_table(args.data, columns=["data_source", "reward_model", "extra_info"]).slice(0, args.count)
    metrics: dict[str, list[float]] = defaultdict(list)
    for row in table.to_pylist():
        result = compute_score(
            data_source=row["data_source"],
            solution_str=args.action,
            ground_truth=row["reward_model"]["ground_truth"],
            extra_info=row["extra_info"],
            reward_mode=args.mode,
            clip_model_path=args.clip_model_path,
            clip_device=args.clip_device,
        )
        for key, value in result.items():
            metrics[key].append(float(value))

    report = {
        "data": str(args.data.resolve()),
        "mode": args.mode,
        "rows": table.num_rows,
        "action": args.action,
        "mean_metrics": {key: sum(values) / len(values) for key, values in sorted(metrics.items())},
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()