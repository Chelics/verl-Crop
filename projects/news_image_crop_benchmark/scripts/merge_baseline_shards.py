#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
from pathlib import Path


def load_baseline_module():
    script_path = Path(__file__).with_name("evaluate_vllm_baseline.py")
    spec = importlib.util.spec_from_file_location("news_crop_baseline_merge", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load baseline module: {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge distributed Qwen crop baseline shards.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--shard-count", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"merge output directory is not empty: {args.output_dir}")

    baseline = load_baseline_module()
    selected_rows = baseline.select_complete_groups(args.data, None, args.seed)
    expected_sample_ids = {str(row["extra_info"]["sample_id"]) for row in selected_rows}

    details = []
    center_scores = {}
    center_baselines = {}
    compatible_config_keys = {
        "model",
        "model_fingerprint",
        "data",
        "data_fingerprint",
        "groups",
        "n",
        "seed",
        "temperature",
        "top_p",
        "max_model_len",
        "max_tokens",
        "image_max_pixels",
        "image_min_pixels",
        "reward_file",
        "reward_file_fingerprint",
        "clip_model_path",
        "clip_model_fingerprint",
        "clip_device",
        "gpu_memory_utilization",
        "tensor_parallel_size",
        "prompt_batch_size",
        "max_num_seqs",
        "image_path_maps",
        "shard_count",
        "run_id",
    }
    reference_config = None
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = args.output_dir / "renders"
    render_dir.mkdir(parents=True, exist_ok=True)

    for shard_index in range(args.shard_count):
        shard_dir = args.shard_root / f"shard-{shard_index}"
        marker_path = shard_dir / "_SHARD_COMPLETE.json"
        if not marker_path.is_file():
            raise FileNotFoundError(f"shard is incomplete: {marker_path}")
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
        if marker["run_id"] != args.run_id:
            raise ValueError(f"shard {shard_index} run ID does not match: {marker['run_id']}")
        if marker["shard_count"] != args.shard_count or marker["shard_index"] != shard_index:
            raise ValueError(f"shard marker metadata does not match shard {shard_index}")
        shard_config = baseline.yaml.safe_load((shard_dir / "baseline_config.yaml").read_text(encoding="utf-8"))
        compatible_config = {key: shard_config[key] for key in compatible_config_keys}
        if reference_config is None:
            reference_config = compatible_config
        elif compatible_config != reference_config:
            raise ValueError(f"shard {shard_index} configuration does not match the other shards")
        for line in (shard_dir / "details.jsonl").read_text(encoding="utf-8").splitlines():
            if line:
                details.append(json.loads(line))
        for progress_path in (shard_dir / ".score_progress").glob("*.json"):
            progress = json.loads(progress_path.read_text(encoding="utf-8"))
            sample_id = str(progress["sample_id"])
            center_baselines[sample_id] = progress["center_baseline"]
            center_scores[sample_id] = float(progress["center_baseline"]["score"])
        for render_path in (shard_dir / "renders").glob("*.jpg"):
            shutil.copy2(render_path, render_dir / render_path.name)

    actual_sample_ids = {str(detail["sample_id"]) for detail in details}
    if actual_sample_ids != expected_sample_ids:
        missing = sorted(expected_sample_ids - actual_sample_ids)
        unexpected = sorted(actual_sample_ids - expected_sample_ids)
        raise ValueError(f"shard coverage mismatch: missing={missing[:10]}, unexpected={unexpected[:10]}")
    expected_candidates = len(selected_rows) * 8
    if len(details) != expected_candidates:
        raise ValueError(f"expected {expected_candidates} candidates, found {len(details)}")
    candidate_keys = [(str(detail["sample_id"]), int(detail["candidate_index"])) for detail in details]
    expected_candidate_keys = {
        (sample_id, candidate_index)
        for sample_id in expected_sample_ids
        for candidate_index in range(8)
    }
    if len(candidate_keys) != len(set(candidate_keys)) or set(candidate_keys) != expected_candidate_keys:
        raise ValueError("candidate coverage must contain indices 0..7 exactly once for every sample")
    if set(center_scores) != expected_sample_ids:
        raise ValueError("center baseline coverage does not match the full test manifest")

    details.sort(key=lambda item: (item["sample_id"], item["candidate_index"]))
    summary = {
        "data": str(args.data.resolve()),
        "groups": len(selected_rows) // 4,
        "prompts": len(selected_rows),
        "candidates_per_prompt": 8,
        "seed": args.seed,
        "distributed_shards": args.shard_count,
        "run_id": args.run_id,
        **baseline.summarize(details, center_scores),
    }
    (args.output_dir / "details.jsonl").write_text(
        "".join(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n" for detail in details),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "sample_manifest.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "sample_id": row["extra_info"]["sample_id"],
                    "title": row["extra_info"]["title"],
                    "target_ratio": row["extra_info"]["target_ratio"],
                    "image_path": row["images"][0],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for row in selected_rows
        ),
        encoding="utf-8",
    )
    with (args.output_dir / "human_review.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["sample_id", "title", "target_ratio", "judgement", "comment"],
        )
        writer.writeheader()
        for row in selected_rows:
            info = row["extra_info"]
            writer.writerow(
                {
                    "sample_id": info["sample_id"],
                    "title": info["title"],
                    "target_ratio": info["target_ratio"],
                    "judgement": "",
                    "comment": "",
                }
            )
    baseline.render_html_report(args.output_dir, summary, selected_rows, details, center_baselines)
    baseline.write_json_atomic(
        args.output_dir / "_MERGE_COMPLETE.json",
        {
            "run_id": args.run_id,
            "prompts": len(selected_rows),
            "candidates": len(details),
            "shards": args.shard_count,
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
