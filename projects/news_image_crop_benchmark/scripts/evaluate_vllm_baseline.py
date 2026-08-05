#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pyarrow.parquet as pq
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from news_crop_benchmark.protocol import parse_crop_action


def load_reward_function(path: Path):
    spec = importlib.util.spec_from_file_location("news_crop_baseline_reward", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reward file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def select_complete_groups(data_path: Path, group_count: int, seed: int) -> list[dict]:
    rows = pq.read_table(data_path).to_pylist()
    grouped: dict[tuple[str, str], dict[float, dict]] = defaultdict(dict)
    for row in rows:
        info = row["extra_info"]
        grouped[(row["images"][0], info["title"])][float(info["target_ratio"])] = row

    complete_groups = [group for group in grouped.values() if len(group) == 4]
    if group_count > len(complete_groups):
        raise ValueError(f"requested {group_count} groups, but only {len(complete_groups)} complete groups exist")
    selected = random.Random(seed).sample(complete_groups, group_count)
    return [group[ratio] for group in selected for ratio in sorted(group)]


def build_vllm_request(processor, row: dict, image: Image.Image, image_max_pixels: int, image_min_pixels: int):
    prompt_text = row["prompt"][0]["content"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text.replace("<image>\n", "", 1)},
            ],
        }
    ]
    rendered_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {
        "prompt": rendered_prompt,
        "multi_modal_data": {"image": image},
        "mm_processor_kwargs": {
            "size": {"longest_edge": image_max_pixels, "shortest_edge": image_min_pixels}
        },
    }


def summarize(details: list[dict], center_scores: dict[str, float]) -> dict:
    ratios = sorted({detail["target_ratio"] for detail in details})

    def summarize_subset(subset: list[dict]) -> dict:
        valid = [detail for detail in subset if detail["valid"]]
        groups: dict[str, list[dict]] = defaultdict(list)
        for detail in subset:
            groups[detail["sample_id"]].append(detail)
        best_scores = [max(item["score"] for item in group) for group in groups.values()]
        center = [center_scores[sample_id] for sample_id in groups]
        return {
            "outputs": len(subset),
            "valid_rate": len(valid) / len(subset),
            "mean_score": mean(detail["score"] for detail in subset),
            "best_of_n_mean_score": mean(best_scores),
            "center_crop_mean_score": mean(center),
            "best_of_n_win_rate_vs_center": mean(
                best_score > center_score for best_score, center_score in zip(best_scores, center, strict=True)
            ),
            "mean_action_area": mean(detail["action"]["area"] for detail in valid) if valid else None,
            "mean_action_cx": mean(detail["action"]["cx"] for detail in valid) if valid else None,
            "mean_action_cy": mean(detail["action"]["cy"] for detail in valid) if valid else None,
            "near_full_image_rate": mean(detail["action"]["area"] >= 950 for detail in valid) if valid else None,
            "tiny_crop_rate": mean(detail["action"]["area"] <= 50 for detail in valid) if valid else None,
        }

    return {
        "overall": summarize_subset(details),
        "by_ratio": {
            f"{ratio:g}": summarize_subset([detail for detail in details if detail["target_ratio"] == ratio])
            for ratio in ratios
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the untrained Qwen3.5 vLLM crop baseline.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reward-file", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--clip-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=2176)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    parser.add_argument("--image-min-pixels", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    args = parser.parse_args()
    if args.groups <= 0 or args.n <= 0:
        raise ValueError("groups and n must be positive")

    selected_rows = select_complete_groups(args.data, args.groups, args.seed)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    images = []
    requests = []
    for row in selected_rows:
        with Image.open(row["images"][0]) as source:
            image = source.convert("RGB")
        images.append(image)
        requests.append(
            build_vllm_request(processor, row, image, args.image_max_pixels, args.image_min_pixels)
        )

    asset_root = Path("/mnt/blob_output/v-yukunban/news_image_crop_assets/original")
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=max(16, len(selected_rows) * args.n),
        enforce_eager=True,
        allowed_local_media_path=str(asset_root),
        limit_mm_per_prompt={"image": 1},
    )
    outputs = llm.generate(
        requests,
        sampling_params=SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            n=args.n,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
    )

    compute_score = load_reward_function(args.reward_file.resolve())
    details = []
    center_scores = {}
    for row, request_output in zip(selected_rows, outputs, strict=True):
        info = row["extra_info"]
        sample_id = info["sample_id"]
        center_result = compute_score(
            data_source=row["data_source"],
            solution_str='<crop>{"cx":500,"cy":500,"area":1000}</crop>',
            ground_truth=row["reward_model"]["ground_truth"],
            extra_info=info,
            reward_mode="proxy",
            clip_model_path=str(args.clip_model_path),
            clip_device=args.clip_device,
        )
        center_scores[sample_id] = float(center_result["score"])
        for candidate_index, output in enumerate(request_output.outputs):
            try:
                action = parse_crop_action(output.text)
                action_dict = {"cx": action.center_x, "cy": action.center_y, "area": action.area}
                valid = True
            except ValueError:
                action_dict = None
                valid = False
            result = compute_score(
                data_source=row["data_source"],
                solution_str=output.text,
                ground_truth=row["reward_model"]["ground_truth"],
                extra_info=info,
                reward_mode="proxy",
                clip_model_path=str(args.clip_model_path),
                clip_device=args.clip_device,
            )
            details.append(
                {
                    "sample_id": sample_id,
                    "title": info["title"],
                    "target_ratio": float(info["target_ratio"]),
                    "image_path": row["images"][0],
                    "candidate_index": candidate_index,
                    "response": output.text,
                    "valid": valid,
                    "action": action_dict,
                    **{key: float(value) for key, value in result.items()},
                }
            )

    summary = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "groups": args.groups,
        "prompts": len(selected_rows),
        "candidates_per_prompt": args.n,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "clip_device": args.clip_device,
        **summarize(details, center_scores),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "details.jsonl").write_text(
        "".join(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n" for detail in details)
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
