#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

from news_crop_benchmark.protocol import parse_crop_action


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one Qwen3.5 multimodal crop-action rollout with vLLM.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    parser.add_argument("--image-min-pixels", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--n", type=int, default=1)
    args = parser.parse_args()

    row = pq.read_table(args.data).slice(0, 1).to_pylist()[0]
    image_path = Path(row["images"][0])
    with Image.open(image_path) as source:
        image = source.convert("RGB")

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": row["prompt"][0]["content"].replace("<image>\n", "", 1)},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        allowed_local_media_path=str(image_path.parent),
        limit_mm_per_prompt={"image": 1},
    )
    sampling = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        max_tokens=args.max_tokens,
    )
    outputs = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"image": image},
            "mm_processor_kwargs": {
                "size": {
                    "longest_edge": args.image_max_pixels,
                    "shortest_edge": args.image_min_pixels,
                }
            },
        },
        sampling_params=sampling,
    )
    responses = [output.text for output in outputs[0].outputs]
    valid_actions = []
    for response in responses:
        try:
            valid_actions.append(parse_crop_action(response).__dict__)
        except ValueError:
            valid_actions.append(None)
    print(
        json.dumps(
            {
                "data": str(args.data.resolve()),
                "image": str(image_path),
                "prompt": row["prompt"][0]["content"],
                "responses": responses,
                "valid_actions": valid_actions,
                "valid_rate": sum(action is not None for action in valid_actions) / len(valid_actions),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()