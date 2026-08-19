#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


EXPECTED_TARGET_FIELDS = ("target_ratio", "is_cropped", "is_filled", "crop_box", "fill_color")
TITLE_PATTERN = re.compile(r"^News headline:\s*(.*)$", re.MULTILINE)
CAPTION_PATTERN = re.compile(r"^Image caption:\s*(.*)$", re.MULTILINE)


def validate_action_row(row: dict[str, Any], row_index: int, prompt_template: str | None = None) -> str:
    messages = row["messages"]
    if len(messages) != 2 or [message["role"] for message in messages] != ["user", "assistant"]:
        raise ValueError(f"row {row_index} must contain one user and one assistant message")
    prompt = messages[0]["content"]
    if not isinstance(prompt, str) or prompt.count("<image>") != 1:
        raise ValueError(f"row {row_index} must contain exactly one image placeholder")
    if "Do not output a description or any additional fields." not in prompt:
        raise ValueError(f"row {row_index} prompt does not forbid description output")
    if prompt_template is not None:
        title_match = TITLE_PATTERN.search(prompt)
        caption_match = CAPTION_PATTERN.search(prompt)
        if title_match is None or caption_match is None:
            raise ValueError(f"row {row_index} prompt is missing headline or caption")
        expected_prompt = prompt_template.format(
            title=title_match.group(1).strip(),
            caption=caption_match.group(1).strip(),
            target_ratio=f"{float(row['target_ratio']):g}",
        )
        if prompt != expected_prompt:
            raise ValueError(f"row {row_index} prompt does not match the configured template")
    try:
        target = json.loads(messages[1]["content"])
    except json.JSONDecodeError as error:
        raise ValueError(f"row {row_index} assistant target is not valid JSON") from error
    if tuple(target) != EXPECTED_TARGET_FIELDS:
        raise ValueError(f"row {row_index} target fields are not ordered as {EXPECTED_TARGET_FIELDS}")

    ratio = target["target_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, int | float) or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError(f"row {row_index} target_ratio must be a positive finite number")
    if float(ratio) != float(row["target_ratio"]):
        raise ValueError(f"row {row_index} target_ratio differs from top-level metadata")
    is_cropped = target["is_cropped"]
    is_filled = target["is_filled"]
    if not isinstance(is_cropped, bool) or not isinstance(is_filled, bool):
        raise ValueError(f"row {row_index} action flags must be booleans")
    if is_cropped != row["is_cropped"] or is_filled != row["is_filled"]:
        raise ValueError(f"row {row_index} action flags differ from top-level metadata")

    box = target["crop_box"]
    if is_cropped:
        if (
            not isinstance(box, list)
            or len(box) != 4
            or any(isinstance(value, bool) or not isinstance(value, int | float) for value in box)
            or not all(math.isfinite(float(value)) for value in box)
            or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)
        ):
            raise ValueError(f"row {row_index} cropped action must contain a valid normalized crop_box")
    elif box is not None:
        raise ValueError(f"row {row_index} non-cropped action must use crop_box=null")

    color = target["fill_color"]
    if is_filled:
        if (
            not isinstance(color, list)
            or len(color) != 3
            or any(isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255 for channel in color)
        ):
            raise ValueError(f"row {row_index} filled action must contain integer RGB fill_color")
    elif color is not None:
        raise ValueError(f"row {row_index} non-filled action must use fill_color=null")

    images = row["images"]
    if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], dict):
        raise ValueError(f"row {row_index} must contain exactly one verl image descriptor")
    image = images[0]
    if set(image) != {"image", "min_pixels", "max_pixels"}:
        raise ValueError(f"row {row_index} image descriptor has unexpected fields")
    if not isinstance(image["image"], str) or not image["image"]:
        raise ValueError(f"row {row_index} image descriptor must contain a path")
    if not (
        isinstance(image["min_pixels"], int)
        and isinstance(image["max_pixels"], int)
        and 0 < image["min_pixels"] <= image["max_pixels"]
    ):
        raise ValueError(f"row {row_index} image descriptor has invalid pixel limits")
    description = row.get("reference_description")
    if not isinstance(description, str) or not description.strip():
        raise ValueError(f"row {row_index} must retain a non-empty reference_description")
    return {
        (True, False): "crop",
        (True, True): "crop_fill",
        (False, True): "fill",
        (False, False): "keep",
    }[(is_cropped, is_filled)]


def validate_dataset(
    *,
    data_path: Path,
    model_path: Path,
    prompt_template_path: Path,
    max_length: int,
    max_samples: int,
) -> dict:
    if max_length <= 0 or max_samples == 0 or max_samples < -1:
        raise ValueError("max_length must be positive and max_samples must be -1 or positive")
    required_columns = {
        "messages",
        "images",
        "image_id",
        "source_index",
        "target_ratio",
        "is_cropped",
        "is_filled",
        "reference_description",
    }
    parquet = pq.ParquetFile(data_path)
    missing = sorted(required_columns - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"action SFT data is missing required columns: {missing}")
    source_rows = pq.read_table(data_path, columns=sorted(required_columns), pre_buffer=False).to_pylist()
    if not source_rows:
        raise ValueError("SFT dataset is empty")
    prompt_template = prompt_template_path.read_text(encoding="utf-8").strip()
    operations = Counter(
        validate_action_row(row, index, prompt_template=prompt_template) for index, row in enumerate(source_rows)
    )
    missing_images = []
    unique_image_paths = sorted({row["images"][0]["image"] for row in source_rows})
    for image_value in unique_image_paths:
        image_path = Path(image_value)
        if not image_path.is_file():
            missing_images.append(str(image_path))
            continue
        with Image.open(image_path) as image:
            image.verify()
    if missing_images:
        raise FileNotFoundError(f"SFT image paths do not exist: {missing_images[:3]}")

    from verl.utils import hf_processor, hf_tokenizer
    from verl.utils.dataset.multiturn_sft_dataset import MultiTurnSFTDataset

    tokenizer = hf_tokenizer(str(model_path))
    processor = hf_processor(str(model_path))
    if processor is None:
        raise RuntimeError("model did not produce a supported multimodal processor")
    config = {
        "max_length": max_length,
        "pad_mode": "no_padding",
        "truncation": "error",
        "messages_key": "messages",
        "image_key": "images",
        "image_patch_size": 16,
        "enable_thinking_key": "enable_thinking",
        "enable_thinking_default": False,
        "apply_chat_template_kwargs": {"enable_thinking": False},
        "ignore_input_ids_mismatch": False,
    }
    dataset = MultiTurnSFTDataset(
        parquet_files=str(data_path),
        tokenizer=tokenizer,
        processor=processor,
        config=config,
        max_samples=max_samples,
    )
    token_lengths = []
    loss_token_lengths = []
    image_token_lengths = []
    position_id_shapes = Counter()
    for index in range(len(dataset)):
        item = dataset[index]
        token_length = int(item["input_ids"].shape[0])
        loss_tokens = int(item["loss_mask"].sum().item())
        if loss_tokens <= 0 or loss_tokens >= token_length:
            raise ValueError(f"row {index} has an invalid assistant loss mask")
        image_grid_thw = item.get("multi_modal_inputs", {}).get("image_grid_thw")
        if image_grid_thw is None or image_grid_thw.shape[0] != 1:
            raise ValueError(f"row {index} must contain exactly one processed image")
        image_tokens = int((item["input_ids"] == processor.image_token_id).sum().item())
        if image_tokens <= 0:
            raise ValueError(f"row {index} contains no image tokens")
        token_lengths.append(token_length)
        loss_token_lengths.append(loss_tokens)
        image_token_lengths.append(image_tokens)
        position_id_shapes[str(tuple(item["position_ids"].shape))] += 1

    return {
        "protocol": "cropped-v3-action-sft-v1",
        "data_path": os.path.abspath(os.path.expanduser(str(data_path))),
        "model_path": os.path.abspath(os.path.expanduser(str(model_path))),
        "prompt_template_path": os.path.abspath(os.path.expanduser(str(prompt_template_path))),
        "source_rows": len(source_rows),
        "unique_images": len(unique_image_paths),
        "validated_rows": len(dataset),
        "processor_class": processor.__class__.__name__,
        "image_processor_class": processor.image_processor.__class__.__name__,
        "position_id_shapes": dict(sorted(position_id_shapes.items())),
        "operation_counts": dict(sorted(operations.items())),
        "description_in_assistant": False,
        "description_metadata_field": "reference_description",
        "max_length": max_length,
        "token_lengths": {"min": min(token_lengths), "mean": mean(token_lengths), "max": max(token_lengths)},
        "assistant_loss_tokens": {
            "min": min(loss_token_lengths),
            "mean": mean(loss_token_lengths),
            "max": max(loss_token_lengths),
        },
        "image_tokens": {
            "min": min(image_token_lengths),
            "mean": mean(image_token_lengths),
            "max": max(image_token_lengths),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate five-field crop/fill action SFT data.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_dataset(
        data_path=args.data.expanduser(),
        model_path=args.model.expanduser(),
        prompt_template_path=args.prompt_template.expanduser(),
        max_length=args.max_length,
        max_samples=args.max_samples,
    )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()