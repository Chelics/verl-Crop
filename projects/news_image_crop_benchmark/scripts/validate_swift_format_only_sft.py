#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


TARGET_FIELDS = ("target_ratio", "is_cropped", "is_filled", "crop_box", "fill_color", "description")


def _messages_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row["messages"], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def validate_source_identity(source_rows: list[dict[str, Any]], converted_rows: list[dict[str, Any]]) -> None:
    source_by_key = {(row["image_id"], float(row["target_ratio"])): row for row in source_rows}
    converted_by_key = {(row["image_id"], float(row["target_ratio"])): row for row in converted_rows}
    if source_by_key.keys() != converted_by_key.keys():
        raise ValueError("converted train/validation keys differ from the Swift source")
    for key, source_row in source_by_key.items():
        converted_row = converted_by_key[key]
        for column, value in source_row.items():
            if column == "images":
                image = converted_row[column]
                if not isinstance(image, list) or len(image) != 1 or not isinstance(image[0], dict):
                    raise ValueError(f"converted row {key} does not contain one verl image descriptor")
                if image[0].get("image") != value[0]:
                    raise ValueError(f"converted row {key} changed its image path")
            elif converted_row.get(column) != value:
                raise ValueError(f"converted row {key} changed column {column}")


def validate_target(row: dict[str, Any], row_index: int) -> None:
    messages = row["messages"]
    if len(messages) != 2 or [message["role"] for message in messages] != ["user", "assistant"]:
        raise ValueError(f"row {row_index} must contain one user and one assistant message")
    if messages[0]["content"].count("<image>") != 1:
        raise ValueError(f"row {row_index} must contain exactly one image placeholder")
    try:
        target = json.loads(messages[1]["content"])
    except json.JSONDecodeError as error:
        raise ValueError(f"row {row_index} assistant target is not valid JSON") from error
    if tuple(target) != TARGET_FIELDS:
        raise ValueError(f"row {row_index} target fields are not ordered as {TARGET_FIELDS}")
    if not isinstance(target["description"], str) or not target["description"].strip():
        raise ValueError(f"row {row_index} description must be non-empty")
    if float(target["target_ratio"]) != float(row["target_ratio"]):
        raise ValueError(f"row {row_index} target ratio differs from top-level metadata")
    if target["is_cropped"] != row["is_cropped"] or target["is_filled"] != row["is_filled"]:
        raise ValueError(f"row {row_index} action flags differ from top-level metadata")
    box = target["crop_box"]
    if target["is_cropped"]:
        if not (
            isinstance(box, list)
            and len(box) == 4
            and all(isinstance(value, int | float) and not isinstance(value, bool) for value in box)
            and all(math.isfinite(float(value)) for value in box)
            and 0 <= box[0] < box[2] <= 1
            and 0 <= box[1] < box[3] <= 1
        ):
            raise ValueError(f"row {row_index} has an invalid crop_box")
    elif box is not None:
        raise ValueError(f"row {row_index} must use crop_box=null when not cropped")
    color = target["fill_color"]
    if target["is_filled"]:
        if not (
            isinstance(color, list)
            and len(color) == 3
            and all(isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 255 for value in color)
        ):
            raise ValueError(f"row {row_index} has an invalid fill_color")
    elif color is not None:
        raise ValueError(f"row {row_index} must use fill_color=null when not filled")


def validate_dataset(
    *,
    source_path: Path,
    train_path: Path,
    validation_path: Path,
    model_path: Path,
    max_length: int,
    max_samples: int,
) -> dict[str, Any]:
    if max_length <= 0 or max_samples == 0 or max_samples < -1:
        raise ValueError("max_length must be positive and max_samples must be -1 or positive")
    source_rows = pq.read_table(source_path, pre_buffer=False).to_pylist()
    split_rows = {
        "train": pq.read_table(train_path, pre_buffer=False).to_pylist(),
        "validation": pq.read_table(validation_path, pre_buffer=False).to_pylist(),
    }
    converted_rows = split_rows["train"] + split_rows["validation"]
    validate_source_identity(source_rows, converted_rows)
    for index, row in enumerate(converted_rows):
        validate_target(row, index)

    unique_images = {}
    for row in converted_rows:
        descriptor = row["images"][0]
        if set(descriptor) != {"image", "min_pixels", "max_pixels"}:
            raise ValueError("image descriptor fields differ from image/min_pixels/max_pixels")
        image_path = Path(descriptor["image"])
        unique_images[str(image_path)] = image_path
    missing_images = [str(path) for path in unique_images.values() if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"SFT image paths do not exist: {missing_images[:3]}")
    for image_path in unique_images.values():
        with Image.open(image_path) as image:
            image.verify()

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
    token_lengths = []
    loss_token_lengths = []
    for split, path in (("train", train_path), ("validation", validation_path)):
        dataset = MultiTurnSFTDataset(
            parquet_files=str(path),
            tokenizer=tokenizer,
            processor=processor,
            config=config,
            max_samples=max_samples,
        )
        for index in range(len(dataset)):
            item = dataset[index]
            token_length = int(item["input_ids"].shape[0])
            loss_tokens = int(item["loss_mask"].sum().item())
            if loss_tokens <= 0 or loss_tokens >= token_length:
                raise ValueError(f"{split} row {index} has an invalid assistant loss mask")
            image_grid_thw = item.get("multi_modal_inputs", {}).get("image_grid_thw")
            if image_grid_thw is None or image_grid_thw.shape[0] != 1:
                raise ValueError(f"{split} row {index} must contain exactly one processed image")
            token_lengths.append(token_length)
            loss_token_lengths.append(loss_tokens)

    return {
        "protocol": "swift-v4-verl-format-only-sft-preflight-v1",
        "source_path": os.path.abspath(os.path.expanduser(str(source_path))),
        "train_path": os.path.abspath(os.path.expanduser(str(train_path))),
        "validation_path": os.path.abspath(os.path.expanduser(str(validation_path))),
        "model_path": os.path.abspath(os.path.expanduser(str(model_path))),
        "source_rows": len(source_rows),
        "split_rows": {split: len(rows) for split, rows in split_rows.items()},
        "unique_images": len(unique_images),
        "messages_sha256": _messages_sha256(source_rows),
        "messages_unchanged": True,
        "description_in_assistant": True,
        "max_length": max_length,
        "validated_processor_rows": len(token_lengths),
        "token_lengths": {"min": min(token_lengths), "mean": mean(token_lengths), "max": max(token_lengths)},
        "assistant_loss_tokens": {
            "min": min(loss_token_lengths),
            "mean": mean(loss_token_lengths),
            "max": max(loss_token_lengths),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate format-only Swift SFT data with verl's processor.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=-1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = validate_dataset(
        source_path=args.source.expanduser(),
        train_path=args.train.expanduser(),
        validation_path=args.validation.expanduser(),
        model_path=args.model.expanduser(),
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