#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _messages_sha256(rows: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(json.dumps(row["messages"], ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _write_rows(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(rows), temporary_path, compression="zstd", row_group_size=256)
    temporary_path.replace(output_path)


def _convert_rows(
    source_rows: list[dict[str, Any]], image_min_pixels: int, image_max_pixels: int
) -> list[dict[str, Any]]:
    output_rows = []
    for row_index, source_row in enumerate(source_rows):
        images = source_row["images"]
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str) or not images[0]:
            raise ValueError(f"row {row_index} must contain exactly one Swift image path")
        output_row = dict(source_row)
        output_row["images"] = [
            {
                "image": images[0],
                "min_pixels": image_min_pixels,
                "max_pixels": image_max_pixels,
            }
        ]
        output_rows.append(output_row)
    return output_rows


def convert_swift_sft_to_verl(
    *,
    input_path: Path,
    output_path: Path,
    image_min_pixels: int = 65536,
    image_max_pixels: int = 1048576,
) -> dict[str, Any]:
    if image_min_pixels <= 0 or image_max_pixels < image_min_pixels:
        raise ValueError("image pixel limits must be positive and ordered")

    table = pq.read_table(input_path, pre_buffer=False)
    required = {"messages", "images"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"Swift SFT data is missing required columns: {missing}")

    source_rows = table.to_pylist()
    if not source_rows:
        raise ValueError("Swift SFT dataset is empty")
    output_rows = _convert_rows(source_rows, image_min_pixels, image_max_pixels)

    source_message_hash = _messages_sha256(source_rows)
    output_message_hash = _messages_sha256(output_rows)
    if source_message_hash != output_message_hash:
        raise AssertionError("messages changed during format-only conversion")

    _write_rows(output_rows, output_path)
    return {
        "protocol": "swift-sft-to-verl-format-only-v1",
        "input_path": os.path.abspath(os.path.expanduser(str(input_path))),
        "output_path": os.path.abspath(os.path.expanduser(str(output_path))),
        "rows": len(output_rows),
        "columns": table.column_names,
        "messages_sha256": source_message_hash,
        "messages_unchanged": True,
        "only_transformed_column": "images",
        "source_image_schema": "list<string>",
        "output_image_schema": "list<struct<image:string,min_pixels:int64,max_pixels:int64>>",
        "image_min_pixels": image_min_pixels,
        "image_max_pixels": image_max_pixels,
    }


def convert_and_split_swift_sft_to_verl(
    *,
    input_path: Path,
    output_dir: Path,
    validation_ratio: float = 0.01,
    seed: int = 42,
    image_min_pixels: int = 65536,
    image_max_pixels: int = 1048576,
) -> dict[str, Any]:
    if not 0 < validation_ratio < 1:
        raise ValueError("validation_ratio must be between zero and one")
    if image_min_pixels <= 0 or image_max_pixels < image_min_pixels:
        raise ValueError("image pixel limits must be positive and ordered")

    table = pq.read_table(input_path, pre_buffer=False)
    required = {"messages", "images"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"Swift SFT data is missing required columns: {missing}")
    source_rows = table.to_pylist()
    if len(source_rows) < 2:
        raise ValueError("Swift SFT dataset must contain at least two rows for splitting")
    output_rows = _convert_rows(source_rows, image_min_pixels, image_max_pixels)

    validation_rows = max(int(len(output_rows) * validation_ratio), 1)
    swift_random_state = np.random.RandomState(seed)
    swift_split_seed = int(swift_random_state.randint(0, np.iinfo(np.int32).max))
    permutation = np.random.default_rng(swift_split_seed).permutation(len(output_rows))
    validation_indices = permutation[:validation_rows].tolist()
    train_indices = permutation[validation_rows:].tolist()
    split_rows = {
        "train": [output_rows[index] for index in train_indices],
        "validation": [output_rows[index] for index in validation_indices],
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    for split, rows in split_rows.items():
        _write_rows(rows, output_dir / f"{split}.parquet")
    manifest_path = output_dir / "split_manifest.jsonl"
    manifest_path.write_text(
        "".join(
            json.dumps({"source_row": index, "split": split}, sort_keys=True) + "\n"
            for split, indices in (("train", train_indices), ("validation", validation_indices))
            for index in indices
        ),
        encoding="utf-8",
    )

    source_message_hash = _messages_sha256(source_rows)
    restored_rows = [None] * len(output_rows)
    for split, indices in (("train", train_indices), ("validation", validation_indices)):
        for source_index, row in zip(indices, split_rows[split], strict=True):
            restored_rows[source_index] = row
    output_message_hash = _messages_sha256(restored_rows)
    if source_message_hash != output_message_hash:
        raise AssertionError("messages changed during format-only conversion and split")

    return {
        "protocol": "swift-sft-to-verl-format-only-v1",
        "input_path": os.path.abspath(os.path.expanduser(str(input_path))),
        "output_dir": os.path.abspath(os.path.expanduser(str(output_dir))),
        "source_rows": len(source_rows),
        "split_rows": {split: len(rows) for split, rows in split_rows.items()},
        "validation_ratio": validation_ratio,
        "seed": seed,
        "swift_split_seed": swift_split_seed,
        "split_algorithm": "swift-4.4.2-hf-train-test-split",
        "messages_sha256": source_message_hash,
        "messages_unchanged": True,
        "only_transformed_column": "images",
        "source_image_schema": "list<string>",
        "output_image_schema": "list<struct<image:string,min_pixels:int64,max_pixels:int64>>",
        "image_min_pixels": image_min_pixels,
        "image_max_pixels": image_max_pixels,
        "split_manifest": str(manifest_path.resolve()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Swift multimodal SFT data to verl format only.")
    parser.add_argument("--input", type=Path, required=True)
    output_group = parser.add_mutually_exclusive_group(required=True)
    output_group.add_argument("--output", type=Path)
    output_group.add_argument("--output-dir", type=Path)
    parser.add_argument("--validation-ratio", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-min-pixels", type=int, default=65536)
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if args.output is not None:
        report = convert_swift_sft_to_verl(
            input_path=args.input.expanduser(),
            output_path=args.output.expanduser(),
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
        )
    else:
        report = convert_and_split_swift_sft_to_verl(
            input_path=args.input.expanduser(),
            output_dir=args.output_dir.expanduser(),
            validation_ratio=args.validation_ratio,
            seed=args.seed,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
        )
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(payload, encoding="utf-8")
    print(payload, end="")


if __name__ == "__main__":
    main()