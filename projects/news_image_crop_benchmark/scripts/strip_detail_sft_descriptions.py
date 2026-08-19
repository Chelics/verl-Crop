#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


DETAIL_INSTRUCTION = (
    "Return only one JSON object with these fields: target_ratio, is_cropped, is_filled, crop_box, "
    "fill_color, and description. Use normalized [left, top, right, bottom] coordinates for crop_box. "
    "crop_box is applied directly to the original image. When is_filled is false, crop_box itself must "
    "have the target aspect ratio. The description must explain in detail which visual elements are "
    "preserved or removed, why the crop is safe, and why color fill is needed when applicable."
)
ACTION_INSTRUCTION = (
    "Inspect only visible image content. The headline and caption may help identify the primary subject, but "
    "never invent content from them. Preserve complete faces, heads, chins, identity-defining features, and "
    "meaningful actions or interactions. Avoid crop boundaries through faces, necks, major joints, important "
    "hands, or held objects that matter to the story. Preserve title-relevant logos, wordmarks, names, text "
    "blocks, lower-thirds, and product marks completely; never cut through a letter or essential logo element. "
    "For charts, maps, diagrams, screenshots, posters, and other informational graphics, preserve the title, "
    "labels, legends, axes, data, and structural sections needed for understanding. Low-value empty margins, "
    "decorative borders, repetitive background, unrelated edge clutter, and clearly secondary peripheral "
    "content may be removed. Keep a natural safety margin around protected content. If a direct target-ratio "
    "crop would damage protected content, use crop plus fill or retain the full image and fill instead. "
    "fill_color must continue the visible outer background adjacent to the padding edge, not a foreground "
    "subject, text, logo, border artifact, shadow, or black bar. "
    "Choose exactly one action: crop uses is_cropped=true and is_filled=false, with a non-null crop_box that "
    "itself has the target aspect ratio and fill_color=null; crop plus fill uses is_cropped=true and "
    "is_filled=true, retaining crop_box exactly before padding with fill_color; fill uses is_cropped=false "
    "and is_filled=true, retaining the complete original image with crop_box=null before padding with "
    "fill_color; keep uses is_cropped=false and is_filled=false only when the complete original image already "
    "matches the requested ratio, with crop_box=null and fill_color=null. "
    "Return only one JSON object with exactly these fields in this order: target_ratio, is_cropped, "
    "is_filled, crop_box, and fill_color. Use normalized [left, top, right, bottom] coordinates for "
    "crop_box. crop_box is applied directly to the original image. When is_cropped is true and "
    "is_filled is false, crop_box itself must have the target aspect ratio. Use crop_box=null when is_cropped is false and "
    "fill_color=null when is_filled is false. Do not output a description or any additional fields."
)
DETAIL_FIELDS = ("target_ratio", "is_cropped", "is_filled", "crop_box", "fill_color", "description")
ACTION_FIELDS = DETAIL_FIELDS[:-1]


def strip_descriptions(
    *,
    input_path: Path,
    output_path: Path,
    image_min_pixels: int = 65536,
    image_max_pixels: int = 1048576,
) -> dict:
    if image_min_pixels <= 0 or image_max_pixels < image_min_pixels:
        raise ValueError("image pixel limits must be positive and ordered")
    table = pq.read_table(input_path, pre_buffer=False)
    required = {"messages", "images", "image_id", "source_index", "target_ratio", "is_cropped", "is_filled"}
    missing = sorted(required - set(table.column_names))
    if missing:
        raise ValueError(f"detail SFT data is missing required columns: {missing}")

    output_rows = []
    description_lengths = []
    for row_index, row in enumerate(table.to_pylist()):
        messages = row["messages"]
        if len(messages) != 2 or [message["role"] for message in messages] != ["user", "assistant"]:
            raise ValueError(f"row {row_index} must contain one user and one assistant message")
        prompt = messages[0]["content"]
        if prompt.count(DETAIL_INSTRUCTION) != 1:
            raise ValueError(f"row {row_index} does not contain the expected detail instruction exactly once")
        try:
            target = json.loads(messages[1]["content"])
        except json.JSONDecodeError as error:
            raise ValueError(f"row {row_index} assistant target is not valid JSON") from error
        if tuple(target) != DETAIL_FIELDS:
            raise ValueError(f"row {row_index} target fields are not ordered as {DETAIL_FIELDS}")
        description = target.pop("description")
        if not isinstance(description, str) or not description.strip():
            raise ValueError(f"row {row_index} description must be non-empty text")
        if tuple(target) != ACTION_FIELDS:
            raise AssertionError("action target field order changed")
        if target["target_ratio"] != row["target_ratio"]:
            raise ValueError(f"row {row_index} target_ratio differs from top-level metadata")
        if target["is_cropped"] != row["is_cropped"] or target["is_filled"] != row["is_filled"]:
            raise ValueError(f"row {row_index} action flags differ from top-level metadata")
        images = row["images"]
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise ValueError(f"row {row_index} detail source must contain exactly one image path")

        row["messages"] = [
            {"role": "user", "content": prompt.replace(DETAIL_INSTRUCTION, ACTION_INSTRUCTION)},
            {
                "role": "assistant",
                "content": json.dumps(target, ensure_ascii=False, separators=(",", ":")),
            },
        ]
        row["images"] = [
            {
                "image": images[0],
                "min_pixels": image_min_pixels,
                "max_pixels": image_max_pixels,
            }
        ]
        row["reference_description"] = description
        output_rows.append(row)
        description_lengths.append(len(description))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    pq.write_table(pa.Table.from_pylist(output_rows), temporary_path, compression="zstd", row_group_size=256)
    temporary_path.replace(output_path)
    return {
        "input_path": os.path.abspath(os.path.expanduser(str(input_path))),
        "output_path": os.path.abspath(os.path.expanduser(str(output_path))),
        "rows": len(output_rows),
        "assistant_target_fields": list(ACTION_FIELDS),
        "description_in_assistant": False,
        "description_metadata_field": "reference_description",
        "image_schema": "verl_descriptor",
        "image_min_pixels": image_min_pixels,
        "image_max_pixels": image_max_pixels,
        "description_length_chars": {
            "min": min(description_lengths),
            "max": max(description_lengths),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Create action-only SFT Parquets from detailed crop/fill data.")
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-min-pixels", type=int, default=65536)
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    args = parser.parse_args()
    reports = [
        strip_descriptions(
            input_path=input_path.expanduser(),
            output_path=args.output_dir.expanduser() / input_path.name,
            image_min_pixels=args.image_min_pixels,
            image_max_pixels=args.image_max_pixels,
        )
        for input_path in args.input
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "conversion_report.json"
    report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(reports, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()