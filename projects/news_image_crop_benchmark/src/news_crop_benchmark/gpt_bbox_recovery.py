from __future__ import annotations

import io
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps

from news_crop_benchmark.cropped_overrides import load_override_specs
from news_crop_benchmark.layout import edge_median_color, pad_image_to_ratio, render_layout_action
from news_crop_benchmark.manual_crop_merge import _select_manual_rows
from news_crop_benchmark.protocol import LayoutAction, parse_layout_action
from news_crop_benchmark.vlm_scorer import (
    DEFAULT_AZURE_API_VERSION,
    DEFAULT_AZURE_DEPLOYMENT,
    DEFAULT_AZURE_ENDPOINT,
    DEFAULT_MANAGED_IDENTITY_CLIENT_ID,
    _get_bearer_token_provider,
    _pil_image_to_data_url,
    extract_response_text,
    load_env_files,
)


RECOVERY_PROMPT = """You are recovering an executable editorial layout action from two images.

Image A is the full original news image. Image B is a human-approved final layout for the requested aspect ratio. Reproduce the approved composition from Image A as closely as possible while preserving complete faces, logos, text, diagrams, and other title-relevant content.

Choose exactly one operation:
- crop: retain a source rectangle and fit the largest target-ratio crop inside it.
- crop_pad: retain the selected source rectangle exactly, then add background outside it.
- pad: retain the complete original and add background outside it.

Return exactly one JSON object with exactly five fields:
{"operation":"crop_pad","x1_pct":X1,"y1_pct":Y1,"x2_pct":X2,"y2_pct":Y2}

Coordinates are integer percentages of Image A. For pad, return exactly 0,0,100,100. Do not return explanations, Markdown, confidence, pixel coordinates, or additional fields."""


def recover_gpt_layouts(
    train_path: Path,
    manual_path: Path,
    audited_manifest_path: Path,
    output_dir: Path,
    *,
    max_gpt: int | None = None,
    max_attempts: int = 3,
    client_factory: Callable[[], tuple[Any, str]] | None = None,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if max_gpt is not None and max_gpt <= 0:
        raise ValueError("max_gpt must be positive")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")
    manual_rows = pq.read_table(manual_path).to_pylist()
    selected, rejected, superseded = _select_manual_rows(manual_rows)
    if rejected:
        ratio_rejections = [row for row in rejected if row["reason"] == "ratio_mismatch"]
        if len(ratio_rejections) != len(rejected):
            raise ValueError(f"manual file has unexpected rejected rows: {rejected}")

    source_table = pq.read_table(
        train_path,
        columns=["image_id", "original_image", "title", "ImageCaption", "source_original_url"],
        pre_buffer=False,
    )
    source_rows = {str(row["image_id"]): row for row in source_table.to_pylist() if str(row["image_id"]) in {key[0] for key in selected}}
    if set(source_rows) != {key[0] for key in selected}:
        raise KeyError("not every selected manual image exists in the train source")

    audited = {
        (str(spec["trace_id"]), str(spec["aspect_ratio"])): spec
        for spec in load_override_specs(audited_manifest_path)
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_dir = output_dir / "progress"
    images_dir = output_dir / "images"
    progress_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    pending_gpt = [
        key
        for key in sorted(selected)
        if (key[0], _aspect_ratio(key[1])) not in audited
    ]
    if max_gpt is not None:
        pending_gpt = pending_gpt[:max_gpt]
    included_keys = sorted(set(audited) & {(key[0], _aspect_ratio(key[1])) for key in selected}) + pending_gpt
    included_normalized = {(key[0], _ratio_value(key[1])) for key in included_keys}

    client: Any | None = None
    model: str | None = None
    records = []
    for selected_key, manual in sorted(selected.items()):
        normalized_key = (selected_key[0], _ratio_value(selected_key[1]))
        aspect_ratio = _aspect_ratio(selected_key[1])
        if normalized_key not in included_normalized and (selected_key[0], aspect_ratio) not in audited:
            continue
        task_id = f"{selected_key[0]}__ratio_{selected_key[1]:g}"
        progress_path = progress_dir / f"{task_id}.json"
        source_row = source_rows[selected_key[0]]
        if progress_path.is_file():
            record = json.loads(progress_path.read_text(encoding="utf-8"))
            if record.get("status") == "completed":
                records.append(record)
                continue

        audited_spec = audited.get((selected_key[0], aspect_ratio))
        with Image.open(io.BytesIO(source_row["original_image"])) as source_file:
            original = ImageOps.exif_transpose(source_file).convert("RGB")
        with Image.open(io.BytesIO(manual["manual_crop"])) as manual_file:
            approved = ImageOps.exif_transpose(manual_file).convert("RGB")
        try:
            if audited_spec is not None:
                action = _action_from_audited(audited_spec)
                provenance = "audited_override_v1"
                response_text = None
                response_id = None
                attempts = 0
            else:
                if client is None:
                    client, model = (client_factory or create_azure_openai_client)()
                action, response_text, response_id, attempts = _request_action(
                    client,
                    str(model),
                    original,
                    approved,
                    str(source_row["title"]),
                    str(source_row["ImageCaption"]),
                    aspect_ratio,
                    max_attempts,
                )
                provenance = "gpt_pair_recovery"
            rendered, metadata = _render_action(original, action, _ratio_value(selected_key[1]), manual)
            try:
                image_path = images_dir / f"{task_id}.jpg"
                rendered.save(image_path, format="JPEG", quality=95, optimize=True)
                record = {
                    "status": "completed",
                    "task_id": task_id,
                    "trace_id": selected_key[0],
                    "aspect_ratio": aspect_ratio,
                    "target_ratio": _ratio_value(selected_key[1]),
                    "headline": source_row["title"],
                    "caption": source_row["ImageCaption"],
                    "original_image_url": source_row["source_original_url"],
                    "provenance": provenance,
                    "action": {
                        "operation": action.operation,
                        "x1_pct": action.x1_pct,
                        "y1_pct": action.y1_pct,
                        "x2_pct": action.x2_pct,
                        "y2_pct": action.y2_pct,
                    },
                    "response_text": response_text,
                    "response_id": response_id,
                    "attempt_count": attempts,
                    "image_path": image_path.relative_to(output_dir).as_posix(),
                    "manual_save_id": manual["save_id"],
                    "manual_saved_at": str(manual["saved_at"]),
                    **metadata,
                }
            finally:
                rendered.close()
        finally:
            original.close()
            approved.close()
        _write_json_atomic(progress_path, record)
        records.append(record)

    records.sort(key=lambda record: (record["trace_id"], float(record["target_ratio"])))
    (output_dir / "recoveries.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    (output_dir / "selection_report.json").write_text(
        json.dumps(
            {
                "manual_input_rows": len(manual_rows),
                "selected_rows": len(selected),
                "rejected_rows": rejected,
                "superseded_rows": superseded,
                "audited_available": len(set(audited) & {(key[0], _aspect_ratio(key[1])) for key in selected}),
                "gpt_requested": len(pending_gpt),
                "recoveries": len(records),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return records


def create_azure_openai_client() -> tuple[Any, str]:
    load_env_files()
    from azure.identity import DefaultAzureCredential
    from openai import AzureOpenAI

    endpoint = os.getenv("GPT5_AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT).strip()
    model = os.getenv("GPT5_AZURE_OPENAI_DEPLOYMENT", DEFAULT_AZURE_DEPLOYMENT).strip()
    api_version = os.getenv("GPT5_AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION).strip()
    api_key = os.getenv("GPT5_AZURE_OPENAI_API_KEY", "").strip()
    kwargs: dict[str, Any] = {"api_version": api_version, "azure_endpoint": endpoint, "max_retries": 0}
    if api_key:
        kwargs["api_key"] = api_key
    else:
        client_id = os.getenv("GPT5_AZURE_MANAGED_IDENTITY_CLIENT_ID", DEFAULT_MANAGED_IDENTITY_CLIENT_ID).strip()
        credential = DefaultAzureCredential(managed_identity_client_id=client_id)
        kwargs["azure_ad_token_provider"] = _get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )
    return AzureOpenAI(**kwargs), model


def _request_action(
    client: Any,
    model: str,
    original: Image.Image,
    approved: Image.Image,
    headline: str,
    caption: str,
    aspect_ratio: str,
    max_attempts: int,
) -> tuple[LayoutAction, str, str | None, int]:
    original_url = _pil_image_to_data_url(_preview(original), "JPEG", 80)
    approved_url = _pil_image_to_data_url(_preview(approved), "JPEG", 85)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        retry = "" if attempt == 1 else f"\nPrevious output was invalid: {last_error}. Return corrected JSON only."
        try:
            with client.responses.stream(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": "Image A (full original)"},
                            {"type": "input_image", "image_url": original_url},
                            {"type": "input_text", "text": "Image B (human-approved final layout)"},
                            {"type": "input_image", "image_url": approved_url},
                            {
                                "type": "input_text",
                                "text": (
                                    f"{RECOVERY_PROMPT}\n\nHeadline: {headline}\nCaption: {caption}\n"
                                    f"Target aspect ratio: {aspect_ratio}{retry}"
                                ),
                            },
                        ],
                    }
                ],
                reasoning={"effort": "low"},
                text={"verbosity": "low"},
                max_output_tokens=256,
                timeout=float(os.getenv("CROP_VLM_TIMEOUT", "60")),
            ) as stream:
                response = stream.get_final_response()
            text = extract_response_text(response)
            action = parse_layout_action(text).action
            return action, text, getattr(response, "id", None), attempt
        except Exception as error:
            last_error = error
            if attempt < max_attempts:
                time.sleep(min(8.0, 1.5**attempt))
    raise RuntimeError("GPT layout action recovery exhausted retries") from last_error


def _render_action(
    original: Image.Image,
    action: LayoutAction,
    target_ratio: float,
    manual: dict[str, Any],
) -> tuple[Image.Image, dict[str, Any]]:
    background = _parse_hex_color(manual.get("fill_color_code"))
    if action.operation == "crop_pad" and background is not None:
        box = _action_box(action, original.size)
        selected = original.crop(box)
        try:
            padded = pad_image_to_ratio(selected, target_ratio, background_color=background)
        finally:
            selected.close()
        image = padded.image
        source_box = box
        content_box = padded.content_box
        padding_fraction = 1.0 - ((box[2] - box[0]) * (box[3] - box[1])) / (image.width * image.height)
        background_color = padded.background_color
    elif action.operation == "pad" and background is not None:
        padded = pad_image_to_ratio(original, target_ratio, background_color=background)
        image = padded.image
        source_box = (0, 0, original.width, original.height)
        content_box = padded.content_box
        padding_fraction = 1.0 - (original.width * original.height) / (image.width * image.height)
        background_color = padded.background_color
    else:
        rendered = render_layout_action(original, action, target_ratio)
        image = rendered.image
        source_box = rendered.source_box
        content_box = rendered.content_box
        padding_fraction = rendered.padding_fraction
        background_color = rendered.background_color
    width, height = original.size
    normalized = [
        source_box[0] / width,
        source_box[1] / height,
        source_box[2] / width,
        source_box[3] / height,
    ]
    return image, {
        "source_box_pixels": list(source_box),
        "bbox_normalized": normalized,
        "content_box_pixels": list(content_box),
        "background_color_rgb": list(edge_median_color(original)),
        "padding_color_rgb": list(background_color) if background_color is not None else None,
        "padding_fraction": padding_fraction,
        "output_width": image.width,
        "output_height": image.height,
        "render_mode": "padded" if action.operation in {"crop_pad", "pad"} else "cropped",
        "was_cropped": action.operation in {"crop", "crop_pad"},
        "was_padded": action.operation in {"crop_pad", "pad"},
        "feasible": action.operation == "crop",
    }


def _action_from_audited(spec: dict[str, Any]) -> LayoutAction:
    box = spec["source_box_normalized"]
    return LayoutAction(
        operation=str(spec["operation"]),
        x1_pct=round(float(box[0]) * 100),
        y1_pct=round(float(box[1]) * 100),
        x2_pct=round(float(box[2]) * 100),
        y2_pct=round(float(box[3]) * 100),
    )


def _action_box(action: LayoutAction, size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    return (
        round(action.x1_pct / 100 * width),
        round(action.y1_pct / 100 * height),
        round(action.x2_pct / 100 * width),
        round(action.y2_pct / 100 * height),
    )


def _preview(image: Image.Image, maximum_side: int = 1024) -> Image.Image:
    preview = image.convert("RGB").copy()
    preview.thumbnail((maximum_side, maximum_side), Image.Resampling.LANCZOS)
    return preview


def _parse_hex_color(value: Any) -> tuple[int, int, int] | None:
    if value in (None, ""):
        return None
    text = str(value)
    if len(text) != 7 or not text.startswith("#"):
        return None
    return tuple(int(text[index : index + 2], 16) for index in (1, 3, 5))


def _aspect_ratio(value: float) -> str:
    return f"{float(value):g}:1"


def _ratio_value(value: Any) -> float:
    if isinstance(value, str) and ":" in value:
        numerator, denominator = value.split(":", maxsplit=1)
        return float(numerator) / float(denominator)
    return float(value)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)