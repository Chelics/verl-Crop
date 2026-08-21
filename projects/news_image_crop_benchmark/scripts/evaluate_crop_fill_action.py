#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import os
import re
import threading
import time
from collections import Counter
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageOps

from news_crop_benchmark.geometry import TARGET_RATIOS
from news_crop_benchmark.layout import render_crop_fill_action
from news_crop_benchmark.policy_model_adapter import PolicyModelAdapter
from news_crop_benchmark.protocol import CropFillAction, parse_crop_fill_action, parse_crop_fill_detail_action

SOURCE_COLUMNS = ("image_id", "original_image", "title", "ImageCaption")
FINAL_STATUSES = {"valid", "retry_exhausted"}
RATIO_TOLERANCE = 0.002
TITLE_PATTERN = re.compile(r"^News headline:\s*(.*)$", re.MULTILINE)
CAPTION_PATTERN = re.compile(r"^Image caption:\s*(.*)$", re.MULTILINE)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_pixel_hash(image: Image.Image) -> str:
    normalized = ImageOps.exif_transpose(image).convert("RGB")
    payload = normalized.tobytes() + f"{normalized.width}x{normalized.height}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    temporary_path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def save_image_atomic(image: Image.Image, path: Path, *, format_name: str, **kwargs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    image.save(temporary_path, format=format_name, **kwargs)
    temporary_path.replace(path)


def task_id(image_id: str, target_ratio: float) -> str:
    return f"{image_id}__ratio_{target_ratio:g}"


def load_prompt_template(path: Path) -> str:
    template = path.read_text(encoding="utf-8").strip()
    for variable in ("title", "caption", "target_ratio"):
        if f"{{{variable}}}" not in template:
            raise ValueError(f"prompt template is missing {{{variable}}}")
    if not template.startswith("<image>\n"):
        raise ValueError("prompt template must begin with <image> on its own line")
    return template


def load_and_materialize_tasks(
    data_path: Path,
    output_dir: Path,
    prompt_template: str,
    max_images: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parquet = pq.ParquetFile(data_path)
    missing = sorted(set(SOURCE_COLUMNS) - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"test data is missing required columns: {missing}")
    table = pq.read_table(data_path, columns=list(SOURCE_COLUMNS), pre_buffer=False)
    if any(table[name].null_count for name in SOURCE_COLUMNS):
        nulls = {name: table[name].null_count for name in SOURCE_COLUMNS if table[name].null_count}
        raise ValueError(f"test data contains null required fields: {nulls}")
    rows = table.to_pylist()
    if max_images is not None:
        rows = rows[:max_images]

    source_dir = output_dir / ".source_images"
    preview_dir = output_dir / "renders" / "originals"
    tasks = []
    manifest = []
    seen_ids = set()
    for source_index, row in enumerate(rows):
        image_id = str(row["image_id"])
        title = " ".join(str(row["title"]).split())
        caption = " ".join(str(row["ImageCaption"]).split())
        if not image_id or image_id in seen_ids:
            raise ValueError(f"image_id must be non-empty and unique: {image_id!r}")
        if not title or not caption:
            raise ValueError(f"title and caption must be non-empty for {image_id}")
        with Image.open(BytesIO(bytes(row["original_image"]))) as encoded:
            encoded.load()
            if normalized_pixel_hash(encoded) != image_id:
                raise ValueError(f"normalized pixel hash does not match image_id: {image_id}")
            original = ImageOps.exif_transpose(encoded).convert("RGB")
        width, height = original.size
        source_path = (source_dir / f"{image_id}.webp").resolve()
        if source_path.exists():
            with Image.open(source_path) as existing:
                if normalized_pixel_hash(existing) != image_id:
                    raise ValueError(f"materialized source image differs: {source_path}")
        else:
            save_image_atomic(original, source_path, format_name="WEBP", lossless=True, method=6)
        preview_path = preview_dir / f"{image_id}.jpg"
        if not preview_path.exists():
            preview = original.copy()
            if max(preview.size) > 1200:
                preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
            save_image_atomic(preview, preview_path, format_name="JPEG", quality=90, optimize=True)
            preview.close()
        original.close()
        manifest.append(
            {
                "source_index": source_index,
                "image_id": image_id,
                "title": title,
                "caption": caption,
                "image_width": width,
                "image_height": height,
                "source_image_path": str(source_path),
                "original_render_path": preview_path.relative_to(output_dir).as_posix(),
            }
        )
        for ratio in TARGET_RATIOS:
            tasks.append(
                {
                    "task_id": task_id(image_id, ratio),
                    "source_index": source_index,
                    "image_id": image_id,
                    "title": title,
                    "caption": caption,
                    "target_ratio": float(ratio),
                    "image_width": width,
                    "image_height": height,
                    "image_path": str(source_path),
                    "original_render_path": preview_path.relative_to(output_dir).as_posix(),
                    "prompt": prompt_template.format(title=title, caption=caption, target_ratio=f"{ratio:g}"),
                }
            )
        seen_ids.add(image_id)
    return tasks, manifest


def load_swift_message_tasks(
    data_path: Path,
    output_dir: Path,
    max_images: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    required = {"messages", "images", "image_id", "source_index", "target_ratio"}
    parquet = pq.ParquetFile(data_path)
    missing = sorted(required - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"Swift test data is missing required columns: {missing}")
    rows = pq.read_table(data_path, columns=sorted(required), pre_buffer=False).to_pylist()
    selected_ids = []
    for row in rows:
        image_id = str(row["image_id"])
        if image_id not in selected_ids:
            selected_ids.append(image_id)
    if max_images is not None:
        selected_ids = selected_ids[:max_images]
    selected_id_set = set(selected_ids)
    rows = [row for row in rows if str(row["image_id"]) in selected_id_set]

    preview_dir = output_dir / "renders" / "originals"
    image_metadata = {}
    manifest = []
    tasks = []
    seen_keys = set()
    for row_index, row in enumerate(rows):
        image_id = str(row["image_id"])
        messages = row["messages"]
        images = row["images"]
        if len(messages) != 1 or messages[0]["role"] != "user" or not isinstance(messages[0]["content"], str):
            raise ValueError(f"row {row_index} must contain exactly one user message")
        prompt = messages[0]["content"]
        if prompt.count("<image>") != 1:
            raise ValueError(f"row {row_index} must contain exactly one image placeholder")
        if not isinstance(images, list) or len(images) != 1 or not isinstance(images[0], str):
            raise ValueError(f"row {row_index} must contain exactly one image path")
        image_path = Path(images[0])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        ratio = float(row["target_ratio"])
        key = (image_id, ratio)
        if key in seen_keys:
            raise ValueError(f"duplicate Swift test task: {key}")
        seen_keys.add(key)
        title_match = TITLE_PATTERN.search(prompt)
        caption_match = CAPTION_PATTERN.search(prompt)
        if title_match is None or caption_match is None:
            raise ValueError(f"row {row_index} prompt is missing headline or caption")

        if image_id not in image_metadata:
            with Image.open(image_path) as source:
                original = ImageOps.exif_transpose(source).convert("RGB")
            try:
                if normalized_pixel_hash(original) != image_id:
                    raise ValueError(f"normalized pixel hash does not match image_id: {image_id}")
                preview_path = preview_dir / f"{image_id}.jpg"
                preview = original.copy()
                if max(preview.size) > 1200:
                    preview.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
                save_image_atomic(preview, preview_path, format_name="JPEG", quality=90, optimize=True)
                preview.close()
                image_metadata[image_id] = {
                    "image_width": original.width,
                    "image_height": original.height,
                    "image_path": str(image_path),
                    "original_render_path": preview_path.relative_to(output_dir).as_posix(),
                }
                manifest.append(
                    {
                        "source_index": int(row["source_index"]),
                        "image_id": image_id,
                        "title": title_match.group(1).strip(),
                        "caption": caption_match.group(1).strip(),
                        **image_metadata[image_id],
                    }
                )
            finally:
                original.close()
        metadata = image_metadata[image_id]
        tasks.append(
            {
                "task_id": task_id(image_id, ratio),
                "source_index": int(row["source_index"]),
                "image_id": image_id,
                "title": title_match.group(1).strip(),
                "caption": caption_match.group(1).strip(),
                "target_ratio": ratio,
                "prompt": prompt,
                **metadata,
            }
        )
    expected_tasks = len(selected_ids) * len(TARGET_RATIOS)
    if len(tasks) != expected_tasks:
        raise ValueError(f"Swift test data contains {len(tasks)} tasks, expected {expected_tasks}")
    return tasks, manifest


def progress_path(output_dir: Path, current_task_id: str) -> Path:
    return output_dir / "progress" / f"{current_task_id}.json"


def build_retry_prompt(prompt: str, attempts: Sequence[dict[str, Any]], response_protocol: str) -> str:
    if not attempts:
        return prompt
    error = attempts[-1].get("parse_error") or "output did not satisfy the action-v4 protocol"
    fields = "six-field" if response_protocol == "detail-v4" else "five-field"
    description = " and non-empty description" if response_protocol == "detail-v4" else ""
    return (
        f"{prompt}\n\nYour previous output was rejected by the validator: {error}\n"
        f"Generate a new answer containing only the required {fields} JSON object. Recheck the target ratio, "
        f"boolean flags, conditional null fields, normalized crop box, integer RGB fill color{description}."
    )


def parse_response(response: str, response_protocol: str):
    if response_protocol == "detail-v4":
        return parse_crop_fill_detail_action(response)
    return parse_crop_fill_action(response)


def action_record(action: CropFillAction) -> dict[str, Any]:
    return {
        "predicted_target_ratio": action.target_ratio,
        "is_cropped": action.is_cropped,
        "is_filled": action.is_filled,
        "crop_box": list(action.crop_box) if action.crop_box is not None else None,
        "fill_color": list(action.fill_color) if action.fill_color is not None else None,
        "operation": action.operation,
    }


def generate_actions(
    tasks: list[dict[str, Any]],
    output_dir: Path,
    model_path: Path,
    adapter_path: Path,
    args: argparse.Namespace,
) -> None:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapter = PolicyModelAdapter.create("qwen35")
    renderer = adapter.load_renderer(model_path)
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        enable_lora=True,
        max_lora_rank=args.lora_rank,
        **adapter.llm_kwargs(internvl_max_dynamic_patch=4),
    )
    lora_request = LoRARequest("crop-fill-action-v4", 1, str(adapter_path))
    write_json_atomic(
        output_dir / "runtime.json",
        {
            "renderer_class": type(renderer).__name__,
            "vllm_version": package_version("vllm"),
            "transformers_version": package_version("transformers"),
            "adapter_path": str(adapter_path),
            "lora_rank": args.lora_rank,
        },
    )

    attempts_by_id = {}
    active = {}
    for task in tasks:
        path = progress_path(output_dir, task["task_id"])
        attempts = []
        if path.exists():
            progress = json.loads(path.read_text(encoding="utf-8"))
            attempts = list(progress.get("attempts", []))
            if progress.get("status") in FINAL_STATUSES:
                continue
        attempts_by_id[task["task_id"]] = attempts
        active[task["task_id"]] = task

    for attempt_number in range(1, args.max_attempts + 1):
        attempt_tasks = [
            task for current_id, task in active.items() if len(attempts_by_id[current_id]) == attempt_number - 1
        ]
        for start in range(0, len(attempt_tasks), args.prompt_batch_size):
            batch = attempt_tasks[start : start + args.prompt_batch_size]
            images = []
            requests = []
            try:
                for task in batch:
                    with Image.open(task["image_path"]) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                    images.append(image)
                    requests.append(
                        adapter.build_request(
                            renderer,
                            build_retry_prompt(
                                task["prompt"], attempts_by_id[task["task_id"]], args.response_protocol
                            ),
                            image,
                            image_max_pixels=args.image_max_pixels,
                            image_min_pixels=args.image_min_pixels,
                        )
                    )
                params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_tokens,
                    seed=args.seed + attempt_number - 1,
                )
                outputs = llm.generate(requests, sampling_params=params, lora_request=lora_request)
                for task, output in zip(batch, outputs, strict=True):
                    current_id = task["task_id"]
                    response = output.outputs[0].text if output.outputs else ""
                    record = {
                        "attempt": attempt_number,
                        "seed": args.seed + attempt_number - 1,
                        "response": response,
                        "valid": False,
                        "parse_error": None,
                    }
                    try:
                        action = parse_response(response, args.response_protocol).action
                        if not abs(action.target_ratio - task["target_ratio"]) <= 1e-6:
                            raise ValueError(
                                f"target_ratio differs from task: {action.target_ratio} != {task['target_ratio']}"
                            )
                    except ValueError as error:
                        record["parse_error"] = str(error)
                        attempts_by_id[current_id].append(record)
                        status = "retry_exhausted" if attempt_number == args.max_attempts else "retrying"
                        write_json_atomic(
                            progress_path(output_dir, current_id),
                            {
                                "task_id": current_id,
                                "status": status,
                                "attempts": attempts_by_id[current_id],
                                "action": None,
                            },
                        )
                        if status == "retry_exhausted":
                            active.pop(current_id)
                        continue
                    record["valid"] = True
                    attempts_by_id[current_id].append(record)
                    write_json_atomic(
                        progress_path(output_dir, current_id),
                        {
                            "task_id": current_id,
                            "status": "valid",
                            "attempts": attempts_by_id[current_id],
                            "action": action_record(action),
                        },
                    )
                    active.pop(current_id)
            finally:
                for image in images:
                    image.close()
    if active:
        raise RuntimeError(f"generation left {len(active)} tasks unfinished")


def generate_actions_transformers(
    tasks: list[dict[str, Any]],
    output_dir: Path,
    model_path: Path,
    adapter_path: Path,
    args: argparse.Namespace,
) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor

    load_started = time.perf_counter()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, adapter_path, is_trainable=False)
    model.eval()
    model_load_seconds = time.perf_counter() - load_started
    write_json_atomic(
        output_dir / "runtime.json",
        {
            "backend": "transformers",
            "processor_class": type(processor).__name__,
            "model_class": type(model.base_model.model).__name__,
            "torch_version": torch.__version__,
            "transformers_version": package_version("transformers"),
            "peft_version": package_version("peft"),
            "adapter_path": str(adapter_path),
            "lora_rank": args.lora_rank,
            "model_load_seconds": model_load_seconds,
        },
    )

    attempts_by_id = {}
    active = {}
    for task in tasks:
        path = progress_path(output_dir, task["task_id"])
        attempts = []
        if path.exists():
            progress = json.loads(path.read_text(encoding="utf-8"))
            attempts = list(progress.get("attempts", []))
            if progress.get("status") in FINAL_STATUSES:
                continue
        attempts_by_id[task["task_id"]] = attempts
        active[task["task_id"]] = task

    for attempt_number in range(1, args.max_attempts + 1):
        attempt_tasks = [
            task for current_id, task in active.items() if len(attempts_by_id[current_id]) == attempt_number - 1
        ]
        for task in attempt_tasks:
            current_id = task["task_id"]
            with Image.open(task["image_path"]) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
            try:
                prompt = build_retry_prompt(task["prompt"], attempts_by_id[current_id], args.response_protocol)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt.replace("<image>\n", "", 1)},
                        ],
                    }
                ]
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(model.device) if hasattr(value, "to") else value for key, value in inputs.items()}
                generation_started = time.perf_counter()
                generation_kwargs = {
                    "max_new_tokens": args.max_tokens,
                    "do_sample": args.temperature > 0,
                    "use_cache": True,
                }
                if args.temperature > 0:
                    generation_kwargs.update({"temperature": args.temperature, "top_p": args.top_p})
                with torch.inference_mode():
                    generated = model.generate(**inputs, **generation_kwargs)
                generation_seconds = time.perf_counter() - generation_started
                input_length = inputs["input_ids"].shape[-1]
                response = processor.batch_decode(
                    generated[:, input_length:],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]
            finally:
                image.close()
            record = {
                "attempt": attempt_number,
                "seed": args.seed + attempt_number - 1,
                "response": response,
                "valid": False,
                "parse_error": None,
                "generation_seconds": generation_seconds,
                "generated_tokens": int(generated.shape[-1] - input_length),
            }
            try:
                action = parse_response(response, args.response_protocol).action
                if not abs(action.target_ratio - task["target_ratio"]) <= 1e-6:
                    raise ValueError(f"target_ratio differs from task: {action.target_ratio} != {task['target_ratio']}")
            except ValueError as error:
                record["parse_error"] = str(error)
                attempts_by_id[current_id].append(record)
                status = "retry_exhausted" if attempt_number == args.max_attempts else "retrying"
                write_json_atomic(
                    progress_path(output_dir, current_id),
                    {
                        "task_id": current_id,
                        "status": status,
                        "attempts": attempts_by_id[current_id],
                        "action": None,
                    },
                )
                if status == "retry_exhausted":
                    active.pop(current_id)
                continue
            record["valid"] = True
            attempts_by_id[current_id].append(record)
            write_json_atomic(
                progress_path(output_dir, current_id),
                {
                    "task_id": current_id,
                    "status": "valid",
                    "attempts": attempts_by_id[current_id],
                    "action": action_record(action),
                },
            )
            active.pop(current_id)
    if active:
        raise RuntimeError(f"generation left {len(active)} tasks unfinished")


def render_results(
    tasks: list[dict[str, Any]], output_dir: Path, response_protocol: str = "action-v4"
) -> list[dict[str, Any]]:
    details = []
    for task in tasks:
        progress = json.loads(progress_path(output_dir, task["task_id"]).read_text(encoding="utf-8"))
        attempts = progress["attempts"]
        valid_attempt = next((attempt for attempt in attempts if attempt["valid"]), None)
        detail = {
            "task_id": task["task_id"],
            "source_index": task["source_index"],
            "image_id": task["image_id"],
            "title": task["title"],
            "caption": task["caption"],
            "target_ratio": task["target_ratio"],
            "image_width": task["image_width"],
            "image_height": task["image_height"],
            "original_render_path": task["original_render_path"],
            "generation_status": progress["status"],
            "attempt_count": len(attempts),
            "first_attempt_valid": bool(attempts and attempts[0]["valid"]),
            "final_response": valid_attempt["response"] if valid_attempt else None,
            "last_parse_error": attempts[-1].get("parse_error") if attempts else None,
            "generation_seconds": sum(attempt.get("generation_seconds", 0.0) for attempt in attempts),
            "generated_tokens": sum(attempt.get("generated_tokens", 0) for attempt in attempts),
            "operation": None,
            "predicted_target_ratio": None,
            "is_cropped": None,
            "is_filled": None,
            "crop_box": None,
            "fill_color": None,
            "render_status": "not_rendered",
            "render_error": None,
            "candidate_path": None,
            "source_box": None,
            "content_box": None,
            "background_color": None,
            "padding_fraction": None,
            "render_width": None,
            "render_height": None,
            "render_ratio": None,
            "output_ratio_error": None,
            "ratio_compliant": False,
        }
        if progress["status"] == "valid":
            action = parse_response(valid_attempt["response"], response_protocol).action
            detail.update(action_record(action))
            with Image.open(task["image_path"]) as source:
                original = ImageOps.exif_transpose(source).convert("RGB")
            try:
                rendered = render_crop_fill_action(original, action, task["target_ratio"])
            except ValueError as error:
                detail["render_error"] = str(error)
            else:
                candidate_path = output_dir / "renders" / "candidates" / f"{task['task_id']}.jpg"
                save_image_atomic(rendered.image, candidate_path, format_name="JPEG", quality=95, optimize=True)
                detail.update(
                    {
                        "render_status": "rendered",
                        "candidate_path": candidate_path.relative_to(output_dir).as_posix(),
                        "source_box": list(rendered.source_box),
                        "content_box": list(rendered.content_box),
                        "background_color": list(rendered.background_color)
                        if rendered.background_color is not None
                        else None,
                        "padding_fraction": rendered.padding_fraction,
                        "render_width": rendered.image.width,
                        "render_height": rendered.image.height,
                        "render_ratio": rendered.image.width / rendered.image.height,
                        "output_ratio_error": rendered.output_ratio_error,
                        "ratio_compliant": rendered.output_ratio_error <= RATIO_TOLERANCE,
                    }
                )
                rendered.image.close()
            finally:
                original.close()
        details.append(detail)
    return details


def summarize_subset(details: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [detail for detail in details if detail["generation_status"] == "valid"]
    rendered = [detail for detail in details if detail["render_status"] == "rendered"]
    crop_areas = []
    for detail in valid:
        if detail["crop_box"] is not None:
            left, top, right, bottom = detail["crop_box"]
            crop_areas.append((right - left) * (bottom - top))
    return {
        "tasks": len(details),
        "first_attempt_valid_count": sum(detail["first_attempt_valid"] for detail in details),
        "first_attempt_valid_rate": sum(detail["first_attempt_valid"] for detail in details) / len(details)
        if details
        else 0.0,
        "eventual_valid_count": len(valid),
        "eventual_valid_rate": len(valid) / len(details) if details else 0.0,
        "retry_exhausted_count": sum(detail["generation_status"] == "retry_exhausted" for detail in details),
        "mean_attempt_count": mean(detail["attempt_count"] for detail in details) if details else 0.0,
        "mean_generation_seconds": mean(detail.get("generation_seconds", 0.0) for detail in details)
        if details
        else 0.0,
        "total_generation_seconds": sum(detail.get("generation_seconds", 0.0) for detail in details),
        "total_generated_tokens": sum(detail.get("generated_tokens", 0) for detail in details),
        "operation_counts": dict(sorted(Counter(detail["operation"] or "invalid" for detail in details).items())),
        "rendered_count": len(rendered),
        "render_success_rate": len(rendered) / len(details) if details else 0.0,
        "ratio_compliant_count": sum(detail["ratio_compliant"] for detail in rendered),
        "ratio_compliance_rate": sum(detail["ratio_compliant"] for detail in rendered) / len(rendered)
        if rendered
        else 0.0,
        "mean_crop_area_fraction": mean(crop_areas) if crop_areas else None,
        "mean_padding_fraction": mean(
            detail["padding_fraction"] for detail in rendered if detail["padding_fraction"] is not None
        )
        if rendered
        else None,
    }


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": summarize_subset(details),
        "by_ratio": {
            f"{ratio:g}": summarize_subset([detail for detail in details if detail["target_ratio"] == ratio])
            for ratio in TARGET_RATIOS
        },
    }


def write_results(details: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    details.sort(key=lambda item: (item["source_index"], TARGET_RATIOS.index(item["target_ratio"])))
    (output_dir / "details.jsonl").write_text(
        "".join(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n" for detail in details),
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist(details), output_dir / "details.parquet", compression="zstd")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.writer(output)
        writer.writerow(["scope", "tasks", "first_attempt_valid_rate", "eventual_valid_rate", "render_success_rate", "ratio_compliance_rate", "operation_counts"])
        for scope, metrics in [("overall", summary["overall"]), *summary["by_ratio"].items()]:
            writer.writerow(
                [
                    scope,
                    metrics["tasks"],
                    metrics["first_attempt_valid_rate"],
                    metrics["eventual_valid_rate"],
                    metrics["render_success_rate"],
                    metrics["ratio_compliance_rate"],
                    json.dumps(metrics["operation_counts"], sort_keys=True),
                ]
            )


def render_html_report(details: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    groups = []
    by_index = {}
    for detail in details:
        by_index.setdefault(detail["source_index"], []).append(detail)
    for source_index, rows in sorted(by_index.items()):
        rows.sort(key=lambda item: TARGET_RATIOS.index(item["target_ratio"]))
        candidates = []
        for row in rows:
            image_html = (
                f'<img src="{html.escape(row["candidate_path"])}" alt="candidate">'
                if row["candidate_path"]
                else '<div class="missing">Not rendered</div>'
            )
            candidates.append(
                f'<figure>{image_html}<figcaption>{row["target_ratio"]:g}:1 · '
                f'{html.escape(str(row["operation"] or "invalid"))} · attempts {row["attempt_count"]} · '
                f'ratio ok {row["ratio_compliant"]}</figcaption></figure>'
            )
        groups.append(
            f'<section><h2>{html.escape(rows[0]["title"])}</h2><p>{html.escape(rows[0]["caption"])}</p>'
            f'<div class="grid"><figure><img src="{html.escape(rows[0]["original_render_path"])}" alt="original">'
            f'<figcaption>Original</figcaption></figure>{"".join(candidates)}</div></section>'
        )
    overall = summary["overall"]
    document = f"""<!doctype html><html><head><meta charset="utf-8"><title>Crop/Fill No-Judge Test</title>
<style>body{{font-family:Arial,sans-serif;margin:24px;background:#f5f5f2;color:#171717}}section{{border-top:1px solid #aaa;padding:18px 0}}.grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:12px}}img{{width:100%;height:220px;object-fit:contain;background:white}}figure{{margin:0}}figcaption{{font-size:12px;margin-top:5px}}.missing{{height:220px;display:grid;place-items:center;background:#ddd}}@media(max-width:900px){{.grid{{grid-template-columns:1fr 1fr}}}}</style></head><body>
<h1>Crop/Fill Action-v4 No-Judge Test</h1><p>Tasks: {overall['tasks']} · first valid: {overall['first_attempt_valid_rate']:.3f} · eventual valid: {overall['eventual_valid_rate']:.3f} · rendered: {overall['render_success_rate']:.3f} · ratio compliant: {overall['ratio_compliance_rate']:.3f}</p>{''.join(groups)}</body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def prepare_output(output_dir: Path, config: dict[str, Any], resume: bool) -> None:
    config_path = output_dir / "run_config.yaml"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"output directory is not empty: {output_dir}")
        if not config_path.is_file() or yaml.safe_load(config_path.read_text(encoding="utf-8")) != config:
            raise ValueError("resume configuration does not match run_config.yaml")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate action-v4 crop/fill generation and rendering without a judge.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lora-adapter-path", type=Path, required=True)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path)
    parser.add_argument("--data-format", choices=("raw-image-once", "swift-messages"), default="raw-image-once")
    parser.add_argument("--response-protocol", choices=("action-v4", "detail-v4"), default="action-v4")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--backend", choices=("vllm", "transformers"), default="vllm")
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    parser.add_argument("--image-min-pixels", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--prompt-batch-size", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in ("lora_rank", "max_attempts", "max_model_len", "max_tokens", "prompt_batch_size", "max_num_seqs"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("max-images must be positive")
    if args.data_format == "raw-image-once" and args.prompt_template is None:
        raise ValueError("prompt-template is required for raw-image-once data")
    if args.data_format == "swift-messages" and args.response_protocol != "detail-v4":
        raise ValueError("swift-messages data requires response-protocol=detail-v4")
    if not 0 <= args.temperature:
        raise ValueError("temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    for path in (args.model, args.lora_adapter_path, args.data):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.prompt_template is not None and not args.prompt_template.exists():
        raise FileNotFoundError(args.prompt_template)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        if not (args.lora_adapter_path / filename).is_file():
            raise FileNotFoundError(args.lora_adapter_path / filename)
    template = load_prompt_template(args.prompt_template) if args.prompt_template is not None else None
    args.output_dir = args.output_dir.resolve()
    config = {
        "run_id": args.run_id,
        "protocol": "crop-fill-action-v4-no-judge-v1",
        "backend": args.backend,
        "quality_metrics_available": False,
        "judge_enabled": False,
        "data_format": args.data_format,
        "response_protocol": args.response_protocol,
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "lora_adapter_path": str(args.lora_adapter_path.resolve()),
        "lora_adapter_config_sha256": sha256_file(args.lora_adapter_path / "adapter_config.json"),
        "lora_adapter_model_sha256": sha256_file(args.lora_adapter_path / "adapter_model.safetensors"),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "prompt_template": str(args.prompt_template.resolve()) if args.prompt_template is not None else None,
        "prompt_sha256": hashlib.sha256(template.encode("utf-8")).hexdigest() if template is not None else None,
        "target_ratios": list(TARGET_RATIOS),
        "max_images": args.max_images,
        "max_attempts": args.max_attempts,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "prompt_batch_size": args.prompt_batch_size,
        "max_num_seqs": args.max_num_seqs,
    }
    prepare_output(args.output_dir, config, args.resume)
    if args.data_format == "swift-messages":
        tasks, manifest = load_swift_message_tasks(args.data, args.output_dir, args.max_images)
    else:
        tasks, manifest = load_and_materialize_tasks(args.data, args.output_dir, template, args.max_images)
    (args.output_dir / "source_manifest.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n" for item in manifest),
        encoding="utf-8",
    )
    if args.backend == "transformers":
        generate_actions_transformers(tasks, args.output_dir, args.model, args.lora_adapter_path, args)
    else:
        generate_actions(tasks, args.output_dir, args.model, args.lora_adapter_path, args)
    details = render_results(tasks, args.output_dir, args.response_protocol)
    summary = {
        "run_id": args.run_id,
        "protocol": "crop-fill-action-v4-no-judge-v1",
        "judge_enabled": False,
        "quality_metrics_available": False,
        "images": len(manifest),
        "target_ratios": list(TARGET_RATIOS),
        **summarize(details),
    }
    write_results(details, summary, args.output_dir)
    render_html_report(details, summary, args.output_dir)
    write_json_atomic(
        args.output_dir / "_NO_JUDGE_EVAL_COMPLETE.json",
        {
            "run_id": args.run_id,
            "images": len(manifest),
            "tasks": len(tasks),
            "valid": summary["overall"]["eventual_valid_count"],
            "rendered": summary["overall"]["rendered_count"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
