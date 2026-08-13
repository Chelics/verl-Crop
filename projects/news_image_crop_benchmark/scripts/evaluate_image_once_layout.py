#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.metadata
import json
import multiprocessing
import os
import subprocess
import sys
import threading
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageOps

from news_crop_benchmark.data import build_prompt, load_policy_prompt_template
from news_crop_benchmark.geometry import TARGET_RATIOS, CropAction, action_to_bbox
from news_crop_benchmark.layout import edge_median_color, pad_image_to_ratio
from news_crop_benchmark.policy_model_adapter import MODEL_FAMILIES, PolicyModelAdapter
from news_crop_benchmark.protocol import parse_percent_crop_action
from news_crop_benchmark.proxy_scorer import crop_image

SOURCE_COLUMNS = ("image_id", "original_image", "title", "ImageCaption")
FINAL_CROP_STATUSES = {"valid", "retry_exhausted"}


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
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary_path.write_bytes(payload)
    temporary_path.replace(path)


def prepare_output_directory(output_dir: Path, config: dict[str, Any], resume: bool) -> None:
    config_path = output_dir / "run_config.yaml"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"output directory is not empty; pass --resume to continue: {output_dir}")
        if not config_path.is_file():
            raise FileNotFoundError(f"resume requires an existing run config: {config_path}")
        existing_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise ValueError("resume configuration does not match run_config.yaml")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True),
        encoding="utf-8",
    )


def save_jpeg(image: Image.Image, path: Path, *, maximum_side: int | None = None, quality: int = 95) -> None:
    output = image.convert("RGB")
    if maximum_side is not None and max(output.size) > maximum_side:
        scale = maximum_side / max(output.size)
        output = output.resize(
            (max(1, round(output.width * scale)), max(1, round(output.height * scale))),
            Image.Resampling.LANCZOS,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    output.save(temporary_path, format="JPEG", quality=quality, optimize=True)
    temporary_path.replace(path)


def task_id(image_id: str, target_ratio: float) -> str:
    return f"{image_id}__ratio_{target_ratio:g}"


def load_and_materialize_tasks(
    data_path: Path,
    output_dir: Path,
    crop_prompt_template: str,
    max_images: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parquet = pq.ParquetFile(data_path)
    missing_columns = sorted(set(SOURCE_COLUMNS) - set(parquet.schema_arrow.names))
    if missing_columns:
        raise ValueError(f"dataset is missing required columns: {missing_columns}")
    table = pq.read_table(data_path, columns=list(SOURCE_COLUMNS))
    if any(table[name].null_count for name in SOURCE_COLUMNS):
        null_counts = {name: table[name].null_count for name in SOURCE_COLUMNS if table[name].null_count}
        raise ValueError(f"dataset contains null required fields: {null_counts}")

    rows = table.to_pylist()
    if max_images is not None:
        rows = rows[:max_images]
    source_dir = output_dir / ".source_images"
    preview_dir = output_dir / "renders" / "originals"
    tasks: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    seen_image_ids: set[str] = set()
    for source_index, row in enumerate(rows):
        image_id = str(row["image_id"])
        title = " ".join(str(row["title"]).split())
        caption = " ".join(str(row["ImageCaption"]).split())
        payload = bytes(row["original_image"])
        if not image_id or image_id in seen_image_ids:
            raise ValueError(f"image_id must be non-empty and unique: {image_id!r}")
        if not title:
            raise ValueError(f"title must be non-empty for image {image_id}")
        with Image.open(BytesIO(payload)) as encoded_image:
            encoded_image.load()
            if normalized_pixel_hash(encoded_image) != image_id:
                raise ValueError(f"normalized pixel hash does not match image_id: {image_id}")
            original = ImageOps.exif_transpose(encoded_image).convert("RGB")
        image_width, image_height = original.size
        source_path = (source_dir / f"{image_id}.webp").resolve()
        if source_path.exists():
            if source_path.read_bytes() != payload:
                raise ValueError(f"materialized source image differs from dataset: {source_path}")
        else:
            write_bytes_atomic(source_path, payload)
        preview_path = preview_dir / f"{image_id}.jpg"
        if not preview_path.exists():
            save_jpeg(original, preview_path, maximum_side=1200, quality=90)
        original.close()
        manifest.append(
            {
                "source_index": source_index,
                "image_id": image_id,
                "title": title,
                "caption": caption,
                "image_width": image_width,
                "image_height": image_height,
                "source_image_path": str(source_path),
                "original_render_path": preview_path.relative_to(output_dir).as_posix(),
            }
        )
        for target_ratio in TARGET_RATIOS:
            tasks.append(
                {
                    "task_id": task_id(image_id, target_ratio),
                    "source_index": source_index,
                    "image_id": image_id,
                    "title": title,
                    "caption": caption,
                    "target_ratio": float(target_ratio),
                    "image_width": image_width,
                    "image_height": image_height,
                    "image_path": str(source_path),
                    "crop_prompt": build_prompt(title, float(target_ratio), crop_prompt_template),
                    "original_render_path": preview_path.relative_to(output_dir).as_posix(),
                }
            )
        seen_image_ids.add(image_id)
    return tasks, manifest


def batched(items: Sequence[Any], batch_size: int) -> Iterable[Sequence[Any]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def resolve_gpu_devices(required_gpus: int, visible_devices: str | None = None) -> list[str]:
    if required_gpus <= 0:
        raise ValueError("required_gpus must be positive")
    if visible_devices is None:
        visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        devices = [str(index) for index in range(required_gpus)]
    else:
        devices = [
            device.strip()
            for device in visible_devices.split(",")
            if device.strip() and device.strip() != "-1"
        ]
    if len(devices) < required_gpus:
        raise ValueError(
            f"requested {required_gpus} GPUs, but CUDA_VISIBLE_DEVICES exposes only {len(devices)}: {devices}"
        )
    return devices[:required_gpus]


def run_mode_stage(args: argparse.Namespace, mode_output_dir: Path) -> None:
    command = [
        sys.executable,
        str(Path(__file__).with_name("evaluate_image_once_mode.py")),
        "--model",
        str(args.model),
        "--model-family",
        args.model_family,
        "--model-name",
        args.model_name,
        "--data",
        str(args.data),
        "--mode-prompt-path",
        str(args.mode_prompt_path),
        "--output-dir",
        str(mode_output_dir),
        "--max-attempts",
        str(args.mode_max_attempts),
        "--seed",
        str(args.seed),
        "--temperature",
        str(args.mode_temperature),
        "--top-p",
        str(args.mode_top_p),
        "--max-model-len",
        str(args.max_model_len),
        "--max-tokens",
        str(args.mode_max_tokens),
        "--image-max-pixels",
        str(args.image_max_pixels),
        "--image-min-pixels",
        str(args.image_min_pixels),
        "--internvl-max-dynamic-patch",
        str(args.internvl_max_dynamic_patch),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--data-parallel-size",
        str(args.data_parallel_size),
        "--tensor-parallel-size",
        str(args.tensor_parallel_size),
        "--prompt-batch-size",
        str(args.prompt_batch_size),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--run-id",
        f"{args.run_id}-mode",
        "--resume",
    ]
    if args.max_images is not None:
        command.extend(("--max-images", str(args.max_images)))
    subprocess.run(command, check=True)


def load_mode_results(
    mode_results_dir: Path,
    tasks: Sequence[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    completion_path = mode_results_dir / "_MODE_EVAL_COMPLETE.json"
    details_path = mode_results_dir / "details.jsonl"
    if not completion_path.is_file() or not details_path.is_file():
        raise FileNotFoundError(f"mode results are incomplete: {mode_results_dir}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    expected_task_ids = {task["task_id"] for task in tasks}
    if int(completion.get("tasks", -1)) != len(expected_task_ids):
        raise ValueError("mode result task count does not match the layout task set")
    records: dict[str, dict[str, Any]] = {}
    for line in details_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        current_task_id = str(record["task_id"])
        if current_task_id in records:
            raise ValueError(f"duplicate mode result: {current_task_id}")
        mode = record.get("predicted_mode")
        if mode is not None and mode not in {"crop", "pad"}:
            raise ValueError(f"invalid predicted mode for {current_task_id}: {mode!r}")
        records[current_task_id] = record
    if set(records) != expected_task_ids:
        missing = sorted(expected_task_ids - set(records))[:5]
        extra = sorted(set(records) - expected_task_ids)[:5]
        raise ValueError(f"mode result task IDs do not match; missing={missing}, extra={extra}")
    return records


def crop_progress_path(output_dir: Path, current_task_id: str) -> Path:
    return output_dir / "progress" / "crop" / f"{current_task_id}.json"


def build_crop_retry_prompt(base_prompt: str, previous_attempts: Sequence[dict[str, Any]]) -> str:
    if not previous_attempts:
        return base_prompt
    error = str(previous_attempts[-1].get("parse_error") or "output did not satisfy the crop protocol")
    return (
        f"{base_prompt}\n\nYour previous output was rejected by the validator: {error}\n"
        "Generate a new answer containing only one JSON object. Recheck that cx_pct and cy_pct "
        "are integers in [0, 100], area_pct is an integer in [1, 100], and there are no other fields."
    )


def _load_crop_state(
    output_dir: Path,
    tasks: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    active: dict[str, dict[str, Any]] = {}
    attempts: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        path = crop_progress_path(output_dir, task["task_id"])
        if path.exists():
            progress = json.loads(path.read_text(encoding="utf-8"))
            if progress["status"] in FINAL_CROP_STATUSES:
                continue
            attempts[task["task_id"]] = list(progress["attempts"])
        else:
            attempts[task["task_id"]] = []
        active[task["task_id"]] = task
    return active, attempts


def run_crop_worker(
    rank: int,
    gpu_devices: list[str],
    tasks: list[dict[str, Any]],
    output_dir: Path,
    model_path: Path,
    args: argparse.Namespace,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_devices)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from vllm import LLM, SamplingParams

    active, attempts = _load_crop_state(output_dir, tasks)
    if not active:
        return
    adapter = PolicyModelAdapter.create(args.model_family)
    renderer = adapter.load_renderer(model_path)
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        **adapter.llm_kwargs(internvl_max_dynamic_patch=args.internvl_max_dynamic_patch),
    )
    write_json_atomic(
        output_dir / "runtime" / f"crop_model_rank_{rank}.json",
        {
            "rank": rank,
            "gpu_devices": gpu_devices,
            **adapter.runtime_metadata(renderer),
            "vllm_version": package_version("vllm"),
            "transformers_version": package_version("transformers"),
            "sentencepiece_version": package_version("sentencepiece"),
        },
    )
    for attempt_number in range(1, args.crop_max_attempts + 1):
        attempt_tasks = [
            task
            for current_task_id, task in active.items()
            if len(attempts[current_task_id]) == attempt_number - 1
        ]
        for task_batch in batched(attempt_tasks, args.prompt_batch_size):
            images: list[Image.Image] = []
            try:
                requests = []
                for task in task_batch:
                    with Image.open(task["image_path"]) as source:
                        image = ImageOps.exif_transpose(source).convert("RGB")
                    images.append(image)
                    requests.append(
                        adapter.build_request(
                            renderer,
                            build_crop_retry_prompt(task["crop_prompt"], attempts[task["task_id"]]),
                            image,
                            image_max_pixels=args.image_max_pixels,
                            image_min_pixels=args.image_min_pixels,
                        )
                    )
                attempt_seed = args.seed + attempt_number - 1
                sampling_params = SamplingParams(
                    temperature=args.crop_temperature,
                    top_p=args.crop_top_p,
                    n=1,
                    max_tokens=args.crop_max_tokens,
                    seed=attempt_seed,
                    **adapter.sampling_kwargs(renderer),
                )
                outputs = llm.generate(requests, sampling_params=sampling_params)
                for task, request_output in zip(task_batch, outputs, strict=True):
                    current_task_id = task["task_id"]
                    response = request_output.outputs[0].text if request_output.outputs else ""
                    attempt_record: dict[str, Any] = {
                        "attempt": attempt_number,
                        "seed": attempt_seed,
                        "response": response,
                        "valid": False,
                        "strict_format": False,
                        "parse_error": None,
                    }
                    try:
                        parse_result = parse_percent_crop_action(response)
                    except ValueError as error:
                        attempt_record["parse_error"] = str(error)
                        attempts[current_task_id].append(attempt_record)
                        status = "retry_exhausted" if attempt_number == args.crop_max_attempts else "retrying"
                        write_json_atomic(
                            crop_progress_path(output_dir, current_task_id),
                            {
                                "task_id": current_task_id,
                                "rank": rank,
                                "status": status,
                                "attempts": attempts[current_task_id],
                                "action": None,
                            },
                        )
                        if status == "retry_exhausted":
                            active.pop(current_task_id)
                        continue
                    action = parse_result.action
                    attempt_record["valid"] = True
                    attempt_record["strict_format"] = parse_result.strict_format
                    attempts[current_task_id].append(attempt_record)
                    write_json_atomic(
                        crop_progress_path(output_dir, current_task_id),
                        {
                            "task_id": current_task_id,
                            "rank": rank,
                            "status": "valid",
                            "attempts": attempts[current_task_id],
                            "action": {
                                "cx": action.center_x,
                                "cy": action.center_y,
                                "area": action.area,
                                "cx_pct": int(round(action.center_x / 10)),
                                "cy_pct": int(round(action.center_y / 10)),
                                "area_pct": int(round(action.area / 10)),
                            },
                        },
                    )
                    active.pop(current_task_id)
            finally:
                for image in images:
                    image.close()
    if active:
        raise RuntimeError(f"crop worker left {len(active)} tasks unfinished")


def run_parallel_crop_generation(
    tasks: list[dict[str, Any]],
    output_dir: Path,
    model_path: Path,
    args: argparse.Namespace,
) -> None:
    if not tasks:
        return
    required_gpus = args.data_parallel_size * args.tensor_parallel_size
    devices = resolve_gpu_devices(required_gpus)
    partitions = [tasks[rank :: args.data_parallel_size] for rank in range(args.data_parallel_size)]
    context = multiprocessing.get_context("spawn")
    processes = []
    for rank, rank_tasks in enumerate(partitions):
        if not rank_tasks:
            continue
        start = rank * args.tensor_parallel_size
        process = context.Process(
            target=run_crop_worker,
            args=(
                rank,
                devices[start : start + args.tensor_parallel_size],
                rank_tasks,
                output_dir,
                model_path,
                args,
            ),
            name=f"crop-model-rank-{rank}",
        )
        process.start()
        processes.append(process)
    failed = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append(f"{process.name}={process.exitcode}")
    if failed:
        raise RuntimeError(f"crop generation workers failed: {', '.join(failed)}")


def render_layouts(
    tasks: Sequence[dict[str, Any]],
    mode_results: dict[str, dict[str, Any]],
    output_dir: Path,
    edge_fraction: float,
) -> dict[str, dict[str, Any]]:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task in tasks:
        by_image[task["image_id"]].append(task)
    rendered: dict[str, dict[str, Any]] = {}
    for image_tasks in by_image.values():
        with Image.open(image_tasks[0]["image_path"]) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
        needs_padding = any(mode_results[task["task_id"]].get("predicted_mode") == "pad" for task in image_tasks)
        border_color = edge_median_color(original, edge_fraction=edge_fraction) if needs_padding else None
        for task in image_tasks:
            current_task_id = task["task_id"]
            mode = mode_results[current_task_id].get("predicted_mode")
            if mode not in {"crop", "pad"}:
                rendered[current_task_id] = {"status": "not_rendered", "reason": "invalid_mode"}
                continue
            output_path = output_dir / "renders" / "candidates" / f"{current_task_id}.jpg"
            if mode == "pad":
                pad_render = pad_image_to_ratio(
                    original,
                    task["target_ratio"],
                    background_color=border_color,
                    edge_fraction=edge_fraction,
                )
                candidate = pad_render.image
                metadata = {
                    "background_color": list(pad_render.background_color),
                    "background_hex": "#" + "".join(f"{channel:02X}" for channel in pad_render.background_color),
                    "content_box": list(pad_render.content_box),
                    "padding_fraction": 1.0 - (original.width * original.height) / (candidate.width * candidate.height),
                    "crop_action": None,
                }
            else:
                crop_path = crop_progress_path(output_dir, current_task_id)
                crop_progress = json.loads(crop_path.read_text(encoding="utf-8"))
                if crop_progress["status"] != "valid":
                    rendered[current_task_id] = {"status": "not_rendered", "reason": "invalid_crop"}
                    continue
                action = crop_progress["action"]
                bbox = action_to_bbox(
                    CropAction(center_x=action["cx"], center_y=action["cy"], area=action["area"]),
                    image_width=task["image_width"],
                    image_height=task["image_height"],
                    target_ratio=task["target_ratio"],
                )
                candidate = crop_image(original, bbox)
                metadata = {
                    "background_color": None,
                    "background_hex": None,
                    "content_box": None,
                    "padding_fraction": 0.0,
                    "crop_action": action,
                }
            save_jpeg(candidate, output_path, quality=95)
            rendered[current_task_id] = {
                "status": "rendered",
                "candidate_path": output_path.relative_to(output_dir).as_posix(),
                "render_width": candidate.width,
                "render_height": candidate.height,
                "render_ratio": candidate.width / candidate.height,
                **metadata,
            }
            candidate.close()
        original.close()
    return rendered


def build_details(
    tasks: Sequence[dict[str, Any]],
    mode_results: dict[str, dict[str, Any]],
    render_results: dict[str, dict[str, Any]],
    output_dir: Path,
) -> list[dict[str, Any]]:
    details = []
    for task in tasks:
        current_task_id = task["task_id"]
        mode_result = mode_results[current_task_id]
        mode = mode_result.get("predicted_mode")
        crop_progress = None
        if mode == "crop":
            crop_progress = json.loads(
                crop_progress_path(output_dir, current_task_id).read_text(encoding="utf-8")
            )
        valid_crop_attempt = next(
            (attempt for attempt in (crop_progress or {}).get("attempts", []) if attempt["valid"]),
            None,
        )
        crop_action = (crop_progress or {}).get("action") or {}
        render = render_results[current_task_id]
        details.append(
            {
                "task_id": current_task_id,
                "source_index": task["source_index"],
                "image_id": task["image_id"],
                "title": task["title"],
                "caption": task["caption"],
                "target_ratio": task["target_ratio"],
                "image_width": task["image_width"],
                "image_height": task["image_height"],
                "original_render_path": task["original_render_path"],
                "predicted_mode": mode,
                "mode_status": mode_result["generation_status"],
                "mode_response": mode_result.get("final_response"),
                "mode_attempt_count": mode_result["total_attempt_count"],
                "crop_status": (crop_progress or {}).get("status"),
                "crop_response": valid_crop_attempt.get("response") if valid_crop_attempt else None,
                "crop_attempt_count": len((crop_progress or {}).get("attempts", [])),
                "crop_strict_format": valid_crop_attempt.get("strict_format") if valid_crop_attempt else None,
                "cx_pct": crop_action.get("cx_pct"),
                "cy_pct": crop_action.get("cy_pct"),
                "area_pct": crop_action.get("area_pct"),
                "render_status": render["status"],
                "render_reason": render.get("reason"),
                "candidate_path": render.get("candidate_path"),
                "render_width": render.get("render_width"),
                "render_height": render.get("render_height"),
                "render_ratio": render.get("render_ratio"),
                "background_color": render.get("background_color"),
                "background_hex": render.get("background_hex"),
                "content_box": render.get("content_box"),
                "padding_fraction": render.get("padding_fraction"),
            }
        )
    return details


def summarize_subset(details: Sequence[dict[str, Any]]) -> dict[str, Any]:
    mode_counts = Counter(detail["predicted_mode"] or "invalid" for detail in details)
    crop_details = [detail for detail in details if detail["predicted_mode"] == "crop"]
    pad_details = [detail for detail in details if detail["predicted_mode"] == "pad"]
    rendered = [detail for detail in details if detail["render_status"] == "rendered"]
    return {
        "tasks": len(details),
        "mode_counts": dict(sorted(mode_counts.items())),
        "crop_count": mode_counts["crop"],
        "pad_count": mode_counts["pad"],
        "invalid_mode_count": mode_counts["invalid"],
        "crop_rate": mode_counts["crop"] / len(details) if details else 0.0,
        "pad_rate": mode_counts["pad"] / len(details) if details else 0.0,
        "crop_generation_success_count": sum(detail["crop_status"] == "valid" for detail in crop_details),
        "crop_generation_failure_count": sum(detail["crop_status"] != "valid" for detail in crop_details),
        "crop_strict_format_count": sum(detail["crop_strict_format"] is True for detail in crop_details),
        "rendered_count": len(rendered),
        "render_success_rate": len(rendered) / len(details) if details else 0.0,
        "mean_crop_area_pct": mean(detail["area_pct"] for detail in crop_details if detail["area_pct"] is not None)
        if any(detail["area_pct"] is not None for detail in crop_details)
        else None,
        "mean_padding_fraction": mean(detail["padding_fraction"] for detail in pad_details)
        if pad_details
        else None,
        "background_color_counts": dict(sorted(Counter(detail["background_hex"] for detail in pad_details).items())),
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
    details.sort(key=lambda detail: (detail["source_index"], TARGET_RATIOS.index(detail["target_ratio"])))
    (output_dir / "details.jsonl").write_text(
        "".join(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n" for detail in details),
        encoding="utf-8",
    )
    parquet_rows = [
        {
            **detail,
            "background_color": json.dumps(detail["background_color"]),
            "content_box": json.dumps(detail["content_box"]),
        }
        for detail in details
    ]
    pq.write_table(pa.Table.from_pylist(parquet_rows), output_dir / "details.parquet", compression="zstd")
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as output:
        rows = [("overall", summary["overall"]), *sorted(summary["by_ratio"].items())]
        scalar_keys = [key for key, value in summary["overall"].items() if not isinstance(value, dict)]
        writer = csv.DictWriter(output, fieldnames=["scope", *scalar_keys, "mode_counts"])
        writer.writeheader()
        for scope, metrics in rows:
            writer.writerow(
                {
                    "scope": scope,
                    **{key: metrics[key] for key in scalar_keys},
                    "mode_counts": json.dumps(metrics["mode_counts"], sort_keys=True),
                }
            )
    with (output_dir / "review_template.csv").open("w", newline="", encoding="utf-8-sig") as output:
        fieldnames = [
            "source_index",
            "image_id",
            "title",
            "target_ratio",
            "predicted_mode",
            "background_hex",
            "cx_pct",
            "cy_pct",
            "area_pct",
            "human_mode_label",
            "layout_rating",
            "notes",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for detail in details:
            writer.writerow(
                {
                    **{key: detail[key] for key in fieldnames[:-3]},
                    "human_mode_label": "",
                    "layout_rating": "",
                    "notes": "",
                }
            )


def _format(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return str(value)


def render_html_report(details: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        by_image[detail["image_id"]].append(detail)
    metric_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_format(value))}</td></tr>"
        for key, value in summary["overall"].items()
    )
    sections = []
    ordered_images = sorted(by_image.values(), key=lambda group: group[0]["source_index"])
    for image_details in ordered_images:
        image_details.sort(key=lambda detail: TARGET_RATIOS.index(detail["target_ratio"]))
        first = image_details[0]
        cards = []
        for detail in image_details:
            mode = detail["predicted_mode"] or "invalid"
            candidate = (
                f'<img src="{html.escape(detail["candidate_path"])}" alt="Final {html.escape(mode)} layout">'
                if detail["candidate_path"]
                else '<div class="missing">No rendered layout</div>'
            )
            if mode == "pad":
                swatch = (
                    f'<span class="swatch" style="background:{html.escape(detail["background_hex"])}"></span>'
                    f'{html.escape(detail["background_hex"])}'
                )
                operation = f"Edge background: {swatch}; padding: {_format(detail['padding_fraction'])}"
            elif mode == "crop":
                operation = (
                    f"Crop: ({detail['cx_pct']}, {detail['cy_pct']}), area {detail['area_pct']}%"
                )
            else:
                operation = html.escape(str(detail["render_reason"]))
            cards.append(
                f"""
                <article class="candidate {html.escape(mode)}">
                  <h3>Ratio {detail['target_ratio']:g} · {html.escape(mode.upper())}</h3>
                  {candidate}
                  <p>{operation}</p>
                  <p><strong>Rendered:</strong> {detail['render_width']} × {detail['render_height']}</p>
                  <details><summary>Policy outputs</summary><pre>{html.escape(json.dumps({
                      'mode_response': detail['mode_response'],
                      'crop_response': detail['crop_response'],
                  }, ensure_ascii=False, indent=2))}</pre></details>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="sample">
              <header><h2>{html.escape(first['title'])}</h2><p class="id">{html.escape(first['image_id'])}</p></header>
              <div class="layout">
                <article class="original"><h3>Original</h3>
                  <img src="{html.escape(first['original_render_path'])}" alt="Original image">
                  <p>{html.escape(first['caption'])}</p>
                </article>
                <div class="candidates">{''.join(cards)}</div>
              </div>
            </section>
            """
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(summary['model_name'])} Crop-or-Pad Pipeline</title>
<style>
:root {{ --ink:#202522; --muted:#65706a; --line:#ccd4cf; --paper:#edf1ee; --crop:#176b4d; --pad:#9a5418; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper); font-family:"Segoe UI",sans-serif; }}
main {{ width:min(1640px,96vw); margin:auto; padding:24px 0 64px; }} .summary,.sample {{ background:white; border:1px solid var(--line); padding:20px; margin-bottom:20px; }}
table {{ border-collapse:collapse; width:min(940px,100%); }} th,td {{ border-bottom:1px solid var(--line); padding:7px 10px; text-align:left; }}
.layout {{ display:grid; grid-template-columns:minmax(260px,.8fr) minmax(0,2.2fr); gap:18px; }} .candidates {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.candidate {{ border:1px solid var(--line); border-top:5px solid var(--line); padding:12px; min-width:0; }} .candidate.crop {{ border-top-color:var(--crop); }} .candidate.pad {{ border-top-color:var(--pad); }}
img {{ display:block; width:100%; max-height:540px; object-fit:contain; background:#e6ebe8; }} .candidate img {{ height:340px; }} .swatch {{ display:inline-block; width:18px; height:18px; border:1px solid #777; vertical-align:middle; margin-right:6px; }}
.missing {{ height:220px; display:grid; place-items:center; background:#eee; color:var(--muted); }} .id,p,pre {{ overflow-wrap:anywhere; }} pre {{ white-space:pre-wrap; font-size:12px; }}
@media (max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} .candidates {{ grid-template-columns:1fr; }} main {{ width:100%; padding:10px; }} }}
</style></head><body><main>
<section class="summary"><h1>{html.escape(summary['model_name'])} Crop-or-Pad Pipeline</h1><p>Mode-first routing; crop uses the v1 percentage prompt, pad preserves the complete source on an edge-median background.</p><table>{metric_rows}</table></section>
{''.join(sections)}
</main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def render_markdown_report(details: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    overall = summary["overall"]
    lines = [
        f"# {summary['model_name']} Crop-or-Pad Pipeline",
        "",
        "Mode-first routing. Crop tasks use the v1 percentage prompt; pad tasks preserve the full source image on an edge-median background.",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "tasks",
        "crop_count",
        "pad_count",
        "invalid_mode_count",
        "crop_generation_success_count",
        "crop_generation_failure_count",
        "crop_strict_format_count",
        "rendered_count",
        "render_success_rate",
        "mean_crop_area_pct",
        "mean_padding_fraction",
    ):
        lines.append(f"| `{key}` | {_format(overall[key])} |")
    lines.extend(
        [
            "",
            "## By Ratio",
            "",
            "| Ratio | Crop | Pad | Crop Success | Rendered | Mean Crop Area | Mean Padding |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ratio in TARGET_RATIOS:
        metrics = summary["by_ratio"][f"{ratio:g}"]
        lines.append(
            f"| {ratio:g} | {metrics['crop_count']} | {metrics['pad_count']} | "
            f"{metrics['crop_generation_success_count']} | {metrics['rendered_count']} | "
            f"{_format(metrics['mean_crop_area_pct'])} | {_format(metrics['mean_padding_fraction'])} |"
        )
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        by_image[detail["image_id"]].append(detail)
    lines.extend(["", "## Visual Results", ""])
    ordered_images = sorted(by_image.values(), key=lambda group: group[0]["source_index"])
    for image_details in ordered_images:
        image_details.sort(key=lambda detail: TARGET_RATIOS.index(detail["target_ratio"]))
        first = image_details[0]
        safe_title = str(first["title"]).replace("|", "\\|").replace("\n", " ")
        lines.extend(
            [
                f"### {safe_title}",
                "",
                f"![Original]({Path(first['original_render_path']).as_posix()})",
                "",
                "| Ratio | Final Layout | Mode | Operation |",
                "|---:|---|---|---|",
            ]
        )
        for detail in image_details:
            candidate = (
                f"![Ratio {detail['target_ratio']:g}]({Path(detail['candidate_path']).as_posix()})"
                if detail["candidate_path"]
                else "Not rendered"
            )
            operation = (
                f"background `{detail['background_hex']}`, padding {_format(detail['padding_fraction'])}"
                if detail["predicted_mode"] == "pad"
                else f"center ({detail['cx_pct']}, {detail['cy_pct']}), area {detail['area_pct']}%"
            )
            lines.append(
                f"| {detail['target_ratio']:g} | {candidate} | `{detail['predicted_mode'] or 'invalid'}` | {operation} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Artifacts",
            "",
            "- `details.jsonl` and `details.parquet`: mode, crop or pad parameters, and render metadata",
            "- `review_template.csv`: blank human mode and layout review fields",
            "- `summary.json` and `summary.csv`: descriptive pipeline statistics",
            "- `renders/candidates/`: final crop-or-pad layouts",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a mode-first crop-or-edge-pad image layout pipeline.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-family", choices=MODEL_FAMILIES, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mode-prompt-path", type=Path, required=True)
    parser.add_argument("--crop-prompt-path", type=Path, required=True)
    parser.add_argument("--mode-results-dir", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode-max-attempts", type=int, default=3)
    parser.add_argument("--crop-max-attempts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode-temperature", type=float, default=0.0)
    parser.add_argument("--mode-top-p", type=float, default=1.0)
    parser.add_argument("--crop-temperature", type=float, default=0.7)
    parser.add_argument("--crop-top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=2176)
    parser.add_argument("--mode-max-tokens", type=int, default=32)
    parser.add_argument("--crop-max-tokens", type=int, default=128)
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    parser.add_argument("--image-min-pixels", type=int, default=65536)
    parser.add_argument("--internvl-max-dynamic-patch", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--prompt-batch-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--edge-fraction", type=float, default=0.05)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in (
        "mode_max_attempts",
        "crop_max_attempts",
        "data_parallel_size",
        "tensor_parallel_size",
        "prompt_batch_size",
        "max_num_seqs",
        "internvl_max_dynamic_patch",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("max-images must be positive")
    if not 0.0 < args.edge_fraction <= 0.5:
        raise ValueError("edge-fraction must be in (0, 0.5]")
    for name in ("mode_top_p", "crop_top_p"):
        if not 0.0 < getattr(args, name) <= 1.0:
            raise ValueError(f"{name.replace('_', '-')} must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    for path in (args.model, args.data, args.mode_prompt_path, args.crop_prompt_path):
        if not path.exists():
            raise FileNotFoundError(path)
    if args.mode_results_dir is not None and not args.mode_results_dir.is_dir():
        raise FileNotFoundError(args.mode_results_dir)
    model_config_path = args.model / "config.json"
    if not model_config_path.is_file():
        raise FileNotFoundError(model_config_path)
    args.output_dir = args.output_dir.resolve()
    mode_prompt_template = load_policy_prompt_template(args.mode_prompt_path)
    crop_prompt_template = load_policy_prompt_template(args.crop_prompt_path)
    external_mode_details = args.mode_results_dir / "details.jsonl" if args.mode_results_dir else None
    config = {
        "run_id": args.run_id,
        "model_name": args.model_name,
        "model_family": args.model_family,
        "pipeline_version": "mode-crop-or-edge-pad-v1",
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256_file(model_config_path),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "mode_prompt_path": str(args.mode_prompt_path.resolve()),
        "mode_prompt_sha256": hashlib.sha256(mode_prompt_template.encode("utf-8")).hexdigest(),
        "crop_prompt_path": str(args.crop_prompt_path.resolve()),
        "crop_prompt_sha256": hashlib.sha256(crop_prompt_template.encode("utf-8")).hexdigest(),
        "mode_results_dir": str(args.mode_results_dir.resolve()) if args.mode_results_dir else None,
        "mode_details_sha256": sha256_file(external_mode_details) if external_mode_details else None,
        "target_ratios": list(TARGET_RATIOS),
        "mode_max_attempts": args.mode_max_attempts,
        "crop_max_attempts": args.crop_max_attempts,
        "seed": args.seed,
        "mode_temperature": args.mode_temperature,
        "mode_top_p": args.mode_top_p,
        "crop_temperature": args.crop_temperature,
        "crop_top_p": args.crop_top_p,
        "max_model_len": args.max_model_len,
        "mode_max_tokens": args.mode_max_tokens,
        "crop_max_tokens": args.crop_max_tokens,
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
        "internvl_max_dynamic_patch": args.internvl_max_dynamic_patch,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "data_parallel_size": args.data_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "prompt_batch_size": args.prompt_batch_size,
        "max_num_seqs": args.max_num_seqs,
        "edge_fraction": args.edge_fraction,
        "max_images": args.max_images,
    }
    prepare_output_directory(args.output_dir, config, args.resume)
    tasks, manifest = load_and_materialize_tasks(
        args.data,
        args.output_dir,
        crop_prompt_template,
        max_images=args.max_images,
    )
    (args.output_dir / "source_manifest.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in manifest),
        encoding="utf-8",
    )
    (args.output_dir / "task_manifest.jsonl").write_text(
        "".join(
            json.dumps(
                {
                    "task_id": task["task_id"],
                    "image_id": task["image_id"],
                    "target_ratio": task["target_ratio"],
                    "crop_prompt": task["crop_prompt"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for task in tasks
        ),
        encoding="utf-8",
    )
    if args.mode_results_dir is None:
        mode_results_dir = args.output_dir / "stages" / "mode"
        run_mode_stage(args, mode_results_dir)
    else:
        mode_results_dir = args.mode_results_dir.resolve()
    mode_results = load_mode_results(mode_results_dir, tasks)
    crop_tasks = [task for task in tasks if mode_results[task["task_id"]].get("predicted_mode") == "crop"]
    run_parallel_crop_generation(crop_tasks, args.output_dir, args.model, args)
    render_results = render_layouts(tasks, mode_results, args.output_dir, args.edge_fraction)
    details = build_details(tasks, mode_results, render_results, args.output_dir)
    summary = {
        "run_id": args.run_id,
        "model_name": args.model_name,
        "model_family": args.model_family,
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "images": len(manifest),
        "target_ratios": list(TARGET_RATIOS),
        "mode_results_dir": str(mode_results_dir),
        "edge_fraction": args.edge_fraction,
        "quality_metrics_available": False,
        **summarize(details),
    }
    write_results(details, summary, args.output_dir)
    render_html_report(details, summary, args.output_dir)
    render_markdown_report(details, summary, args.output_dir)
    write_json_atomic(
        args.output_dir / "_LAYOUT_PIPELINE_COMPLETE.json",
        {
            "run_id": args.run_id,
            "images": len(manifest),
            "tasks": len(tasks),
            "crop_count": summary["overall"]["crop_count"],
            "pad_count": summary["overall"]["pad_count"],
            "rendered_count": summary["overall"]["rendered_count"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()