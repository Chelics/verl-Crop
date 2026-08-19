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
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
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
from news_crop_benchmark.layout import render_layout_action
from news_crop_benchmark.policy_model_adapter import (
    MODEL_FAMILIES,
    PolicyModelAdapter,
)
from news_crop_benchmark.protocol import (
    LayoutAction,
    parse_crop_action_with_format,
    parse_layout_action,
    parse_percent_crop_action,
)
from news_crop_benchmark.proxy_scorer import crop_image
from news_crop_benchmark.vlm_scorer import CropVLMScorer

SOURCE_COLUMNS = ("image_id", "original_image", "title", "ImageCaption")
FINAL_GENERATION_STATUSES = {"valid", "retry_exhausted"}
COUNTED_JUDGE_STATUS = "completed"


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


def save_report_image(image: Image.Image, path: Path, maximum_side: int | None = None, quality: int = 92) -> None:
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
    target_ratios: Sequence[float] = TARGET_RATIOS,
    policy_prompt_template: str | None = None,
    max_images: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    parquet = pq.ParquetFile(data_path)
    available_columns = set(parquet.schema_arrow.names)
    missing_columns = sorted(set(SOURCE_COLUMNS) - available_columns)
    if missing_columns:
        raise ValueError(f"dataset is missing required columns: {missing_columns}")

    table = pq.read_table(data_path, columns=list(SOURCE_COLUMNS))
    if any(table[name].null_count for name in SOURCE_COLUMNS):
        null_counts = {name: table[name].null_count for name in SOURCE_COLUMNS if table[name].null_count}
        raise ValueError(f"dataset contains null required fields: {null_counts}")

    source_dir = output_dir / ".source_images"
    preview_dir = output_dir / "renders" / "originals"
    tasks: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    seen_image_ids: set[str] = set()
    source_rows = table.to_pylist()
    if max_images is not None:
        source_rows = source_rows[:max_images]
    for source_index, row in enumerate(source_rows):
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
            save_report_image(original, preview_path, maximum_side=1200, quality=90)
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
        for target_ratio in target_ratios:
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
                    "prompt": build_prompt(title, float(target_ratio), policy_prompt_template),
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
        devices = []
        for device in visible_devices.split(","):
            device = device.strip()
            if device == "-1":
                break
            if device:
                devices.append(device)
    if len(devices) < required_gpus:
        raise ValueError(
            f"requested {required_gpus} GPUs, but CUDA_VISIBLE_DEVICES exposes only {len(devices)}: {devices}"
        )
    return devices[:required_gpus]


def generation_progress_path(output_dir: Path, current_task_id: str) -> Path:
    return output_dir / "progress" / "generation" / f"{current_task_id}.json"


def judge_progress_path(output_dir: Path, current_task_id: str) -> Path:
    return output_dir / "progress" / "judge" / f"{current_task_id}.json"


def build_attempt_prompt(
    base_prompt: str,
    previous_attempts: Sequence[dict[str, Any]],
    action_protocol: str = "legacy-crop-json",
) -> str:
    if not previous_attempts:
        return base_prompt
    previous = previous_attempts[-1]
    error = str(previous.get("parse_error") or "output did not satisfy the required crop protocol")
    if action_protocol == "layout-json-v1":
        requirements = (
            "Generate a new answer containing only one JSON object. Recheck that operation is crop, "
            "crop_pad, or pad; all four percentage coordinates are integers in [0, 100]; lower bounds "
            "are smaller than upper bounds; pad uses exactly [0, 0, 100, 100]; and there are no other fields."
        )
    elif action_protocol == "percent-json-v1":
        requirements = (
            "Generate a new answer containing only one JSON object. Recheck that cx_pct and cy_pct "
            "are integers in [0, 100], area_pct is an integer in [1, 100], and there are no other fields."
        )
    else:
        requirements = (
            "Generate a new answer. Recheck that cx and cy are normalized integers in [0, 1000], "
            "area is a normalized integer in [1, 1000], and output contains exactly one complete <crop> line."
        )
    return f"{base_prompt}\n\nYour previous output was rejected by the validator: {error}\n{requirements}"


def _load_generation_state(
    output_dir: Path,
    tasks: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    active: dict[str, dict[str, Any]] = {}
    attempts: dict[str, list[dict[str, Any]]] = {}
    for task in tasks:
        path = generation_progress_path(output_dir, task["task_id"])
        if path.exists():
            progress = json.loads(path.read_text(encoding="utf-8"))
            if progress["status"] in FINAL_GENERATION_STATUSES:
                continue
            attempts[task["task_id"]] = list(progress["attempts"])
        else:
            attempts[task["task_id"]] = []
        active[task["task_id"]] = task
    return active, attempts


def run_generation_worker(
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

    active, attempts = _load_generation_state(output_dir, tasks)
    if not active:
        return

    adapter = PolicyModelAdapter.create(args.model_family)
    renderer = adapter.load_renderer(model_path)
    lora_adapter_path = getattr(args, "lora_adapter_path", None)
    lora_rank = getattr(args, "lora_rank", 32)
    lora_request = None
    lora_kwargs = {}
    if lora_adapter_path is not None:
        from vllm.lora.request import LoRARequest

        lora_kwargs = {"enable_lora": True, "max_lora_rank": lora_rank}
        lora_request = LoRARequest("news-crop-sft", 1, str(lora_adapter_path))
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=args.tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
        **lora_kwargs,
        **adapter.llm_kwargs(internvl_max_dynamic_patch=args.internvl_max_dynamic_patch),
    )
    write_json_atomic(
        output_dir / "runtime" / f"model_rank_{rank}.json",
        {
            "rank": rank,
            "gpu_devices": gpu_devices,
            "lora_adapter_path": str(lora_adapter_path) if lora_adapter_path is not None else None,
            "lora_rank": lora_rank if lora_adapter_path is not None else None,
            **adapter.runtime_metadata(renderer),
            "vllm_version": package_version("vllm"),
            "transformers_version": package_version("transformers"),
            "sentencepiece_version": package_version("sentencepiece"),
        },
    )

    for attempt_number in range(1, args.max_attempts + 1):
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
                            build_attempt_prompt(
                                task["prompt"],
                                attempts[task["task_id"]],
                                args.action_protocol,
                            ),
                            image,
                            image_max_pixels=args.image_max_pixels,
                            image_min_pixels=args.image_min_pixels,
                        )
                    )

                attempt_seed = args.seed + attempt_number - 1
                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    n=1,
                    max_tokens=args.max_tokens,
                    seed=attempt_seed,
                    **adapter.sampling_kwargs(renderer),
                )
                if lora_request is None:
                    outputs = llm.generate(requests, sampling_params=sampling_params)
                else:
                    outputs = llm.generate(
                        requests,
                        sampling_params=sampling_params,
                        lora_request=lora_request,
                    )
                for task, request_output in zip(task_batch, outputs, strict=True):
                    current_task_id = task["task_id"]
                    response = request_output.outputs[0].text if request_output.outputs else ""
                    if args.canonicalize_bare_json:
                        canonical_response, response_normalized = adapter.canonicalize_response(response)
                    else:
                        canonical_response, response_normalized = response, False
                    attempt_record: dict[str, Any] = {
                        "attempt": attempt_number,
                        "seed": attempt_seed,
                        "response": response,
                        "canonical_response": canonical_response,
                        "response_normalized": response_normalized,
                        "action_protocol": args.action_protocol,
                        "valid": False,
                        "strict_format": False,
                        "canonical_format": False,
                        "parse_error": None,
                    }
                    try:
                        if args.action_protocol == "layout-json-v1":
                            parse_result = parse_layout_action(response)
                        elif args.action_protocol == "percent-json-v1":
                            parse_result = parse_percent_crop_action(response)
                        else:
                            parse_result = parse_crop_action_with_format(canonical_response)
                    except ValueError as error:
                        attempt_record["parse_error"] = str(error)
                        attempts[current_task_id].append(attempt_record)
                        status = "retry_exhausted" if attempt_number == args.max_attempts else "retrying"
                        write_json_atomic(
                            generation_progress_path(output_dir, current_task_id),
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
                    attempt_record["strict_format"] = parse_result.strict_format and not response_normalized
                    attempt_record["canonical_format"] = (
                        True
                        if args.action_protocol in {"percent-json-v1", "layout-json-v1"}
                        else parse_result.strict_format
                    )
                    if args.action_protocol in {"percent-json-v1", "layout-json-v1"} and not parse_result.strict_format:
                        attempt_record["response_normalized"] = True
                    attempts[current_task_id].append(attempt_record)
                    if args.action_protocol == "layout-json-v1":
                        serialized_action = {
                            "operation": action.operation,
                            "x1_pct": action.x1_pct,
                            "y1_pct": action.y1_pct,
                            "x2_pct": action.x2_pct,
                            "y2_pct": action.y2_pct,
                        }
                    else:
                        serialized_action = {
                            "cx": action.center_x,
                            "cy": action.center_y,
                            "area": action.area,
                        }
                    write_json_atomic(
                        generation_progress_path(output_dir, current_task_id),
                        {
                            "task_id": current_task_id,
                            "rank": rank,
                            "status": "valid",
                            "attempts": attempts[current_task_id],
                            "action": serialized_action,
                        },
                    )
                    active.pop(current_task_id)
            finally:
                for image in images:
                    image.close()
    if active:
        raise RuntimeError(f"generation worker left {len(active)} tasks unfinished")


def run_parallel_generation(
    tasks: list[dict[str, Any]],
    output_dir: Path,
    model_path: Path,
    args: argparse.Namespace,
) -> None:
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
            target=run_generation_worker,
            args=(
                rank,
                devices[start : start + args.tensor_parallel_size],
                rank_tasks,
                output_dir,
                model_path,
                args,
            ),
            name=f"policy-model-rank-{rank}",
        )
        process.start()
        processes.append(process)

    failed = []
    for process in processes:
        process.join()
        if process.exitcode != 0:
            failed.append(f"{process.name}={process.exitcode}")
    if failed:
        raise RuntimeError(f"vLLM generation workers failed: {', '.join(failed)}")


def parse_judge_metadata(output_text: str | None) -> dict[str, Any]:
    if not output_text:
        return {"rules": [], "confidence_score": None, "tier_name": None}
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", output_text):
        try:
            payload, _ = decoder.raw_decode(output_text[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("evaluation"), dict):
            continue
        evaluation = payload["evaluation"]
        rules = evaluation.get("rules", [])
        return {
            "rules": [str(rule) for rule in rules] if isinstance(rules, list) else [],
            "confidence_score": evaluation.get("confidence_score"),
            "tier_name": evaluation.get("tier_name"),
        }
    return {"rules": [], "confidence_score": None, "tier_name": None}


def render_candidate(task: dict[str, Any], action: dict[str, float], output_dir: Path) -> Path:
    output_path = output_dir / "renders" / "candidates" / f"{task['task_id']}.jpg"
    if output_path.exists():
        return output_path
    crop_action = CropAction(center_x=action["cx"], center_y=action["cy"], area=action["area"])
    bbox = action_to_bbox(
        crop_action,
        image_width=task["image_width"],
        image_height=task["image_height"],
        target_ratio=task["target_ratio"],
    )
    with Image.open(task["image_path"]) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    candidate = crop_image(original, bbox)
    save_report_image(candidate, output_path, quality=95)
    candidate.close()
    original.close()
    return output_path


def render_unified_layout_candidate(
    task: dict[str, Any],
    action: dict[str, Any],
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    output_path = output_dir / "renders" / "candidates" / f"{task['task_id']}.jpg"
    layout_action = LayoutAction(
        operation=str(action["operation"]),
        x1_pct=int(action["x1_pct"]),
        y1_pct=int(action["y1_pct"]),
        x2_pct=int(action["x2_pct"]),
        y2_pct=int(action["y2_pct"]),
    )
    with Image.open(task["image_path"]) as source:
        original = ImageOps.exif_transpose(source).convert("RGB")
    try:
        rendered = render_layout_action(original, layout_action, task["target_ratio"])
        try:
            if not output_path.exists():
                save_report_image(rendered.image, output_path, quality=95)
            metadata = {
                "selected_operation": rendered.operation,
                "source_box": list(rendered.source_box),
                "content_box": list(rendered.content_box),
                "background_color": list(rendered.background_color) if rendered.background_color else None,
                "background_hex": (
                    "#" + "".join(f"{channel:02X}" for channel in rendered.background_color)
                    if rendered.background_color
                    else None
                ),
                "padding_fraction": rendered.padding_fraction,
                "render_width": rendered.image.width,
                "render_height": rendered.image.height,
            }
        finally:
            rendered.image.close()
    finally:
        original.close()
    return output_path, metadata


def run_judge(
    tasks: list[dict[str, Any]],
    output_dir: Path,
    prompt_path: Path,
    judge_workers: int,
) -> None:
    os.environ.setdefault("CROP_VLM_LOG_PATH", str(output_dir / "judge_responses.jsonl"))
    thread_state = threading.local()

    def score_task(task: dict[str, Any]) -> None:
        judge_path = judge_progress_path(output_dir, task["task_id"])
        if judge_path.exists():
            try:
                existing = json.loads(judge_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if existing.get("status") in {"completed", "not_run"}:
                return
        generation = json.loads(
            generation_progress_path(output_dir, task["task_id"]).read_text(encoding="utf-8")
        )
        if generation["status"] != "valid":
            write_json_atomic(
                judge_path,
                {
                    "task_id": task["task_id"],
                    "status": "not_run",
                    "reason": "generation_retry_exhausted",
                },
            )
            return

        if "operation" in generation["action"]:
            candidate_path, render_metadata = render_unified_layout_candidate(
                task, generation["action"], output_dir
            )
        else:
            candidate_path = render_candidate(task, generation["action"], output_dir)
            render_metadata = {}
        with Image.open(task["image_path"]) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(candidate_path) as source:
            candidate = source.convert("RGB")
        if not hasattr(thread_state, "scorer"):
            thread_state.scorer = CropVLMScorer(str(prompt_path))
        score_kwargs = {
            "log_context": {
                "task_id": task["task_id"],
                "sample_id": task["image_id"],
                "target_ratio": task["target_ratio"],
                "action": generation["action"],
            }
        }
        if render_metadata:
            score_kwargs["evaluation_context"] = {
                "requested_aspect_ratio": task["target_ratio"],
                **render_metadata,
            }
        try:
            result = thread_state.scorer.score_detailed(
                original,
                candidate,
                task["caption"],
                task["title"],
                **score_kwargs,
            )
        finally:
            original.close()
            candidate.close()
        metadata = parse_judge_metadata(result.output_text)
        write_json_atomic(
            judge_path,
            {
                "task_id": task["task_id"],
                "status": result.status,
                "label": result.label,
                "reward": result.reward,
                "output_text": result.output_text,
                "response_id": result.response_id,
                "request_attempt_count": result.attempt_count,
                "latency_ms": result.latency_ms,
                "error_type": result.error_type,
                "candidate_path": candidate_path.relative_to(output_dir).as_posix(),
                "render_metadata": render_metadata,
                **metadata,
            },
        )

    with ThreadPoolExecutor(max_workers=judge_workers) as executor:
        list(executor.map(score_task, tasks))


def build_details(tasks: Sequence[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    details = []
    for task in tasks:
        generation = json.loads(
            generation_progress_path(output_dir, task["task_id"]).read_text(encoding="utf-8")
        )
        judge = json.loads(judge_progress_path(output_dir, task["task_id"]).read_text(encoding="utf-8"))
        attempts = generation["attempts"]
        invalid_attempt_count = sum(not attempt["valid"] for attempt in attempts)
        valid_attempt = next((attempt for attempt in attempts if attempt["valid"]), None)
        action = generation.get("action") or {}
        details.append(
            {
                "task_id": task["task_id"],
                "source_index": task["source_index"],
                "image_id": task["image_id"],
                "title": task["title"],
                "caption": task["caption"],
                "target_ratio": task["target_ratio"],
                "image_width": task["image_width"],
                "image_height": task["image_height"],
                "original_render_path": task["original_render_path"],
                "candidate_path": judge.get("candidate_path"),
                "generation_status": generation["status"],
                "had_invalid_output": invalid_attempt_count > 0,
                "invalid_attempt_count": invalid_attempt_count,
                "total_attempt_count": len(attempts),
                "strict_format": valid_attempt["strict_format"] if valid_attempt else False,
                "canonical_format": (
                    valid_attempt.get("canonical_format", valid_attempt["strict_format"])
                    if valid_attempt
                    else False
                ),
                "response_normalized": (
                    bool(valid_attempt.get("response_normalized", False)) if valid_attempt else False
                ),
                "final_response": valid_attempt["response"] if valid_attempt else None,
                "action_cx": action.get("cx"),
                "action_cy": action.get("cy"),
                "action_area": action.get("area"),
                "layout_operation": action.get("operation"),
                "layout_x1_pct": action.get("x1_pct"),
                "layout_y1_pct": action.get("y1_pct"),
                "layout_x2_pct": action.get("x2_pct"),
                "layout_y2_pct": action.get("y2_pct"),
                "render_metadata": judge.get("render_metadata", {}),
                "judge_status": judge["status"],
                "judge_label": judge.get("label"),
                "judge_reward": judge.get("reward"),
                "judge_rules": judge.get("rules", []),
                "judge_tier_name": judge.get("tier_name"),
                "judge_confidence_score": judge.get("confidence_score"),
                "judge_latency_ms": judge.get("latency_ms"),
                "judge_error_type": judge.get("error_type"),
                "judge_output_text": judge.get("output_text"),
            }
        )
    return details


def percentile(values: Sequence[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_subset(details: Sequence[dict[str, Any]]) -> dict[str, Any]:
    generated = [detail for detail in details if detail["generation_status"] == "valid"]
    counted = [detail for detail in details if detail["judge_status"] == COUNTED_JUDGE_STATUS]
    labels = [float(detail["judge_label"]) for detail in counted]
    rewards = [float(detail["judge_reward"]) for detail in counted]
    latencies = [float(detail["judge_latency_ms"]) for detail in counted]
    tier_counts = Counter(str(int(label)) if label.is_integer() else str(label) for label in labels)
    rule_counts = Counter(rule for detail in counted for rule in detail["judge_rules"])
    total = len(details)
    action_cx = [detail["action_cx"] for detail in generated if detail["action_cx"] is not None]
    action_cy = [detail["action_cy"] for detail in generated if detail["action_cy"] is not None]
    action_area = [detail["action_area"] for detail in generated if detail["action_area"] is not None]
    operation_counts = Counter(
        detail.get("layout_operation") for detail in generated if detail.get("layout_operation") is not None
    )
    return {
        "tasks": total,
        "generation_success_count": len(generated),
        "generation_success_rate": len(generated) / total if total else 0.0,
        "first_attempt_valid_count": sum(
            detail["generation_status"] == "valid" and detail["total_attempt_count"] == 1
            for detail in details
        ),
        "had_invalid_output_count": sum(detail["had_invalid_output"] for detail in details),
        "had_invalid_output_rate": mean(detail["had_invalid_output"] for detail in details) if details else 0.0,
        "invalid_output_count": sum(detail["invalid_attempt_count"] for detail in details),
        "mean_attempt_count": mean(detail["total_attempt_count"] for detail in details) if details else 0.0,
        "retry_recovered_count": sum(
            detail["generation_status"] == "valid" and detail["had_invalid_output"] for detail in details
        ),
        "retry_exhausted_count": sum(detail["generation_status"] == "retry_exhausted" for detail in details),
        "strict_format_count": sum(detail["strict_format"] for detail in generated),
        "strict_format_rate": mean(detail["strict_format"] for detail in generated) if generated else 0.0,
        "canonical_format_count": sum(detail.get("canonical_format", False) for detail in generated),
        "canonical_format_rate": (
            mean(detail.get("canonical_format", False) for detail in generated) if generated else 0.0
        ),
        "response_normalized_count": sum(
            detail.get("response_normalized", False) for detail in generated
        ),
        "response_normalized_rate": (
            mean(detail.get("response_normalized", False) for detail in generated) if generated else 0.0
        ),
        "judge_completed_count": len(counted),
        "judge_parse_fallback_count": sum(detail["judge_status"] == "parse_fallback" for detail in details),
        "judge_failed_count": sum(detail["judge_status"] == "failed" for detail in details),
        "judge_not_run_count": sum(detail["judge_status"] == "not_run" for detail in details),
        "judge_completed_rate": len(counted) / len(generated) if generated else 0.0,
        "tier_counts": dict(sorted(tier_counts.items())),
        "rule_counts": dict(sorted(rule_counts.items())),
        "mean_judge_label": mean(labels) if labels else None,
        "mean_judge_reward": mean(rewards) if rewards else None,
        "tier_0_1_acceptable_rate": mean(label <= 1 for label in labels) if labels else None,
        "tier_3_5_severe_rate": mean(label >= 3 for label in labels) if labels else None,
        "mean_action_cx": mean(action_cx) if action_cx else None,
        "mean_action_cy": mean(action_cy) if action_cy else None,
        "mean_action_area": mean(action_area) if action_area else None,
        "operation_counts": dict(sorted(operation_counts.items())),
        "judge_latency_ms_mean": mean(latencies) if latencies else None,
        "judge_latency_ms_p50": percentile(latencies, 0.50),
        "judge_latency_ms_p95": percentile(latencies, 0.95),
    }


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": summarize_subset(details),
        "by_ratio": {
            f"{ratio:g}": summarize_subset([detail for detail in details if detail["target_ratio"] == ratio])
            for ratio in TARGET_RATIOS
        },
    }


def write_result_tables(details: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    details.sort(key=lambda detail: (detail["source_index"], TARGET_RATIOS.index(detail["target_ratio"])))
    (output_dir / "details.jsonl").write_text(
        "".join(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n" for detail in details),
        encoding="utf-8",
    )
    parquet_rows = [
        {
            **detail,
            "judge_rules": json.dumps(detail["judge_rules"], ensure_ascii=False),
            "render_metadata": json.dumps(detail.get("render_metadata", {}), sort_keys=True),
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
        scalar_keys = [
            key for key, value in summary["overall"].items() if not isinstance(value, dict)
        ]
        writer = csv.DictWriter(
            output,
            fieldnames=["scope", *scalar_keys, "tier_counts", "rule_counts"],
        )
        writer.writeheader()
        for scope, metrics in rows:
            writer.writerow(
                {
                    "scope": scope,
                    **{key: metrics[key] for key in scalar_keys},
                    "tier_counts": json.dumps(metrics["tier_counts"], sort_keys=True),
                    "rule_counts": json.dumps(metrics["rule_counts"], sort_keys=True),
                }
            )


def render_html_report(details: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    layout_evaluation = any(detail.get("layout_operation") is not None for detail in details)
    evaluation_name = "Unified Layout Evaluation" if layout_evaluation else "Four-Ratio Crop Evaluation"
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        by_image[detail["image_id"]].append(detail)
    metric_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary["overall"].items()
    )
    sections = []
    for image_id, image_details in by_image.items():
        image_details.sort(key=lambda detail: TARGET_RATIOS.index(detail["target_ratio"]))
        first = image_details[0]
        cards = []
        for detail in image_details:
            candidate_html = (
                f'<img src="{html.escape(detail["candidate_path"])}" alt="Policy result">'
                if detail["candidate_path"]
                else '<div class="missing">No valid result</div>'
            )
            cards.append(
                f"""
                <article class="candidate">
                  <h3>Ratio {detail['target_ratio']:g}</h3>
                  {candidate_html}
                  <p><strong>Generation:</strong> {html.escape(detail['generation_status'])}</p>
                  <p><strong>Attempts:</strong> {detail['total_attempt_count']}
                     (invalid: {detail['invalid_attempt_count']})</p>
                  <p><strong>Judge:</strong> {html.escape(detail['judge_status'])}</p>
                  <p><strong>Tier:</strong> {html.escape(str(detail['judge_label']))}</p>
                  <p><strong>Rules:</strong> {html.escape(', '.join(detail['judge_rules']))}</p>
                  <details><summary>Responses and scoring data</summary><pre>{html.escape(json.dumps({
                      'policy_response': detail['final_response'],
                      'action': (
                          {
                              'operation': detail['layout_operation'],
                              'x1_pct': detail['layout_x1_pct'],
                              'y1_pct': detail['layout_y1_pct'],
                              'x2_pct': detail['layout_x2_pct'],
                              'y2_pct': detail['layout_y2_pct'],
                          }
                          if detail['layout_operation'] is not None
                          else {
                              'cx': detail['action_cx'],
                              'cy': detail['action_cy'],
                              'area': detail['action_area'],
                          }
                      ),
                      'render_metadata': detail['render_metadata'],
                      'judge_output': detail['judge_output_text'],
                  }, ensure_ascii=False, indent=2))}</pre></details>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="sample">
              <header><h2>{html.escape(first['title'])}</h2><p>{html.escape(image_id)}</p></header>
              <div class="layout">
                <article class="original"><h3>Original</h3>
                  <img src="{html.escape(first['original_render_path'])}" alt="Original">
                  <p>{html.escape(first['caption'])}</p>
                </article>
                <div class="candidates">{''.join(cards)}</div>
              </div>
            </section>
            """
        )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(summary['model_name'])} {evaluation_name}</title>
<style>
:root {{ --ink:#1d2522; --muted:#63706a; --line:#d3dad6; --paper:#f1f4f2; --accent:#006b54; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper); font-family:"Segoe UI",sans-serif; }}
main {{ width:min(1600px,96vw); margin:auto; padding:24px 0 64px; }} .summary,.sample {{ background:white; border:1px solid var(--line); padding:20px; margin-bottom:20px; }}
table {{ border-collapse:collapse; width:min(900px,100%); }} th,td {{ border-bottom:1px solid var(--line); padding:7px 10px; text-align:left; }}
.layout {{ display:grid; grid-template-columns:minmax(260px,0.8fr) minmax(0,2.2fr); gap:18px; }} .candidates {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.candidate {{ border:1px solid var(--line); padding:12px; }} img {{ display:block; width:100%; max-height:520px; object-fit:contain; background:#e9eeeb; }}
.candidate img {{ height:320px; }} p {{ overflow-wrap:anywhere; }} pre {{ white-space:pre-wrap; overflow-wrap:anywhere; font-size:12px; }} .missing {{ min-height:180px; display:grid; place-items:center; background:#eee; color:var(--muted); }}
@media (max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} .candidates {{ grid-template-columns:1fr; }} main {{ width:100%; padding:10px; }} }}
</style></head><body><main>
<section class="summary"><h1>{html.escape(summary['model_name'])} {evaluation_name}</h1><table>{metric_rows}</table></section>
{''.join(sections)}
</main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def _format_metric(value: Any) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def _escape_markdown_text(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def _markdown_image(alt: str, path: str | None) -> str:
    if not path:
        return "No valid crop"
    return f"![{alt}]({Path(path).as_posix()})"


def render_markdown_report(details: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    overall = summary["overall"]
    layout_evaluation = any(detail.get("layout_operation") is not None for detail in details)
    evaluation_name = "Unified Layout Evaluation" if layout_evaluation else "Four-Ratio Crop Evaluation"
    lines = [
        f"# {summary['model_name']} {evaluation_name}",
        "",
        "## Run",
        "",
        f"- Run ID: `{summary.get('run_id', 'N/A')}`",
        f"- Model family: `{summary.get('model_family', 'N/A')}`",
        f"- Model: `{summary.get('model', 'N/A')}`",
        f"- Data: `{summary.get('data', 'N/A')}`",
        f"- Images: {summary.get('images', 'N/A')}",
        f"- Tasks: {overall['tasks']}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    metric_keys = (
        "generation_success_count",
        "generation_success_rate",
        "first_attempt_valid_count",
        "had_invalid_output_count",
        "invalid_output_count",
        "retry_recovered_count",
        "retry_exhausted_count",
        "strict_format_rate",
        "canonical_format_rate",
        "judge_completed_count",
        "judge_completed_rate",
        "judge_failed_count",
        "judge_parse_fallback_count",
        "mean_judge_label",
        "mean_judge_reward",
        "tier_0_1_acceptable_rate",
        "tier_3_5_severe_rate",
        "mean_action_cx",
        "mean_action_cy",
        "mean_action_area",
        "judge_latency_ms_p50",
        "judge_latency_ms_p95",
    )
    lines.extend(f"| `{key}` | {_format_metric(overall.get(key))} |" for key in metric_keys)
    if layout_evaluation:
        lines.append(f"| `operation_counts` | `{json.dumps(overall['operation_counts'], sort_keys=True)}` |")
    lines.extend(
        [
            "",
            "## By Ratio",
            "",
            "| Ratio | Tasks | Acceptable Tier 0-1 | Severe Tier 3-5 | Mean Label | Mean Reward | Mean Area |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ratio in TARGET_RATIOS:
        metrics = summary["by_ratio"][f"{ratio:g}"]
        lines.append(
            f"| {ratio:g} | {metrics['tasks']} | {_format_metric(metrics['tier_0_1_acceptable_rate'])} | "
            f"{_format_metric(metrics['tier_3_5_severe_rate'])} | "
            f"{_format_metric(metrics['mean_judge_label'])} | "
            f"{_format_metric(metrics['mean_judge_reward'])} | "
            f"{_format_metric(metrics['mean_action_area'])} |"
        )

    lines.extend(["", "## Tier Distribution", "", "| Tier | Count |", "|---:|---:|"])
    lines.extend(f"| {tier} | {count} |" for tier, count in overall["tier_counts"].items())

    lines.extend(["", "## Most Frequent Judge Rules", "", "| Rule | Count |", "|---|---:|"])
    rule_counts = sorted(overall["rule_counts"].items(), key=lambda item: (-item[1], item[0]))
    lines.extend(f"| `{rule}` | {count} |" for rule, count in rule_counts)

    retried = [detail for detail in details if detail["invalid_attempt_count"]]
    lines.extend(["", "## Generation Retries", ""])
    if retried:
        lines.extend(
            [
                "| Task | Ratio | Invalid Attempts | Final Tier | Title |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for detail in retried:
            title = str(detail["title"]).replace("|", "\\|")
            lines.append(
                f"| `{detail['task_id']}` | {detail['target_ratio']:g} | "
                f"{detail['invalid_attempt_count']} | {_format_metric(detail['judge_label'])} | {title} |"
            )
    else:
        lines.append("No task required a generation retry.")

    severe = sorted(
        (detail for detail in details if detail["judge_label"] is not None and detail["judge_label"] >= 4),
        key=lambda detail: (-detail["judge_label"], detail["source_index"], detail["target_ratio"]),
    )
    lines.extend(
        [
            "",
            "## Severe Examples",
            "",
            "The table lists up to 30 Tier 4-5 tasks. Use the recorded relative render path to retrieve only a sample that needs visual review.",
            "",
            "| Task | Tier | Ratio | Rules | Candidate | Title |",
            "|---|---:|---:|---|---|---|",
        ]
    )
    for detail in severe[:30]:
        title = str(detail["title"]).replace("|", "\\|")
        rules = ", ".join(detail["judge_rules"])
        lines.append(
            f"| `{detail['task_id']}` | {_format_metric(detail['judge_label'])} | "
            f"{detail['target_ratio']:g} | {rules} | `{detail['candidate_path']}` | {title} |"
        )

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        by_image[detail["image_id"]].append(detail)
    lines.extend(
        [
            "",
            "## Visual Results",
            "",
            "Image links are relative to this report. Open `report.md` with JupyterLab or a remote VS Code Markdown Preview from the result directory so `renders/` is reachable.",
            "",
        ]
    )
    ordered_images = sorted(by_image.values(), key=lambda group: group[0]["source_index"])
    for image_details in ordered_images:
        image_details.sort(key=lambda detail: TARGET_RATIOS.index(detail["target_ratio"]))
        first = image_details[0]
        lines.extend(
            [
                f"### {_escape_markdown_text(first['title'])}",
                "",
                f"`image_id: {first['image_id']}`",
                "",
                f"**Original**  ",
                _markdown_image("Original", first["original_render_path"]),
                "",
                "| Ratio | Candidate | Tier | Rules | Action | Attempts |",
                "|---:|---|---:|---|---|---:|",
            ]
        )
        for detail in image_details:
            rules = _escape_markdown_text(", ".join(detail["judge_rules"]))
            candidate_alt = f"Ratio {detail['target_ratio']:g}"
            if detail.get("layout_operation") is not None:
                action = (
                    f"{detail['layout_operation']} "
                    f"({detail['layout_x1_pct']}, {detail['layout_y1_pct']}, "
                    f"{detail['layout_x2_pct']}, {detail['layout_y2_pct']})"
                )
            else:
                action = (
                    f"({detail['action_cx']}, {detail['action_cy']}, {detail['action_area']})"
                    if detail["action_cx"] is not None
                    else "N/A"
                )
            lines.append(
                f"| {detail['target_ratio']:g} | "
                f"{_markdown_image(candidate_alt, detail['candidate_path'])} | "
                f"{_format_metric(detail['judge_label'])} | {rules} | {action} | "
                f"{detail['total_attempt_count']} |"
            )
        lines.append("")

    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `summary.json`: machine-readable aggregate metrics",
            "- `summary.csv`: aggregate metrics table",
            "- `details.parquet`: structured task-level results",
            "- `details.jsonl`: task-level results including judge text",
            "- `generation_attempts.jsonl`: every policy response and retry",
            "- `judge_responses.jsonl`: raw judge responses and latency",
            "- `renders/originals/`: source previews",
            "- `renders/candidates/`: final policy results",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_generation_attempts(tasks: Sequence[dict[str, Any]], output_dir: Path) -> None:
    records = []
    for task in tasks:
        generation = json.loads(
            generation_progress_path(output_dir, task["task_id"]).read_text(encoding="utf-8")
        )
        for attempt in generation["attempts"]:
            records.append({"task_id": task["task_id"], "image_id": task["image_id"], **attempt})
    (output_dir / "generation_attempts.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )


def load_runtime_metadata(output_dir: Path) -> list[dict[str, Any]]:
    runtime_dir = output_dir / "runtime"
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(runtime_dir.glob("model_rank_*.json"))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate raw image_once Parquet data with a vision-language policy and GPT visual judge."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--lora-adapter-path", type=Path)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--model-family", choices=MODEL_FAMILIES, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--policy-prompt-path", type=Path)
    parser.add_argument("--vlm-prompt-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=2176)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    parser.add_argument("--image-min-pixels", type=int, default=65536)
    parser.add_argument("--internvl-max-dynamic-patch", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--data-parallel-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--prompt-batch-size", type=int, default=16)
    parser.add_argument("--max-num-seqs", type=int, default=16)
    parser.add_argument("--judge-workers", type=int, default=2)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--judge-only-resume",
        action="store_true",
        help="Reuse complete generation progress and retry unfinished judge tasks without starting vLLM.",
    )
    parser.add_argument("--canonicalize-bare-json", action="store_true")
    parser.add_argument(
        "--action-protocol",
        choices=("legacy-crop-json", "percent-json-v1", "layout-json-v1"),
        default="legacy-crop-json",
    )
    args = parser.parse_args()
    for name in (
        "max_attempts",
        "data_parallel_size",
        "tensor_parallel_size",
        "prompt_batch_size",
        "max_num_seqs",
        "judge_workers",
        "internvl_max_dynamic_patch",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError("max-images must be positive")
    if args.lora_adapter_path is not None and args.lora_rank <= 0:
        raise ValueError("lora-rank must be positive when lora-adapter-path is set")
    if args.judge_only_resume and not args.resume:
        raise ValueError("--judge-only-resume requires --resume")
    return args


def main() -> None:
    args = parse_args()
    required_paths = (args.model, args.data, args.vlm_prompt_path)
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    if args.policy_prompt_path is not None and not args.policy_prompt_path.is_file():
        raise FileNotFoundError(args.policy_prompt_path)
    if args.lora_adapter_path is not None:
        if not args.lora_adapter_path.is_dir():
            raise FileNotFoundError(args.lora_adapter_path)
        for filename in ("adapter_config.json", "adapter_model.safetensors"):
            if not (args.lora_adapter_path / filename).is_file():
                raise FileNotFoundError(args.lora_adapter_path / filename)
    args.output_dir = args.output_dir.resolve()
    policy_prompt_template = load_policy_prompt_template(args.policy_prompt_path)
    config = {
        "run_id": args.run_id,
        "model_name": args.model_name,
        "model_family": args.model_family,
        "output_protocol_version": (
            "layout-json-v1-fenced-recovery-v1"
            if args.action_protocol == "layout-json-v1"
            else (
                "percent-json-v1-fenced-recovery-v1"
                if args.action_protocol == "percent-json-v1"
                else (
                "crop-json-canonicalization-v1"
                if args.canonicalize_bare_json
                else "native-crop-protocol-v1"
                )
            )
        ),
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256_file(args.model / "config.json"),
        "model_index_sha256": (
            sha256_file(args.model / "model.safetensors.index.json")
            if (args.model / "model.safetensors.index.json").is_file()
            else None
        ),
        "lora_adapter_path": str(args.lora_adapter_path.resolve()) if args.lora_adapter_path else None,
        "lora_rank": args.lora_rank if args.lora_adapter_path else None,
        "lora_adapter_config_sha256": (
            sha256_file(args.lora_adapter_path / "adapter_config.json") if args.lora_adapter_path else None
        ),
        "lora_adapter_model_sha256": (
            sha256_file(args.lora_adapter_path / "adapter_model.safetensors") if args.lora_adapter_path else None
        ),
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "policy_prompt_path": (
            str(args.policy_prompt_path.resolve()) if args.policy_prompt_path is not None else None
        ),
        "policy_prompt_sha256": hashlib.sha256(policy_prompt_template.encode("utf-8")).hexdigest(),
        "vlm_prompt_path": str(args.vlm_prompt_path.resolve()),
        "vlm_prompt_sha256": sha256_file(args.vlm_prompt_path),
        "target_ratios": list(TARGET_RATIOS),
        "max_attempts": args.max_attempts,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
        "internvl_max_dynamic_patch": args.internvl_max_dynamic_patch,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "data_parallel_size": args.data_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "prompt_batch_size": args.prompt_batch_size,
        "max_num_seqs": args.max_num_seqs,
        "judge_workers": args.judge_workers,
        "max_images": args.max_images,
        "judge_config": {
            "deployment": os.getenv("CROP_VLM_MODEL", os.getenv("GPT5_AZURE_OPENAI_DEPLOYMENT", "gpt-5.6-sol")),
            "api_version": os.getenv("GPT5_AZURE_OPENAI_API_VERSION", "2025-04-01-preview"),
            "timeout": float(os.getenv("CROP_VLM_TIMEOUT", "45")),
            "max_retries": int(os.getenv("CROP_VLM_MAX_RETRIES", "2")),
            "fallback_label": float(os.getenv("CROP_VLM_FALLBACK_LABEL", "5.0")),
            "parse_fallback_label": float(os.getenv("CROP_VLM_PARSE_FALLBACK_LABEL", "2.5")),
            "image_size": int(os.getenv("CROP_VLM_IMAGE_SIZE", "384")),
            "image_format": os.getenv("CROP_VLM_IMAGE_FORMAT", "JPEG"),
            "jpeg_quality": int(os.getenv("CROP_VLM_JPEG_QUALITY", "70")),
            "reasoning_effort": os.getenv("CROP_VLM_REASONING_EFFORT", "low"),
            "output_verbosity": os.getenv("CROP_VLM_OUTPUT_VERBOSITY", "low"),
            "preprocess_mode": os.getenv("CROP_VLM_PREPROCESS_MODE", "letterbox"),
        },
    }
    prepare_output_directory(args.output_dir, config, args.resume)
    tasks, manifest = load_and_materialize_tasks(
        args.data,
        args.output_dir,
        policy_prompt_template=policy_prompt_template,
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
                    "policy_prompt": task["prompt"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for task in tasks
        ),
        encoding="utf-8",
    )

    if args.judge_only_resume:
        missing_generation = [
            task["task_id"]
            for task in tasks
            if not generation_progress_path(args.output_dir, task["task_id"]).is_file()
        ]
        if missing_generation:
            raise FileNotFoundError(
                f"judge-only resume requires complete generation progress; missing={missing_generation[:5]}"
            )
    else:
        run_parallel_generation(tasks, args.output_dir, args.model, args)
    run_judge(tasks, args.output_dir, args.vlm_prompt_path, args.judge_workers)
    details = build_details(tasks, args.output_dir)
    summary = {
        "run_id": args.run_id,
        "model_name": args.model_name,
        "model_family": args.model_family,
        "model": str(args.model.resolve()),
        "provenance": {
            "model_config_sha256": config["model_config_sha256"],
            "model_index_sha256": config["model_index_sha256"],
            "data_sha256": config["data_sha256"],
            "policy_prompt_sha256": config["policy_prompt_sha256"],
            "vlm_prompt_sha256": config["vlm_prompt_sha256"],
            "runtime_ranks": load_runtime_metadata(args.output_dir),
        },
        "data": str(args.data.resolve()),
        "images": len(manifest),
        "target_ratios": list(TARGET_RATIOS),
        "max_attempts": args.max_attempts,
        **summarize(details),
    }
    write_generation_attempts(tasks, args.output_dir)
    write_result_tables(details, summary, args.output_dir)
    render_html_report(details, summary, args.output_dir)
    render_markdown_report(details, summary, args.output_dir)
    write_json_atomic(
        args.output_dir / "_EVAL_COMPLETE.json",
        {
            "run_id": args.run_id,
            "images": len(manifest),
            "tasks": len(tasks),
            "generation_success_count": summary["overall"]["generation_success_count"],
            "judge_completed_count": summary["overall"]["judge_completed_count"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()