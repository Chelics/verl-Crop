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
import threading
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from PIL import Image, ImageOps

from news_crop_benchmark.data import build_prompt, load_policy_prompt_template
from news_crop_benchmark.geometry import TARGET_RATIOS
from news_crop_benchmark.policy_model_adapter import MODEL_FAMILIES, PolicyModelAdapter
from news_crop_benchmark.protocol import parse_mode_decision

SOURCE_COLUMNS = ("image_id", "original_image", "title", "ImageCaption")
FINAL_GENERATION_STATUSES = {"valid", "retry_exhausted"}


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


def save_report_image(image: Image.Image, path: Path, maximum_side: int = 1200) -> None:
    output = image.convert("RGB")
    if max(output.size) > maximum_side:
        scale = maximum_side / max(output.size)
        output = output.resize(
            (max(1, round(output.width * scale)), max(1, round(output.height * scale))),
            Image.Resampling.LANCZOS,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    output.save(temporary_path, format="JPEG", quality=90, optimize=True)
    temporary_path.replace(path)


def task_id(image_id: str, target_ratio: float) -> str:
    return f"{image_id}__ratio_{target_ratio:g}"


def load_and_materialize_tasks(
    data_path: Path,
    output_dir: Path,
    target_ratios: Sequence[float] = TARGET_RATIOS,
    mode_prompt_template: str | None = None,
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

    source_rows = table.to_pylist()
    if max_images is not None:
        source_rows = source_rows[:max_images]
    source_dir = output_dir / ".source_images"
    preview_dir = output_dir / "renders" / "originals"
    tasks: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    seen_image_ids: set[str] = set()
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
            save_report_image(original, preview_path)
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
                    "prompt": build_prompt(title, float(target_ratio), mode_prompt_template),
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


def generation_progress_path(output_dir: Path, current_task_id: str) -> Path:
    return output_dir / "progress" / "generation" / f"{current_task_id}.json"


def build_retry_prompt(base_prompt: str, previous_attempts: Sequence[dict[str, Any]]) -> str:
    if not previous_attempts:
        return base_prompt
    error = str(previous_attempts[-1].get("parse_error") or "output did not satisfy the mode protocol")
    return (
        f"{base_prompt}\n\nYour previous output was rejected by the validator: {error}\n"
        "Generate a new answer containing only one JSON object. It must be exactly "
        '{"mode":"crop"} or {"mode":"pad"}, with no other fields or text.'
    )


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
        output_dir / "runtime" / f"model_rank_{rank}.json",
        {
            "rank": rank,
            "gpu_devices": gpu_devices,
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
                            build_retry_prompt(task["prompt"], attempts[task["task_id"]]),
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
                outputs = llm.generate(requests, sampling_params=sampling_params)
                for task, request_output in zip(task_batch, outputs, strict=True):
                    current_task_id = task["task_id"]
                    response = request_output.outputs[0].text if request_output.outputs else ""
                    attempt_record: dict[str, Any] = {
                        "attempt": attempt_number,
                        "seed": attempt_seed,
                        "response": response,
                        "valid": False,
                        "parse_error": None,
                    }
                    try:
                        decision = parse_mode_decision(response)
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
                                "mode": None,
                            },
                        )
                        if status == "retry_exhausted":
                            active.pop(current_task_id)
                        continue

                    attempt_record["valid"] = True
                    attempts[current_task_id].append(attempt_record)
                    write_json_atomic(
                        generation_progress_path(output_dir, current_task_id),
                        {
                            "task_id": current_task_id,
                            "rank": rank,
                            "status": "valid",
                            "attempts": attempts[current_task_id],
                            "mode": decision.mode,
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
            name=f"mode-model-rank-{rank}",
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


def build_details(tasks: Sequence[dict[str, Any]], output_dir: Path) -> list[dict[str, Any]]:
    details = []
    for task in tasks:
        generation = json.loads(
            generation_progress_path(output_dir, task["task_id"]).read_text(encoding="utf-8")
        )
        attempts = generation["attempts"]
        invalid_attempt_count = sum(not attempt["valid"] for attempt in attempts)
        valid_attempt = next((attempt for attempt in attempts if attempt["valid"]), None)
        final_attempt = valid_attempt or (attempts[-1] if attempts else None)
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
                "generation_status": generation["status"],
                "predicted_mode": generation.get("mode"),
                "strict_format": valid_attempt is not None,
                "had_invalid_output": invalid_attempt_count > 0,
                "invalid_attempt_count": invalid_attempt_count,
                "total_attempt_count": len(attempts),
                "final_response": final_attempt["response"] if final_attempt else None,
                "final_parse_error": final_attempt.get("parse_error") if final_attempt else None,
            }
        )
    return details


def summarize_subset(details: Sequence[dict[str, Any]]) -> dict[str, Any]:
    generated = [detail for detail in details if detail["generation_status"] == "valid"]
    mode_counts = Counter(detail["predicted_mode"] for detail in generated)
    total = len(details)
    return {
        "tasks": total,
        "generation_success_count": len(generated),
        "generation_success_rate": len(generated) / total if total else 0.0,
        "first_attempt_valid_count": sum(
            detail["generation_status"] == "valid" and detail["total_attempt_count"] == 1
            for detail in details
        ),
        "had_invalid_output_count": sum(detail["had_invalid_output"] for detail in details),
        "invalid_output_count": sum(detail["invalid_attempt_count"] for detail in details),
        "retry_recovered_count": sum(
            detail["generation_status"] == "valid" and detail["had_invalid_output"]
            for detail in details
        ),
        "retry_exhausted_count": sum(
            detail["generation_status"] == "retry_exhausted" for detail in details
        ),
        "mode_counts": dict(sorted(mode_counts.items())),
        "crop_count": mode_counts["crop"],
        "pad_count": mode_counts["pad"],
        "crop_rate": mode_counts["crop"] / len(generated) if generated else 0.0,
        "pad_rate": mode_counts["pad"] / len(generated) if generated else 0.0,
    }


def summarize_image_patterns(details: Sequence[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        grouped[detail["image_id"]].append(detail)
    pattern_counts: Counter[str] = Counter()
    incomplete = 0
    for image_details in grouped.values():
        image_details.sort(key=lambda detail: TARGET_RATIOS.index(detail["target_ratio"]))
        if any(detail["predicted_mode"] is None for detail in image_details):
            incomplete += 1
            continue
        pattern_counts["/".join(detail["predicted_mode"] for detail in image_details)] += 1
    uniform_crop = sum(count for pattern, count in pattern_counts.items() if set(pattern.split("/")) == {"crop"})
    uniform_pad = sum(count for pattern, count in pattern_counts.items() if set(pattern.split("/")) == {"pad"})
    return {
        "images": len(grouped),
        "complete_image_count": len(grouped) - incomplete,
        "incomplete_image_count": incomplete,
        "uniform_crop_image_count": uniform_crop,
        "uniform_pad_image_count": uniform_pad,
        "mixed_mode_image_count": sum(pattern_counts.values()) - uniform_crop - uniform_pad,
        "mode_pattern_counts": dict(sorted(pattern_counts.items())),
    }


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": summarize_subset(details),
        "by_ratio": {
            f"{ratio:g}": summarize_subset(
                [detail for detail in details if detail["target_ratio"] == ratio]
            )
            for ratio in TARGET_RATIOS
        },
        "image_patterns": summarize_image_patterns(details),
    }


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


def write_result_tables(details: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    details.sort(key=lambda detail: (detail["source_index"], TARGET_RATIOS.index(detail["target_ratio"])))
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
        rows = [("overall", summary["overall"]), *sorted(summary["by_ratio"].items())]
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "scope",
                "tasks",
                "generation_success_count",
                "generation_success_rate",
                "crop_count",
                "pad_count",
                "crop_rate",
                "pad_rate",
                "invalid_output_count",
                "retry_exhausted_count",
            ],
        )
        writer.writeheader()
        for scope, metrics in rows:
            writer.writerow({"scope": scope, **{key: metrics[key] for key in writer.fieldnames[1:]}})

    with (output_dir / "review_template.csv").open("w", newline="", encoding="utf-8-sig") as output:
        fieldnames = [
            "source_index",
            "image_id",
            "title",
            "caption",
            "target_ratio",
            "predicted_mode",
            "generation_status",
            "human_label",
            "notes",
        ]
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for detail in details:
            writer.writerow(
                {
                    **{key: detail[key] for key in fieldnames[:-2]},
                    "human_label": "",
                    "notes": "",
                }
            )


def _format_metric(value: Any) -> str:
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
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_format_metric(value))}</td></tr>"
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
            cards.append(
                f"""
                <article class="decision {html.escape(mode)}">
                  <p class="ratio">Target {detail['target_ratio']:g}</p>
                  <p class="mode">{html.escape(mode.upper())}</p>
                  <p><strong>Status:</strong> {html.escape(detail['generation_status'])}</p>
                  <p><strong>Attempts:</strong> {detail['total_attempt_count']}
                     (invalid: {detail['invalid_attempt_count']})</p>
                  <p><strong>Human label:</strong> __________________</p>
                  <details><summary>Model response</summary><pre>{html.escape(str(detail['final_response']))}</pre></details>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="sample">
              <header><h2>{html.escape(first['title'])}</h2><p class="id">{html.escape(first['image_id'])}</p></header>
              <div class="layout">
                <article class="original">
                  <img src="{html.escape(first['original_render_path'])}" alt="Original image">
                  <p>{html.escape(first['caption'])}</p>
                </article>
                <div class="decisions">{''.join(cards)}</div>
              </div>
            </section>
            """
        )

    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(summary['model_name'])} Crop-or-Pad Diagnosis</title>
<style>
:root {{ --ink:#202522; --muted:#66706a; --line:#ccd3cf; --paper:#eef1ef; --crop:#176b4d; --pad:#9a5418; --bad:#9b2c2c; }}
* {{ box-sizing:border-box; }} body {{ margin:0; color:var(--ink); background:var(--paper); font-family:"Segoe UI",sans-serif; }}
main {{ width:min(1560px,96vw); margin:auto; padding:24px 0 64px; }} .summary,.sample {{ background:white; border:1px solid var(--line); padding:20px; margin-bottom:20px; }}
table {{ border-collapse:collapse; width:min(900px,100%); }} th,td {{ border-bottom:1px solid var(--line); padding:7px 10px; text-align:left; }}
.layout {{ display:grid; grid-template-columns:minmax(280px,0.9fr) minmax(0,2.1fr); gap:18px; }} .decisions {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:12px; }}
.original img {{ display:block; width:100%; max-height:560px; object-fit:contain; background:#e7ebe8; }} .decision {{ border:1px solid var(--line); border-top:5px solid var(--line); padding:14px; min-width:0; }}
.decision.crop {{ border-top-color:var(--crop); }} .decision.pad {{ border-top-color:var(--pad); }} .decision.invalid {{ border-top-color:var(--bad); }} .ratio {{ color:var(--muted); margin-bottom:4px; }} .mode {{ font-size:26px; font-weight:700; margin:0 0 18px; }}
.id,p,pre {{ overflow-wrap:anywhere; }} pre {{ white-space:pre-wrap; font-size:12px; }}
@media (max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} .decisions {{ grid-template-columns:1fr; }} main {{ width:100%; padding:10px; }} }}
</style></head><body><main>
<section class="summary"><h1>{html.escape(summary['model_name'])} Crop-or-Pad Diagnosis</h1><p>Mode-only zero-shot output. No crop rendering, quality judge, or background-color selection was run.</p><table>{metric_rows}</table></section>
{''.join(sections)}
</main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def render_markdown_report(details: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    overall = summary["overall"]
    lines = [
        f"# {summary['model_name']} Crop-or-Pad Diagnosis",
        "",
        "This is an unlabeled mode-only report. It does not render crops, score quality, or select padding colors.",
        "",
        "## Run",
        "",
        f"- Run ID: `{summary['run_id']}`",
        f"- Model family: `{summary['model_family']}`",
        f"- Images: {summary['images']}",
        f"- Tasks: {overall['tasks']}",
        "",
        "## Descriptive Summary",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "generation_success_count",
        "generation_success_rate",
        "first_attempt_valid_count",
        "invalid_output_count",
        "retry_recovered_count",
        "retry_exhausted_count",
        "crop_count",
        "pad_count",
        "crop_rate",
        "pad_rate",
    ):
        lines.append(f"| `{key}` | {_format_metric(overall[key])} |")
    lines.extend(
        [
            "",
            "## By Ratio",
            "",
            "| Ratio | Valid | Crop | Pad | Crop Rate | Pad Rate |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ratio in TARGET_RATIOS:
        metrics = summary["by_ratio"][f"{ratio:g}"]
        lines.append(
            f"| {ratio:g} | {metrics['generation_success_count']} | {metrics['crop_count']} | "
            f"{metrics['pad_count']} | {_format_metric(metrics['crop_rate'])} | "
            f"{_format_metric(metrics['pad_rate'])} |"
        )

    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        by_image[detail["image_id"]].append(detail)
    lines.extend(["", "## Visual Review", ""])
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
                "| Ratio | Prediction | Status | Attempts | Human Label |",
                "|---:|---|---|---:|---|",
            ]
        )
        for detail in image_details:
            lines.append(
                f"| {detail['target_ratio']:g} | `{detail['predicted_mode'] or 'invalid'}` | "
                f"{detail['generation_status']} | {detail['total_attempt_count']} |  |"
            )
        lines.append("")
    lines.extend(
        [
            "## Artifacts",
            "",
            "- `review_template.csv`: predictions with blank `human_label` and `notes` columns",
            "- `details.jsonl` and `details.parquet`: task-level mode predictions",
            "- `summary.json` and `summary.csv`: descriptive output statistics",
            "- `generation_attempts.jsonl`: all raw model responses and retries",
            "- `renders/originals/`: report previews of source images",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def load_runtime_metadata(output_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((output_dir / "runtime").glob("model_rank_*.json"))
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a standalone zero-shot crop-versus-pad diagnosis on image_once Parquet data."
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-family", choices=MODEL_FAMILIES, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--mode-prompt-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-model-len", type=int, default=2176)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    parser.add_argument("--image-min-pixels", type=int, default=65536)
    parser.add_argument("--internvl-max-dynamic-patch", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--prompt-batch-size", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--max-images", type=int)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    for name in (
        "max_attempts",
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
    if args.temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")
    return args


def main() -> None:
    args = parse_args()
    for path in (args.model, args.data, args.mode_prompt_path):
        if not path.exists():
            raise FileNotFoundError(path)
    model_config_path = args.model / "config.json"
    if not model_config_path.is_file():
        raise FileNotFoundError(model_config_path)
    args.output_dir = args.output_dir.resolve()
    mode_prompt_template = load_policy_prompt_template(args.mode_prompt_path)
    model_index_path = args.model / "model.safetensors.index.json"
    config = {
        "run_id": args.run_id,
        "model_name": args.model_name,
        "model_family": args.model_family,
        "output_protocol_version": "mode-json-v1",
        "background_color_selection": False,
        "model": str(args.model.resolve()),
        "model_config_sha256": sha256_file(model_config_path),
        "model_index_sha256": sha256_file(model_index_path) if model_index_path.is_file() else None,
        "data": str(args.data.resolve()),
        "data_sha256": sha256_file(args.data),
        "mode_prompt_path": str(args.mode_prompt_path.resolve()),
        "mode_prompt_sha256": hashlib.sha256(mode_prompt_template.encode("utf-8")).hexdigest(),
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
        "max_images": args.max_images,
    }
    prepare_output_directory(args.output_dir, config, args.resume)
    tasks, manifest = load_and_materialize_tasks(
        args.data,
        args.output_dir,
        mode_prompt_template=mode_prompt_template,
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
                    "mode_prompt": task["prompt"],
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
            for task in tasks
        ),
        encoding="utf-8",
    )

    run_parallel_generation(tasks, args.output_dir, args.model, args)
    details = build_details(tasks, args.output_dir)
    summary = {
        "run_id": args.run_id,
        "model_name": args.model_name,
        "model_family": args.model_family,
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "images": len(manifest),
        "target_ratios": list(TARGET_RATIOS),
        "quality_metrics_available": False,
        "background_color_selection": False,
        "provenance": {
            "model_config_sha256": config["model_config_sha256"],
            "model_index_sha256": config["model_index_sha256"],
            "data_sha256": config["data_sha256"],
            "mode_prompt_sha256": config["mode_prompt_sha256"],
            "runtime_ranks": load_runtime_metadata(args.output_dir),
        },
        **summarize(details),
    }
    write_generation_attempts(tasks, args.output_dir)
    write_result_tables(details, summary, args.output_dir)
    render_html_report(details, summary, args.output_dir)
    render_markdown_report(details, summary, args.output_dir)
    write_json_atomic(
        args.output_dir / "_MODE_EVAL_COMPLETE.json",
        {
            "run_id": args.run_id,
            "images": len(manifest),
            "tasks": len(tasks),
            "generation_success_count": summary["overall"]["generation_success_count"],
            "crop_count": summary["overall"]["crop_count"],
            "pad_count": summary["overall"]["pad_count"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()