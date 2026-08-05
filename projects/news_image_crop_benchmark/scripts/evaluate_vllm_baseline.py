#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import importlib.util
import json
import multiprocessing
import os
import random
from collections import defaultdict
from pathlib import Path, PurePosixPath
from statistics import mean
from typing import Any, Iterator, Sequence

import yaml
from PIL import Image

from news_crop_benchmark.geometry import CropAction, action_to_bbox
from news_crop_benchmark.protocol import parse_crop_action_with_format
from news_crop_benchmark.proxy_scorer import crop_image


def load_reward_function(path: Path):
    spec = importlib.util.spec_from_file_location("news_crop_baseline_reward", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load reward file: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint_path(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_file():
        return sha256_file(path)

    digest = hashlib.sha256()
    digest.update(b"directory-manifest-v1\0")
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        stat = item.stat()
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(stat.st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(str(stat.st_mtime_ns).encode("ascii"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def fingerprint_selected_rows(rows: Sequence[dict]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        info = row["extra_info"]
        record = {
            "sample_id": info["sample_id"],
            "title": info["title"],
            "target_ratio": info["target_ratio"],
            "image_path": row["images"][0],
            "ground_truth": row["reward_model"]["ground_truth"],
        }
        digest.update(json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def fingerprint_selected_images(rows: Sequence[dict]) -> str:
    digest = hashlib.sha256()
    for image_path_text in sorted({str(row["images"][0]) for row in rows}):
        image_path = Path(image_path_text)
        digest.update(image_path_text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(image_path).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def remap_dataset_paths(rows: Sequence[dict], source_prefix: str, destination_prefix: str) -> None:
    source = PurePosixPath(source_prefix)
    destination = PurePosixPath(destination_prefix)
    for row in rows:
        image_path = PurePosixPath(str(row["images"][0]))
        try:
            relative_path = image_path.relative_to(source)
        except ValueError:
            continue
        remapped_path = str(destination / relative_path)
        row["images"][0] = remapped_path
        extra_info = row.get("extra_info", {})
        if extra_info.get("original_image_path") == str(image_path):
            extra_info["original_image_path"] = remapped_path


def select_complete_groups(data_path: Path, group_count: int | None, seed: int) -> list[dict]:
    import pyarrow.parquet as pq

    rows = pq.read_table(data_path).to_pylist()
    grouped: dict[tuple[str, str], dict[float, dict]] = defaultdict(dict)
    for row in rows:
        info = row["extra_info"]
        grouped[(row["images"][0], info["title"])][float(info["target_ratio"])] = row

    complete_groups = [group for _, group in sorted(grouped.items()) if len(group) == 4]
    if group_count is None:
        selected = complete_groups
    elif group_count > len(complete_groups):
        raise ValueError(f"requested {group_count} groups, but only {len(complete_groups)} complete groups exist")
    else:
        selected = random.Random(seed).sample(complete_groups, group_count)
    return [group[ratio] for group in selected for ratio in sorted(group)]


def batched(items: Sequence[Any], batch_size: int) -> Iterator[Sequence[Any]]:
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


def partition_rows(rows: Sequence[dict], data_parallel_size: int) -> list[list[dict]]:
    if data_parallel_size <= 0:
        raise ValueError("data_parallel_size must be positive")
    return [list(rows[rank::data_parallel_size]) for rank in range(data_parallel_size)]


def build_vllm_request(processor, row: dict, image: Image.Image, image_max_pixels: int, image_min_pixels: int):
    prompt_text = row["prompt"][0]["content"]
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text.replace("<image>\n", "", 1)},
            ],
        }
    ]
    rendered_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    return {
        "prompt": rendered_prompt,
        "multi_modal_data": {"image": image},
        "mm_processor_kwargs": {
            "size": {"longest_edge": image_max_pixels, "shortest_edge": image_min_pixels}
        },
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary_path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    temporary_path.replace(path)


def generation_path(raw_dir: Path, sample_id: str) -> Path:
    return raw_dir / f"{sample_id}.json"


def score_progress_path(progress_dir: Path, sample_id: str) -> Path:
    return progress_dir / f"{sample_id}.json"


def run_generation_worker(
    rank: int,
    gpu_devices: list[str],
    rows: list[dict],
    raw_dir: Path,
    model_path: Path,
    tensor_parallel_size: int,
    prompt_batch_size: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    image_max_pixels: int,
    image_min_pixels: int,
    temperature: float,
    top_p: float,
    n: int,
    max_tokens: int,
    seed: int,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_devices)
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    pending_rows = [
        row for row in rows if not generation_path(raw_dir, str(row["extra_info"]["sample_id"])).exists()
    ]
    if not pending_rows:
        return

    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True)
    llm = LLM(
        model=str(model_path),
        tensor_parallel_size=tensor_parallel_size,
        dtype="bfloat16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        enforce_eager=True,
        limit_mm_per_prompt={"image": 1},
    )
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        n=n,
        max_tokens=max_tokens,
        seed=seed,
    )

    for row_batch in batched(pending_rows, prompt_batch_size):
        images: list[Image.Image] = []
        try:
            requests = []
            for row in row_batch:
                with Image.open(row["images"][0]) as source:
                    image = source.convert("RGB")
                images.append(image)
                requests.append(build_vllm_request(processor, row, image, image_max_pixels, image_min_pixels))

            outputs = llm.generate(requests, sampling_params=sampling_params)
            for row, request_output in zip(row_batch, outputs, strict=True):
                sample_id = str(row["extra_info"]["sample_id"])
                write_json_atomic(
                    generation_path(raw_dir, sample_id),
                    {
                        "sample_id": sample_id,
                        "rank": rank,
                        "responses": [output.text for output in request_output.outputs],
                    },
                )
        finally:
            for image in images:
                image.close()


def run_parallel_generation(
    rows: list[dict],
    raw_dir: Path,
    model_path: Path,
    data_parallel_size: int,
    tensor_parallel_size: int,
    prompt_batch_size: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    image_max_pixels: int,
    image_min_pixels: int,
    temperature: float,
    top_p: float,
    n: int,
    max_tokens: int,
    seed: int,
) -> None:
    required_gpus = data_parallel_size * tensor_parallel_size
    devices = resolve_gpu_devices(required_gpus)
    partitions = partition_rows(rows, data_parallel_size)
    context = multiprocessing.get_context("spawn")
    processes = []
    for rank, rank_rows in enumerate(partitions):
        if not rank_rows:
            continue
        start = rank * tensor_parallel_size
        process = context.Process(
            target=run_generation_worker,
            args=(
                rank,
                devices[start : start + tensor_parallel_size],
                rank_rows,
                raw_dir,
                model_path,
                tensor_parallel_size,
                prompt_batch_size,
                max_num_seqs,
                gpu_memory_utilization,
                max_model_len,
                image_max_pixels,
                image_min_pixels,
                temperature,
                top_p,
                n,
                max_tokens,
                seed,
            ),
            name=f"qwen-baseline-rank-{rank}",
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


def prepare_output_directory(output_dir: Path, config: dict, resume: bool) -> None:
    config_path = output_dir / "baseline_config.yaml"
    if output_dir.exists() and any(output_dir.iterdir()):
        if not resume:
            raise FileExistsError(f"output directory is not empty; pass --resume to continue: {output_dir}")
        if not config_path.is_file():
            raise FileNotFoundError(f"resume requires an existing baseline config: {config_path}")
        existing_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if existing_config != config:
            raise ValueError("resume configuration does not match the existing baseline_config.yaml")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=True), encoding="utf-8")


def summarize(details: list[dict], center_scores: dict[str, float]) -> dict:
    ratios = sorted({detail["target_ratio"] for detail in details})

    def summarize_subset(subset: list[dict]) -> dict:
        valid = [detail for detail in subset if detail["valid"]]
        strict_format = [detail for detail in subset if detail["strict_format"]]
        groups: dict[str, list[dict]] = defaultdict(list)
        for detail in subset:
            groups[detail["sample_id"]].append(detail)
        best_scores = [max(item["score"] for item in group) for group in groups.values()]
        pass_at_1_scores = [
            min(group, key=lambda item: item["candidate_index"])["score"] for group in groups.values()
        ]
        center = [center_scores[sample_id] for sample_id in groups]
        return {
            "outputs": len(subset),
            "valid_rate": len(valid) / len(subset),
            "strict_format_rate": len(strict_format) / len(subset),
            "mean_score": mean(detail["score"] for detail in subset),
            "pass_at_1_mean_score": mean(pass_at_1_scores),
            "best_of_n_mean_score": mean(best_scores),
            "center_crop_mean_score": mean(center),
            "best_of_n_win_rate_vs_center": mean(
                best_score > center_score for best_score, center_score in zip(best_scores, center, strict=True)
            ),
            "mean_action_area": mean(detail["action"]["area"] for detail in valid) if valid else None,
            "mean_action_cx": mean(detail["action"]["cx"] for detail in valid) if valid else None,
            "mean_action_cy": mean(detail["action"]["cy"] for detail in valid) if valid else None,
            "near_full_image_rate": mean(detail["action"]["area"] >= 950 for detail in valid) if valid else None,
            "tiny_crop_rate": mean(detail["action"]["area"] <= 50 for detail in valid) if valid else None,
        }

    return {
        "overall": summarize_subset(details),
        "by_ratio": {
            f"{ratio:g}": summarize_subset([detail for detail in details if detail["target_ratio"] == ratio])
            for ratio in ratios
        },
    }


def save_preview(image: Image.Image, path: Path, maximum_side: int = 900) -> None:
    preview = image.convert("RGB")
    if max(preview.size) > maximum_side:
        scale = maximum_side / max(preview.size)
        preview = preview.resize(
            (max(1, round(preview.width * scale)), max(1, round(preview.height * scale))),
            Image.Resampling.LANCZOS,
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    preview.save(path, format="JPEG", quality=88, optimize=True)


def render_html_report(
    output_dir: Path,
    summary: dict,
    selected_rows: list[dict],
    details: list[dict],
    center_baselines: dict[str, dict],
) -> None:
    details_by_sample: dict[str, list[dict]] = defaultdict(list)
    for detail in details:
        details_by_sample[detail["sample_id"]].append(detail)

    metric_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in summary["overall"].items()
    )
    sections = []
    for row in selected_rows:
        info = row["extra_info"]
        sample_id = info["sample_id"]
        candidates = sorted(details_by_sample[sample_id], key=lambda item: item["candidate_index"])
        best = max(candidates, key=lambda item: item["score"])
        candidate_cards = []
        for candidate in candidates:
            action = candidate["action"]
            action_text = "解析失败" if action is None else json.dumps(action, ensure_ascii=False)
            image_html = (
                f'<img src="{html.escape(candidate["render_path"])}" alt="候选裁剪">'
                if candidate.get("render_path")
                else '<div class="missing">无有效裁剪图</div>'
            )
            best_badge = '<span class="badge">best</span>' if candidate is best else ""
            format_text = "严格合法" if candidate["strict_format"] else "恢复解析"
            candidate_cards.append(
                f"""
                <article class="candidate">
                  <h4>候选 {candidate['candidate_index'] + 1} {best_badge}</h4>
                  {image_html}
                  <p><strong>总分：</strong>{candidate['score']:.4f}</p>
                  <p><strong>格式：</strong>{format_text}（{candidate['format_reward']:.2f}）</p>
                  <p><strong>动作：</strong><code>{html.escape(action_text)}</code></p>
                  <details><summary>Reward 分项与原始响应</summary>
                    <pre>{html.escape(json.dumps({
                        'response': candidate['response'],
                        'title_relevance': candidate['title_relevance'],
                        'saliency': candidate['saliency'],
                        'composition': candidate['composition'],
                        'integrity': candidate['integrity'],
                        'area': candidate['area'],
                        'area_fraction': candidate['area_fraction'],
                    }, ensure_ascii=False, indent=2))}</pre>
                  </details>
                </article>
                """
            )

        center = center_baselines[sample_id]
        sections.append(
            f"""
            <section class="sample">
              <header>
                <div><span class="ratio">比例 {info['target_ratio']:g}</span></div>
                <h2>{html.escape(info['title'])}</h2>
                <p class="id">sample_id: {html.escape(sample_id)}</p>
              </header>
              <div class="reference-grid">
                <article><h3>原图</h3><img src="{html.escape(center['original_render_path'])}" alt="原图"></article>
                <article><h3>最大中心裁剪</h3><img src="{html.escape(center['render_path'])}" alt="中心裁剪"><p>Proxy Reward：{center['score']:.4f}</p></article>
              </div>
              <div class="candidate-grid">{''.join(candidate_cards)}</div>
              <div class="review"><strong>人工判断：</strong>□ Qwen 更好　□ 中心裁剪更好　□ 两者都可以　□ 两者都不好</div>
            </section>
            """
        )

    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3.5-9B 裁剪预诊断</title>
<style>
:root {{ color-scheme: light; --ink:#17201f; --muted:#64706d; --line:#d8dedb; --paper:#f5f7f5; --accent:#006d5b; }}
* {{ box-sizing:border-box; }} body {{ margin:0; font-family:"Noto Sans SC","Microsoft YaHei",sans-serif; color:var(--ink); background:var(--paper); }}
main {{ width:min(1500px,96vw); margin:0 auto; padding:28px 0 64px; }}
h1,h2,h3,h4,p {{ margin-top:0; }} .summary,.sample {{ background:white; border:1px solid var(--line); margin-bottom:22px; padding:22px; }}
.summary table {{ border-collapse:collapse; width:min(760px,100%); }} th,td {{ border-bottom:1px solid var(--line); padding:8px 10px; text-align:left; }}
.sample header {{ border-bottom:1px solid var(--line); margin-bottom:18px; }} .ratio,.badge {{ display:inline-block; color:white; background:var(--accent); padding:3px 8px; font-size:13px; }}
.id {{ color:var(--muted); font-size:12px; word-break:break-all; }} img {{ display:block; max-width:100%; max-height:440px; object-fit:contain; background:#edf0ee; }}
.reference-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:16px; margin-bottom:20px; }}
.candidate-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; }}
.candidate {{ border:1px solid var(--line); padding:12px; min-width:0; }} .candidate img {{ width:100%; height:260px; object-fit:contain; }}
code,pre {{ font-family:"Noto Sans Mono",monospace; }} code {{ overflow-wrap:anywhere; }} pre {{ white-space:pre-wrap; overflow-wrap:anywhere; font-size:12px; }}
.review {{ margin-top:18px; padding:14px; border:1px dashed var(--accent); }} .warning {{ color:#8a3c00; }}
@media (max-width:720px) {{ .reference-grid {{ grid-template-columns:1fr; }} main {{ width:100%; padding:12px; }} .summary,.sample {{ padding:14px; }} }}
</style>
</head>
<body><main>
<section class="summary"><h1>Qwen3.5-9B 裁剪预诊断</h1>
<p class="warning">注意：Proxy Reward 同时用于候选排序和自动评价，本报告必须结合人工视觉判断。</p>
<table>{metric_rows}</table></section>
{''.join(sections)}
</main></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the untrained Qwen3.5 vLLM crop baseline.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--reward-file", type=Path, required=True)
    parser.add_argument("--clip-model-path", type=Path, required=True)
    parser.add_argument("--clip-device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    group_selection = parser.add_mutually_exclusive_group()
    group_selection.add_argument("--groups", type=int, default=4)
    group_selection.add_argument("--all-groups", action="store_true")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=2176)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    parser.add_argument("--image-min-pixels", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    parser.add_argument("--data-parallel-size", type=int, default=8)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--prompt-batch-size", type=int, default=32)
    parser.add_argument("--max-num-seqs", type=int, default=32)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--run-id", default="standalone")
    parser.add_argument(
        "--image-path-map",
        action="append",
        default=[],
        metavar="SOURCE=DESTINATION",
        help="Remap absolute image paths stored in the dataset; may be repeated.",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.groups <= 0 or args.n <= 0:
        raise ValueError("groups and n must be positive")
    if args.data_parallel_size <= 0 or args.tensor_parallel_size <= 0:
        raise ValueError("data-parallel-size and tensor-parallel-size must be positive")
    if args.prompt_batch_size <= 0 or args.max_num_seqs <= 0:
        raise ValueError("prompt-batch-size and max-num-seqs must be positive")
    if args.max_num_seqs < args.n:
        raise ValueError("max-num-seqs must be at least n")
    if args.shard_count <= 0:
        raise ValueError("shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count)")

    group_count = None if args.all_groups else args.groups
    selected_rows = select_complete_groups(args.data, group_count, args.seed)
    parsed_path_maps = []
    for path_map in args.image_path_map:
        if "=" not in path_map:
            raise ValueError(f"image-path-map must use SOURCE=DESTINATION: {path_map}")
        source_prefix, destination_prefix = path_map.split("=", 1)
        if not source_prefix or not destination_prefix:
            raise ValueError(f"image-path-map must use non-empty SOURCE=DESTINATION: {path_map}")
        remap_dataset_paths(selected_rows, source_prefix, destination_prefix)
        parsed_path_maps.append({"source": source_prefix, "destination": destination_prefix})
    selected_rows = selected_rows[args.shard_index :: args.shard_count]
    if not selected_rows:
        raise ValueError(f"shard {args.shard_index} has no selected rows")
    config = {
        "model": str(args.model.resolve()),
        "model_fingerprint": fingerprint_path(args.model),
        "data": str(args.data.resolve()),
        "data_fingerprint": fingerprint_path(args.data),
        "groups": "all" if args.all_groups else args.groups,
        "sample_manifest_fingerprint": fingerprint_selected_rows(selected_rows),
        "selected_images_fingerprint": fingerprint_selected_images(selected_rows),
        "n": args.n,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
        "reward_file": str(args.reward_file.resolve()),
        "reward_file_fingerprint": fingerprint_path(args.reward_file),
        "clip_model_path": str(args.clip_model_path.resolve()),
        "clip_model_fingerprint": fingerprint_path(args.clip_model_path),
        "clip_device": args.clip_device,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "data_parallel_size": args.data_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        "prompt_batch_size": args.prompt_batch_size,
        "max_num_seqs": args.max_num_seqs,
        "image_path_maps": parsed_path_maps,
        "shard_count": args.shard_count,
        "shard_index": args.shard_index,
        "run_id": args.run_id,
    }
    prepare_output_directory(args.output_dir, config, args.resume)
    raw_dir = args.output_dir / ".raw_generations"
    progress_dir = args.output_dir / ".score_progress"
    raw_dir.mkdir(parents=True, exist_ok=True)
    progress_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = args.output_dir / "sample_manifest.jsonl"
    if not manifest_path.exists():
        manifest_path.write_text(
            "".join(
                json.dumps(
                    {
                        "sample_id": row["extra_info"]["sample_id"],
                        "title": row["extra_info"]["title"],
                        "target_ratio": row["extra_info"]["target_ratio"],
                        "image_path": row["images"][0],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
                for row in selected_rows
            ),
            encoding="utf-8",
        )

    human_review_path = args.output_dir / "human_review.csv"
    if not human_review_path.exists():
        with human_review_path.open("w", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=["sample_id", "title", "target_ratio", "judgement", "comment"],
            )
            writer.writeheader()
            for row in selected_rows:
                info = row["extra_info"]
                writer.writerow(
                    {
                        "sample_id": info["sample_id"],
                        "title": info["title"],
                        "target_ratio": info["target_ratio"],
                        "judgement": "",
                        "comment": "",
                    }
                )

    run_parallel_generation(
        rows=selected_rows,
        raw_dir=raw_dir,
        model_path=args.model,
        data_parallel_size=args.data_parallel_size,
        tensor_parallel_size=args.tensor_parallel_size,
        prompt_batch_size=args.prompt_batch_size,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        image_max_pixels=args.image_max_pixels,
        image_min_pixels=args.image_min_pixels,
        temperature=args.temperature,
        top_p=args.top_p,
        n=args.n,
        max_tokens=args.max_tokens,
        seed=args.seed,
    )

    compute_score = load_reward_function(args.reward_file.resolve())
    for row in selected_rows:
        info = row["extra_info"]
        sample_id = str(info["sample_id"])
        progress_path = score_progress_path(progress_dir, sample_id)
        if progress_path.exists():
            continue

        raw_path = generation_path(raw_dir, sample_id)
        if not raw_path.is_file():
            raise FileNotFoundError(f"missing generation output for {sample_id}: {raw_path}")
        generation = json.loads(raw_path.read_text(encoding="utf-8"))
        responses = generation["responses"]
        if len(responses) != args.n:
            raise ValueError(f"{sample_id} has {len(responses)} responses, expected {args.n}")

        with Image.open(row["images"][0]) as source:
            original = source.convert("RGB")
        original_relative_path = Path("renders") / f"{sample_id}_original.jpg"
        center_relative_path = Path("renders") / f"{sample_id}_center.jpg"
        save_preview(original, args.output_dir / original_relative_path)
        center_bbox = action_to_bbox(
            CropAction(center_x=500, center_y=500, area=1000),
            image_width=info["image_width"],
            image_height=info["image_height"],
            target_ratio=float(info["target_ratio"]),
        )
        save_preview(crop_image(original, center_bbox), args.output_dir / center_relative_path)
        center_result = compute_score(
            data_source=row["data_source"],
            solution_str='<crop>{"cx":500,"cy":500,"area":1000}</crop>',
            ground_truth=row["reward_model"]["ground_truth"],
            extra_info=info,
            reward_mode="proxy",
            clip_model_path=str(args.clip_model_path),
            clip_device=args.clip_device,
        )
        center_baseline = {
            "score": float(center_result["score"]),
            "render_path": center_relative_path.as_posix(),
            "original_render_path": original_relative_path.as_posix(),
        }
        sample_details = []
        for candidate_index, response in enumerate(responses):
            try:
                parse_result = parse_crop_action_with_format(response)
                action = parse_result.action
                action_dict = {"cx": action.center_x, "cy": action.center_y, "area": action.area}
                valid = True
                strict_format = parse_result.strict_format
                bbox = action_to_bbox(
                    action,
                    image_width=info["image_width"],
                    image_height=info["image_height"],
                    target_ratio=float(info["target_ratio"]),
                )
                render_relative_path = Path("renders") / f"{sample_id}_candidate_{candidate_index}.jpg"
                save_preview(crop_image(original, bbox), args.output_dir / render_relative_path)
            except ValueError:
                action_dict = None
                valid = False
                strict_format = False
                render_relative_path = None
            result = compute_score(
                data_source=row["data_source"],
                solution_str=response,
                ground_truth=row["reward_model"]["ground_truth"],
                extra_info=info,
                reward_mode="proxy",
                clip_model_path=str(args.clip_model_path),
                clip_device=args.clip_device,
            )
            sample_details.append(
                {
                    "sample_id": sample_id,
                    "title": info["title"],
                    "target_ratio": float(info["target_ratio"]),
                    "image_path": row["images"][0],
                    "candidate_index": candidate_index,
                    "response": response,
                    "valid": valid,
                    "strict_format": strict_format,
                    "action": action_dict,
                    "render_path": render_relative_path.as_posix() if render_relative_path else None,
                    **{key: float(value) for key, value in result.items()},
                }
            )
        original.close()
        write_json_atomic(
            progress_path,
            {
                "sample_id": sample_id,
                "center_baseline": center_baseline,
                "details": sample_details,
            },
        )

    details = []
    center_scores = {}
    center_baselines = {}
    for row in selected_rows:
        sample_id = str(row["extra_info"]["sample_id"])
        progress = json.loads(score_progress_path(progress_dir, sample_id).read_text(encoding="utf-8"))
        details.extend(progress["details"])
        center_baselines[sample_id] = progress["center_baseline"]
        center_scores[sample_id] = float(progress["center_baseline"]["score"])
    details.sort(key=lambda item: (item["sample_id"], item["candidate_index"]))

    summary = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "groups": len(selected_rows) // 4,
        "prompts": len(selected_rows),
        "candidates_per_prompt": args.n,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "clip_device": args.clip_device,
        "data_parallel_size": args.data_parallel_size,
        "tensor_parallel_size": args.tensor_parallel_size,
        **summarize(details, center_scores),
    }
    (args.output_dir / "details.jsonl").write_text(
        "".join(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n" for detail in details),
        encoding="utf-8",
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    render_html_report(args.output_dir, summary, selected_rows, details, center_baselines)
    write_json_atomic(
        args.output_dir / "_SHARD_COMPLETE.json",
        {
            "run_id": args.run_id,
            "shard_count": args.shard_count,
            "shard_index": args.shard_index,
            "prompts": len(selected_rows),
            "candidates": len(details),
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
