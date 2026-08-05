#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean

import pyarrow.parquet as pq
import yaml
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

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


def select_complete_groups(data_path: Path, group_count: int, seed: int) -> list[dict]:
    rows = pq.read_table(data_path).to_pylist()
    grouped: dict[tuple[str, str], dict[float, dict]] = defaultdict(dict)
    for row in rows:
        info = row["extra_info"]
        grouped[(row["images"][0], info["title"])][float(info["target_ratio"])] = row

    complete_groups = [group for group in grouped.values() if len(group) == 4]
    if group_count > len(complete_groups):
        raise ValueError(f"requested {group_count} groups, but only {len(complete_groups)} complete groups exist")
    selected = random.Random(seed).sample(complete_groups, group_count)
    return [group[ratio] for group in selected for ratio in sorted(group)]


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
    parser.add_argument("--groups", type=int, default=4)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--max-model-len", type=int, default=2176)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--image-max-pixels", type=int, default=1048576)
    parser.add_argument("--image-min-pixels", type=int, default=65536)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.60)
    args = parser.parse_args()
    if args.groups <= 0 or args.n <= 0:
        raise ValueError("groups and n must be positive")

    selected_rows = select_complete_groups(args.data, args.groups, args.seed)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    images = []
    requests = []
    for row in selected_rows:
        with Image.open(row["images"][0]) as source:
            image = source.convert("RGB")
        images.append(image)
        requests.append(
            build_vllm_request(processor, row, image, args.image_max_pixels, args.image_min_pixels)
        )

    asset_root = Path("/mnt/blob_output/v-yukunban/news_image_crop_assets/original")
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=1,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        max_num_seqs=max(16, len(selected_rows) * args.n),
        enforce_eager=True,
        allowed_local_media_path=str(asset_root),
        limit_mm_per_prompt={"image": 1},
    )
    outputs = llm.generate(
        requests,
        sampling_params=SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            n=args.n,
            max_tokens=args.max_tokens,
            seed=args.seed,
        ),
    )

    compute_score = load_reward_function(args.reward_file.resolve())
    args.output_dir.mkdir(parents=True, exist_ok=True)
    render_dir = args.output_dir / "renders"
    details = []
    center_scores = {}
    center_baselines = {}
    for row, original, request_output in zip(selected_rows, images, outputs, strict=True):
        info = row["extra_info"]
        sample_id = info["sample_id"]
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
        center_scores[sample_id] = float(center_result["score"])
        center_baselines[sample_id] = {
            "score": float(center_result["score"]),
            "render_path": center_relative_path.as_posix(),
            "original_render_path": original_relative_path.as_posix(),
        }
        for candidate_index, output in enumerate(request_output.outputs):
            try:
                parse_result = parse_crop_action_with_format(output.text)
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
                solution_str=output.text,
                ground_truth=row["reward_model"]["ground_truth"],
                extra_info=info,
                reward_mode="proxy",
                clip_model_path=str(args.clip_model_path),
                clip_device=args.clip_device,
            )
            details.append(
                {
                    "sample_id": sample_id,
                    "title": info["title"],
                    "target_ratio": float(info["target_ratio"]),
                    "image_path": row["images"][0],
                    "candidate_index": candidate_index,
                    "response": output.text,
                    "valid": valid,
                    "strict_format": strict_format,
                    "action": action_dict,
                    "render_path": render_relative_path.as_posix() if render_relative_path else None,
                    **{key: float(value) for key, value in result.items()},
                }
            )

    summary = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "groups": args.groups,
        "prompts": len(selected_rows),
        "candidates_per_prompt": args.n,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "clip_device": args.clip_device,
        **summarize(details, center_scores),
    }
    (args.output_dir / "details.jsonl").write_text(
        "".join(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n" for detail in details)
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    config = {
        "model": str(args.model.resolve()),
        "data": str(args.data.resolve()),
        "groups": args.groups,
        "n": args.n,
        "seed": args.seed,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "image_max_pixels": args.image_max_pixels,
        "image_min_pixels": args.image_min_pixels,
        "clip_model_path": str(args.clip_model_path.resolve()),
        "clip_device": args.clip_device,
    }
    (args.output_dir / "baseline_config.yaml").write_text(
        yaml.safe_dump(config, allow_unicode=True, sort_keys=True)
    )
    (args.output_dir / "sample_manifest.jsonl").write_text(
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
        )
    )
    with (args.output_dir / "human_review.csv").open("w", newline="", encoding="utf-8-sig") as output:
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
    render_html_report(args.output_dir, summary, selected_rows, details, center_baselines)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
