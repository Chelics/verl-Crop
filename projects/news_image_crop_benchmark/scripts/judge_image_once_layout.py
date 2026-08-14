#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image, ImageOps

from news_crop_benchmark.geometry import TARGET_RATIOS
from news_crop_benchmark.vlm_scorer import CropVLMScorer

COUNTED_JUDGE_STATUS = "completed"


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def load_layout_details(layout_results_dir: Path) -> list[dict[str, Any]]:
    completion_path = layout_results_dir / "_LAYOUT_PIPELINE_COMPLETE.json"
    details_path = layout_results_dir / "details.jsonl"
    if not completion_path.is_file() or not details_path.is_file():
        raise FileNotFoundError(f"layout results are incomplete: {layout_results_dir}")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    details = [json.loads(line) for line in details_path.read_text(encoding="utf-8").splitlines()]
    if int(completion.get("tasks", -1)) != len(details):
        raise ValueError("layout completion task count does not match details.jsonl")
    task_ids = [str(detail["task_id"]) for detail in details]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("layout details contain duplicate task IDs")
    for detail in details:
        if detail.get("predicted_mode") not in {"crop", "pad"}:
            raise ValueError(f"task has no valid layout mode: {detail['task_id']}")
        if detail.get("render_status") != "rendered" or not detail.get("candidate_path"):
            raise ValueError(f"task has no rendered candidate: {detail['task_id']}")
        source_path = layout_results_dir / ".source_images" / f"{detail['image_id']}.webp"
        candidate_path = layout_results_dir / str(detail["candidate_path"])
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        if not candidate_path.is_file():
            raise FileNotFoundError(candidate_path)
    return details


def judge_progress_path(
    layout_results_dir: Path,
    task_id: str,
    judge_id: str = "layout_judge_v1",
) -> Path:
    return layout_results_dir / "progress" / judge_id / f"{task_id}.json"


def parse_judge_metadata(output_text: str | None) -> dict[str, Any]:
    default = {
        "rules": [],
        "confidence_score": None,
        "tier_name": None,
        "mode_appropriateness": None,
        "layout_relationship": None,
    }
    if not output_text:
        return default
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", output_text):
        try:
            payload, _ = decoder.raw_decode(output_text[match.start() :])
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict) or not isinstance(payload.get("evaluation"), dict):
            continue
        evaluation = payload["evaluation"]
        comparison = payload.get("comparison", {})
        rules = evaluation.get("rules", [])
        return {
            "rules": [str(rule) for rule in rules] if isinstance(rules, list) else [],
            "confidence_score": evaluation.get("confidence_score"),
            "tier_name": evaluation.get("tier_name"),
            "mode_appropriateness": evaluation.get("mode_appropriateness"),
            "layout_relationship": (
                comparison.get("layout_relationship") if isinstance(comparison, dict) else None
            ),
        }
    return default


def run_judge(
    details: Sequence[dict[str, Any]],
    layout_results_dir: Path,
    prompt_path: Path,
    judge_workers: int,
    judge_id: str = "layout_judge_v1",
    response_log_name: str = "layout_judge_responses.jsonl",
    include_evaluation_context: bool = True,
) -> None:
    os.environ["CROP_VLM_LOG_PATH"] = str(layout_results_dir / response_log_name)
    thread_state = threading.local()

    def score_task(detail: dict[str, Any]) -> None:
        progress_path = judge_progress_path(layout_results_dir, detail["task_id"], judge_id)
        if progress_path.exists():
            try:
                existing = json.loads(progress_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                existing = {}
            if existing.get("status") in {"completed", "parse_fallback"}:
                return
        source_path = layout_results_dir / ".source_images" / f"{detail['image_id']}.webp"
        candidate_path = layout_results_dir / str(detail["candidate_path"])
        with Image.open(source_path) as source:
            original = ImageOps.exif_transpose(source).convert("RGB")
        with Image.open(candidate_path) as source:
            candidate = source.convert("RGB")
        if not hasattr(thread_state, "scorer"):
            thread_state.scorer = CropVLMScorer(str(prompt_path))
        evaluation_context = {
            "target_ratio": detail["target_ratio"],
            "selected_mode": detail["predicted_mode"],
            "background_hex": detail.get("background_hex"),
            "padding_fraction": detail.get("padding_fraction"),
            "crop_action": (
                {
                    "cx_pct": detail.get("cx_pct"),
                    "cy_pct": detail.get("cy_pct"),
                    "area_pct": detail.get("area_pct"),
                }
                if detail["predicted_mode"] == "crop"
                else None
            ),
        }
        result = thread_state.scorer.score_detailed(
            original,
            candidate,
            str(detail.get("caption", "")),
            str(detail["title"]),
            log_context={
                "task_id": detail["task_id"],
                "sample_id": detail["image_id"],
                "target_ratio": detail["target_ratio"],
                "selected_mode": detail["predicted_mode"],
            },
            evaluation_context=evaluation_context if include_evaluation_context else None,
        )
        original.close()
        candidate.close()
        write_json_atomic(
            progress_path,
            {
                "task_id": detail["task_id"],
                "status": result.status,
                "label": result.label,
                "reward": result.reward,
                "output_text": result.output_text,
                "response_id": result.response_id,
                "request_attempt_count": result.attempt_count,
                "latency_ms": result.latency_ms,
                "error_type": result.error_type,
                **parse_judge_metadata(result.output_text),
            },
        )

    with ThreadPoolExecutor(max_workers=judge_workers) as executor:
        list(executor.map(score_task, details))


def build_judged_details(
    details: Sequence[dict[str, Any]],
    layout_results_dir: Path,
    judge_id: str = "layout_judge_v1",
) -> list[dict[str, Any]]:
    judged = []
    for detail in details:
        judge = json.loads(
            judge_progress_path(layout_results_dir, detail["task_id"], judge_id).read_text(encoding="utf-8")
        )
        judged.append(
            {
                **detail,
                "judge_status": judge["status"],
                "judge_label": judge.get("label"),
                "judge_reward": judge.get("reward"),
                "judge_rules": judge.get("rules", []),
                "judge_tier_name": judge.get("tier_name"),
                "judge_confidence_score": judge.get("confidence_score"),
                "judge_mode_appropriateness": judge.get("mode_appropriateness"),
                "judge_layout_relationship": judge.get("layout_relationship"),
                "judge_latency_ms": judge.get("latency_ms"),
                "judge_error_type": judge.get("error_type"),
                "judge_output_text": judge.get("output_text"),
            }
        )
    return judged


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
    counted = [detail for detail in details if detail["judge_status"] == COUNTED_JUDGE_STATUS]
    labels = [float(detail["judge_label"]) for detail in counted]
    rewards = [float(detail["judge_reward"]) for detail in counted]
    latencies = [float(detail["judge_latency_ms"]) for detail in counted]
    tier_counts = Counter(str(int(label)) if label.is_integer() else str(label) for label in labels)
    rule_counts = Counter(rule for detail in counted for rule in detail["judge_rules"])
    appropriateness = [detail for detail in counted if detail["judge_mode_appropriateness"] is not None]
    relationships = [detail for detail in counted if detail["judge_layout_relationship"] is not None]
    appropriateness_counts = Counter(str(detail["judge_mode_appropriateness"]) for detail in appropriateness)
    relationship_counts = Counter(str(detail["judge_layout_relationship"]) for detail in relationships)
    return {
        "tasks": len(details),
        "judge_completed_count": len(counted),
        "judge_completed_rate": len(counted) / len(details) if details else 0.0,
        "judge_parse_fallback_count": sum(
            detail["judge_status"] == "parse_fallback" for detail in details
        ),
        "judge_failed_count": sum(detail["judge_status"] == "failed" for detail in details),
        "tier_counts": dict(sorted(tier_counts.items())),
        "rule_counts": dict(sorted(rule_counts.items())),
        "mode_appropriateness_counts": dict(sorted(appropriateness_counts.items())),
        "layout_relationship_counts": dict(sorted(relationship_counts.items())),
        "mean_judge_label": mean(labels) if labels else None,
        "mean_judge_reward": mean(rewards) if rewards else None,
        "tier_0_1_acceptable_rate": mean(label <= 1 for label in labels) if labels else None,
        "tier_3_5_severe_rate": mean(label >= 3 for label in labels) if labels else None,
        "mode_appropriate_rate": (
            mean(detail["judge_mode_appropriateness"] == "appropriate" for detail in appropriateness)
            if appropriateness
            else None
        ),
        "mode_inappropriate_rate": (
            mean(detail["judge_mode_appropriateness"] == "inappropriate" for detail in appropriateness)
            if appropriateness
            else None
        ),
        "judge_latency_ms_mean": mean(latencies) if latencies else None,
        "judge_latency_ms_p50": percentile(latencies, 0.50),
        "judge_latency_ms_p95": percentile(latencies, 0.95),
    }


def summarize(details: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "overall": summarize_subset(details),
        "by_mode": {
            mode: summarize_subset([detail for detail in details if detail["predicted_mode"] == mode])
            for mode in ("crop", "pad")
        },
        "by_ratio": {
            f"{ratio:g}": summarize_subset(
                [detail for detail in details if detail["target_ratio"] == ratio]
            )
            for ratio in TARGET_RATIOS
        },
        "by_mode_and_ratio": {
            mode: {
                f"{ratio:g}": summarize_subset(
                    [
                        detail
                        for detail in details
                        if detail["predicted_mode"] == mode and detail["target_ratio"] == ratio
                    ]
                )
                for ratio in TARGET_RATIOS
            }
            for mode in ("crop", "pad")
        },
    }


def write_results(
    details: list[dict[str, Any]],
    summary: dict[str, Any],
    layout_results_dir: Path,
    output_prefix: str = "judge",
) -> None:
    details.sort(key=lambda detail: (detail["source_index"], TARGET_RATIOS.index(detail["target_ratio"])))
    (layout_results_dir / f"{output_prefix}_details.jsonl").write_text(
        "".join(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n" for detail in details),
        encoding="utf-8",
    )
    parquet_rows = [
        {
            **detail,
            "judge_rules": json.dumps(detail["judge_rules"], ensure_ascii=False),
            "background_color": json.dumps(detail.get("background_color")),
            "content_box": json.dumps(detail.get("content_box")),
        }
        for detail in details
    ]
    pq.write_table(
        pa.Table.from_pylist(parquet_rows),
        layout_results_dir / f"{output_prefix}_details.parquet",
        compression="zstd",
    )
    (layout_results_dir / f"{output_prefix}_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with (layout_results_dir / f"{output_prefix}_summary.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as output:
        rows = [("overall", summary["overall"])]
        rows.extend((f"mode:{mode}", metrics) for mode, metrics in summary["by_mode"].items())
        rows.extend((f"ratio:{ratio}", metrics) for ratio, metrics in summary["by_ratio"].items())
        scalar_keys = [key for key, value in summary["overall"].items() if not isinstance(value, dict)]
        writer = csv.DictWriter(output, fieldnames=["scope", *scalar_keys, "tier_counts"])
        writer.writeheader()
        for scope, metrics in rows:
            writer.writerow(
                {
                    "scope": scope,
                    **{key: metrics[key] for key in scalar_keys},
                    "tier_counts": json.dumps(metrics["tier_counts"], sort_keys=True),
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


def render_html_report(
    details: list[dict[str, Any]],
    summary: dict[str, Any],
    layout_results_dir: Path,
    output_prefix: str = "judge",
    report_title: str = "Crop-or-Pad Layout Judge",
) -> None:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for detail in details:
        by_image[detail["image_id"]].append(detail)
    metric_rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(_format(value))}</td></tr>"
        for key, value in summary["overall"].items()
    )
    mode_rows = "".join(
        f"<tr><th>{mode}</th><td>{metrics['tasks']}</td><td>{_format(metrics['mean_judge_label'])}</td>"
        f"<td>{_format(metrics['tier_0_1_acceptable_rate'])}</td><td>{_format(metrics['tier_3_5_severe_rate'])}</td>"
        f"<td>{_format(metrics['mode_appropriate_rate'])}</td><td>{_format(metrics['mode_inappropriate_rate'])}</td></tr>"
        for mode, metrics in summary["by_mode"].items()
    )
    sections = []
    for image_details in sorted(by_image.values(), key=lambda group: group[0]["source_index"]):
        image_details.sort(key=lambda detail: TARGET_RATIOS.index(detail["target_ratio"]))
        first = image_details[0]
        cards = []
        for detail in image_details:
            mode = detail["predicted_mode"]
            cards.append(
                f"""
                <article class="candidate {html.escape(mode)}">
                  <h3>Ratio {detail['target_ratio']:g} · {html.escape(mode.upper())}</h3>
                  <img src="{html.escape(str(detail['candidate_path']))}" alt="Judged layout">
                  <p class="tier">Tier {html.escape(str(detail['judge_label']))} · {html.escape(str(detail['judge_tier_name']))}</p>
                  <p><strong>Mode:</strong> {html.escape(str(detail['judge_mode_appropriateness']))}</p>
                  <p><strong>Rules:</strong> {html.escape(', '.join(detail['judge_rules']))}</p>
                  <details><summary>Judge response</summary><pre>{html.escape(str(detail['judge_output_text']))}</pre></details>
                </article>
                """
            )
        sections.append(
            f"""
            <section class="sample">
              <header><h2>{html.escape(first['title'])}</h2><p>{html.escape(first['image_id'])}</p></header>
              <div class="layout">
                <article class="original"><h3>Original</h3><img src="{html.escape(first['original_render_path'])}" alt="Original"></article>
                <div class="candidates">{''.join(cards)}</div>
              </div>
            </section>
            """
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(report_title)}</title><style>
:root {{ --ink:#202522; --line:#ccd4cf; --paper:#edf1ee; --crop:#176b4d; --pad:#9a5418; }} * {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--paper); font-family:"Segoe UI",sans-serif; }} main {{ width:min(1640px,96vw); margin:auto; padding:24px 0 64px; }}
.summary,.sample {{ background:white; border:1px solid var(--line); padding:20px; margin-bottom:20px; }} table {{ border-collapse:collapse; width:min(1000px,100%); margin-bottom:24px; }} th,td {{ border-bottom:1px solid var(--line); padding:7px 10px; text-align:left; }}
.layout {{ display:grid; grid-template-columns:minmax(260px,.8fr) minmax(0,2.2fr); gap:18px; }} .candidates {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }}
.candidate {{ border:1px solid var(--line); border-top:5px solid var(--line); padding:12px; }} .candidate.crop {{ border-top-color:var(--crop); }} .candidate.pad {{ border-top-color:var(--pad); }} img {{ display:block; width:100%; max-height:540px; object-fit:contain; background:#e6ebe8; }} .candidate img {{ height:340px; }}
.tier {{ font-size:18px; font-weight:700; }} p,pre {{ overflow-wrap:anywhere; }} pre {{ white-space:pre-wrap; font-size:12px; }} @media(max-width:900px) {{ .layout {{ grid-template-columns:1fr; }} .candidates {{ grid-template-columns:1fr; }} }}
</style></head><body><main><section class="summary"><h1>{html.escape(report_title)}</h1><table>{metric_rows}</table>
<h2>By Mode</h2><table><tr><th>Mode</th><th>Tasks</th><th>Mean Tier</th><th>Tier 0-1</th><th>Tier 3-5</th><th>Appropriate</th><th>Inappropriate</th></tr>{mode_rows}</table></section>
{''.join(sections)}</main></body></html>"""
    (layout_results_dir / f"{output_prefix}_report.html").write_text(document, encoding="utf-8")


def render_markdown_report(
    details: list[dict[str, Any]],
    summary: dict[str, Any],
    layout_results_dir: Path,
    output_prefix: str = "judge",
    report_title: str = "Crop-or-Pad Layout Judge",
) -> None:
    overall = summary["overall"]
    lines = [
        f"# {report_title}",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in (
        "tasks",
        "judge_completed_count",
        "judge_completed_rate",
        "judge_parse_fallback_count",
        "judge_failed_count",
        "mean_judge_label",
        "mean_judge_reward",
        "tier_0_1_acceptable_rate",
        "tier_3_5_severe_rate",
        "mode_appropriate_rate",
        "mode_inappropriate_rate",
    ):
        lines.append(f"| `{key}` | {_format(overall[key])} |")
    lines.extend(
        [
            "",
            "## By Mode",
            "",
            "| Mode | Tasks | Mean Tier | Tier 0-1 | Tier 3-5 | Appropriate | Inappropriate |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for mode, metrics in summary["by_mode"].items():
        lines.append(
            f"| {mode} | {metrics['tasks']} | {_format(metrics['mean_judge_label'])} | "
            f"{_format(metrics['tier_0_1_acceptable_rate'])} | {_format(metrics['tier_3_5_severe_rate'])} | "
            f"{_format(metrics['mode_appropriate_rate'])} | {_format(metrics['mode_inappropriate_rate'])} |"
        )
    lines.extend(
        [
            "",
            "## By Ratio",
            "",
            "| Ratio | Tasks | Mean Tier | Tier 0-1 | Tier 3-5 |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for ratio in TARGET_RATIOS:
        metrics = summary["by_ratio"][f"{ratio:g}"]
        lines.append(
            f"| {ratio:g} | {metrics['tasks']} | {_format(metrics['mean_judge_label'])} | "
            f"{_format(metrics['tier_0_1_acceptable_rate'])} | {_format(metrics['tier_3_5_severe_rate'])} |"
        )
    lines.extend(["", "## Tier Distribution", "", "| Tier | Count |", "|---:|---:|"])
    lines.extend(f"| {tier} | {count} |" for tier, count in overall["tier_counts"].items())
    lines.extend(["", "## Most Frequent Rules", "", "| Rule | Count |", "|---|---:|"])
    for rule, count in sorted(overall["rule_counts"].items(), key=lambda item: (-item[1], item[0])):
        lines.append(f"| `{rule}` | {count} |")
    (layout_results_dir / f"{output_prefix}_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Judge completed crop-or-pad layout results.")
    parser.add_argument("--layout-results-dir", type=Path, required=True)
    parser.add_argument("--vlm-prompt-path", type=Path, required=True)
    parser.add_argument("--judge-workers", type=int, default=2)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--judge-id", default="layout_judge_v1")
    parser.add_argument("--output-prefix", default="judge")
    parser.add_argument("--response-log-name", default="layout_judge_responses.jsonl")
    parser.add_argument("--completion-name", default="_LAYOUT_JUDGE_COMPLETE.json")
    parser.add_argument("--report-title", default="Crop-or-Pad Layout Judge")
    parser.add_argument("--omit-evaluation-context", action="store_true")
    args = parser.parse_args()
    if args.judge_workers <= 0:
        raise ValueError("judge-workers must be positive")
    safe_name = re.compile(r"^[A-Za-z0-9_.-]+$")
    for name, value in (
        ("judge-id", args.judge_id),
        ("output-prefix", args.output_prefix),
        ("response-log-name", args.response_log_name),
        ("completion-name", args.completion_name),
    ):
        if not safe_name.fullmatch(value):
            raise ValueError(f"{name} contains unsupported characters")
    return args


def main() -> None:
    args = parse_args()
    if not args.layout_results_dir.is_dir():
        raise FileNotFoundError(args.layout_results_dir)
    if not args.vlm_prompt_path.is_file():
        raise FileNotFoundError(args.vlm_prompt_path)
    layout_results_dir = args.layout_results_dir.resolve()
    details = load_layout_details(layout_results_dir)
    run_judge(
        details,
        layout_results_dir,
        args.vlm_prompt_path,
        args.judge_workers,
        judge_id=args.judge_id,
        response_log_name=args.response_log_name,
        include_evaluation_context=not args.omit_evaluation_context,
    )
    judged_details = build_judged_details(details, layout_results_dir, judge_id=args.judge_id)
    summary = {
        "run_id": args.run_id,
        "layout_results_dir": str(layout_results_dir),
        "vlm_prompt_path": str(args.vlm_prompt_path.resolve()),
        "judge_model": os.getenv(
            "CROP_VLM_MODEL",
            os.getenv("GPT5_AZURE_OPENAI_DEPLOYMENT", "gpt-5.6-sol"),
        ),
        "judge_id": args.judge_id,
        "evaluation_context_included": not args.omit_evaluation_context,
        **summarize(judged_details),
    }
    write_results(judged_details, summary, layout_results_dir, output_prefix=args.output_prefix)
    render_html_report(
        judged_details,
        summary,
        layout_results_dir,
        output_prefix=args.output_prefix,
        report_title=args.report_title,
    )
    render_markdown_report(
        judged_details,
        summary,
        layout_results_dir,
        output_prefix=args.output_prefix,
        report_title=args.report_title,
    )
    write_json_atomic(
        layout_results_dir / args.completion_name,
        {
            "run_id": args.run_id,
            "tasks": len(judged_details),
            "judge_completed_count": summary["overall"]["judge_completed_count"],
            "judge_failed_count": summary["overall"]["judge_failed_count"],
            "judge_parse_fallback_count": summary["overall"]["judge_parse_fallback_count"],
        },
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()