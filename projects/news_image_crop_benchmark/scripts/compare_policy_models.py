#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

COMPATIBILITY_KEYS = (
    "data_sha256",
    "policy_prompt_sha256",
    "vlm_prompt_sha256",
    "output_protocol_version",
    "target_ratios",
    "max_attempts",
    "seed",
    "temperature",
    "top_p",
    "max_tokens",
    "max_images",
    "max_model_len",
    "image_max_pixels",
    "image_min_pixels",
    "internvl_max_dynamic_patch",
    "judge_config",
)


def parse_run(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise ValueError(f"run must use NAME=PATH: {value}")
    name, path_text = value.split("=", 1)
    if not name or not path_text:
        raise ValueError(f"run must use non-empty NAME=PATH: {value}")
    return name, Path(path_text).expanduser().resolve()


def load_run(name: str, path: Path) -> dict[str, Any]:
    config_path = path / "run_config.yaml"
    details_path = path / "details.jsonl"
    complete_path = path / "_EVAL_COMPLETE.json"
    for required_path in (config_path, details_path, complete_path):
        if not required_path.is_file():
            raise FileNotFoundError(required_path)

    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    details = {}
    for line in details_path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        detail = json.loads(line)
        task_id = str(detail["task_id"])
        if task_id in details:
            raise ValueError(f"duplicate task_id in {name}: {task_id}")
        details[task_id] = detail
    return {"name": name, "path": path, "config": config, "details": details}


def validate_compatible_runs(runs: Sequence[dict[str, Any]]) -> list[str]:
    if len(runs) < 2:
        raise ValueError("at least two runs are required")
    reference = runs[0]
    for run in runs[1:]:
        for key in COMPATIBILITY_KEYS:
            if run["config"].get(key) != reference["config"].get(key):
                raise ValueError(
                    f"incompatible {key}: {reference['name']}={reference['config'].get(key)!r}, "
                    f"{run['name']}={run['config'].get(key)!r}"
                )
        if set(run["details"]) != set(reference["details"]):
            missing = sorted(set(reference["details"]) - set(run["details"]))
            unexpected = sorted(set(run["details"]) - set(reference["details"]))
            raise ValueError(
                f"task coverage differs for {run['name']}: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )
    return sorted(reference["details"])


def outcome(left: dict[str, Any], right: dict[str, Any]) -> str:
    left_valid = left.get("judge_status") == "completed"
    right_valid = right.get("judge_status") == "completed"
    if not left_valid or not right_valid:
        return "unscored"
    left_label = float(left["judge_label"])
    right_label = float(right["judge_label"])
    if left_label < right_label:
        return "win"
    if left_label > right_label:
        return "loss"
    return "tie"


def build_paired_rows(runs: Sequence[dict[str, Any]], task_ids: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    reference_name = runs[0]["name"]
    for task_id in task_ids:
        reference_detail = runs[0]["details"][task_id]
        row: dict[str, Any] = {
            "task_id": task_id,
            "image_id": reference_detail["image_id"],
            "title": reference_detail["title"],
            "target_ratio": float(reference_detail["target_ratio"]),
        }
        for run in runs:
            detail = run["details"][task_id]
            prefix = run["name"]
            row[f"{prefix}_generation_status"] = detail["generation_status"]
            row[f"{prefix}_invalid_attempt_count"] = detail["invalid_attempt_count"]
            row[f"{prefix}_strict_format"] = detail.get("strict_format")
            row[f"{prefix}_response_normalized"] = detail.get("response_normalized")
            row[f"{prefix}_judge_status"] = detail["judge_status"]
            row[f"{prefix}_judge_label"] = detail.get("judge_label")
            row[f"{prefix}_judge_reward"] = detail.get("judge_reward")
            row[f"{prefix}_candidate_path"] = detail.get("candidate_path")
            if run is not runs[0]:
                row[f"{prefix}_vs_{reference_name}"] = outcome(detail, reference_detail)
        rows.append(row)
    return rows


def summarize_run(run: dict[str, Any], task_ids: Sequence[str]) -> dict[str, Any]:
    details = [run["details"][task_id] for task_id in task_ids]
    completed = [detail for detail in details if detail["judge_status"] == "completed"]
    labels = [float(detail["judge_label"]) for detail in completed]
    tier_counts = Counter(str(int(label)) if label.is_integer() else str(label) for label in labels)
    return {
        "tasks": len(details),
        "generation_success_rate": mean(detail["generation_status"] == "valid" for detail in details),
        "had_invalid_output_rate": mean(detail["had_invalid_output"] for detail in details),
        "invalid_output_count": sum(detail["invalid_attempt_count"] for detail in details),
        "strict_format_rate": mean(detail.get("strict_format", False) for detail in details),
        "response_normalized_rate": mean(
            detail.get("response_normalized", False) for detail in details
        ),
        "judge_completed_rate": len(completed) / len(details) if details else 0.0,
        "mean_judge_label": mean(labels) if labels else None,
        "tier_0_1_acceptable_rate": mean(label <= 1 for label in labels) if labels else None,
        "tier_3_5_severe_rate": mean(label >= 3 for label in labels) if labels else None,
        "tier_counts": dict(sorted(tier_counts.items())),
    }


def summarize_comparison(
    runs: Sequence[dict[str, Any]],
    task_ids: Sequence[str],
    paired_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    reference_name = runs[0]["name"]

    def summarize_scope(scope_ids: Sequence[str]) -> dict[str, Any]:
        scope_id_set = set(scope_ids)
        scope_rows = [row for row in paired_rows if row["task_id"] in scope_id_set]
        metrics: dict[str, Any] = {
            "models": {run["name"]: summarize_run(run, scope_ids) for run in runs}
        }
        for run in runs[1:]:
            key = f"{run['name']}_vs_{reference_name}"
            counts = Counter(row[key] for row in scope_rows)
            scored = counts["win"] + counts["tie"] + counts["loss"]
            metrics[key] = {
                "win": counts["win"],
                "tie": counts["tie"],
                "loss": counts["loss"],
                "unscored": counts["unscored"],
                "win_rate": counts["win"] / scored if scored else None,
                "tie_rate": counts["tie"] / scored if scored else None,
                "loss_rate": counts["loss"] / scored if scored else None,
            }
        return metrics

    ratios = sorted({float(row["target_ratio"]) for row in paired_rows})
    return {
        "reference_model": reference_name,
        "runs": {
            run["name"]: {
                "path": str(run["path"]),
                "model_name": run["config"].get("model_name"),
                "model_family": run["config"].get("model_family"),
                "model": run["config"].get("model"),
                "model_config_sha256": run["config"].get("model_config_sha256"),
                "model_index_sha256": run["config"].get("model_index_sha256"),
            }
            for run in runs
        },
        "overall": summarize_scope(task_ids),
        "by_ratio": {
            f"{ratio:g}": summarize_scope(
                [row["task_id"] for row in paired_rows if row["target_ratio"] == ratio]
            )
            for ratio in ratios
        },
    }


def write_outputs(
    runs: Sequence[dict[str, Any]],
    paired_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "paired_details.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in paired_rows),
        encoding="utf-8",
    )
    pq.write_table(pa.Table.from_pylist(paired_rows), output_dir / "paired_details.parquet", compression="zstd")
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=list(paired_rows[0]))
        writer.writeheader()
        writer.writerows(paired_rows)

    headers = "".join(f"<th>{html.escape(run['name'])}</th>" for run in runs)
    model_rows = []
    for metric in (
        "generation_success_rate",
        "had_invalid_output_rate",
        "invalid_output_count",
        "strict_format_rate",
        "response_normalized_rate",
        "judge_completed_rate",
        "mean_judge_label",
        "tier_0_1_acceptable_rate",
        "tier_3_5_severe_rate",
        "tier_counts",
    ):
        cells = "".join(
            f"<td>{html.escape(str(summary['overall']['models'][run['name']][metric]))}</td>"
            for run in runs
        )
        model_rows.append(f"<tr><th>{html.escape(metric)}</th>{cells}</tr>")
    pair_rows = "".join(
        f"<tr><th>{html.escape(key)}</th><td colspan='{len(runs)}'>{html.escape(str(value))}</td></tr>"
        for key, value in summary["overall"].items()
        if key != "models"
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Policy Model Crop Comparison</title>
<style>body{{font-family:"Segoe UI",sans-serif;margin:24px;color:#17201f}}table{{border-collapse:collapse;width:100%;max-width:1200px}}th,td{{border:1px solid #d6ddda;padding:8px;text-align:left}}th{{background:#eef2f0}}pre{{white-space:pre-wrap}}</style>
</head><body><h1>Policy Model Crop Comparison</h1>
<p>Reference model: {html.escape(summary['reference_model'])}</p>
<table><thead><tr><th>Metric</th>{headers}</tr></thead><tbody>{''.join(model_rows)}{pair_rows}</tbody></table>
<h2>By Ratio</h2><pre>{html.escape(json.dumps(summary['by_ratio'], ensure_ascii=False, indent=2))}</pre>
</body></html>"""
    (output_dir / "comparison.html").write_text(document, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare completed crop evaluation runs by task_id.")
    parser.add_argument("--run", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    runs = [load_run(*parse_run(value)) for value in args.run]
    task_ids = validate_compatible_runs(runs)
    paired_rows = build_paired_rows(runs, task_ids)
    summary = summarize_comparison(runs, task_ids, paired_rows)
    write_outputs(runs, paired_rows, summary, args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()