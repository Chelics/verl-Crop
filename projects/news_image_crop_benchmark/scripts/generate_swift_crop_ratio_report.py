#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
import shutil
from collections import Counter
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image

from export_swift_crop_renders import image_reference, load_source_image, normalized_pixel_box, parse_response


SEVERITY_ORDER = ("negligible", "small", "noticeable", "large", "severe")
SEVERITY_LABELS = {
    "negligible": "Negligible",
    "small": "Small",
    "noticeable": "Noticeable",
    "large": "Large",
    "severe": "Severe",
}
DETAIL_FIELDS = (
    "image_id",
    "target_ratio",
    "actual_ratio",
    "relative_ratio_error_percent",
    "direction",
    "prediction_width",
    "prediction_height",
    "reference_width",
    "reference_height",
    "remove_axis",
    "remove_pixels",
    "remove_percent",
    "severity",
    "final_render_action",
    "final_output_width",
    "final_output_height",
    "final_ratio",
    "final_ratio_error_percent",
    "title",
    "caption",
    "description",
    "source_reference",
    "thumbnail_path",
)


def prompt_field(prompt: str, label: str) -> str:
    match = re.search(rf"^{re.escape(label)}:\s*(.*)$", prompt, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def operation_name(plan: dict[str, Any]) -> str:
    return {
        (True, False): "crop",
        (True, True): "crop_fill",
        (False, True): "fill",
        (False, False): "keep",
    }[(bool(plan["is_cropped"]), bool(plan["is_filled"]))]


def severity_name(correction_fraction: float) -> str:
    if correction_fraction <= 0.005:
        return "negligible"
    if correction_fraction <= 0.02:
        return "small"
    if correction_fraction <= 0.05:
        return "noticeable"
    if correction_fraction <= 0.10:
        return "large"
    return "severe"


def inscribed_reference_box(
    prediction_box: tuple[int, int, int, int],
    target_ratio: float,
    *,
    alignment: str,
) -> tuple[int, int, int, int]:
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")
    if alignment not in {"top_left", "center"}:
        raise ValueError("alignment must be top_left or center")
    left, top, right, bottom = prediction_box
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("prediction_box must have positive area")

    actual_ratio = width / height
    if actual_ratio > target_ratio:
        reference_width = max(1, min(width, round(height * target_ratio)))
        reference_height = height
    elif actual_ratio < target_ratio:
        reference_width = width
        reference_height = max(1, min(height, round(width / target_ratio)))
    else:
        reference_width, reference_height = width, height

    x_offset = 0 if alignment == "top_left" else (width - reference_width) // 2
    y_offset = 0 if alignment == "top_left" else (height - reference_height) // 2
    return (
        left + x_offset,
        top + y_offset,
        left + x_offset + reference_width,
        top + y_offset + reference_height,
    )


def build_detail(
    *,
    image_id: str,
    source_reference: str,
    source_width: int,
    source_height: int,
    plan: dict[str, Any],
    title: str,
    caption: str,
) -> dict[str, Any]:
    if operation_name(plan) != "crop":
        raise ValueError("ratio-box diagnostics only apply to crop-only plans")
    crop_box = plan["crop_box"]
    if not isinstance(crop_box, list):
        raise ValueError("crop-only plan must contain crop_box")

    prediction_box = normalized_pixel_box(Image.new("RGB", (source_width, source_height)), crop_box)
    left, top, right, bottom = prediction_box
    prediction_width, prediction_height = right - left, bottom - top
    target_ratio = float(plan["target_ratio"])
    actual_ratio = prediction_width / prediction_height
    relative_ratio_error = abs(actual_ratio - target_ratio) / target_ratio

    top_left_reference_box = inscribed_reference_box(prediction_box, target_ratio, alignment="top_left")
    center_reference_box = inscribed_reference_box(prediction_box, target_ratio, alignment="center")
    reference_width = top_left_reference_box[2] - top_left_reference_box[0]
    reference_height = top_left_reference_box[3] - top_left_reference_box[1]

    if actual_ratio > target_ratio:
        direction = "too_wide"
        remove_axis = "width"
        remove_pixels = prediction_width - target_ratio * prediction_height
        correction_fraction = remove_pixels / prediction_width
    elif actual_ratio < target_ratio:
        direction = "too_tall"
        remove_axis = "height"
        remove_pixels = prediction_height - prediction_width / target_ratio
        correction_fraction = remove_pixels / prediction_height
    else:
        direction = "exact"
        remove_axis = "none"
        remove_pixels = 0.0
        correction_fraction = 0.0

    return {
        "image_id": image_id,
        "source_reference": source_reference,
        "source_width": source_width,
        "source_height": source_height,
        "target_ratio": target_ratio,
        "actual_ratio": actual_ratio,
        "relative_ratio_error_percent": relative_ratio_error * 100,
        "crop_box": [float(value) for value in crop_box],
        "prediction_box": list(prediction_box),
        "prediction_width": prediction_width,
        "prediction_height": prediction_height,
        "top_left_reference_box": list(top_left_reference_box),
        "center_reference_box": list(center_reference_box),
        "reference_width": reference_width,
        "reference_height": reference_height,
        "reference_ratio": reference_width / reference_height,
        "direction": direction,
        "remove_axis": remove_axis,
        "remove_pixels": remove_pixels,
        "remove_percent": correction_fraction * 100,
        "severity": severity_name(correction_fraction),
        "title": title,
        "caption": caption,
        "description": str(plan.get("description") or ""),
    }


def load_details(results_path: Path, image_root: Path | None) -> tuple[list[dict[str, Any]], dict[str, int]]:
    details = []
    operation_counts: Counter[str] = Counter()
    seen: set[tuple[str, float]] = set()
    with results_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            plan = parse_response(row["response"])
            operation = operation_name(plan)
            operation_counts[operation] += 1
            if operation != "crop":
                continue

            source_reference = image_reference(row)
            source_path = image_root / Path(source_reference).name if image_root else Path(source_reference)
            if not source_path.is_file():
                raise FileNotFoundError(f"line {line_number}: source image not found: {source_path}")
            image_id = source_path.stem
            key = (image_id, float(plan["target_ratio"]))
            if key in seen:
                raise ValueError(f"line {line_number}: duplicate image/ratio key {key}")
            seen.add(key)

            source = load_source_image(source_path)
            try:
                source_width, source_height = source.size
            finally:
                source.close()
            prompt = row.get("messages", [{}])[0].get("content", "")
            detail = build_detail(
                image_id=image_id,
                source_reference=source_reference,
                source_width=source_width,
                source_height=source_height,
                plan=plan,
                title=prompt_field(prompt, "News headline"),
                caption=prompt_field(prompt, "Image caption"),
            )
            detail["source_path"] = str(source_path)
            details.append(detail)
    if not details:
        raise ValueError(f"no crop-only records found in {results_path}")
    return details, dict(sorted(operation_counts.items()))


def load_render_manifest(manifest_path: Path) -> dict[tuple[str, float], dict[str, Any]]:
    manifest = {}
    with manifest_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (str(row["image_id"]), float(row["target_ratio"]))
            if key in manifest:
                raise ValueError(f"line {line_number}: duplicate render manifest key {key}")
            manifest[key] = row
    if not manifest:
        raise ValueError(f"render manifest is empty: {manifest_path}")
    return manifest


def attach_final_renders(
    details: list[dict[str, Any]],
    *,
    manifest_path: Path | None,
) -> Counter[str]:
    if manifest_path is None:
        for detail in details:
            detail.update(
                {
                    "final_render_action": None,
                    "final_output_width": None,
                    "final_output_height": None,
                    "final_ratio": None,
                    "final_ratio_error_percent": None,
                }
            )
        return Counter()

    manifest = load_render_manifest(manifest_path)
    counts: Counter[str] = Counter()
    missing = []
    for detail in details:
        key = (detail["image_id"], float(detail["target_ratio"]))
        rendered = manifest.get(key)
        if rendered is None:
            missing.append(key)
            continue
        output_width = int(rendered["output_width"])
        output_height = int(rendered["output_height"])
        final_ratio = output_width / output_height
        final_action = str(rendered["render_action"])
        counts[final_action] += 1
        detail.update(
            {
                "final_render_action": final_action,
                "final_output_width": output_width,
                "final_output_height": output_height,
                "final_ratio": final_ratio,
                "final_ratio_error_percent": abs(final_ratio - detail["target_ratio"])
                / detail["target_ratio"]
                * 100,
            }
        )
    if missing:
        raise ValueError(f"render manifest is missing crop-only keys: {missing[:3]}")
    return counts


def materialize_thumbnails(details: list[dict[str, Any]], output_dir: Path, max_edge: int) -> None:
    if max_edge <= 0:
        raise ValueError("max_edge must be positive")
    thumbnail_dir = output_dir / "assets" / "thumbnails"
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    paths_by_image: dict[str, Path] = {}
    for detail in details:
        image_id = detail["image_id"]
        output_path = thumbnail_dir / f"{image_id}.jpg"
        if image_id not in paths_by_image:
            source = load_source_image(Path(detail["source_path"]))
            try:
                source.thumbnail((max_edge, max_edge), resample=Image.Resampling.LANCZOS)
                source.save(output_path, format="JPEG", quality=88, optimize=True)
            finally:
                source.close()
            paths_by_image[image_id] = output_path
        detail["thumbnail_path"] = paths_by_image[image_id].relative_to(output_dir).as_posix()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def summarize(
    details: list[dict[str, Any]],
    *,
    results_path: Path,
    image_root: Path | None,
    operation_counts: dict[str, int],
    render_manifest_path: Path | None,
    final_render_counts: Counter[str],
) -> dict[str, Any]:
    correction_values = [detail["remove_percent"] for detail in details]
    return {
        "protocol": "swift-crop-ratio-diagnostics-v1",
        "results_path": str(results_path),
        "image_root": str(image_root) if image_root else None,
        "source_records": sum(operation_counts.values()),
        "crop_only_records": len(details),
        "unique_crop_images": len({detail["image_id"] for detail in details}),
        "operation_counts": operation_counts,
        "render_manifest_path": str(render_manifest_path) if render_manifest_path else None,
        "crop_only_final_render_counts": dict(sorted(final_render_counts.items())),
        "direction_counts": dict(sorted(Counter(detail["direction"] for detail in details).items())),
        "severity_counts": {
            severity: sum(detail["severity"] == severity for detail in details) for severity in SEVERITY_ORDER
        },
        "correction_percent": {
            "mean": mean(correction_values),
            "p25": percentile(correction_values, 0.25),
            "median": median(correction_values),
            "p75": percentile(correction_values, 0.75),
            "p90": percentile(correction_values, 0.90),
            "p95": percentile(correction_values, 0.95),
            "max": max(correction_values),
        },
        "over_5_percent": sum(value > 5 for value in correction_values),
        "over_10_percent": sum(value > 10 for value in correction_values),
        "reference_box": {
            "default_alignment": "top_left",
            "definition": "largest target-ratio rectangle inscribed in the model prediction box",
        },
    }


def box_style(box: list[int], width: int, height: int) -> str:
    left, top, right, bottom = box
    return (
        f"left:{left / width * 100:.6f}%;top:{top / height * 100:.6f}%;"
        f"width:{(right - left) / width * 100:.6f}%;height:{(bottom - top) / height * 100:.6f}%"
    )


def excess_boxes(detail: dict[str, Any], alignment: str) -> list[list[int]]:
    prediction = detail["prediction_box"]
    reference = detail[f"{alignment}_reference_box"]
    left, top, right, bottom = prediction
    ref_left, ref_top, ref_right, ref_bottom = reference
    if detail["direction"] == "too_wide":
        return [[left, top, ref_left, bottom], [ref_right, top, right, bottom]]
    if detail["direction"] == "too_tall":
        return [[left, top, right, ref_top], [left, ref_bottom, right, bottom]]
    return []


def format_direction(direction: str) -> str:
    return {"too_wide": "Too wide", "too_tall": "Too tall", "exact": "Exact"}[direction]


def render_case(detail: dict[str, Any]) -> str:
    width, height = detail["source_width"], detail["source_height"]
    overlay_parts = [
        f'<span class="prediction-box" style="{box_style(detail["prediction_box"], width, height)}"></span>',
        f'<span class="reference-box top-left-only" style="{box_style(detail["top_left_reference_box"], width, height)}"></span>',
        f'<span class="reference-box center-only" style="{box_style(detail["center_reference_box"], width, height)}"></span>',
    ]
    for alignment, css_class in (("top_left", "top-left-only"), ("center", "center-only")):
        for box in excess_boxes(detail, alignment):
            if box[2] > box[0] and box[3] > box[1]:
                overlay_parts.append(
                    f'<span class="excess {css_class}" style="{box_style(box, width, height)}"></span>'
                )

    target = detail["target_ratio"]
    actual = detail["actual_ratio"]
    remove_side = "width" if detail["remove_axis"] == "width" else "height"
    searchable = " ".join((detail["image_id"], detail["title"], detail["caption"])).lower()
    final_action = detail["final_render_action"] or "not_attached"
    final_action_label = {
        "explicit_plan": "Used as predicted",
        "safe_box_expanded_crop": "Expanded crop",
        "safe_box_padding": "Fallback padding",
        "not_attached": "Final render unavailable",
    }.get(final_action, final_action.replace("_", " ").title())
    final_metrics = (
        f"<div><dt>Final output</dt><dd>{detail['final_output_width']} × {detail['final_output_height']} px</dd></div>"
        f"<div><dt>Final ratio</dt><dd>{detail['final_ratio']:.4f}:1 · {detail['final_ratio_error_percent']:.3f}% error</dd></div>"
        if detail["final_ratio"] is not None
        else ""
    )
    return f"""
<article class="case" data-ratio="{target:g}" data-severity="{detail['severity']}"
  data-direction="{detail['direction']}" data-correction="{detail['remove_percent']:.9f}"
    data-final-action="{html.escape(final_action, quote=True)}"
  data-search="{html.escape(searchable, quote=True)}">
  <header class="case-header">
    <div><span class="ratio-label">{target:g}:1 target</span><h2>{html.escape(detail['title'] or detail['image_id'])}</h2></div>
    <span class="severity severity-{detail['severity']}">{SEVERITY_LABELS[detail['severity']]}</span>
  </header>
  <div class="case-body">
    <figure>
      <div class="image-stage" style="aspect-ratio:{width}/{height}">
        <img src="{html.escape(detail['thumbnail_path'])}" loading="lazy" alt="{html.escape(detail['title'] or detail['image_id'])}">
        {''.join(overlay_parts)}
      </div>
      <figcaption><span class="legend prediction-legend"></span>Model box <span class="legend reference-legend"></span>Target box <span class="legend excess-legend"></span>Excess</figcaption>
    </figure>
    <div class="metrics">
    <div class="pipeline"><span>Model JSON: crop-only</span><strong>→</strong><span>{html.escape(final_action_label)}</span></div>
    <div class="ratio-comparison"><div><span>Target</span><strong>{target:.4g}:1</strong></div><span class="comparison-arrow">→</span><div><span>Raw model box</span><strong>{actual:.4f}:1</strong></div></div>
      <dl>
        <div><dt>Model box</dt><dd>{detail['prediction_width']} × {detail['prediction_height']} px</dd></div>
        <div><dt>Target box</dt><dd>{detail['reference_width']} × {detail['reference_height']} px</dd></div>
        <div><dt>Shape</dt><dd>{format_direction(detail['direction'])}</dd></div>
        <div><dt>Remove</dt><dd>{detail['remove_pixels']:.1f} px of {remove_side}</dd></div>
        <div><dt>Dimension change</dt><dd>{detail['remove_percent']:.2f}%</dd></div>
        <div><dt>Ratio error</dt><dd>{detail['relative_ratio_error_percent']:.2f}%</dd></div>
        {final_metrics}
      </dl>
      <p class="description">{html.escape(detail['description'])}</p>
      <p class="image-id">{html.escape(detail['image_id'])}</p>
    </div>
  </div>
</article>"""


def render_report(details: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    cards = "".join(render_case(detail) for detail in sorted(details, key=lambda item: item["remove_percent"], reverse=True))
    ratio_options = "".join(
        f'<option value="{ratio:g}">{ratio:g}:1</option>' for ratio in sorted({detail["target_ratio"] for detail in details})
    )
    severity_options = "".join(
        f'<option value="{severity}">{SEVERITY_LABELS[severity]}</option>' for severity in SEVERITY_ORDER
    )
    direction_counts = summary["direction_counts"]
    correction = summary["correction_percent"]
    final_counts = summary["crop_only_final_render_counts"]
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Swift Raw Crop Geometry Diagnostics</title>
<style>
:root{{--ink:#202221;--muted:#626763;--paper:#f7f7f3;--panel:#fff;--line:#d7d9d3;--red:#d53b2f;--green:#168353;--amber:#c8811a;--blue:#176b87}}
*{{box-sizing:border-box}}body{{margin:0;color:var(--ink);font-family:Aptos,"Trebuchet MS",sans-serif;background-color:var(--paper);background-image:linear-gradient(#e9ebe5 1px,transparent 1px),linear-gradient(90deg,#e9ebe5 1px,transparent 1px);background-size:24px 24px}}
main{{width:min(1500px,calc(100% - 32px));margin:0 auto 64px}}.page-header{{padding:28px 0 18px;border-bottom:3px solid var(--ink)}}h1,h2{{font-family:Bahnschrift,"Arial Narrow",sans-serif;letter-spacing:0}}h1{{font-size:clamp(28px,4vw,48px);margin:0 0 8px}}.subtitle{{margin:0;color:var(--muted)}}
.summary{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin:18px 0}}.summary div{{background:var(--panel);padding:14px}}.summary span{{display:block;color:var(--muted);font-size:12px;text-transform:uppercase}}.summary strong{{display:block;margin-top:4px;font-size:25px}}
.toolbar{{position:sticky;top:0;z-index:20;display:flex;flex-wrap:wrap;gap:10px;align-items:end;padding:12px;border:1px solid var(--line);background:rgba(255,255,255,.96);box-shadow:0 3px 12px #00000012}}label{{display:grid;gap:4px;color:var(--muted);font-size:12px}}select,input,button{{height:36px;border:1px solid #aeb2ac;background:#fff;color:var(--ink);font:inherit;padding:0 10px}}input{{min-width:220px}}.segmented{{display:flex}}.segmented button{{margin-left:-1px;cursor:pointer}}.segmented button:first-child{{margin-left:0}}.segmented button[aria-pressed=true]{{background:var(--ink);color:#fff}}.visible-count{{margin-left:auto;align-self:center;font-variant-numeric:tabular-nums}}
.case-list{{display:grid;gap:18px;margin-top:18px}}.case{{border:1px solid var(--line);border-left:5px solid var(--red);background:var(--panel);padding:16px;box-shadow:0 5px 18px #2328200d}}.case[hidden]{{display:none}}.case-header{{display:flex;align-items:start;justify-content:space-between;gap:12px;margin-bottom:12px}}.case-header h2{{font-size:18px;margin:4px 0 0;line-height:1.25}}.ratio-label{{color:var(--muted);font-size:12px}}.severity{{padding:5px 8px;border:1px solid currentColor;font-size:12px;white-space:nowrap}}.severity-negligible{{color:var(--green)}}.severity-small{{color:var(--blue)}}.severity-noticeable{{color:var(--amber)}}.severity-large,.severity-severe{{color:var(--red)}}
.case-body{{display:grid;grid-template-columns:minmax(0,1.55fr) minmax(280px,.75fr);gap:18px}}figure{{margin:0;min-width:0}}.image-stage{{position:relative;width:100%;overflow:hidden;background:#e7e8e3}}.image-stage img{{display:block;width:100%;height:100%;object-fit:fill}}.prediction-box,.reference-box,.excess{{position:absolute;pointer-events:none}}.prediction-box{{border:3px solid var(--red);z-index:4;box-shadow:0 0 0 1px #fff8 inset}}.reference-box{{border:3px dashed var(--green);z-index:5;box-shadow:0 0 0 1px #fff8}}.excess{{background:rgba(213,59,47,.32);z-index:3}}body[data-alignment=top_left] .center-only{{display:none}}body[data-alignment=center] .top-left-only{{display:none}}figcaption{{display:flex;gap:16px;align-items:center;margin-top:8px;color:var(--muted);font-size:12px}}.legend{{display:inline-block;width:22px;height:10px;margin-right:-11px}}.prediction-legend{{border-top:3px solid var(--red)}}.reference-legend{{border-top:3px dashed var(--green)}}.excess-legend{{background:rgba(213,59,47,.32)}}
.metrics{{min-width:0}}.pipeline{{display:flex;align-items:center;justify-content:center;gap:10px;padding:8px;border:1px solid #b8c3bb;background:#edf4ef;color:#24563d;font-size:13px;margin-bottom:12px}}.ratio-comparison{{display:grid;grid-template-columns:1fr auto 1fr;align-items:center;border-bottom:2px solid var(--ink);padding-bottom:12px;margin-bottom:8px}}.ratio-comparison div{{display:grid;gap:3px}}.ratio-comparison span{{font-size:12px;color:var(--muted)}}.ratio-comparison strong{{font-size:24px;font-variant-numeric:tabular-nums}}.comparison-arrow{{padding:0 10px;font-size:22px!important}}dl{{margin:0}}dl div{{display:flex;justify-content:space-between;gap:16px;padding:8px 0;border-bottom:1px solid var(--line)}}dt{{color:var(--muted)}}dd{{margin:0;text-align:right;font-variant-numeric:tabular-nums}}.description{{font-family:Georgia,serif;font-size:13px;line-height:1.45;color:#454945}}.image-id{{font-family:Consolas,monospace;font-size:10px;color:var(--muted);overflow-wrap:anywhere}}
.empty{{display:none;padding:48px;text-align:center;background:#fff;margin-top:18px;border:1px solid var(--line)}}
@media(max-width:900px){{main{{width:min(100% - 18px,1500px)}}.summary{{grid-template-columns:1fr 1fr}}.case-body{{grid-template-columns:1fr}}.toolbar{{position:static}}.visible-count{{width:100%;margin-left:0}}input{{min-width:150px;max-width:100%}}}}
</style></head>
<body data-alignment="top_left"><main>
<header class="page-header"><h1>Swift Raw Crop Geometry Diagnostics</h1><p class="subtitle">Model-declared crop-only boxes before fallback · downstream padding or crop expansion is shown per record</p></header>
<section class="summary">
    <div><span>Model-declared crop-only</span><strong>{summary['crop_only_records']}</strong></div>
    <div><span>Used as predicted</span><strong>{final_counts.get('explicit_plan', 0)}</strong></div>
    <div><span>Expanded crop</span><strong>{final_counts.get('safe_box_expanded_crop', 0)}</strong></div>
    <div><span>Fallback padding</span><strong>{final_counts.get('safe_box_padding', 0)}</strong></div>
</section>
<section class="toolbar" aria-label="Report controls">
  <label>Reference alignment<div class="segmented"><button type="button" data-alignment="top_left" aria-pressed="true">Top-left</button><button type="button" data-alignment="center" aria-pressed="false">Centered</button></div></label>
  <label>Target ratio<select id="ratio-filter"><option value="all">All ratios</option>{ratio_options}</select></label>
  <label>Severity<select id="severity-filter"><option value="all">All severities</option>{severity_options}</select></label>
  <label>Shape<select id="direction-filter"><option value="all">All shapes</option><option value="too_wide">Too wide ({direction_counts.get('too_wide', 0)})</option><option value="too_tall">Too tall ({direction_counts.get('too_tall', 0)})</option><option value="exact">Exact ({direction_counts.get('exact', 0)})</option></select></label>
    <label>Final action<select id="action-filter"><option value="all">All final actions</option><option value="explicit_plan">Used as predicted</option><option value="safe_box_expanded_crop">Expanded crop</option><option value="safe_box_padding">Fallback padding</option></select></label>
  <label>Sort<select id="sort"><option value="correction-desc">Largest change</option><option value="correction-asc">Smallest change</option><option value="ratio">Target ratio</option></select></label>
  <label>Search<input id="search" type="search" placeholder="Headline or image ID"></label>
  <strong class="visible-count"><span id="visible-count">{len(details)}</span> shown</strong>
</section>
<section class="case-list" id="case-list">{cards}</section><div class="empty" id="empty">No matching records.</div>
</main>
<script>
const list=document.getElementById('case-list');const cases=[...list.querySelectorAll('.case')];
for(const button of document.querySelectorAll('button[data-alignment]'))button.addEventListener('click',()=>{{document.body.dataset.alignment=button.dataset.alignment;for(const other of document.querySelectorAll('button[data-alignment]'))other.setAttribute('aria-pressed',String(other===button));}});
function update(){{const ratio=document.getElementById('ratio-filter').value;const severity=document.getElementById('severity-filter').value;const direction=document.getElementById('direction-filter').value;const action=document.getElementById('action-filter').value;const search=document.getElementById('search').value.trim().toLowerCase();const sort=document.getElementById('sort').value;let visible=0;for(const card of cases){{const show=(ratio==='all'||card.dataset.ratio===ratio)&&(severity==='all'||card.dataset.severity===severity)&&(direction==='all'||card.dataset.direction===direction)&&(action==='all'||card.dataset.finalAction===action)&&(!search||card.dataset.search.includes(search));card.hidden=!show;if(show)visible++;}}const ordered=[...cases].sort((a,b)=>sort==='correction-asc'?Number(a.dataset.correction)-Number(b.dataset.correction):sort==='ratio'?Number(a.dataset.ratio)-Number(b.dataset.ratio):Number(b.dataset.correction)-Number(a.dataset.correction));for(const card of ordered)list.appendChild(card);document.getElementById('visible-count').textContent=visible;document.getElementById('empty').style.display=visible?'none':'block';}}
for(const id of ['ratio-filter','severity-filter','direction-filter','action-filter','sort','search'])document.getElementById(id).addEventListener(id==='search'?'input':'change',update);
</script></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def write_structured_outputs(details: list[dict[str, Any]], summary: dict[str, Any], output_dir: Path) -> None:
    (output_dir / "details.jsonl").write_text(
        "".join(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n" for detail in details),
        encoding="utf-8",
    )
    with (output_dir / "details.csv").open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=DETAIL_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(details)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def generate_report(
    *,
    results_path: Path,
    image_root: Path | None,
    output_dir: Path,
    max_thumbnail_edge: int,
    overwrite: bool,
    render_manifest_path: Path | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        if not overwrite:
            raise FileExistsError(f"output directory is not empty: {output_dir}; pass --overwrite to replace it")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    details, operation_counts = load_details(results_path, image_root)
    final_render_counts = attach_final_renders(details, manifest_path=render_manifest_path)
    materialize_thumbnails(details, output_dir, max_thumbnail_edge)
    summary = summarize(
        details,
        results_path=results_path,
        image_root=image_root,
        operation_counts=operation_counts,
        render_manifest_path=render_manifest_path,
        final_render_counts=final_render_counts,
    )
    write_structured_outputs(details, summary, output_dir)
    render_report(details, summary, output_dir)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a static crop-only ratio diagnostic report from Swift output.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, help="Resolve source images by basename under this directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-thumbnail-edge", type=int, default=1400)
    parser.add_argument("--render-manifest", type=Path, help="Optional downstream render manifest.jsonl.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = generate_report(
        results_path=args.results.expanduser(),
        image_root=args.image_root.expanduser() if args.image_root else None,
        output_dir=args.output_dir.expanduser(),
        max_thumbnail_edge=args.max_thumbnail_edge,
        overwrite=args.overwrite,
        render_manifest_path=args.render_manifest.expanduser() if args.render_manifest else None,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()