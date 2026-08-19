#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
import re
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import Any, Sequence

import pyarrow.parquet as pq
from PIL import Image, ImageOps

from news_crop_benchmark.vlm_scorer import (
    DEFAULT_AZURE_API_VERSION,
    DEFAULT_AZURE_DEPLOYMENT,
    DEFAULT_AZURE_ENDPOINT,
    DEFAULT_MANAGED_IDENTITY_CLIENT_ID,
    _get_bearer_token_provider,
    extract_response_text,
    load_env_files,
)


RATIO_DIRECTORIES = {
    1.0: "gaic_1p00",
    1.59: "gaic_1p59",
    1.77: "gaic_1p77",
    1.91: "gaic_1p91",
}
ORDERS = ("visual_a", "mllm_a")
REQUIRED_RESPONSE_FIELDS = {
    "winner",
    "title_relevance",
    "crop_quality",
    "title_relevant_elements",
    "A_missing_or_damaged_elements",
    "B_missing_or_damaged_elements",
    "reason",
    "confidence",
}


@dataclass(frozen=True)
class PairTask:
    task_id: str
    image_id: str
    source_index: int
    title: str
    caption: str
    target_ratio: float
    original_path: Path
    visual_path: Path
    mllm_path: Path


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + f".tmp.{os.getpid()}.{threading.get_ident()}")
    temporary_path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)


def absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(path))


def ratio_key(value: float) -> float:
    for expected in RATIO_DIRECTORIES:
        if abs(float(value) - expected) < 1e-6:
            return expected
    raise ValueError(f"unsupported target ratio: {value}")


def parse_title(prompt: str) -> str:
    match = re.search(r"^News headline:\s*(.+?)\s*$", prompt, flags=re.MULTILINE)
    if not match:
        raise ValueError("prompt does not contain a News headline field")
    return match.group(1).strip()


def parse_caption(prompt: str) -> str:
    match = re.search(r"^Image caption:\s*(.+?)\s*$", prompt, flags=re.MULTILINE)
    return match.group(1).strip() if match else ""


def resolve_original(original_root: Path, image_id: str) -> Path:
    matches = [path for path in original_root.glob(f"{image_id}.*") if path.is_file()]
    if len(matches) != 1:
        raise ValueError(f"expected one original image for {image_id}, found {len(matches)}")
    return matches[0]


def load_mllm_paths(mllm_root: Path) -> dict[tuple[str, float], Path]:
    manifest_path = mllm_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    paths: dict[tuple[str, float], Path] = {}
    for line_number, line in enumerate(manifest_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            continue
        row = json.loads(line)
        key = (str(row["image_id"]), ratio_key(row["target_ratio"]))
        if key in paths:
            raise ValueError(f"duplicate MLLM manifest key at line {line_number}: {key}")
        path = mllm_root / str(row["render_path"])
        if not path.is_file():
            raise FileNotFoundError(path)
        paths[key] = absolute_path(path)
    return paths


def build_tasks(
    inference_data: Path,
    original_root: Path,
    visual_root: Path,
    mllm_root: Path,
) -> list[PairTask]:
    mllm_paths = load_mllm_paths(mllm_root)
    rows = pq.read_table(
        inference_data,
        columns=["messages", "image_id", "source_index", "target_ratio"],
    ).to_pylist()
    tasks = []
    seen: set[tuple[str, float]] = set()
    original_paths: dict[str, Path] = {}
    for row in rows:
        image_id = str(row["image_id"])
        target_ratio = ratio_key(row["target_ratio"])
        key = (image_id, target_ratio)
        if key in seen:
            raise ValueError(f"duplicate inference key: {key}")
        seen.add(key)
        prompt = str(row["messages"][0]["content"])
        title = parse_title(prompt)
        caption = parse_caption(prompt)
        original_path = original_paths.setdefault(image_id, resolve_original(original_root, image_id))
        visual_path = visual_root / RATIO_DIRECTORIES[target_ratio] / f"{image_id}.png"
        if not visual_path.is_file():
            raise FileNotFoundError(visual_path)
        if key not in mllm_paths:
            raise FileNotFoundError(f"missing MLLM render for {key}")
        tasks.append(
            PairTask(
                task_id=f"{image_id}__ratio_{target_ratio:g}",
                image_id=image_id,
                source_index=int(row["source_index"]),
                title=title,
                caption=caption,
                target_ratio=target_ratio,
                original_path=absolute_path(original_path),
                visual_path=absolute_path(visual_path),
                mllm_path=mllm_paths[key],
            )
        )
    if set(mllm_paths) != seen:
        extras = sorted(set(mllm_paths) - seen)
        raise ValueError(f"MLLM manifest has {len(extras)} unmatched keys")
    tasks.sort(key=lambda task: (task.source_index, list(RATIO_DIRECTORIES).index(task.target_ratio)))
    return tasks


def task_dict(task: PairTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "image_id": task.image_id,
        "source_index": task.source_index,
        "title": task.title,
        "caption": task.caption,
        "target_ratio": task.target_ratio,
        "original_path": str(task.original_path),
        "visual_path": str(task.visual_path),
        "mllm_path": str(task.mllm_path),
    }


def write_task_manifest(tasks: Sequence[PairTask], output_dir: Path) -> None:
    (output_dir / "tasks.jsonl").write_text(
        "".join(json.dumps(task_dict(task), ensure_ascii=False, sort_keys=True) + "\n" for task in tasks),
        encoding="utf-8",
    )


def image_data_url(path: Path, max_side: int, jpeg_quality: int) -> str:
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    if max(image.size) > max_side:
        image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
    import base64

    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def parse_judge_response(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    payload = None
    for match in re.finditer(r"\{", text):
        try:
            candidate, _ = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if isinstance(candidate, dict) and REQUIRED_RESPONSE_FIELDS <= set(candidate):
            payload = candidate
            break
    if payload is None:
        raise ValueError("response does not contain the required JSON object")
    if payload["winner"] not in {"A", "B", "tie"}:
        raise ValueError(f"invalid winner: {payload['winner']}")
    for field in ("title_relevance", "crop_quality"):
        scores = payload[field]
        if not isinstance(scores, dict) or set(scores) != {"A", "B"}:
            raise ValueError(f"{field} must contain exactly A and B")
        for candidate, value in scores.items():
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
                raise ValueError(f"invalid {field}.{candidate}: {value}")
    confidence = payload["confidence"]
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)) or not 0 <= confidence <= 1:
        raise ValueError(f"invalid confidence: {confidence}")
    for field in (
        "title_relevant_elements",
        "A_missing_or_damaged_elements",
        "B_missing_or_damaged_elements",
    ):
        if not isinstance(payload[field], list) or not all(isinstance(value, str) for value in payload[field]):
            raise ValueError(f"{field} must be a list of strings")
    if not isinstance(payload["reason"], str) or not payload["reason"].strip():
        raise ValueError("reason must be a non-empty string")
    return payload


def create_client() -> tuple[Any, str]:
    load_env_files()
    from openai import AzureOpenAI

    endpoint = os.getenv("GPT5_AZURE_OPENAI_ENDPOINT", DEFAULT_AZURE_ENDPOINT).strip()
    deployment = os.getenv(
        "PAIRWISE_JUDGE_MODEL",
        os.getenv("GPT5_AZURE_OPENAI_DEPLOYMENT", DEFAULT_AZURE_DEPLOYMENT),
    ).strip()
    api_version = os.getenv("GPT5_AZURE_OPENAI_API_VERSION", DEFAULT_AZURE_API_VERSION).strip()
    api_key = os.getenv("GPT5_AZURE_OPENAI_API_KEY", "").strip()
    client_kwargs: dict[str, Any] = {
        "api_version": api_version,
        "azure_endpoint": endpoint,
        "max_retries": 0,
    }
    if api_key:
        client_kwargs["api_key"] = api_key
    else:
        from azure.identity import DefaultAzureCredential

        credential = DefaultAzureCredential(
            managed_identity_client_id=os.getenv(
                "GPT5_AZURE_MANAGED_IDENTITY_CLIENT_ID",
                DEFAULT_MANAGED_IDENTITY_CLIENT_ID,
            ).strip()
        )
        client_kwargs["azure_ad_token_provider"] = _get_bearer_token_provider(
            credential,
            "https://cognitiveservices.azure.com/.default",
        )
    return AzureOpenAI(**client_kwargs), deployment


class PairwiseJudge:
    def __init__(self, prompt_path: Path) -> None:
        self.prompt = prompt_path.read_text(encoding="utf-8").strip()
        if not self.prompt:
            raise ValueError("pairwise judge prompt must not be empty")
        self.client, self.model = create_client()
        self.max_side = int(os.getenv("PAIRWISE_JUDGE_IMAGE_SIZE", "1024"))
        self.jpeg_quality = int(os.getenv("PAIRWISE_JUDGE_JPEG_QUALITY", "85"))
        self.timeout = float(os.getenv("PAIRWISE_JUDGE_TIMEOUT", "90"))
        self.max_retries = int(os.getenv("PAIRWISE_JUDGE_MAX_RETRIES", "2"))
        self.backoff = float(os.getenv("PAIRWISE_JUDGE_RETRY_BACKOFF", "1.5"))
        self.max_output_tokens = int(os.getenv("PAIRWISE_JUDGE_MAX_OUTPUT_TOKENS", "2048"))

    def judge(self, task: PairTask, order: str) -> dict[str, Any]:
        if order not in ORDERS:
            raise ValueError(f"unsupported candidate order: {order}")
        candidate_a = task.visual_path if order == "visual_a" else task.mllm_path
        candidate_b = task.mllm_path if order == "visual_a" else task.visual_path
        content = [
            {"type": "input_text", "text": "Original Image"},
            {"type": "input_image", "image_url": image_data_url(task.original_path, self.max_side, self.jpeg_quality)},
            {"type": "input_text", "text": "Candidate A"},
            {"type": "input_image", "image_url": image_data_url(candidate_a, self.max_side, self.jpeg_quality)},
            {"type": "input_text", "text": "Candidate B"},
            {"type": "input_image", "image_url": image_data_url(candidate_b, self.max_side, self.jpeg_quality)},
            {
                "type": "input_text",
                "text": (
                    f"{self.prompt}\n\n"
                    f"News Title: {task.title}\n"
                    f"Target Aspect Ratio (width/height): {task.target_ratio:g}\n\n"
                    "Return only the required JSON object."
                ),
            },
        ]
        last_error: Exception | None = None
        started = time.perf_counter()
        for attempt in range(self.max_retries + 1):
            try:
                with self.client.responses.stream(
                    model=self.model,
                    input=[{"role": "user", "content": content}],
                    reasoning={"effort": "low"},
                    text={"verbosity": "low"},
                    max_output_tokens=self.max_output_tokens,
                    timeout=self.timeout,
                ) as stream:
                    response = stream.get_final_response()
                output_text = extract_response_text(response)
                evaluation = parse_judge_response(output_text)
                return {
                    "status": "completed",
                    "order": order,
                    "model": self.model,
                    "response_id": getattr(response, "id", None),
                    "attempt_count": attempt + 1,
                    "latency_ms": round((time.perf_counter() - started) * 1000, 2),
                    "evaluation": evaluation,
                    "output_text": output_text,
                }
            except Exception as error:  # noqa: BLE001 - credential, SDK, and parse errors are retried alike.
                last_error = error
                if attempt < self.max_retries:
                    time.sleep(self.backoff**attempt)
        return {
            "status": "failed",
            "order": order,
            "model": self.model,
            "attempt_count": self.max_retries + 1,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error_type": type(last_error).__name__,
            "error": str(last_error),
        }


def progress_path(output_dir: Path, task: PairTask, order: str) -> Path:
    return output_dir / "progress" / task.task_id / f"{order}.json"


def source_outcome(result: dict[str, Any]) -> str:
    winner = result["evaluation"]["winner"]
    if winner == "tie":
        return "tie"
    if result["order"] == "visual_a":
        return "visual" if winner == "A" else "mllm"
    return "mllm" if winner == "A" else "visual"


def source_scores(result: dict[str, Any], field: str) -> dict[str, float]:
    scores = result["evaluation"][field]
    if result["order"] == "visual_a":
        return {"visual": float(scores["A"]), "mllm": float(scores["B"])}
    return {"mllm": float(scores["A"]), "visual": float(scores["B"])}


def combine_task(task: PairTask, results: dict[str, dict[str, Any]]) -> dict[str, Any]:
    complete = all(results.get(order, {}).get("status") == "completed" for order in ORDERS)
    detail = {**task_dict(task), "orders": results, "complete": complete}
    if not complete:
        detail.update(stable=False, final_outcome="incomplete")
        return detail
    outcomes = [source_outcome(results[order]) for order in ORDERS]
    stable = outcomes[0] == outcomes[1]
    detail.update(stable=stable, final_outcome=outcomes[0] if stable else "unstable")
    for field in ("title_relevance", "crop_quality"):
        scores = [source_scores(results[order], field) for order in ORDERS]
        visual = mean(score["visual"] for score in scores)
        mllm = mean(score["mllm"] for score in scores)
        detail[f"visual_{field}"] = visual
        detail[f"mllm_{field}"] = mllm
        detail[f"mllm_minus_visual_{field}"] = mllm - visual
    return detail


def summarize_details(details: Sequence[dict[str, Any]]) -> dict[str, Any]:
    complete = [detail for detail in details if detail["complete"]]
    stable = [detail for detail in complete if detail["stable"]]
    outcome_counts = Counter(detail["final_outcome"] for detail in stable)
    denominator = len(stable)
    all_order_results = [result for detail in complete for result in detail["orders"].values()]
    return {
        "tasks": len(details),
        "completed_pairs": len(complete),
        "incomplete_pairs": len(details) - len(complete),
        "stable_pairs": len(stable),
        "unstable_pairs": len(complete) - len(stable),
        "unstable_rate": (len(complete) - len(stable)) / len(complete) if complete else None,
        "stable_outcome_counts": dict(sorted(outcome_counts.items())),
        "mllm_win_rate": outcome_counts["mllm"] / denominator if denominator else None,
        "visual_win_rate": outcome_counts["visual"] / denominator if denominator else None,
        "tie_rate": outcome_counts["tie"] / denominator if denominator else None,
        "mean_mllm_minus_visual_title_relevance": (
            mean(detail["mllm_minus_visual_title_relevance"] for detail in stable) if stable else None
        ),
        "mean_mllm_minus_visual_crop_quality": (
            mean(detail["mllm_minus_visual_crop_quality"] for detail in stable) if stable else None
        ),
        "individual_judgment_A_win_rate": (
            mean(result["evaluation"]["winner"] == "A" for result in all_order_results)
            if all_order_results
            else None
        ),
        "mean_confidence": (
            mean(float(result["evaluation"]["confidence"]) for result in all_order_results)
            if all_order_results
            else None
        ),
    }


def review_asset_path(output_dir: Path, image_id: str, name: str) -> tuple[Path, str]:
    relative_path = Path("assets") / image_id / name
    return output_dir / relative_path, relative_path.as_posix()


def write_review_image(source_path: Path, output_path: Path, max_side: int = 1400) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: OSError | None = None
    for _ in range(3):
        try:
            with Image.open(source_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                if max(image.size) > max_side:
                    image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                temporary_path = output_path.with_suffix(".tmp.jpg")
                image.save(temporary_path, format="JPEG", quality=90, optimize=True)
                temporary_path.replace(output_path)
            return
        except OSError as error:
            last_error = error
    raise OSError(f"failed to prepare review image after 3 attempts: {source_path}") from last_error


def prepare_review_assets(details: Sequence[dict[str, Any]], output_dir: Path) -> dict[str, dict[str, str]]:
    assets: dict[str, dict[str, str]] = {}
    prepared_originals: set[str] = set()
    for detail in details:
        image_id = str(detail["image_id"])
        ratio_name = f"{float(detail['target_ratio']):g}".replace(".", "p")
        original_output, original_relative = review_asset_path(output_dir, image_id, "original.jpg")
        if image_id not in prepared_originals:
            write_review_image(Path(detail["original_path"]), original_output)
            prepared_originals.add(image_id)
        visual_output, visual_relative = review_asset_path(output_dir, image_id, f"ratio_{ratio_name}_gaic.jpg")
        mllm_output, mllm_relative = review_asset_path(output_dir, image_id, f"ratio_{ratio_name}_llm.jpg")
        write_review_image(Path(detail["visual_path"]), visual_output)
        write_review_image(Path(detail["mllm_path"]), mllm_output)
        assets[str(detail["task_id"])] = {
            "original": original_relative,
            "visual": visual_relative,
            "mllm": mllm_relative,
        }
    return assets


def review_source_name(source: str) -> str:
    return "GAIC" if source == "visual" else "LLM" if source == "mllm" else source.upper()


def review_outcome_name(outcome: str) -> str:
    return {
        "visual": "GAIC wins",
        "mllm": "LLM wins",
        "tie": "Tie",
        "unstable": "Unstable",
        "incomplete": "Incomplete",
    }.get(outcome, outcome)


def review_order_html(result: dict[str, Any]) -> str:
    order = str(result["order"])
    evaluation = result["evaluation"]
    candidate_a = "GAIC" if order == "visual_a" else "LLM"
    candidate_b = "LLM" if order == "visual_a" else "GAIC"
    outcome = source_outcome(result)
    relevance = source_scores(result, "title_relevance")
    quality = source_scores(result, "crop_quality")
    if order == "visual_a":
        missing = {
            "visual": evaluation["A_missing_or_damaged_elements"],
            "mllm": evaluation["B_missing_or_damaged_elements"],
        }
    else:
        missing = {
            "mllm": evaluation["A_missing_or_damaged_elements"],
            "visual": evaluation["B_missing_or_damaged_elements"],
        }

    def render_items(values: Sequence[str]) -> str:
        return "<ul>" + "".join(f"<li>{html.escape(value)}</li>" for value in values) + "</ul>" if values else '<p class="none">None</p>'

    return f"""
    <article class="order-review">
      <div class="order-heading">
        <strong>A = {candidate_a}, B = {candidate_b}</strong>
        <span class="order-winner">{html.escape(review_outcome_name(outcome))}</span>
        <span>confidence {float(evaluation['confidence']):.2f}</span>
      </div>
      <table class="order-scores">
        <tr><th></th><th>GAIC</th><th>LLM</th></tr>
        <tr><td>Title relevance</td><td>{relevance['visual']:g}</td><td>{relevance['mllm']:g}</td></tr>
        <tr><td>Crop quality</td><td>{quality['visual']:g}</td><td>{quality['mllm']:g}</td></tr>
      </table>
      <p class="reason">{html.escape(str(evaluation['reason']))}</p>
      <div class="damage-grid">
        <div><strong>GAIC missing/damaged</strong>{render_items(missing['visual'])}</div>
        <div><strong>LLM missing/damaged</strong>{render_items(missing['mllm'])}</div>
      </div>
    </article>
    """


def render_review_report(
    details: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    output_dir: Path,
) -> None:
    assets = prepare_review_assets(details, output_dir)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for detail in details:
        grouped.setdefault(str(detail["image_id"]), []).append(detail)
    overall = summary["overall"]
    stable_outcomes = overall["stable_outcome_counts"]
    sections = []
    for image_details in grouped.values():
        image_details.sort(key=lambda detail: list(RATIO_DIRECTORIES).index(float(detail["target_ratio"])))
        first = image_details[0]
        ratio_sections = []
        for detail in image_details:
            task_assets = assets[str(detail["task_id"])]
            outcome = str(detail["final_outcome"])
            ratio = float(detail["target_ratio"])
            title_elements = sorted(
                {
                    str(value)
                    for result in detail["orders"].values()
                    for value in result.get("evaluation", {}).get("title_relevant_elements", [])
                }
            )
            title_elements_html = ", ".join(html.escape(value) for value in title_elements) or "None identified"
            order_reviews = "".join(review_order_html(detail["orders"][order]) for order in ORDERS)
            ratio_sections.append(
                f"""
                <section class="ratio-review" data-ratio="{ratio:g}" data-outcome="{html.escape(outcome)}">
                  <header class="ratio-header">
                    <h3>Target ratio {ratio:g}:1</h3>
                    <span class="outcome outcome-{html.escape(outcome)}">{html.escape(review_outcome_name(outcome))}</span>
                    <span class="stability">{'Stable after swap' if detail['stable'] else 'Order-sensitive'}</span>
                  </header>
                  <div class="candidate-grid">
                    <article class="candidate gaic">
                      <h4>GAIC <span>visual baseline, no title</span></h4>
                      <img src="{html.escape(task_assets['visual'])}" alt="GAIC crop at ratio {ratio:g}">
                      <div class="score-line"><span>Title relevance <b>{float(detail['visual_title_relevance']):g}</b></span><span>Crop quality <b>{float(detail['visual_crop_quality']):g}</b></span></div>
                    </article>
                    <article class="candidate llm">
                      <h4>LLM <span>title-conditioned crop</span></h4>
                      <img src="{html.escape(task_assets['mllm'])}" alt="LLM crop at ratio {ratio:g}">
                      <div class="score-line"><span>Title relevance <b>{float(detail['mllm_title_relevance']):g}</b></span><span>Crop quality <b>{float(detail['mllm_crop_quality']):g}</b></span></div>
                    </article>
                  </div>
                  <p class="elements"><strong>Visible title-relevant elements:</strong> {title_elements_html}</p>
                  <details class="judge-details" {'open' if outcome == 'unstable' else ''}>
                    <summary>Two swapped Judge decisions</summary>
                    <div class="order-grid">{order_reviews}</div>
                  </details>
                </section>
                """
            )
        title_search = html.escape(str(first["title"]).lower(), quote=True)
        sections.append(
            f"""
            <section class="image-review" data-title="{title_search}">
              <header class="story-header">
                <div><span class="source-index">Source {int(first['source_index'])}</span><h2>{html.escape(str(first['title']))}</h2></div>
                <code>{html.escape(str(first['image_id']))}</code>
              </header>
              <p class="caption">{html.escape(str(first.get('caption', '')))}</p>
              <div class="story-layout">
                <aside class="original-panel">
                  <h3>Original image</h3>
                  <img src="{html.escape(assets[str(first['task_id'])]['original'])}" alt="Original source image">
                </aside>
                <div class="ratio-list">{''.join(ratio_sections)}</div>
              </div>
            </section>
            """
        )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>GAIC vs LLM Title-Crop Review</title><style>
:root {{ --paper:#f4f4f0; --surface:#fff; --ink:#20231f; --muted:#676d67; --line:#d7d9d2; --gaic:#176b87; --llm:#9a4d27; --good:#236b45; --warn:#a05a15; }}
* {{ box-sizing:border-box; }} body {{ margin:0; background:var(--paper); color:var(--ink); font-family:"Aptos","Segoe UI",sans-serif; letter-spacing:0; }}
main {{ width:min(1800px,97vw); margin:auto; padding:24px 0 64px; }} h1,h2,h3,h4,p {{ margin-top:0; }}
.summary,.image-review {{ background:var(--surface); border:1px solid var(--line); border-radius:6px; margin-bottom:20px; }}
.summary {{ padding:20px; }} .metric-grid {{ display:grid; grid-template-columns:repeat(6,minmax(120px,1fr)); gap:10px; }}
.metric {{ border-left:3px solid var(--ink); padding:8px 12px; background:#f8f8f5; }} .metric b {{ display:block; font-size:24px; }} .metric span {{ color:var(--muted); font-size:13px; }}
.filters {{ position:sticky; top:0; z-index:5; display:flex; flex-wrap:wrap; gap:12px; align-items:end; background:#eceee9; border:1px solid var(--line); padding:12px; margin:16px 0; }}
.filters label {{ display:grid; gap:4px; color:var(--muted); font-size:12px; }} select,input {{ min-height:36px; border:1px solid #afb4ad; background:white; padding:6px 9px; font:inherit; }}
.story-header {{ display:flex; justify-content:space-between; gap:20px; padding:18px 20px 8px; border-bottom:1px solid var(--line); }} .story-header h2 {{ margin:5px 0 0; font-size:21px; }}
.source-index {{ color:var(--muted); font-size:12px; text-transform:uppercase; }} .story-header code {{ color:var(--muted); font-size:11px; overflow-wrap:anywhere; max-width:35%; }}
.caption {{ color:var(--muted); padding:10px 20px; border-bottom:1px solid var(--line); margin:0; }}
.story-layout {{ display:grid; grid-template-columns:minmax(240px,25%) minmax(0,75%); }} .original-panel {{ padding:16px; border-right:1px solid var(--line); position:sticky; top:82px; align-self:start; }}
.original-panel img,.candidate img {{ display:block; width:100%; object-fit:contain; background:#e4e7e1; }} .original-panel img {{ max-height:620px; }} .ratio-list {{ min-width:0; }}
.ratio-review {{ padding:16px; border-bottom:1px solid var(--line); }} .ratio-review:last-child {{ border-bottom:0; }} .ratio-header {{ display:flex; align-items:center; gap:10px; margin-bottom:12px; }} .ratio-header h3 {{ margin:0 auto 0 0; font-size:17px; }}
.outcome,.stability {{ font-size:12px; font-weight:700; padding:4px 8px; border:1px solid var(--line); }} .outcome-mllm {{ color:var(--llm); border-color:var(--llm); }} .outcome-visual {{ color:var(--gaic); border-color:var(--gaic); }} .outcome-unstable {{ color:var(--warn); border-color:var(--warn); }}
.candidate-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:14px; }} .candidate {{ border:1px solid var(--line); border-top:4px solid; min-width:0; }} .candidate.gaic {{ border-top-color:var(--gaic); }} .candidate.llm {{ border-top-color:var(--llm); }}
.candidate h4 {{ padding:9px 11px; margin:0; }} .candidate h4 span {{ color:var(--muted); font-size:12px; font-weight:400; }} .candidate img {{ height:360px; }} .score-line {{ display:flex; justify-content:space-between; gap:8px; padding:9px 11px; background:#f7f8f5; font-size:13px; }}
.elements {{ margin:12px 0 0; font-size:13px; }} .judge-details {{ margin-top:12px; border:1px solid var(--line); }} .judge-details summary {{ cursor:pointer; padding:10px; font-weight:700; background:#f1f2ee; }} .order-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); }}
.order-review {{ padding:12px; min-width:0; }} .order-review + .order-review {{ border-left:1px solid var(--line); }} .order-heading {{ display:flex; flex-wrap:wrap; gap:8px 12px; align-items:center; font-size:12px; }} .order-winner {{ color:var(--good); font-weight:700; }}
.order-scores {{ border-collapse:collapse; width:100%; margin:10px 0; font-size:12px; }} .order-scores th,.order-scores td {{ border-bottom:1px solid var(--line); padding:5px; text-align:left; }} .reason {{ font-size:13px; line-height:1.45; }} .damage-grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:10px; font-size:12px; }} .damage-grid ul {{ padding-left:17px; }} .none {{ color:var(--muted); }}
@media(max-width:1100px) {{ .metric-grid {{ grid-template-columns:repeat(3,1fr); }} .story-layout {{ grid-template-columns:1fr; }} .original-panel {{ position:static; border-right:0; border-bottom:1px solid var(--line); }} }}
@media(max-width:720px) {{ .candidate-grid,.order-grid,.damage-grid {{ grid-template-columns:1fr; }} .order-review + .order-review {{ border-left:0; border-top:1px solid var(--line); }} .metric-grid {{ grid-template-columns:repeat(2,1fr); }} .candidate img {{ height:auto; max-height:420px; }} }}
</style></head><body><main>
<section class="summary"><h1>GAIC vs LLM Title-Crop Review</h1><p>Anonymous pairwise judging with both A/B orders. Only swap-stable pairs count toward win rates.</p>
<div class="metric-grid">
<div class="metric"><b>{int(overall['tasks'])}</b><span>crop pairs</span></div><div class="metric"><b>{int(overall['stable_pairs'])}</b><span>stable pairs</span></div>
<div class="metric"><b>{float(overall['mllm_win_rate'] or 0):.1%}</b><span>LLM win rate</span></div><div class="metric"><b>{float(overall['visual_win_rate'] or 0):.1%}</b><span>GAIC win rate</span></div>
<div class="metric"><b>{int(stable_outcomes.get('tie', 0))}</b><span>stable ties</span></div><div class="metric"><b>{float(overall['unstable_rate'] or 0):.1%}</b><span>unstable rate</span></div>
</div></section>
<div class="filters"><label>Search title<input id="title-filter" type="search" placeholder="Headline text"></label><label>Ratio<select id="ratio-filter"><option value="all">All</option>{''.join(f'<option value="{ratio:g}">{ratio:g}:1</option>' for ratio in RATIO_DIRECTORIES)}</select></label><label>Outcome<select id="outcome-filter"><option value="all">All</option><option value="mllm">LLM wins</option><option value="visual">GAIC wins</option><option value="tie">Tie</option><option value="unstable">Unstable</option></select></label></div>
{''.join(sections)}
</main><script>
const titleFilter=document.getElementById('title-filter'); const ratioFilter=document.getElementById('ratio-filter'); const outcomeFilter=document.getElementById('outcome-filter');
function applyFilters() {{ const query=titleFilter.value.trim().toLowerCase(); document.querySelectorAll('.image-review').forEach(group => {{ let visible=0; group.querySelectorAll('.ratio-review').forEach(card => {{ const matchesTitle=!query || group.dataset.title.includes(query); const matchesRatio=ratioFilter.value==='all' || card.dataset.ratio===ratioFilter.value; const matchesOutcome=outcomeFilter.value==='all' || card.dataset.outcome===outcomeFilter.value; const show=matchesTitle && matchesRatio && matchesOutcome; card.hidden=!show; if(show) visible++; }}); group.hidden=visible===0; }}); }}
[titleFilter,ratioFilter,outcomeFilter].forEach(control => control.addEventListener('input',applyFilters));
</script></body></html>"""
    (output_dir / "report.html").write_text(document, encoding="utf-8")


def write_reports(details: list[dict[str, Any]], output_dir: Path, model: str) -> dict[str, Any]:
    overall = summarize_details(details)
    by_ratio = {
        f"{ratio:g}": summarize_details([detail for detail in details if detail["target_ratio"] == ratio])
        for ratio in RATIO_DIRECTORIES
    }
    summary = {"judge_model": model, "overall": overall, "by_ratio": by_ratio}
    (output_dir / "details.jsonl").write_text(
        "".join(json.dumps(detail, ensure_ascii=False, sort_keys=True) + "\n" for detail in details),
        encoding="utf-8",
    )
    write_json_atomic(output_dir / "summary.json", summary)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        rows = [("overall", overall), *((f"ratio:{ratio}", values) for ratio, values in by_ratio.items())]
        scalar_keys = [key for key, value in overall.items() if not isinstance(value, dict)]
        writer = csv.DictWriter(handle, fieldnames=["scope", *scalar_keys, "stable_outcome_counts"])
        writer.writeheader()
        for scope, values in rows:
            writer.writerow(
                {
                    "scope": scope,
                    **{key: values.get(key) for key in scalar_keys},
                    "stable_outcome_counts": json.dumps(values["stable_outcome_counts"], sort_keys=True),
                }
            )
    with (output_dir / "details.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        fieldnames = [
            "task_id",
            "image_id",
            "source_index",
            "title",
            "caption",
            "target_ratio",
            "final_outcome",
            "stable",
            "visual_title_relevance",
            "mllm_title_relevance",
            "mllm_minus_visual_title_relevance",
            "visual_crop_quality",
            "mllm_crop_quality",
            "mllm_minus_visual_crop_quality",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for detail in details:
            writer.writerow({field: detail.get(field) for field in fieldnames})
    lines = [
        "# Pairwise Title-Crop Judge",
        "",
        f"Judge model: `{model}`",
        "",
        "Candidate identities were hidden from the judge. Every pair was judged in both A/B orders.",
        "",
        "## Overall",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key, value in overall.items():
        lines.append(f"| `{key}` | {json.dumps(value, ensure_ascii=False, sort_keys=True)} |")
    lines.extend(
        [
            "",
            "## By Ratio",
            "",
            "| Ratio | Complete | Stable | MLLM win | Visual win | Tie | Unstable | Title relevance delta | Crop quality delta |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for ratio, values in by_ratio.items():
        lines.append(
            f"| {ratio} | {values['completed_pairs']} | {values['stable_pairs']} | "
            f"{values['mllm_win_rate']} | {values['visual_win_rate']} | {values['tie_rate']} | "
            f"{values['unstable_rate']} | {values['mean_mllm_minus_visual_title_relevance']} | "
            f"{values['mean_mllm_minus_visual_crop_quality']} |"
        )
    (output_dir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    render_review_report(details, summary, output_dir)
    return summary


def run_judging(
    tasks: Sequence[PairTask],
    output_dir: Path,
    prompt_path: Path,
    workers: int,
) -> str:
    thread_state = threading.local()
    response_log = output_dir / "responses.jsonl"

    def judge_task(task: PairTask) -> None:
        for order in ORDERS:
            path = progress_path(output_dir, task, order)
            if path.is_file():
                try:
                    existing = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    existing = {}
                if existing.get("status") == "completed":
                    continue
            if not hasattr(thread_state, "judge"):
                thread_state.judge = PairwiseJudge(prompt_path)
            result = thread_state.judge.judge(task, order)
            result.update(task_id=task.task_id, timestamp=datetime.now(timezone.utc).isoformat())
            write_json_atomic(path, result)
            append_jsonl(response_log, result)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(judge_task, tasks))
    judge = PairwiseJudge(prompt_path)
    return judge.model


def collect_details(tasks: Sequence[PairTask], output_dir: Path) -> list[dict[str, Any]]:
    details = []
    for task in tasks:
        results = {}
        for order in ORDERS:
            path = progress_path(output_dir, task, order)
            if path.is_file():
                try:
                    results[order] = json.loads(path.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    pass
        details.append(combine_task(task, results))
    return details


def completed_judge_model(details: Sequence[dict[str, Any]]) -> str:
    for detail in details:
        for result in detail["orders"].values():
            if result.get("status") == "completed" and result.get("model"):
                return str(result["model"])
    return "unknown"


def parse_args() -> argparse.Namespace:
    default_report = Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "news-crop-benchmark" / "reports"
    parser = argparse.ArgumentParser(description="Pairwise title-relevance judge for visual and MLLM crops.")
    parser.add_argument("--inference-data", type=Path, required=True)
    parser.add_argument("--original-root", type=Path, required=True)
    parser.add_argument("--visual-root", type=Path, required=True)
    parser.add_argument("--mllm-root", type=Path, required=True)
    parser.add_argument("--prompt-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=default_report / "gaic-vs-swift-v42-title-pairwise")
    parser.add_argument("--judge-workers", type=int, default=2)
    parser.add_argument("--max-tasks", type=int)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare-only", action="store_true")
    mode.add_argument("--report-only", action="store_true")
    args = parser.parse_args()
    if args.judge_workers <= 0:
        raise ValueError("judge-workers must be positive")
    if args.max_tasks is not None and args.max_tasks <= 0:
        raise ValueError("max-tasks must be positive")
    return args


def main() -> None:
    args = parse_args()
    for path in (args.inference_data, args.original_root, args.visual_root, args.mllm_root, args.prompt_path):
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(args.inference_data, args.original_root, args.visual_root, args.mllm_root)
    write_task_manifest(tasks, output_dir)
    config = {
        "inference_data": str(absolute_path(args.inference_data)),
        "original_root": str(absolute_path(args.original_root)),
        "visual_root": str(absolute_path(args.visual_root)),
        "mllm_root": str(absolute_path(args.mllm_root)),
        "prompt_path": str(absolute_path(args.prompt_path)),
        "prompt_sha256": hashlib.sha256(args.prompt_path.read_bytes()).hexdigest(),
        "tasks": len(tasks),
        "judge_calls": len(tasks) * len(ORDERS),
    }
    write_json_atomic(output_dir / "run_config.json", config)
    active_tasks = tasks[: args.max_tasks] if args.max_tasks else tasks
    if args.prepare_only:
        print(json.dumps({**config, "prepare_only": True, "output_dir": str(output_dir)}, indent=2))
        return
    if args.report_only:
        details = collect_details(active_tasks, output_dir)
        summary = write_reports(details, output_dir, completed_judge_model(details))
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return
    model = run_judging(active_tasks, output_dir, args.prompt_path, args.judge_workers)
    details = collect_details(active_tasks, output_dir)
    summary = write_reports(details, output_dir, model)
    print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()