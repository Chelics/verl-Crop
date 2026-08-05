from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from news_crop_benchmark.geometry import action_to_bbox
from news_crop_benchmark.protocol import parse_crop_action_with_format
from news_crop_benchmark.proxy_scorer import (
    compute_visual_proxy_metrics,
    crop_image,
    get_clip_title_scorer,
    relative_title_relevance,
)
from news_crop_benchmark.reward import RewardComponents, combine_proxy_reward
from news_crop_benchmark.vlm_scorer import get_crop_vlm_scorer

DATA_SOURCE = "news_image_crop"


def _empty_result(score: float, format_reward: float = 0.0) -> dict[str, float]:
    return {
        "score": score,
        "format_reward": format_reward,
        "strict_format": float(format_reward == 1.0),
        "title_relevance": 0.0,
        "clip_original_similarity": 0.0,
        "clip_candidate_similarity": 0.0,
        "clip_similarity_delta": 0.0,
        "saliency": 0.0,
        "composition": 0.0,
        "integrity": 0.0,
        "area": 0.0,
        "area_fraction": 0.0,
        "proxy_enabled": 0.0,
        "vlm_enabled": 0.0,
        "vlm_label": -1.0,
    }


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    reward_mode: str = "smoke",
    clip_model_path: str | None = None,
    clip_device: str = "cpu",
    clip_delta_scale: float = 20.0,
    vlm_prompt_path: str | None = None,
    recoverable_format_reward: float = 0.5,
    format_penalty_weight: float = 0.1,
) -> dict[str, float]:
    """Score a normalized crop action for verl.

    ``smoke`` validates the training plumbing only. ``proxy`` enables local proxy metrics, while
    ``vlm`` sends the original and rendered candidate to an Azure OpenAI visual evaluator.
    """
    if data_source != DATA_SOURCE:
        raise ValueError(f"unsupported data_source: {data_source}")
    if reward_mode not in {"smoke", "proxy", "vlm"}:
        raise ValueError("reward_mode must be 'smoke', 'proxy', or 'vlm'")
    if reward_mode == "proxy" and not clip_model_path:
        raise ValueError("clip_model_path is required in proxy mode")
    if reward_mode == "vlm" and not vlm_prompt_path:
        raise ValueError("vlm_prompt_path is required in vlm mode")
    if not 0.0 <= recoverable_format_reward < 1.0:
        raise ValueError("recoverable_format_reward must be in [0, 1)")
    if format_penalty_weight < 0.0:
        raise ValueError("format_penalty_weight must be non-negative")

    extra_info = extra_info or {}
    try:
        metadata = json.loads(ground_truth)
        image_width = int(metadata["image_width"])
        image_height = int(metadata["image_height"])
        target_ratio = float(metadata["target_ratio"])
        parse_result = parse_crop_action_with_format(solution_str)
        bbox = action_to_bbox(
            parse_result.action,
            image_width=image_width,
            image_height=image_height,
            target_ratio=target_ratio,
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return _empty_result(score=-1.0)

    format_reward = 1.0 if parse_result.strict_format else recoverable_format_reward

    if reward_mode == "smoke":
        return _empty_result(score=format_reward, format_reward=format_reward)

    image_path = Path(str(extra_info.get("original_image_path", "")))
    title = str(extra_info.get("title", "")).strip()
    if not image_path.is_absolute() or not title:
        return _empty_result(score=-1.0, format_reward=format_reward)

    try:
        with Image.open(image_path) as source:
            original = source.convert("RGB")
    except OSError:
        return _empty_result(score=-1.0, format_reward=format_reward)
    if original.size != (image_width, image_height):
        return _empty_result(score=-1.0, format_reward=format_reward)

    candidate = crop_image(original, bbox)
    if reward_mode == "vlm":
        caption = str(extra_info.get("caption", "")).strip()
        vlm_scorer = get_crop_vlm_scorer(str(vlm_prompt_path))
        quality_score, label = vlm_scorer.score(
            original,
            candidate,
            caption,
            title,
            log_context={
                "evaluation_id": extra_info.get("evaluation_id"),
                "sample_id": extra_info.get("sample_id"),
                "target_ratio": target_ratio,
                "action": {
                    "cx": parse_result.action.center_x,
                    "cy": parse_result.action.center_y,
                    "area": parse_result.action.area,
                },
            },
        )
        score = max(-1.0, quality_score - format_penalty_weight * (1.0 - format_reward))
        result = _empty_result(score=float(score), format_reward=format_reward)
        result.update(
            strict_format=float(parse_result.strict_format),
            vlm_enabled=1.0,
            vlm_label=float(label),
        )
        return result

    visual = compute_visual_proxy_metrics(original, bbox)
    clip_scorer = get_clip_title_scorer(clip_model_path, device=clip_device)
    original_similarity, candidate_similarity = clip_scorer.score(title, original, candidate)
    title_relevance = relative_title_relevance(
        original_similarity,
        candidate_similarity,
        scale=clip_delta_scale,
    )
    components = RewardComponents(
        valid=True,
        title_relevance=title_relevance,
        saliency=visual.saliency,
        composition=visual.composition,
        integrity=visual.integrity,
        area=visual.area,
    )
    quality_score = combine_proxy_reward(components)
    score = max(-1.0, quality_score - format_penalty_weight * (1.0 - format_reward))

    return {
        "score": float(score),
        "format_reward": format_reward,
        "strict_format": float(parse_result.strict_format),
        "title_relevance": float(title_relevance),
        "clip_original_similarity": float(original_similarity),
        "clip_candidate_similarity": float(candidate_similarity),
        "clip_similarity_delta": float(candidate_similarity - original_similarity),
        "saliency": visual.saliency,
        "composition": visual.composition,
        "integrity": visual.integrity,
        "area": visual.area,
        "area_fraction": visual.area_fraction,
        "proxy_enabled": 1.0,
    }
