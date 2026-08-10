from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from news_crop_benchmark.geometry import TARGET_RATIOS

DATA_SOURCE = "news_image_crop"
ABILITY = "news_image_cropping"
SPLIT_NAMES = ("train", "validation", "test")
DEFAULT_SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
DEFAULT_POLICY_PROMPT_TEMPLATE = """<image>
News title: {title}
Target aspect ratio (width / height): {target_ratio}
Select the crop that best illustrates the news title.
Return exactly one line: <crop>{"cx": CX, "cy": CY, "area": AREA}</crop>
CX is the horizontal crop-center coordinate: 0 is the left edge and 1000 is the right edge.
CY is the vertical crop-center coordinate: 0 is the top edge and 1000 is the bottom edge.
AREA is the crop area as thousandths of the original image area: 1 is 0.1% and 1000 is the full image.
Use integers only. Do not include explanations or any other text."""
_POLICY_PROMPT_VARIABLE_PATTERN = re.compile(r"\{(title|target_ratio)\}")


def stable_id(namespace: str, *parts: str) -> str:
    if not namespace or not parts or any(not part for part in parts):
        raise ValueError("namespace and identity parts must be non-empty")
    payload = "\0".join((namespace, *parts)).encode()
    return hashlib.sha256(payload).hexdigest()


def sample_id(trace_id: str) -> str:
    return stable_id("sample", trace_id)


def asset_id(url: str) -> str:
    return stable_id("asset", url)


def training_sample_id(original_url: str, title: str, target_ratio: float) -> str:
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("title must be non-empty")
    _validate_target_ratio(target_ratio)
    return stable_id("training-sample", original_url, clean_title, f"{target_ratio:g}")


def assign_group_split(
    group_key: str,
    seed: int = 42,
    fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS,
) -> str:
    if not group_key:
        raise ValueError("group_key must be non-empty")
    if len(fractions) != len(SPLIT_NAMES) or any(fraction < 0 for fraction in fractions):
        raise ValueError("fractions must contain three non-negative values")
    if not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must sum to 1")

    digest = hashlib.blake2b(f"{seed}\0{group_key}".encode(), digest_size=8).digest()
    position = int.from_bytes(digest, byteorder="big") / 2**64
    boundary = 0.0
    for name, fraction in zip(SPLIT_NAMES, fractions, strict=True):
        boundary += fraction
        if position < boundary:
            return name
    return SPLIT_NAMES[-1]


def load_policy_prompt_template(path: str | Path | None = None) -> str:
    if path is None:
        return DEFAULT_POLICY_PROMPT_TEMPLATE
    template = Path(path).read_text(encoding="utf-8").strip()
    validate_policy_prompt_template(template)
    return template


def validate_policy_prompt_template(template: str) -> None:
    if not template:
        raise ValueError("policy prompt template must be non-empty")
    if not template.startswith("<image>\n"):
        raise ValueError("policy prompt template must start with '<image>' on its own line")
    for variable in ("title", "target_ratio"):
        if f"{{{variable}}}" not in template:
            raise ValueError(f"policy prompt template must contain '{{{variable}}}'")


def build_prompt(
    title: str,
    target_ratio: float,
    policy_prompt_template: str | None = None,
) -> str:
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("title must be non-empty")
    _validate_target_ratio(target_ratio)
    template = DEFAULT_POLICY_PROMPT_TEMPLATE if policy_prompt_template is None else policy_prompt_template
    validate_policy_prompt_template(template)
    values = {"title": clean_title, "target_ratio": f"{target_ratio:g}"}
    return _POLICY_PROMPT_VARIABLE_PATTERN.sub(
        lambda match: values[match.group(1)],
        template,
    )


def build_verl_row(
    *,
    sample_identifier: str,
    source_index: int,
    split: str,
    title: str,
    original_image_path: str,
    image_width: int,
    image_height: int,
    target_ratio: float,
    policy_prompt_template: str | None = None,
) -> dict[str, Any]:
    if not sample_identifier:
        raise ValueError("sample_identifier must be non-empty")
    if source_index < 0:
        raise ValueError("source_index must be non-negative")
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}")
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not Path(original_image_path).is_absolute():
        raise ValueError("original_image_path must be an absolute shared path")
    _validate_target_ratio(target_ratio)

    ground_truth = json.dumps(
        {
            "image_height": image_height,
            "image_width": image_width,
            "target_ratio": target_ratio,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "data_source": DATA_SOURCE,
        "prompt": [
            {
                "role": "user",
                "content": build_prompt(title, target_ratio, policy_prompt_template),
            }
        ],
        "images": [original_image_path],
        "ability": ABILITY,
        "reward_model": {"style": "proxy", "ground_truth": ground_truth},
        "extra_info": {
            "index": source_index,
            "sample_id": sample_identifier,
            "split": split,
            "title": title.strip(),
            "target_ratio": target_ratio,
            "image_width": image_width,
            "image_height": image_height,
            "original_image_path": original_image_path,
        },
    }


def _validate_target_ratio(target_ratio: float) -> None:
    if not any(math.isclose(target_ratio, ratio, abs_tol=1e-9) for ratio in TARGET_RATIOS):
        raise ValueError(f"target_ratio must be one of {TARGET_RATIOS}")
