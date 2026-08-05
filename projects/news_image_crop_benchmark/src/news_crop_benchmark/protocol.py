from __future__ import annotations

import json
import re

from news_crop_benchmark.geometry import CropAction

_CROP_PATTERN = re.compile(r"<crop>\s*(\{.*?\})\s*</crop>", re.DOTALL)


def parse_crop_action(response: str) -> CropAction:
    """Parse ``<crop>{...}</crop>`` from a model response."""
    match = _CROP_PATTERN.search(response)
    if match is None:
        raise ValueError("response does not contain a <crop> JSON object")

    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError as error:
        raise ValueError("crop payload is not valid JSON") from error

    expected_keys = {"cx", "cy", "area"}
    if set(payload) != expected_keys:
        raise ValueError(f"crop payload must contain exactly {sorted(expected_keys)}")
    if any(isinstance(payload[key], bool) or not isinstance(payload[key], int | float) for key in expected_keys):
        raise ValueError("crop coordinates must be numeric")

    action = CropAction(center_x=float(payload["cx"]), center_y=float(payload["cy"]), area=float(payload["area"]))
    action.validate()
    return action
