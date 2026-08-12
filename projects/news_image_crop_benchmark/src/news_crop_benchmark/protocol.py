from __future__ import annotations

import json
import re
from dataclasses import dataclass

from news_crop_benchmark.geometry import CropAction

_CROP_PATTERN = re.compile(r"<crop>\s*(\{.*?\})\s*</crop>", re.DOTALL)
_MISSING_CLOSING_TAG_PATTERN = re.compile(r"^\s*<crop>\s*(\{.*\})\s*$", re.DOTALL)
_JSON_CODE_BLOCK_PATTERN = re.compile(r"^\s*```json\s*(\{.*\})\s*```\s*$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class CropParseResult:
    action: CropAction
    strict_format: bool


def _parse_payload(payload_text: str) -> CropAction:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ValueError("crop payload is not valid JSON") from error

    expected_keys = {"cx", "cy", "area"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(f"crop payload must contain exactly {sorted(expected_keys)}")
    if any(isinstance(payload[key], bool) or not isinstance(payload[key], int) for key in expected_keys):
        raise ValueError("crop coordinates must be integers")

    action = CropAction(center_x=float(payload["cx"]), center_y=float(payload["cy"]), area=float(payload["area"]))
    action.validate()
    return action


def parse_crop_action(response: str) -> CropAction:
    """Parse ``<crop>{...}</crop>`` from a model response."""
    match = _CROP_PATTERN.search(response)
    if match is None:
        raise ValueError("response does not contain a <crop> JSON object")

    return _parse_payload(match.group(1))


def parse_crop_action_with_format(response: str) -> CropParseResult:
    """Parse a crop action and conservatively recover a missing closing tag."""
    strict_match = _CROP_PATTERN.search(response)
    if strict_match is not None:
        return CropParseResult(action=_parse_payload(strict_match.group(1)), strict_format=True)

    recoverable_match = _MISSING_CLOSING_TAG_PATTERN.fullmatch(response)
    if recoverable_match is None:
        raise ValueError("response does not contain a recoverable <crop> JSON object")
    return CropParseResult(action=_parse_payload(recoverable_match.group(1)), strict_format=False)


def _parse_percent_payload(payload_text: str) -> CropAction:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ValueError("response must be exactly one valid JSON object") from error

    expected_keys = {"cx_pct", "cy_pct", "area_pct"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(f"percentage crop payload must contain exactly {sorted(expected_keys)}")
    if any(isinstance(payload[key], bool) or not isinstance(payload[key], int) for key in expected_keys):
        raise ValueError("percentage crop values must be integers")
    if not 0 <= payload["cx_pct"] <= 100 or not 0 <= payload["cy_pct"] <= 100:
        raise ValueError("cx_pct and cy_pct must be in [0, 100]")
    if not 1 <= payload["area_pct"] <= 100:
        raise ValueError("area_pct must be in [1, 100]")

    action = CropAction(
        center_x=float(payload["cx_pct"] * 10),
        center_y=float(payload["cy_pct"] * 10),
        area=float(payload["area_pct"] * 10),
    )
    action.validate()
    return action


def parse_percent_crop_action(response: str) -> CropParseResult:
    """Parse exact percentage JSON, conservatively recovering one JSON code block."""
    stripped = response.strip()
    try:
        return CropParseResult(action=_parse_percent_payload(stripped), strict_format=True)
    except ValueError as strict_error:
        fenced_match = _JSON_CODE_BLOCK_PATTERN.fullmatch(response)
        if fenced_match is None:
            raise strict_error
        return CropParseResult(
            action=_parse_percent_payload(fenced_match.group(1)),
            strict_format=False,
        )
