from __future__ import annotations

import json
import math
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


@dataclass(frozen=True)
class ModeDecision:
    mode: str


@dataclass(frozen=True)
class LayoutAction:
    operation: str
    x1_pct: int
    y1_pct: int
    x2_pct: int
    y2_pct: int


@dataclass(frozen=True)
class LayoutParseResult:
    action: LayoutAction
    strict_format: bool


@dataclass(frozen=True)
class CropFillAction:
    target_ratio: float
    is_cropped: bool
    is_filled: bool
    crop_box: tuple[float, float, float, float] | None
    fill_color: tuple[int, int, int] | None

    @property
    def operation(self) -> str:
        return {
            (True, False): "crop",
            (True, True): "crop_fill",
            (False, True): "fill",
            (False, False): "keep",
        }[(self.is_cropped, self.is_filled)]


@dataclass(frozen=True)
class CropFillParseResult:
    action: CropFillAction
    strict_format: bool


def parse_mode_decision(response: str) -> ModeDecision:
    """Parse an exact ``{"mode":"crop|pad"}`` JSON response."""
    try:
        payload = json.loads(response)
    except json.JSONDecodeError as error:
        raise ValueError("response must be exactly one valid JSON object") from error

    if not isinstance(payload, dict) or set(payload) != {"mode"}:
        raise ValueError("mode payload must contain exactly ['mode']")
    if payload["mode"] not in {"crop", "pad"}:
        raise ValueError("mode must be exactly 'crop' or 'pad'")
    return ModeDecision(mode=payload["mode"])


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


def _parse_layout_payload(payload_text: str) -> LayoutAction:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ValueError("response must be exactly one valid JSON object") from error

    expected_keys = {"operation", "x1_pct", "y1_pct", "x2_pct", "y2_pct"}
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError(f"layout payload must contain exactly {sorted(expected_keys)}")
    if payload["operation"] not in {"crop", "crop_pad", "pad"}:
        raise ValueError("operation must be exactly 'crop', 'crop_pad', or 'pad'")
    coordinate_keys = expected_keys - {"operation"}
    if any(isinstance(payload[key], bool) or not isinstance(payload[key], int) for key in coordinate_keys):
        raise ValueError("layout coordinates must be integers")
    if any(not 0 <= payload[key] <= 100 for key in coordinate_keys):
        raise ValueError("layout coordinates must be in [0, 100]")
    if payload["x1_pct"] >= payload["x2_pct"] or payload["y1_pct"] >= payload["y2_pct"]:
        raise ValueError("layout lower bounds must be smaller than upper bounds")
    if payload["operation"] == "pad" and any(
        payload[key] != expected
        for key, expected in {"x1_pct": 0, "y1_pct": 0, "x2_pct": 100, "y2_pct": 100}.items()
    ):
        raise ValueError("pad operation must use the full-image box [0, 0, 100, 100]")
    return LayoutAction(**payload)


def parse_layout_action(response: str) -> LayoutParseResult:
    """Parse the unified crop/crop-pad/pad action with conservative fence recovery."""
    stripped = response.strip()
    try:
        return LayoutParseResult(action=_parse_layout_payload(stripped), strict_format=True)
    except ValueError as strict_error:
        fenced_match = _JSON_CODE_BLOCK_PATTERN.fullmatch(response)
        if fenced_match is None:
            raise strict_error
        return LayoutParseResult(
            action=_parse_layout_payload(fenced_match.group(1)),
            strict_format=False,
        )


def _parse_crop_fill_payload(payload_text: str) -> CropFillAction:
    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as error:
        raise ValueError("response must be exactly one valid JSON object") from error
    expected_fields = ("target_ratio", "is_cropped", "is_filled", "crop_box", "fill_color")
    if not isinstance(payload, dict) or tuple(payload) != expected_fields:
        raise ValueError(f"crop/fill payload fields must be ordered as {expected_fields}")
    ratio = payload["target_ratio"]
    if isinstance(ratio, bool) or not isinstance(ratio, int | float) or not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("target_ratio must be a positive finite number")
    is_cropped = payload["is_cropped"]
    is_filled = payload["is_filled"]
    if not isinstance(is_cropped, bool) or not isinstance(is_filled, bool):
        raise ValueError("is_cropped and is_filled must be booleans")

    raw_box = payload["crop_box"]
    crop_box = None
    if is_cropped:
        if (
            not isinstance(raw_box, list)
            or len(raw_box) != 4
            or any(isinstance(value, bool) or not isinstance(value, int | float) for value in raw_box)
            or not all(math.isfinite(float(value)) for value in raw_box)
            or not (0 <= raw_box[0] < raw_box[2] <= 1 and 0 <= raw_box[1] < raw_box[3] <= 1)
        ):
            raise ValueError("cropped action must contain a valid normalized crop_box")
        crop_box = tuple(float(value) for value in raw_box)
    elif raw_box is not None:
        raise ValueError("non-cropped action must use crop_box=null")

    raw_color = payload["fill_color"]
    fill_color = None
    if is_filled:
        if (
            not isinstance(raw_color, list)
            or len(raw_color) != 3
            or any(
                isinstance(channel, bool) or not isinstance(channel, int) or not 0 <= channel <= 255
                for channel in raw_color
            )
        ):
            raise ValueError("filled action must contain integer RGB fill_color")
        fill_color = tuple(raw_color)
    elif raw_color is not None:
        raise ValueError("non-filled action must use fill_color=null")
    return CropFillAction(float(ratio), is_cropped, is_filled, crop_box, fill_color)


def parse_crop_fill_action(response: str) -> CropFillParseResult:
    """Parse the exact action-v4 JSON protocol without Markdown recovery."""
    return CropFillParseResult(action=_parse_crop_fill_payload(response.strip()), strict_format=True)
