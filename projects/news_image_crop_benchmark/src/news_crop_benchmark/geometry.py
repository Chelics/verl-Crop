from __future__ import annotations

import math
from dataclasses import dataclass

TARGET_RATIOS = (1.0, 1.91, 1.77, 1.59)
ACTION_SCALE = 1000.0


@dataclass(frozen=True)
class CropAction:
    """Normalized crop action emitted by the policy."""

    center_x: float
    center_y: float
    area: float

    def validate(self) -> None:
        if not 0.0 <= self.center_x <= ACTION_SCALE:
            raise ValueError(f"center_x must be in [0, {ACTION_SCALE:g}]")
        if not 0.0 <= self.center_y <= ACTION_SCALE:
            raise ValueError(f"center_y must be in [0, {ACTION_SCALE:g}]")
        if not 0.0 < self.area <= ACTION_SCALE:
            raise ValueError(f"area must be in (0, {ACTION_SCALE:g}]")


@dataclass(frozen=True)
class BBox:
    """Axis-aligned crop box in source-image pixels."""

    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def aspect_ratio(self) -> float:
        if self.height <= 0:
            raise ValueError("bbox height must be positive")
        return self.width / self.height


def nearest_target_ratio(width: int, height: int, target_ratios: tuple[float, ...] = TARGET_RATIOS) -> float:
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if not target_ratios or any(ratio <= 0 for ratio in target_ratios):
        raise ValueError("target_ratios must contain positive values")

    observed_ratio = width / height
    return min(target_ratios, key=lambda ratio: abs(math.log(observed_ratio / ratio)))


def action_to_bbox(
    action: CropAction,
    image_width: int,
    image_height: int,
    target_ratio: float,
) -> BBox:
    """Convert a normalized center/area action into an in-bounds, fixed-ratio crop."""
    action.validate()
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")

    requested_area = action.area / ACTION_SCALE * image_width * image_height
    crop_width = math.sqrt(requested_area * target_ratio)
    crop_height = math.sqrt(requested_area / target_ratio)

    fit_scale = min(1.0, image_width / crop_width, image_height / crop_height)
    crop_width *= fit_scale
    crop_height *= fit_scale

    raw_center_x = action.center_x / ACTION_SCALE * image_width
    raw_center_y = action.center_y / ACTION_SCALE * image_height
    center_x = min(max(raw_center_x, crop_width / 2), image_width - crop_width / 2)
    center_y = min(max(raw_center_y, crop_height / 2), image_height - crop_height / 2)

    return BBox(
        x1=center_x - crop_width / 2,
        y1=center_y - crop_height / 2,
        x2=center_x + crop_width / 2,
        y2=center_y + crop_height / 2,
    )


def bbox_iou(left: BBox, right: BBox) -> float:
    intersection_width = max(0.0, min(left.x2, right.x2) - max(left.x1, right.x1))
    intersection_height = max(0.0, min(left.y2, right.y2) - max(left.y1, right.y1))
    intersection = intersection_width * intersection_height
    union = left.area + right.area - intersection
    return intersection / union if union > 0 else 0.0
