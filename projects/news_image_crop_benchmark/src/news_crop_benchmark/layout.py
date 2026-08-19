from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image

from news_crop_benchmark.geometry import BBox
from news_crop_benchmark.protocol import CropFillAction, LayoutAction


@dataclass(frozen=True)
class PadRender:
    image: Image.Image
    background_color: tuple[int, int, int]
    content_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class LayoutRender:
    image: Image.Image
    operation: str
    source_box: tuple[int, int, int, int]
    content_box: tuple[int, int, int, int]
    background_color: tuple[int, int, int] | None
    padding_fraction: float


@dataclass(frozen=True)
class CropFillRender:
    image: Image.Image
    operation: str
    source_box: tuple[int, int, int, int]
    content_box: tuple[int, int, int, int]
    background_color: tuple[int, int, int] | None
    padding_fraction: float
    output_ratio_error: float


def edge_median_color(
    image: Image.Image,
    edge_fraction: float = 0.05,
    maximum_samples: int = 100_000,
) -> tuple[int, int, int]:
    """Return a robust RGB background estimate from the source image border."""
    if not 0.0 < edge_fraction <= 0.5:
        raise ValueError("edge_fraction must be in (0, 0.5]")
    if maximum_samples <= 0:
        raise ValueError("maximum_samples must be positive")

    pixels = np.asarray(image.convert("RGB"), dtype=np.uint8)
    height, width, _ = pixels.shape
    band = max(1, int(round(min(width, height) * edge_fraction)))
    border_mask = np.zeros((height, width), dtype=bool)
    border_mask[:band, :] = True
    border_mask[-band:, :] = True
    border_mask[:, :band] = True
    border_mask[:, -band:] = True
    border_pixels = pixels[border_mask]
    if len(border_pixels) > maximum_samples:
        stride = math.ceil(len(border_pixels) / maximum_samples)
        border_pixels = border_pixels[::stride]
    median = np.median(border_pixels, axis=0)
    return tuple(int(round(float(channel))) for channel in median)


def pad_image_to_ratio(
    image: Image.Image,
    target_ratio: float,
    *,
    background_color: tuple[int, int, int] | None = None,
    edge_fraction: float = 0.05,
) -> PadRender:
    """Center an unscaled source image on the smallest enclosing target-ratio canvas."""
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")
    source = image.convert("RGB")
    width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    try:
        observed_ratio = width / height
        if observed_ratio > target_ratio:
            canvas_width = width
            canvas_height = max(height, math.ceil(width / target_ratio))
        else:
            canvas_height = height
            canvas_width = max(width, math.ceil(height * target_ratio))

        color = background_color or edge_median_color(source, edge_fraction=edge_fraction)
        if len(color) != 3 or any(isinstance(channel, bool) or not 0 <= channel <= 255 for channel in color):
            raise ValueError("background_color must contain three integer RGB values in [0, 255]")
        color = tuple(int(channel) for channel in color)
        left = (canvas_width - width) // 2
        top = (canvas_height - height) // 2
        canvas = Image.new("RGB", (canvas_width, canvas_height), color=color)
        canvas.paste(source, (left, top))
        return PadRender(
            image=canvas,
            background_color=color,
            content_box=(left, top, left + width, top + height),
        )
    finally:
        source.close()


def _percentage_box(action: LayoutAction, width: int, height: int) -> BBox:
    return BBox(
        x1=action.x1_pct / 100 * width,
        y1=action.y1_pct / 100 * height,
        x2=action.x2_pct / 100 * width,
        y2=action.y2_pct / 100 * height,
    )


def _pixel_box(box: BBox, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(round(box.x1))))
    top = max(0, min(height - 1, int(round(box.y1))))
    right = max(left + 1, min(width, int(round(box.x2))))
    bottom = max(top + 1, min(height, int(round(box.y2))))
    return left, top, right, bottom


def _normalized_pixel_box(
    box: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, math.floor(box[0] * width)))
    top = max(0, min(height - 1, math.floor(box[1] * height)))
    right = max(left + 1, min(width, math.ceil(box[2] * width)))
    bottom = max(top + 1, min(height, math.ceil(box[3] * height)))
    return left, top, right, bottom


def render_crop_fill_action(
    image: Image.Image,
    action: CropFillAction,
    expected_target_ratio: float,
) -> CropFillRender:
    """Render action-v4 exactly: crop the selected box, then optionally pad with the predicted color."""
    if expected_target_ratio <= 0:
        raise ValueError("expected_target_ratio must be positive")
    if not math.isclose(action.target_ratio, expected_target_ratio, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError(
            f"predicted target_ratio differs from task: predicted={action.target_ratio}, expected={expected_target_ratio}"
        )
    source = image.convert("RGB")
    width, height = source.size
    if width <= 0 or height <= 0:
        source.close()
        raise ValueError("image dimensions must be positive")
    try:
        if action.is_cropped:
            assert action.crop_box is not None
            source_box = _normalized_pixel_box(action.crop_box, width, height)
            retained = source.crop(source_box)
        else:
            source_box = (0, 0, width, height)
            retained = source.copy()
        try:
            if action.is_filled:
                assert action.fill_color is not None
                padded = pad_image_to_ratio(
                    retained,
                    expected_target_ratio,
                    background_color=action.fill_color,
                )
                output = padded.image
                content_box = padded.content_box
                background_color = padded.background_color
                padding_fraction = 1.0 - (retained.width * retained.height) / (output.width * output.height)
            else:
                output = retained.copy()
                content_box = (0, 0, output.width, output.height)
                background_color = None
                padding_fraction = 0.0
        finally:
            retained.close()
        actual_ratio = output.width / output.height
        return CropFillRender(
            image=output,
            operation=action.operation,
            source_box=source_box,
            content_box=content_box,
            background_color=background_color,
            padding_fraction=padding_fraction,
            output_ratio_error=abs(actual_ratio - expected_target_ratio) / expected_target_ratio,
        )
    finally:
        source.close()


def _largest_target_box_inside(box: BBox, target_ratio: float) -> BBox:
    if box.aspect_ratio > target_ratio:
        crop_height = box.height
        crop_width = crop_height * target_ratio
    else:
        crop_width = box.width
        crop_height = crop_width / target_ratio
    center_x = (box.x1 + box.x2) / 2
    center_y = (box.y1 + box.y2) / 2
    return BBox(
        x1=center_x - crop_width / 2,
        y1=center_y - crop_height / 2,
        x2=center_x + crop_width / 2,
        y2=center_y + crop_height / 2,
    )


def render_layout_action(
    image: Image.Image,
    action: LayoutAction,
    target_ratio: float,
    *,
    edge_fraction: float = 0.05,
) -> LayoutRender:
    """Render one unified layout action without resizing or distorting source pixels."""
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")
    source = image.convert("RGB")
    width, height = source.size
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")

    try:
        if action.operation == "crop":
            requested = _percentage_box(action, width, height)
            source_box = _pixel_box(_largest_target_box_inside(requested, target_ratio), width, height)
            candidate = source.crop(source_box)
            return LayoutRender(
                image=candidate,
                operation=action.operation,
                source_box=source_box,
                content_box=(0, 0, candidate.width, candidate.height),
                background_color=None,
                padding_fraction=0.0,
            )

        if action.operation == "crop_pad":
            source_box = _pixel_box(_percentage_box(action, width, height), width, height)
            cropped = source.crop(source_box)
            try:
                padded = pad_image_to_ratio(cropped, target_ratio, edge_fraction=edge_fraction)
                padding_fraction = 1.0 - (cropped.width * cropped.height) / (padded.image.width * padded.image.height)
            finally:
                cropped.close()
            return LayoutRender(
                image=padded.image,
                operation=action.operation,
                source_box=source_box,
                content_box=padded.content_box,
                background_color=padded.background_color,
                padding_fraction=padding_fraction,
            )

        if action.operation == "pad":
            source_box = (0, 0, width, height)
            padded = pad_image_to_ratio(source, target_ratio, edge_fraction=edge_fraction)
            padding_fraction = 1.0 - (width * height) / (padded.image.width * padded.image.height)
            return LayoutRender(
                image=padded.image,
                operation=action.operation,
                source_box=source_box,
                content_box=padded.content_box,
                background_color=padded.background_color,
                padding_fraction=padding_fraction,
            )

        raise ValueError(f"unsupported layout operation: {action.operation}")
    finally:
        source.close()