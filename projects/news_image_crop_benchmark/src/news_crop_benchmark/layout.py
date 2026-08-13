from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class PadRender:
    image: Image.Image
    background_color: tuple[int, int, int]
    content_box: tuple[int, int, int, int]


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