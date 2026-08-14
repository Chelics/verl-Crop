import math

import numpy as np
import pytest
from PIL import Image

from news_crop_benchmark.layout import edge_median_color, pad_image_to_ratio, render_layout_action
from news_crop_benchmark.protocol import LayoutAction


def test_edge_median_color_ignores_sparse_border_marks():
    pixels = np.full((100, 120, 3), (238, 241, 244), dtype=np.uint8)
    pixels[:5, ::4] = (0, 0, 0)
    pixels[-5:, 1::4] = (255, 20, 20)

    color = edge_median_color(Image.fromarray(pixels), edge_fraction=0.05)

    assert color == (238, 241, 244)


def test_padding_preserves_source_pixels_and_uses_edge_color():
    source = Image.new("RGB", (80, 120), color=(31, 73, 113))
    source.paste((220, 30, 40), (8, 8, 72, 112))

    result = pad_image_to_ratio(source, 1.0)

    assert result.image.size == (120, 120)
    assert result.background_color == (31, 73, 113)
    assert result.content_box == (20, 0, 100, 120)
    assert result.image.crop(result.content_box).tobytes() == source.tobytes()
    assert result.image.getpixel((0, 0)) == result.background_color


def test_wide_padding_expands_canvas_without_resizing_source():
    source = Image.new("RGB", (100, 120), color="white")

    result = pad_image_to_ratio(source, 1.91, background_color=(10, 20, 30))

    assert result.image.size == (math.ceil(120 * 1.91), 120)
    assert result.content_box[2] - result.content_box[0] == source.width
    assert result.content_box[3] - result.content_box[1] == source.height
    assert result.image.crop(result.content_box).tobytes() == source.tobytes()
    assert abs(result.image.width / result.image.height - 1.91) <= 1 / result.image.height


@pytest.mark.parametrize("target_ratio", [0.0, -1.0])
def test_padding_rejects_invalid_ratio(target_ratio):
    with pytest.raises(ValueError, match="target_ratio"):
        pad_image_to_ratio(Image.new("RGB", (10, 10)), target_ratio)


def test_padding_rejects_invalid_background_color():
    with pytest.raises(ValueError, match="background_color"):
        pad_image_to_ratio(Image.new("RGB", (10, 10)), 1.0, background_color=(0, 0, 256))


def test_unified_crop_renders_exact_target_ratio():
    source = Image.new("RGB", (200, 100), color="white")
    action = LayoutAction("crop", 10, 10, 90, 90)

    result = render_layout_action(source, action, 1.0)

    assert result.operation == "crop"
    assert result.image.width == result.image.height
    assert result.background_color is None
    assert result.padding_fraction == 0.0


def test_unified_crop_pad_crops_before_padding_without_resizing():
    source = Image.new("RGB", (200, 100), color=(240, 240, 240))
    source.paste((20, 80, 120), (50, 10, 150, 90))
    action = LayoutAction("crop_pad", 25, 10, 75, 90)

    result = render_layout_action(source, action, 1.91)
    cropped = source.crop((50, 10, 150, 90))

    assert result.operation == "crop_pad"
    assert result.source_box == (50, 10, 150, 90)
    assert result.image.crop(result.content_box).tobytes() == cropped.tobytes()
    assert abs(result.image.width / result.image.height - 1.91) <= 1 / result.image.height
    assert result.padding_fraction > 0.0


def test_unified_pad_uses_full_source_and_can_be_noop():
    source = Image.new("RGB", (100, 100), color=(10, 20, 30))
    action = LayoutAction("pad", 0, 0, 100, 100)

    result = render_layout_action(source, action, 1.0)

    assert result.source_box == (0, 0, 100, 100)
    assert result.content_box == (0, 0, 100, 100)
    assert result.image.tobytes() == source.tobytes()
    assert result.padding_fraction == 0.0


def test_unified_crop_never_expands_outside_selected_rectangle():
    source = Image.new("RGB", (200, 100), color="white")
    action = LayoutAction("crop", 0, 0, 50, 50)

    result = render_layout_action(source, action, 2.0)

    left, top, right, bottom = result.source_box
    assert 0 <= left < right <= 100
    assert 0 <= top < bottom <= 50
    assert result.source_box == (0, 0, 100, 50)