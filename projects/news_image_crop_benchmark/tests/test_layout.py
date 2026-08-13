import math

import numpy as np
import pytest
from PIL import Image

from news_crop_benchmark.layout import edge_median_color, pad_image_to_ratio


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