from __future__ import annotations

import io
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from news_crop_benchmark.layout import pad_image_to_ratio


def load_override_specs(path: Path) -> list[dict[str, Any]]:
    specs = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    keys = set()
    for spec in specs:
        expected = {"trace_id", "aspect_ratio", "operation", "source_box_normalized", "reason"}
        missing = expected - set(spec)
        if missing:
            raise ValueError(f"override spec missing fields: {sorted(missing)}")
        key = (str(spec["trace_id"]), str(spec["aspect_ratio"]))
        if key in keys:
            raise ValueError(f"duplicate override: {key}")
        keys.add(key)
        if spec["operation"] not in {"crop", "crop_pad", "pad"}:
            raise ValueError(f"unsupported override operation: {spec['operation']}")
        box = spec["source_box_normalized"]
        if len(box) != 4 or not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError(f"invalid source_box_normalized: {box}")
    return specs


def render_overrides(
    train_path: Path,
    manifest_path: Path,
    output_dir: Path,
    *,
    quality: int = 95,
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    if not 1 <= quality <= 100:
        raise ValueError("quality must be in [1, 100]")
    specs = load_override_specs(manifest_path)
    required_ids = {str(spec["trace_id"]) for spec in specs}
    table = pq.read_table(train_path, columns=["image_id", "original_image"], pre_buffer=False)
    source_images = {
        str(image_id): bytes(payload)
        for image_id, payload in zip(table.column("image_id").to_pylist(), table.column("original_image").to_pylist())
        if str(image_id) in required_ids
    }
    missing_ids = sorted(required_ids - set(source_images))
    if missing_ids:
        raise KeyError(f"override source IDs not found: {missing_ids}")

    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for spec in specs:
        trace_id = str(spec["trace_id"])
        target_ratio = _parse_ratio(str(spec["aspect_ratio"]))
        with Image.open(io.BytesIO(source_images[trace_id])) as source_file:
            source = ImageOps.exif_transpose(source_file).convert("RGB")
        try:
            source_box = _pixel_box(spec["source_box_normalized"], source.size)
            selected = source.crop(source_box)
            try:
                operation = str(spec["operation"])
                background = spec.get("background_color_rgb")
                background_color = tuple(int(value) for value in background) if background is not None else None
                if operation == "crop":
                    candidate, fitted_box = _fit_box_to_ratio(selected, target_ratio)
                    source_box = (
                        source_box[0] + fitted_box[0],
                        source_box[1] + fitted_box[1],
                        source_box[0] + fitted_box[2],
                        source_box[1] + fitted_box[3],
                    )
                    content_box = (0, 0, candidate.width, candidate.height)
                    padding_fraction = 0.0
                elif operation == "crop_pad":
                    padded = pad_image_to_ratio(selected, target_ratio, background_color=background_color)
                    candidate = padded.image
                    background_color = padded.background_color
                    content_box = padded.content_box
                    padding_fraction = 1.0 - (selected.width * selected.height) / (candidate.width * candidate.height)
                else:
                    padded = pad_image_to_ratio(source, target_ratio, background_color=background_color)
                    candidate = padded.image
                    source_box = (0, 0, source.width, source.height)
                    background_color = padded.background_color
                    content_box = padded.content_box
                    padding_fraction = 1.0 - (source.width * source.height) / (candidate.width * candidate.height)
            finally:
                selected.close()

            ratio_slug = str(spec["aspect_ratio"]).replace(":", "_").replace(".", "p")
            image_path = images_dir / f"{trace_id}__{ratio_slug}.jpg"
            candidate.save(image_path, format="JPEG", quality=quality, optimize=True)
            records.append(
                {
                    **spec,
                    "image_path": image_path.relative_to(output_dir).as_posix(),
                    "source_box_pixels": list(source_box),
                    "content_box_pixels": list(content_box),
                    "background_color_rgb": list(background_color) if background_color is not None else None,
                    "padding_fraction": padding_fraction,
                    "output_width": candidate.width,
                    "output_height": candidate.height,
                }
            )
            candidate.close()
        finally:
            source.close()

    (output_dir / "overrides.jsonl").write_text(
        "".join(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n" for record in records),
        encoding="utf-8",
    )
    return records


def _parse_ratio(value: str) -> float:
    numerator, denominator = value.split(":", maxsplit=1)
    ratio = float(numerator) / float(denominator)
    if ratio <= 0:
        raise ValueError(f"invalid aspect ratio: {value}")
    return ratio


def _pixel_box(box: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    left = max(0, min(width - 1, round(float(box[0]) * width)))
    top = max(0, min(height - 1, round(float(box[1]) * height)))
    right = max(left + 1, min(width, round(float(box[2]) * width)))
    bottom = max(top + 1, min(height, round(float(box[3]) * height)))
    return left, top, right, bottom


def _fit_box_to_ratio(
    image: Image.Image,
    target_ratio: float,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    observed_ratio = image.width / image.height
    if observed_ratio > target_ratio:
        width = max(1, round(image.height * target_ratio))
        left = (image.width - width) // 2
        box = (left, 0, left + width, image.height)
    else:
        height = max(1, round(image.width / target_ratio))
        top = (image.height - height) // 2
        box = (0, top, image.width, top + height)
    candidate = image.crop(box)
    if not math.isclose(candidate.width / candidate.height, target_ratio, abs_tol=1 / candidate.height):
        raise RuntimeError("rendered override does not match target ratio")
    return candidate, box