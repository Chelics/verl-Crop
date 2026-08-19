#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


REQUIRED_PLAN_FIELDS = {
    "target_ratio",
    "is_cropped",
    "is_filled",
    "crop_box",
    "fill_color",
    "description",
}


def parse_response(text: str) -> dict[str, Any]:
    start = text.find("{")
    if start < 0:
        raise ValueError("response does not contain a JSON object")
    decoder = json.JSONDecoder()
    result: dict[str, Any] = {}
    position = start
    while position < len(text):
        fragment, consumed = decoder.raw_decode(text[position:])
        if not isinstance(fragment, dict):
            raise ValueError("response JSON fragment must be an object")
        result.update(fragment)
        position += consumed
        while position < len(text) and text[position].isspace():
            position += 1
        if position >= len(text) or text[position] != ",":
            break
        position += 1
        while position < len(text) and text[position].isspace():
            position += 1
    missing = REQUIRED_PLAN_FIELDS - set(result)
    if missing:
        raise ValueError(f"response is missing fields: {sorted(missing)}")
    return result


def normalized_pixel_box(image: Image.Image, crop_box: list[float]) -> tuple[int, int, int, int]:
    if len(crop_box) != 4:
        raise ValueError("crop_box must contain four values")
    left, top, right, bottom = (min(1.0, max(0.0, float(value))) for value in crop_box)
    if right <= left or bottom <= top:
        raise ValueError(f"invalid crop_box: {crop_box}")
    width, height = image.size
    return (
        max(0, min(width - 1, math.floor(left * width))),
        max(0, min(height - 1, math.floor(top * height))),
        max(1, min(width, math.ceil(right * width))),
        max(1, min(height, math.ceil(bottom * height))),
    )


def pad_to_ratio(image: Image.Image, target_ratio: float, color: list[int]) -> Image.Image:
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")
    if not color or len(color) != 3:
        raise ValueError("fill_color must contain three RGB values when is_filled is true")
    rgb = tuple(max(0, min(255, int(value))) for value in color)
    width, height = image.size
    canvas_size = (width, math.ceil(width / target_ratio)) if width / height > target_ratio else (math.ceil(height * target_ratio), height)
    canvas = Image.new("RGB", canvas_size, rgb)
    canvas.paste(image, ((canvas.width - width) // 2, (canvas.height - height) // 2))
    return canvas


def outer_background_color(image: Image.Image, target_ratio: float) -> list[int]:
    width, height = image.size
    strip_width = max(1, round(width * 0.02))
    strip_height = max(1, round(height * 0.02))
    if width / height > target_ratio:
        edges = [image.crop((0, 0, width, strip_height)), image.crop((0, height - strip_height, width, height))]
    else:
        edges = [image.crop((0, 0, strip_width, height)), image.crop((width - strip_width, 0, width, height))]
    samples = Image.new("RGB", (sum(edge.width for edge in edges), max(edge.height for edge in edges)))
    offset = 0
    for edge in edges:
        samples.paste(edge, (offset, 0))
        offset += edge.width
    return [round(value) for value in ImageStat.Stat(samples).median]


def render_safe_box(
    image: Image.Image,
    crop_box: list[float] | None,
    target_ratio: float,
    color: list[int] | None,
) -> tuple[Image.Image, str, list[int] | None]:
    if crop_box is None:
        left, top, right, bottom = 0, 0, *image.size
    else:
        left, top, right, bottom = normalized_pixel_box(image, crop_box)
    width, height = image.size
    if width / height > target_ratio:
        crop_width, crop_height = round(height * target_ratio), height
    else:
        crop_width, crop_height = width, round(width / target_ratio)

    if right - left <= crop_width and bottom - top <= crop_height:
        center_left = round((left + right - crop_width) / 2)
        center_top = round((top + bottom - crop_height) / 2)
        expanded_left = min(left, max(right - crop_width, center_left))
        expanded_top = min(top, max(bottom - crop_height, center_top))
        expanded_left = max(0, min(width - crop_width, expanded_left))
        expanded_top = max(0, min(height - crop_height, expanded_top))
        return (
            image.crop((expanded_left, expanded_top, expanded_left + crop_width, expanded_top + crop_height)),
            "safe_box_expanded_crop",
            None,
        )

    safe_crop = image.crop((left, top, right, bottom))
    fill_color = color or outer_background_color(safe_crop, target_ratio)
    return pad_to_ratio(safe_crop, target_ratio, fill_color), "safe_box_padding", fill_color


def apply_plan(source: Image.Image, plan: dict[str, Any], safe_fallback_padding: bool) -> tuple[Image.Image, str, list[int] | None]:
    original = source.copy().convert("RGB")
    result = original
    if plan["is_cropped"]:
        if plan["crop_box"] is None:
            raise ValueError("is_cropped is true but crop_box is null")
        result = result.crop(normalized_pixel_box(result, plan["crop_box"]))
    target_ratio = float(plan["target_ratio"])
    if plan["is_filled"]:
        result = pad_to_ratio(result, target_ratio, plan["fill_color"])
    actual_ratio = result.width / result.height
    tolerance = max(0.01, 2 / result.height)
    if abs(actual_ratio - target_ratio) <= tolerance:
        return result, "explicit_plan", plan["fill_color"]
    if safe_fallback_padding and not plan["is_filled"]:
        return render_safe_box(original, plan["crop_box"], target_ratio, plan["fill_color"])
    raise ValueError(
        f"explicit crop/pad operations produced ratio {actual_ratio:.4f}, expected {target_ratio:g}"
    )


def image_reference(row: dict[str, Any]) -> str:
    image = row["images"][0]
    return image["path"] if isinstance(image, dict) else image


def load_records(results_path: Path, image_root: Path | None) -> list[dict[str, Any]]:
    records = []
    seen: set[tuple[str, float]] = set()
    with results_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            source_reference = image_reference(row)
            source_path = (image_root / Path(source_reference).name) if image_root else Path(source_reference)
            plan = parse_response(row["response"])
            image_id = source_path.stem
            target_ratio = float(plan["target_ratio"])
            key = (image_id, target_ratio)
            if key in seen:
                raise ValueError(f"line {line_number}: duplicate image/ratio key {key}")
            if not source_path.is_file():
                raise FileNotFoundError(f"line {line_number}: source image not found: {source_path}")
            seen.add(key)
            records.append(
                {
                    "line_number": line_number,
                    "image_id": image_id,
                    "source_reference": source_reference,
                    "source_path": source_path,
                    "plan": plan,
                    "response": row["response"],
                }
            )
    if not records:
        raise ValueError(f"no records found in {results_path}")
    return records


def ratio_name(target_ratio: float) -> str:
    return f"{target_ratio:g}"


def load_source_image(path: Path, attempts: int = 3) -> Image.Image:
    last_error: OSError | None = None
    for _ in range(attempts):
        try:
            with Image.open(path) as source:
                return source.convert("RGB")
        except OSError as error:
            last_error = error
    raise OSError(f"failed to read source image after {attempts} attempts: {path}") from last_error


def validate_render(record: dict[str, Any], result: Image.Image) -> None:
    target_ratio = float(record["plan"]["target_ratio"])
    tolerance = max(0.01, 2 / result.height)
    if abs(result.width / result.height - target_ratio) > tolerance:
        raise ValueError(
            f"line {record['line_number']}: rendered ratio {result.width / result.height:.4f} "
            f"does not match {target_ratio:g}"
        )


def validate_records(records: list[dict[str, Any]], safe_fallback_padding: bool) -> dict[str, int]:
    action_counts: dict[str, int] = {}
    for record in records:
        source = load_source_image(record["source_path"])
        result, action, _ = apply_plan(source, record["plan"], safe_fallback_padding)
        validate_render(record, result)
        action_counts[action] = action_counts.get(action, 0) + 1
    return action_counts


def export_records(
    records: list[dict[str, Any]],
    output_dir: Path,
    safe_fallback_padding: bool,
    overwrite: bool,
) -> dict[str, int]:
    if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {output_dir}; pass --overwrite to replace files")
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.jsonl"
    temporary_manifest = output_dir / ".manifest.jsonl.tmp"
    action_counts: dict[str, int] = {}
    with temporary_manifest.open("w", encoding="utf-8") as manifest:
        for record in records:
            source = load_source_image(record["source_path"])
            result, action, render_fill_color = apply_plan(source, record["plan"], safe_fallback_padding)
            validate_render(record, result)
            action_counts[action] = action_counts.get(action, 0) + 1
            target_ratio = float(record["plan"]["target_ratio"])
            relative_path = Path(record["image_id"]) / f"{ratio_name(target_ratio)}.png"
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and not overwrite:
                raise FileExistsError(f"output file already exists: {output_path}")
            temporary_path = output_path.with_suffix(".tmp.png")
            result.save(temporary_path, format="PNG", optimize=True)
            temporary_path.replace(output_path)
            manifest.write(
                json.dumps(
                    {
                        "image_id": record["image_id"],
                        "target_ratio": target_ratio,
                        "source_path": record["source_reference"],
                        "render_path": relative_path.as_posix(),
                        "output_width": result.width,
                        "output_height": result.height,
                        "render_action": action,
                        "render_fill_color": render_fill_color,
                        "plan": record["plan"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    temporary_manifest.replace(manifest_path)
    return action_counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render Swift crop inference JSONL into image files.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, help="Resolve each source image by basename under this directory.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--safe-fallback-padding", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate every render without writing output files.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = load_records(args.results, args.image_root)
    action_counts = (
        validate_records(records, args.safe_fallback_padding)
        if args.dry_run
        else export_records(records, args.output_dir, args.safe_fallback_padding, args.overwrite)
    )
    summary = {
        "records": len(records),
        "unique_images": len({record["image_id"] for record in records}),
        "render_actions": action_counts,
        "dry_run": args.dry_run,
    }
    if not args.dry_run:
        summary["output_dir"] = str(args.output_dir.resolve())
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()