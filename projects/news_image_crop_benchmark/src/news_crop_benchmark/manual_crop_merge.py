from __future__ import annotations

import gc
import hashlib
import io
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image


KEY_RATIO_DIGITS = 2
RATIO_TOLERANCE = 0.002
REPLACED_FIELDS = {
    "feasible",
    "render_mode",
    "was_cropped",
    "was_padded",
    "bbox_normalized",
    "bbox_pixels",
    "edge_artifact_trim_pixels",
    "background_color_rgb",
    "padding_color_rgb",
    "output_width",
    "output_height",
    "reason",
    "confidence",
    "cropped_image_format",
    "cropped_image",
    "error",
}


def merge_manual_crops(
    base_path: Path,
    manual_path: Path,
    output_path: Path,
    *,
    report_path: Path | None = None,
    row_group_size: int = 64,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    base_path = Path(os.path.abspath(base_path.expanduser()))
    manual_path = Path(os.path.abspath(manual_path.expanduser()))
    output_path = Path(os.path.abspath(output_path.expanduser()))
    report_path = (
        Path(os.path.abspath(report_path.expanduser()))
        if report_path is not None
        else output_path.with_suffix(".merge_report.json")
    )
    if output_path.exists():
        raise FileExistsError(output_path)
    if report_path.exists():
        raise FileExistsError(report_path)
    if row_group_size <= 0:
        raise ValueError("row_group_size must be positive")

    base = pq.read_table(base_path, pre_buffer=False)
    _require_columns(base.column_names, REPLACED_FIELDS | {"trace_id", "target_ratio", "headline", "caption"})
    manual_rows = pq.read_table(manual_path).to_pylist()
    selected, rejected, superseded = _select_manual_rows(manual_rows)

    base_keys = [
        _key(trace_id, ratio)
        for trace_id, ratio in zip(base.column("trace_id").to_pylist(), base.column("target_ratio").to_pylist())
    ]
    base_key_counts = Counter(base_keys)
    duplicate_base = [key for key, count in base_key_counts.items() if count != 1]
    if duplicate_base:
        raise ValueError(f"baseline contains duplicate keys: {duplicate_base}")
    key_to_index = {key: index for index, key in enumerate(base_keys)}
    missing_keys = sorted(set(selected) - set(key_to_index))
    if missing_keys:
        raise KeyError(f"manual keys not found in baseline: {missing_keys}")

    headlines = base.column("headline").to_pylist()
    captions = base.column("caption").to_pylist()
    replacements: dict[int, dict[str, Any]] = {}
    for key, manual in selected.items():
        index = key_to_index[key]
        if manual["title"] != headlines[index] or manual["ImageCaption"] != captions[index]:
            raise ValueError(f"manual title/caption mismatch for {key}")
        replacements[index] = _replacement_values(manual)

    mask = pa.array([index in replacements for index in range(base.num_rows)], type=pa.bool_())
    arrays = []
    for field in base.schema:
        source = base.column(field.name)
        if field.name in REPLACED_FIELDS:
            replacement_array = pa.array(
                [replacements[index][field.name] if index in replacements else None for index in range(base.num_rows)],
                type=field.type,
            )
            arrays.append(pc.if_else(mask, replacement_array, source))
        else:
            arrays.append(source)
    merged = pa.Table.from_arrays(arrays, schema=base.schema)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".partial")
    if temporary_path.exists():
        temporary_path.unlink()
    try:
        pq.write_table(
            merged,
            temporary_path,
            compression="zstd",
            row_group_size=row_group_size,
            use_dictionary=True,
            write_statistics=True,
        )
        temporary_path.replace(output_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    finally:
        del merged, arrays, mask
        gc.collect()

    validation = _validate_output(base, output_path, replacements)
    report = {
        "base": _file_record(base_path),
        "manual": _file_record(manual_path),
        "output": _file_record(output_path),
        "base_rows": base.num_rows,
        "output_rows": validation["output_rows"],
        "manual_input_rows": len(manual_rows),
        "manual_selected_rows": len(selected),
        "manual_selected_images": len({key[0] for key in selected}),
        "rejected_rows": rejected,
        "superseded_rows": superseded,
        "replacement_states": dict(Counter(_operation(row) for row in selected.values())),
        "transparent_replacements": sum(bool(row["_has_transparency"]) for row in selected.values()),
        "row_group_size": row_group_size,
        **validation,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _select_manual_rows(
    rows: list[dict[str, Any]],
) -> tuple[dict[tuple[str, float], dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    required = {
        "save_id",
        "image_id",
        "manual_crop",
        "ratio",
        "theme_color",
        "fill_color_code",
        "is_filled",
        "is_cropped",
        "crop_reason",
        "title",
        "ImageCaption",
        "saved_at",
    }
    selected: dict[tuple[str, float], dict[str, Any]] = {}
    rejected = []
    superseded = []
    for row_index, row in enumerate(rows):
        _require_columns(row, required)
        payload = row["manual_crop"]
        try:
            with Image.open(io.BytesIO(payload)) as image:
                image.load()
                width, height = image.size
                image_format = image.format
                has_transparency = image.mode in {"RGBA", "LA"} and image.getchannel("A").getextrema()[0] < 255
        except (OSError, TypeError) as error:
            rejected.append({"row_index": row_index, "save_id": row["save_id"], "reason": f"invalid_image: {error}"})
            continue
        observed_ratio = width / height
        declared_ratio = float(row["ratio"])
        if abs(observed_ratio - declared_ratio) > RATIO_TOLERANCE:
            rejected.append(
                {
                    "row_index": row_index,
                    "save_id": row["save_id"],
                    "reason": "ratio_mismatch",
                    "declared_ratio": declared_ratio,
                    "observed_ratio": observed_ratio,
                    "width": width,
                    "height": height,
                }
            )
            continue
        candidate = {
            **row,
            "_row_index": row_index,
            "_width": width,
            "_height": height,
            "_format": image_format,
            "_has_transparency": has_transparency,
        }
        key = _key(row["image_id"], declared_ratio)
        previous = selected.get(key)
        if previous is None or _timestamp(candidate["saved_at"]) > _timestamp(previous["saved_at"]):
            if previous is not None:
                superseded.append(_superseded_record(previous, key))
            selected[key] = candidate
        else:
            superseded.append(_superseded_record(candidate, key))
    if not selected:
        raise ValueError("manual dataset contains no valid replacements")
    return selected, rejected, superseded


def _replacement_values(manual: dict[str, Any]) -> dict[str, Any]:
    is_filled = bool(manual["is_filled"])
    is_cropped = bool(manual["is_cropped"])
    operation = _operation(manual)
    reason = str(manual.get("crop_reason") or "").strip()
    if not reason:
        reason = (
            f"[manual replacement] operation={operation}; save_id={manual['save_id']}; "
            f"saved_at={manual['saved_at']}"
        )
    return {
        "feasible": not is_filled,
        "render_mode": "padded" if is_filled else ("cropped" if is_cropped else "unchanged"),
        "was_cropped": is_cropped,
        "was_padded": is_filled,
        "bbox_normalized": None,
        "bbox_pixels": None,
        "edge_artifact_trim_pixels": [0, 0, 0, 0],
        "background_color_rgb": _parse_color(manual.get("theme_color")),
        "padding_color_rgb": _parse_color(manual.get("fill_color_code")) if is_filled else None,
        "output_width": int(manual["_width"]),
        "output_height": int(manual["_height"]),
        "reason": reason,
        "confidence": "high",
        "cropped_image_format": str(manual["_format"]),
        "cropped_image": bytes(manual["manual_crop"]),
        "error": None,
    }


def _validate_output(base: Any, output_path: Path, replacements: dict[int, dict[str, Any]]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(output_path)
    if not parquet_file.schema_arrow.equals(base.schema, check_metadata=True):
        raise RuntimeError("merged output schema or metadata differs from baseline")
    if parquet_file.metadata.num_rows != base.num_rows:
        raise RuntimeError("merged output row count differs from baseline")

    offset = 0
    untouched_verified = 0
    replacements_verified = 0
    for batch in parquet_file.iter_batches(batch_size=64, use_threads=False):
        output_batch = pa.Table.from_batches([batch], schema=base.schema)
        base_batch = base.slice(offset, output_batch.num_rows)
        untouched_indices = [local for local in range(output_batch.num_rows) if offset + local not in replacements]
        if untouched_indices:
            take = pa.array(untouched_indices, type=pa.int64())
            if not output_batch.take(take).equals(base_batch.take(take), check_metadata=True):
                raise RuntimeError(f"untouched rows differ near output offset {offset}")
            untouched_verified += len(untouched_indices)
        for local in range(output_batch.num_rows):
            global_index = offset + local
            if global_index not in replacements:
                continue
            row = output_batch.slice(local, 1).to_pylist()[0]
            for field, expected in replacements[global_index].items():
                if row[field] != expected:
                    raise RuntimeError(f"replacement mismatch at row {global_index}, field {field}")
            replacements_verified += 1
        offset += output_batch.num_rows
    if offset != base.num_rows or replacements_verified != len(replacements):
        raise RuntimeError("merged output validation did not cover every row")
    return {
        "output_rows": offset,
        "output_row_groups": parquet_file.metadata.num_row_groups,
        "schema_equal": True,
        "untouched_rows_verified": untouched_verified,
        "replacement_rows_verified": replacements_verified,
    }


def _operation(row: dict[str, Any]) -> str:
    if row["is_filled"] and row["is_cropped"]:
        return "crop_pad"
    if row["is_filled"]:
        return "pad"
    if row["is_cropped"]:
        return "crop"
    return "unchanged"


def _parse_color(value: Any) -> list[int] | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    if len(text) != 7 or not text.startswith("#"):
        raise ValueError(f"invalid color: {value}")
    try:
        return [int(text[index : index + 2], 16) for index in (1, 3, 5)]
    except ValueError as error:
        raise ValueError(f"invalid color: {value}") from error


def _key(image_id: Any, ratio: Any) -> tuple[str, float]:
    return str(image_id), round(float(ratio), KEY_RATIO_DIGITS)


def _timestamp(value: Any) -> datetime:
    if not isinstance(value, datetime):
        raise TypeError(f"saved_at must be a datetime, got {type(value).__name__}")
    return value


def _superseded_record(row: dict[str, Any], key: tuple[str, float]) -> dict[str, Any]:
    return {
        "row_index": row["_row_index"],
        "save_id": row["save_id"],
        "key": [key[0], key[1]],
        "saved_at": str(row["saved_at"]),
    }


def _require_columns(container: Any, required: set[str]) -> None:
    missing = sorted(required - set(container))
    if missing:
        raise ValueError(f"missing required columns: {missing}")


def _file_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path),
        "size": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()