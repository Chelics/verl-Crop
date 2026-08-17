from __future__ import annotations

import gc
import hashlib
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


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


def merge_cropped_overrides(
    base_path: Path,
    override_dir: Path,
    output_path: Path,
    *,
    report_path: Path | None = None,
    row_group_size: int = 64,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    base_path = Path(os.path.abspath(base_path.expanduser()))
    override_dir = Path(os.path.abspath(override_dir.expanduser()))
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

    manifest_path = override_dir / "overrides.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    override_records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not override_records:
        raise ValueError("override manifest is empty")
    override_by_key: dict[tuple[str, float], dict[str, Any]] = {}
    for record in override_records:
        key = _key(record["trace_id"], _parse_ratio(record["aspect_ratio"]))
        if key in override_by_key:
            raise ValueError(f"duplicate override key: {key}")
        image_path = override_dir / str(record["image_path"])
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        override_by_key[key] = {**record, "_image_bytes": image_path.read_bytes()}

    base = pq.read_table(base_path, pre_buffer=False)
    required = REPLACED_FIELDS | {"trace_id", "target_ratio", "original_width", "original_height"}
    missing = sorted(required - set(base.column_names))
    if missing:
        raise ValueError(f"baseline missing fields: {missing}")
    base_rows = base.select(["trace_id", "target_ratio", "original_width", "original_height"]).to_pylist()
    base_keys = [_key(row["trace_id"], row["target_ratio"]) for row in base_rows]
    if len(base_keys) != len(set(base_keys)):
        raise ValueError("baseline contains duplicate trace-ratio keys")
    key_to_index = {key: index for index, key in enumerate(base_keys)}
    missing_keys = sorted(set(override_by_key) - set(key_to_index))
    if missing_keys:
        raise KeyError(f"override keys not found in baseline: {missing_keys}")

    replacements: dict[int, dict[str, Any]] = {}
    for key, record in override_by_key.items():
        index = key_to_index[key]
        replacements[index] = _replacement_values(record, base_rows[index])

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
    temporary_path.unlink(missing_ok=True)
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
        "override_manifest": _file_record(manifest_path),
        "output": _file_record(output_path),
        "base_rows": base.num_rows,
        "override_rows": len(replacements),
        "override_images": len({key[0] for key in override_by_key}),
        "operation_counts": dict(Counter(record["operation"] for record in override_records)),
        "reason_source": "local_gpt_layout_v2_manifest",
        "row_group_size": row_group_size,
        **validation,
    }
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def _replacement_values(record: dict[str, Any], base_row: dict[str, Any]) -> dict[str, Any]:
    operation = str(record["operation"])
    if operation not in {"crop", "crop_pad", "pad"}:
        raise ValueError(f"unsupported operation: {operation}")
    box = [int(value) for value in record["source_box_pixels"]]
    width = int(base_row["original_width"])
    height = int(base_row["original_height"])
    if len(box) != 4 or not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height):
        raise ValueError(f"invalid source box for {record['trace_id']}: {box}")
    normalized = [box[0] / width, box[1] / height, box[2] / width, box[3] / height]
    background = record.get("background_color_rgb")
    is_padded = operation in {"crop_pad", "pad"}
    return {
        "feasible": operation == "crop",
        "render_mode": "padded" if is_padded else "cropped",
        "was_cropped": operation in {"crop", "crop_pad"},
        "was_padded": is_padded,
        "bbox_normalized": normalized,
        "bbox_pixels": box,
        "edge_artifact_trim_pixels": [0, 0, 0, 0],
        "background_color_rgb": list(background) if background is not None else None,
        "padding_color_rgb": list(background) if is_padded and background is not None else None,
        "output_width": int(record["output_width"]),
        "output_height": int(record["output_height"]),
        "reason": str(record["reason"]),
        "confidence": "high",
        "cropped_image_format": "JPEG",
        "cropped_image": bytes(record["_image_bytes"]),
        "error": None,
    }


def _validate_output(base: Any, output_path: Path, replacements: dict[int, dict[str, Any]]) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    parquet_file = pq.ParquetFile(output_path)
    if not parquet_file.schema_arrow.equals(base.schema, check_metadata=True):
        raise RuntimeError("output schema differs from baseline")
    if parquet_file.metadata.num_rows != base.num_rows:
        raise RuntimeError("output row count differs from baseline")
    offset = 0
    untouched_verified = 0
    replacement_verified = 0
    for batch in parquet_file.iter_batches(batch_size=64, use_threads=False):
        output_batch = pa.Table.from_batches([batch], schema=base.schema)
        base_batch = base.slice(offset, output_batch.num_rows)
        untouched = [local for local in range(output_batch.num_rows) if offset + local not in replacements]
        if untouched:
            indices = pa.array(untouched, type=pa.int64())
            if not output_batch.take(indices).equals(base_batch.take(indices), check_metadata=True):
                raise RuntimeError(f"untouched rows differ near offset {offset}")
            untouched_verified += len(untouched)
        for local in range(output_batch.num_rows):
            global_index = offset + local
            if global_index not in replacements:
                continue
            row = output_batch.slice(local, 1).to_pylist()[0]
            for field, expected in replacements[global_index].items():
                if row[field] != expected:
                    raise RuntimeError(f"replacement mismatch at row {global_index}, field {field}")
            replacement_verified += 1
        offset += output_batch.num_rows
    if offset != base.num_rows or replacement_verified != len(replacements):
        raise RuntimeError("output validation did not cover every row")
    return {
        "output_rows": offset,
        "output_row_groups": parquet_file.metadata.num_row_groups,
        "schema_equal": True,
        "untouched_rows_verified": untouched_verified,
        "replacement_rows_verified": replacement_verified,
        "non_null_bbox_rows": sum(value["bbox_normalized"] is not None for value in replacements.values()),
    }


def _key(trace_id: Any, ratio: Any) -> tuple[str, float]:
    return str(trace_id), round(float(ratio), 2)


def _parse_ratio(value: str) -> float:
    numerator, denominator = value.split(":", maxsplit=1)
    return float(numerator) / float(denominator)


def _file_record(path: Path) -> dict[str, Any]:
    return {"path": str(path), "size": path.stat().st_size, "sha256": _sha256(path)}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()