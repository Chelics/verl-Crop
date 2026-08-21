from __future__ import annotations

import hashlib
import html
import io
import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageOps, ImageStat

from news_crop_benchmark.result_viewer import ALL_FILTER, sample_index


METADATA_COLUMNS = (
    "sample_id",
    "data_source",
    "ability",
    "messages",
    "enable_thinking",
    "trace_id",
    "aspect_ratio",
    "target_ratio",
    "render_mode",
    "was_cropped",
    "was_padded",
    "edge_artifact_trim_pixels",
    "target_action",
    "reason",
    "confidence",
)
TARGET_RENDERER_VERSION = 1

SWIFT_SFT_VIEWER_CSS = """
.gradio-container { max-width: 1500px !important; }
.sft-header { border-bottom: 1px solid var(--border-color-primary); padding-bottom: 10px; }
.sft-header h1 { font-size: 24px !important; letter-spacing: 0 !important; }
.filter-row, .navigation-row { align-items: end; }
.sample-heading { min-height: 84px; }
.source-image img, .target-image img { object-fit: contain !important; }
button, .form { border-radius: 6px !important; }
"""


def _parse_target_action(action_text: str) -> dict[str, Any]:
    start = action_text.find("{")
    end = action_text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("target_action does not contain a JSON object")
    payload = json.loads(action_text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("target_action JSON must be an object")
    return payload


def _target_box(image: Image.Image, values: Any) -> tuple[int, int, int, int]:
    if not isinstance(values, list) or len(values) != 4:
        raise ValueError("bbox must contain four coordinates")
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
        raise ValueError("bbox coordinates must be numeric")
    left, top, right, bottom = (float(value) for value in values)
    if not (0 <= left < right <= 1000 and 0 <= top < bottom <= 1000):
        raise ValueError(f"invalid bbox: {values}")
    width, height = image.size
    return (
        max(0, min(width - 1, math.floor(left / 1000 * width))),
        max(0, min(height - 1, math.floor(top / 1000 * height))),
        max(1, min(width, math.ceil(right / 1000 * width))),
        max(1, min(height, math.ceil(bottom / 1000 * height))),
    )


def _rgb(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, list) or len(value) != 3:
        raise ValueError(f"{name} must contain three RGB channels")
    if any(
        isinstance(channel, bool) or not isinstance(channel, (int, float)) or not 0 <= channel <= 255
        for channel in value
    ):
        raise ValueError(f"{name} channels must be numeric values in [0, 255]")
    return tuple(int(round(channel)) for channel in value)


def _weighted_split(total: int, first_weight: float, second_weight: float) -> tuple[int, int]:
    if first_weight < 0 or second_weight < 0:
        raise ValueError("padding weights must be non-negative")
    weight_sum = first_weight + second_weight
    first = total // 2 if weight_sum == 0 else round(total * first_weight / weight_sum)
    first = max(0, min(total, first))
    return first, total - first


def _edge_color(image: Image.Image, side: str) -> tuple[int, int, int]:
    width, height = image.size
    strip_width = max(1, round(width * 0.02))
    strip_height = max(1, round(height * 0.02))
    boxes = {
        "top": (0, 0, width, strip_height),
        "right": (width - strip_width, 0, width, height),
        "bottom": (0, height - strip_height, width, height),
        "left": (0, 0, strip_width, height),
    }
    with image.crop(boxes[side]) as edge:
        return tuple(round(channel) for channel in ImageStat.Stat(edge).median)


def _gradient_band(
    size: tuple[int, int],
    side: str,
    outer_color: tuple[int, int, int],
    inner_color: tuple[int, int, int],
) -> Image.Image:
    width, height = size
    length = height if side in {"top", "bottom"} else width
    colors = []
    for index in range(length):
        fraction = 0.0 if length == 1 else index / (length - 1)
        if side in {"bottom", "right"}:
            fraction = 1.0 - fraction
        colors.append(
            tuple(round(outer + (inner - outer) * fraction) for outer, inner in zip(outer_color, inner_color))
        )
    if side in {"top", "bottom"}:
        strip = Image.new("RGB", (1, height))
        strip.putdata(colors)
    else:
        strip = Image.new("RGB", (width, 1))
        strip.putdata(colors)
    try:
        return strip.resize(size, Image.Resampling.BILINEAR)
    finally:
        strip.close()


def _padding_band(image: Image.Image, size: tuple[int, int], side: str, color: Any, style: str) -> Image.Image:
    outer_color = _rgb(color, f"padding color for {side}")
    if style == "solid":
        return Image.new("RGB", size, outer_color)
    if style == "gradient":
        return _gradient_band(size, side, outer_color, _edge_color(image, side))
    raise ValueError(f"unsupported padding_style: {style}")


def render_target_action(image: Image.Image, action_text: str, target_ratio: float) -> Image.Image:
    if target_ratio <= 0:
        raise ValueError("target_ratio must be positive")
    payload = _parse_target_action(action_text)
    mode = payload.get("mode")
    if mode not in {"crop", "crop_then_pad"}:
        raise ValueError(f"unsupported target action mode: {mode}")
    source = image.convert("RGB")
    try:
        retained = source.crop(_target_box(source, payload.get("bbox")))
    finally:
        source.close()
    if mode == "crop":
        return retained

    colors = payload.get("padding_colors_rgb")
    weights = payload.get("padding_weights")
    style = payload.get("padding_style")
    if not isinstance(colors, dict) or not isinstance(weights, dict):
        retained.close()
        raise ValueError("crop_then_pad requires padding colors and weights")
    for side in ("top", "right", "bottom", "left"):
        if side not in colors or side not in weights:
            retained.close()
            raise ValueError(f"crop_then_pad is missing padding data for {side}")
        if isinstance(weights[side], bool) or not isinstance(weights[side], (int, float)):
            retained.close()
            raise ValueError(f"padding weight for {side} must be numeric")

    width, height = retained.size
    observed_ratio = width / height
    if math.isclose(observed_ratio, target_ratio, rel_tol=0.0, abs_tol=1 / max(width, height)):
        return retained
    if observed_ratio > target_ratio:
        canvas_width = width
        canvas_height = max(height, math.ceil(width / target_ratio))
        top, bottom = _weighted_split(canvas_height - height, float(weights["top"]), float(weights["bottom"]))
        canvas = Image.new("RGB", (canvas_width, canvas_height))
        try:
            if top:
                with _padding_band(retained, (canvas_width, top), "top", colors["top"], str(style)) as band:
                    canvas.paste(band, (0, 0))
            canvas.paste(retained, (0, top))
            if bottom:
                with _padding_band(retained, (canvas_width, bottom), "bottom", colors["bottom"], str(style)) as band:
                    canvas.paste(band, (0, top + height))
        finally:
            retained.close()
        return canvas

    canvas_height = height
    canvas_width = max(width, math.ceil(height * target_ratio))
    left, right = _weighted_split(canvas_width - width, float(weights["left"]), float(weights["right"]))
    canvas = Image.new("RGB", (canvas_width, canvas_height))
    try:
        if left:
            with _padding_band(retained, (left, canvas_height), "left", colors["left"], str(style)) as band:
                canvas.paste(band, (0, 0))
        canvas.paste(retained, (left, 0))
        if right:
            with _padding_band(retained, (right, canvas_height), "right", colors["right"], str(style)) as band:
                canvas.paste(band, (left + width, 0))
    finally:
        retained.close()
    return canvas


class SwiftSFTDataset:
    def __init__(self, parquet_path: Path, cache_dir: Path | None = None) -> None:
        import pyarrow.parquet as pq

        self.parquet_path = Path(os.path.abspath(parquet_path.expanduser()))
        if not self.parquet_path.is_file():
            raise FileNotFoundError(self.parquet_path)
        self.rows = pq.read_table(self.parquet_path, columns=list(METADATA_COLUMNS), pre_buffer=False).to_pylist()
        if not self.rows:
            raise ValueError(f"No rows found in {self.parquet_path}")
        for row_index, row in enumerate(self.rows):
            row["_row_index"] = row_index
        self._rows_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            self._rows_by_trace[str(row["trace_id"])].append(row)
        for rows in self._rows_by_trace.values():
            rows.sort(key=lambda row: float(row["target_ratio"]))
        self.trace_ids = sorted(
            self._rows_by_trace,
            key=lambda trace_id: min(int(row["_row_index"]) for row in self._rows_by_trace[trace_id]),
        )
        self.cache_dir = cache_dir or self._default_cache_dir()

    @property
    def ratios(self) -> list[str]:
        return sorted({str(row["target_ratio"]) for row in self.rows}, key=float)

    @property
    def modes(self) -> list[str]:
        return sorted({str(row["render_mode"]) for row in self.rows})

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "rows": len(self.rows),
            "traces": len(self.trace_ids),
            "modes": dict(sorted(Counter(str(row["render_mode"]) for row in self.rows).items())),
            "ratios": dict(sorted(Counter(str(row["aspect_ratio"]) for row in self.rows).items())),
            "confidences": dict(sorted(Counter(str(row["confidence"]) for row in self.rows).items())),
            "cropped": sum(bool(row["was_cropped"]) for row in self.rows),
            "padded": sum(bool(row["was_padded"]) for row in self.rows),
            "data_sources": dict(sorted(Counter(str(row["data_source"]) for row in self.rows).items())),
        }

    def filter_trace_ids(
        self,
        *,
        ratio: str | None = None,
        mode: str | None = None,
        query: str | None = None,
    ) -> list[str]:
        normalized_query = (query or "").strip().casefold()

        def matches(row: dict[str, Any]) -> bool:
            if ratio is not None and str(row["target_ratio"]) != ratio:
                return False
            if mode is not None and str(row["render_mode"]) != mode:
                return False
            if normalized_query:
                searchable = " ".join(
                    [
                        str(row.get("sample_id", "")),
                        str(row.get("trace_id", "")),
                        str(row.get("reason", "")),
                        str(row.get("target_action", "")),
                        self._user_message(row),
                    ]
                ).casefold()
                if normalized_query not in searchable:
                    return False
            return True

        return [trace_id for trace_id in self.trace_ids if any(matches(row) for row in self._rows_by_trace[trace_id])]

    def prepare_previews(
        self,
        *,
        maximum_side: int = 1600,
        quality: int = 92,
        batch_size: int = 8,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        import pyarrow.parquet as pq

        if maximum_side <= 0 or not 1 <= quality <= 100 or batch_size <= 0:
            raise ValueError("invalid preview settings")
        manifest = self._manifest(maximum_side, quality)
        manifest_path = self.cache_dir / "manifest.json"
        if manifest_path.is_file() and json.loads(manifest_path.read_text(encoding="utf-8")) == manifest:
            if all(
                self.preview_path(trace_id).is_file()
                and all(self.target_preview_path(row).is_file() for row in self._rows_by_trace[trace_id])
                for trace_id in self.trace_ids
            ):
                return

        previews_dir = self.cache_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        missing_sources = {trace_id for trace_id in self.trace_ids if not self.preview_path(trace_id).is_file()}
        if missing_sources:
            wanted = {
                min(int(row["_row_index"]) for row in self._rows_by_trace[trace_id]): trace_id
                for trace_id in missing_sources
            }
            parquet_file = pq.ParquetFile(self.parquet_path, pre_buffer=False)
            row_index = 0
            for batch in parquet_file.iter_batches(batch_size=batch_size, columns=["images"], use_threads=False):
                for images in batch.column(0).to_pylist():
                    trace_id = wanted.get(row_index)
                    if trace_id is not None:
                        payload = images[0]["bytes"] if images else None
                        if payload is None:
                            raise ValueError(f"missing image bytes for trace {trace_id}")
                        with Image.open(io.BytesIO(payload)) as image:
                            preview = ImageOps.exif_transpose(image).convert("RGB")
                            preview.thumbnail((maximum_side, maximum_side), Image.Resampling.LANCZOS)
                            preview.save(self.preview_path(trace_id), format="JPEG", quality=quality, optimize=True)
                    row_index += 1

        completed = 0
        for trace_id in self.trace_ids:
            source_path = self.preview_path(trace_id)
            if not source_path.is_file():
                raise RuntimeError(f"source preview was not prepared for trace {trace_id}")
            with Image.open(source_path) as source_image:
                source = ImageOps.exif_transpose(source_image).convert("RGB")
            try:
                for row in self._rows_by_trace[trace_id]:
                    output_path = self.target_preview_path(row)
                    if not output_path.is_file():
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        rendered = render_target_action(source, str(row["target_action"]), float(row["target_ratio"]))
                        try:
                            rendered.thumbnail((maximum_side, maximum_side), Image.Resampling.LANCZOS)
                            rendered.save(output_path, format="JPEG", quality=quality, optimize=True)
                        finally:
                            rendered.close()
            finally:
                source.close()
            completed += 1
            if progress is not None and (completed == len(self.trace_ids) or completed % 50 == 0):
                progress(completed, len(self.trace_ids))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def preview_path(self, trace_id: str) -> Path:
        return self.cache_dir / "previews" / f"{trace_id}.jpg"

    def target_preview_path(self, row: dict[str, Any]) -> Path:
        return (
            self.cache_dir
            / f"targets-v{TARGET_RENDERER_VERSION}"
            / str(row["trace_id"])
            / f"{int(row['_row_index']):06d}.jpg"
        )

    def trace_view(self, trace_id: str) -> dict[str, Any]:
        rows = self._rows_by_trace[trace_id]
        first = rows[0]
        return {
            "trace_id": trace_id,
            "title": self._field_from_user_message(first, "Article headline"),
            "caption": self._field_from_user_message(first, "Image caption"),
            "image": str(self.preview_path(trace_id)) if self.preview_path(trace_id).is_file() else None,
            "targets": [
                {
                    "image": str(self.target_preview_path(row)) if self.target_preview_path(row).is_file() else None,
                    "label": f"{row['aspect_ratio']} · {str(row['render_mode']).upper()}",
                }
                for row in rows
            ],
            "rows": rows,
        }

    @staticmethod
    def _user_message(row: dict[str, Any]) -> str:
        for message in row.get("messages", []):
            if message.get("role") == "user":
                return str(message.get("content", ""))
        return ""

    @classmethod
    def _field_from_user_message(cls, row: dict[str, Any], name: str) -> str:
        prefix = f"{name}:"
        for line in cls._user_message(row).splitlines():
            if line.startswith(prefix):
                return line[len(prefix) :].strip()
        return ""

    def _default_cache_dir(self) -> Path:
        stat = self.parquet_path.stat()
        fingerprint = hashlib.sha256(
            f"{self.parquet_path}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:12]
        return Path(os.getenv("LOCALAPPDATA", Path.home() / ".cache")) / "news-crop-benchmark" / "sft-viewer" / fingerprint

    def _manifest(self, maximum_side: int, quality: int) -> dict[str, Any]:
        stat = self.parquet_path.stat()
        return {
            "source": str(self.parquet_path),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "rows": len(self.rows),
            "traces": len(self.trace_ids),
            "maximum_side": maximum_side,
            "quality": quality,
            "target_renderer_version": TARGET_RENDERER_VERSION,
        }


def build_swift_sft_app(dataset: SwiftSFTDataset) -> Any:
    import gradio as gr

    initial_ids = dataset.trace_ids

    def load_image(path: str | None) -> Any:
        if path is None:
            return None
        try:
            with Image.open(path) as image:
                return image.copy()
        except OSError:
            return None

    def render(trace_ids: list[str], index: int) -> tuple[Any, ...]:
        if not trace_ids:
            return 0, 1, "/ 0", "### No matching samples", None, None, None, None, None, [], "", []
        index = max(0, min(index, len(trace_ids) - 1))
        view = dataset.trace_view(trace_ids[index])
        heading = (
            f"### {html.escape(view['title'])}\n"
            f"`{html.escape(view['trace_id'])}`\n\n"
            f"{html.escape(view['caption'])}"
        )
        table = []
        for row in view["rows"]:
            action = _parse_target_action(str(row["target_action"]))
            bbox = action["bbox"]
            retained_fraction = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / 1_000_000
            table.append(
                [
                    row["aspect_ratio"],
                    action["mode"],
                    ", ".join(str(value) for value in bbox),
                    f"{retained_fraction:.1%}",
                    row["was_cropped"],
                    row["was_padded"],
                    ", ".join(str(value) for value in row["edge_artifact_trim_pixels"]),
                    row["confidence"],
                ]
            )
        actions = "\n\n---\n\n".join(
            f"#### {html.escape(str(row['aspect_ratio']))} · {html.escape(str(row['render_mode']).upper())}\n\n"
            f"**Target action**\n\n<pre><code>{html.escape(str(row['target_action']))}</code></pre>\n\n"
            f"**Editorial reason**\n\n{html.escape(str(row['reason']))}"
            for row in view["rows"]
        )
        raw = [{key: value for key, value in row.items() if key != "_row_index"} for row in view["rows"]]
        target_images = [load_image(target["image"]) for target in view["targets"]]
        if len(target_images) != 4:
            raise ValueError(f"expected four target previews for {view['trace_id']}, found {len(target_images)}")
        return (
            index,
            index + 1,
            f"/ {len(trace_ids)}",
            heading,
            load_image(view["image"]),
            *target_images,
            table,
            actions,
            raw,
        )

    def apply_filters(ratio: str, mode: str, query: str) -> tuple[Any, ...]:
        trace_ids = dataset.filter_trace_ids(
            ratio=None if ratio == ALL_FILTER else ratio,
            mode=None if mode == ALL_FILTER else mode,
            query=query,
        )
        return trace_ids, *render(trace_ids, 0)

    def move(trace_ids: list[str], index: int, offset: int) -> tuple[Any, ...]:
        return render(trace_ids, index + offset)

    def jump(trace_ids: list[str], requested: Any) -> tuple[Any, ...]:
        return render(trace_ids, sample_index(requested, len(trace_ids)))

    initial = render(initial_ids, 0)
    summary = dataset.summary
    mode_text = " &nbsp; ".join(
        f"**Mode {html.escape(mode.title())}:** {count}" for mode, count in summary["modes"].items()
    )
    ratio_text = " &nbsp; ".join(
        f"**{html.escape(ratio)}:** {count}" for ratio, count in summary["ratios"].items()
    )
    with gr.Blocks(title="Swift Crop SFT Viewer") as app:
        gr.Markdown(
            f"# Swift Crop SFT Dataset\n`{html.escape(str(dataset.parquet_path))}`\n\n"
            f"**Traces:** {summary['traces']} &nbsp; **Rows:** {summary['rows']} &nbsp; "
            f"**Rows cropped:** {summary['cropped']} &nbsp; **Rows padded:** {summary['padded']} &nbsp; {mode_text}\n\n"
            f"{ratio_text}",
            elem_classes="sft-header",
        )
        trace_ids_state = gr.State(initial_ids)
        index_state = gr.State(initial[0])
        with gr.Row(elem_classes="filter-row"):
            ratio_filter = gr.Dropdown([ALL_FILTER, *dataset.ratios], value=ALL_FILTER, label="Ratio")
            mode_filter = gr.Dropdown([ALL_FILTER, *dataset.modes], value=ALL_FILTER, label="Render mode")
            query = gr.Textbox(label="Search", placeholder="Headline, caption, reason, trace ID, or action")
        with gr.Row(elem_classes="navigation-row"):
            previous_button = gr.Button("Previous", variant="secondary")
            sample_number = gr.Number(initial[1], label="Sample", minimum=1, precision=0, step=1)
            sample_total = gr.Markdown(initial[2])
            jump_button = gr.Button("Jump", variant="secondary")
            next_button = gr.Button("Next", variant="primary")
        sample_heading = gr.Markdown(initial[3], elem_classes="sample-heading")
        source_image = gr.Image(
            initial[4], label="Source image", interactive=False, height=420, elem_classes="source-image"
        )
        with gr.Row(equal_height=True):
            target_images = [
                gr.Image(
                    initial[5 + offset],
                    label=f"{ratio} rendered target",
                    interactive=False,
                    height=320,
                    min_width=260,
                    elem_classes="target-image",
                )
                for offset, ratio in enumerate(("1:1", "1.59:1", "1.77:1", "1.91:1"))
            ]
        details = gr.Dataframe(
            value=initial[9],
            headers=["Ratio", "Action", "BBox (0-1000)", "Source retained", "Cropped", "Padded", "Edge trim", "Confidence"],
            datatype=["str", "str", "str", "str", "bool", "bool", "str", "str"],
            interactive=False,
            label="Training targets",
        )
        with gr.Accordion("Target actions and reasons", open=True):
            actions = gr.Markdown(initial[10])
        with gr.Accordion("Raw rows", open=False):
            raw_rows = gr.JSON(initial[11], label=None)

        filter_inputs = [ratio_filter, mode_filter, query]
        outputs = [
            trace_ids_state,
            index_state,
            sample_number,
            sample_total,
            sample_heading,
            source_image,
            *target_images,
            details,
            actions,
            raw_rows,
        ]
        for component in filter_inputs:
            component.change(apply_filters, inputs=filter_inputs, outputs=outputs)
        navigation_outputs = outputs[1:]
        previous_button.click(
            lambda trace_ids, index: move(trace_ids, index, -1),
            inputs=[trace_ids_state, index_state],
            outputs=navigation_outputs,
        )
        next_button.click(
            lambda trace_ids, index: move(trace_ids, index, 1),
            inputs=[trace_ids_state, index_state],
            outputs=navigation_outputs,
        )
        jump_inputs = [trace_ids_state, sample_number]
        jump_button.click(jump, inputs=jump_inputs, outputs=navigation_outputs)
        sample_number.submit(jump, inputs=jump_inputs, outputs=navigation_outputs)
    return app