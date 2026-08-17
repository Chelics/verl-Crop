from __future__ import annotations

import hashlib
import html
import io
import json
import os
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from news_crop_benchmark.result_viewer import ALL_FILTER, sample_index


METADATA_COLUMNS = (
    "source_index",
    "trace_id",
    "headline",
    "caption",
    "original_image_url",
    "original_width",
    "original_height",
    "aspect_ratio",
    "target_ratio",
    "crop_required",
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
    "error",
)

DATASET_VIEWER_CSS = """
.gradio-container { max-width: 1500px !important; }
.dataset-header { border-bottom: 1px solid var(--border-color-primary); padding-bottom: 10px; }
.dataset-header h1 { font-size: 24px !important; letter-spacing: 0 !important; }
.filter-row, .navigation-row { align-items: end; }
.sample-heading { min-height: 96px; }
.original-image img, .crop-gallery img { object-fit: contain !important; }
button, .form { border-radius: 6px !important; }
"""


class CroppedDataset:
    def __init__(
        self,
        parquet_path: Path,
        cache_dir: Path | None = None,
        override_dir: Path | None = None,
        reason_prefix: str | None = None,
    ) -> None:
        import pyarrow.parquet as pq

        self.parquet_path = Path(os.path.abspath(parquet_path.expanduser()))
        if not self.parquet_path.is_file():
            raise FileNotFoundError(self.parquet_path)
        table = pq.read_table(self.parquet_path, columns=list(METADATA_COLUMNS), pre_buffer=False)
        self.rows = table.to_pylist()
        if not self.rows:
            raise ValueError(f"No rows found in {self.parquet_path}")
        for row_index, row in enumerate(self.rows):
            row["_row_index"] = row_index
        self.reason_prefix = reason_prefix
        if reason_prefix is not None:
            self.rows = [row for row in self.rows if str(row["reason"]).startswith(reason_prefix)]
            if not self.rows:
                raise ValueError(f"No rows found with reason prefix {reason_prefix!r}")

        self._rows_by_trace: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            self._rows_by_trace[str(row["trace_id"])].append(row)
        for trace_rows in self._rows_by_trace.values():
            trace_rows.sort(key=lambda row: float(row["target_ratio"]))
        self.trace_ids = sorted(
            self._rows_by_trace,
            key=lambda trace_id: (
                min(int(row["source_index"]) for row in self._rows_by_trace[trace_id]),
                trace_id,
            ),
        )
        self.cache_dir = cache_dir or self._default_cache_dir()
        self.override_dir = Path(os.path.abspath(override_dir.expanduser())) if override_dir is not None else None
        self.overrides = self._load_overrides()

    @property
    def render_modes(self) -> list[str]:
        return sorted({str(row["render_mode"]) for row in self.rows})

    @property
    def confidence_values(self) -> list[str]:
        return sorted({str(row["confidence"]) for row in self.rows})

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "rows": len(self.rows),
            "stories": len(self.trace_ids),
            "render_modes": dict(Counter(str(row["render_mode"]) for row in self.rows)),
            "feasible": dict(Counter(str(row["feasible"]) for row in self.rows)),
            "errors": sum(row["error"] is not None for row in self.rows),
        }

    def filter_trace_ids(
        self,
        *,
        render_mode: str | None = None,
        feasible: str | None = None,
        confidence: str | None = None,
        has_override: str | None = None,
        query: str | None = None,
    ) -> list[str]:
        normalized_query = (query or "").strip().casefold()

        def matches(row: dict[str, Any]) -> bool:
            if render_mode is not None and str(row["render_mode"]) != render_mode:
                return False
            if feasible is not None and str(row["feasible"]) != feasible:
                return False
            if confidence is not None and str(row["confidence"]) != confidence:
                return False
            if normalized_query:
                searchable = " ".join(
                    str(row.get(field, ""))
                    for field in ("headline", "caption", "reason", "trace_id")
                ).casefold()
                if normalized_query not in searchable:
                    return False
            return True

        filtered = []
        for trace_id in self.trace_ids:
            trace_has_override = any(key[0] == trace_id for key in self.overrides)
            if has_override == "Yes" and not trace_has_override:
                continue
            if has_override == "No" and trace_has_override:
                continue
            if any(matches(row) for row in self._rows_by_trace[trace_id]):
                filtered.append(trace_id)
        return filtered

    def prepare_previews(
        self,
        *,
        maximum_side: int = 1400,
        quality: int = 90,
        batch_size: int = 8,
        progress: Callable[[int, int], None] | None = None,
    ) -> None:
        import pyarrow.parquet as pq
        from PIL import Image, ImageOps

        if maximum_side <= 0 or not 1 <= quality <= 100 or batch_size <= 0:
            raise ValueError("invalid preview settings")
        expected_manifest = self._manifest(maximum_side, quality)
        manifest_path = self.cache_dir / "manifest.json"
        if manifest_path.is_file():
            current = json.loads(manifest_path.read_text(encoding="utf-8"))
            if current == expected_manifest and all(self.preview_path(row).is_file() for row in self.rows):
                return

        previews_dir = self.cache_dir / "previews"
        previews_dir.mkdir(parents=True, exist_ok=True)
        parquet_file = pq.ParquetFile(self.parquet_path, pre_buffer=False)
        completed = 0
        for row_index, payload in self._iter_preview_payloads(parquet_file, batch_size):
            output_path = previews_dir / f"{row_index:05d}.jpg"
            if payload is not None and not output_path.is_file():
                with Image.open(io.BytesIO(payload)) as image:
                    preview = ImageOps.exif_transpose(image).convert("RGB")
                    preview.thumbnail((maximum_side, maximum_side), Image.Resampling.LANCZOS)
                    preview.save(output_path, format="JPEG", quality=quality, optimize=True)
            completed += 1
            if progress is not None and (completed == len(self.rows) or completed % 50 == 0):
                progress(completed, len(self.rows))
        if completed != len(self.rows):
            raise RuntimeError(f"preview row count mismatch: {completed} != {len(self.rows)}")
        manifest_path.write_text(
            json.dumps(expected_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def preview_path(self, row: dict[str, Any]) -> Path:
        return self.cache_dir / "previews" / f"{int(row['_row_index']):05d}.jpg"

    def story_view(self, trace_id: str) -> dict[str, Any]:
        rows = self._rows_by_trace[trace_id]
        first = rows[0]
        gallery = [
            (
                str(self.preview_path(row)) if self.preview_path(row).is_file() else None,
                f"{row['aspect_ratio']} | {str(row['render_mode']).upper()}",
            )
            for row in rows
        ]
        override_gallery = []
        for row in rows:
            record = self.overrides.get((trace_id, str(row["aspect_ratio"])))
            if record is not None:
                override_gallery.append(
                    (
                        str(record["resolved_image_path"]),
                        f"{record['aspect_ratio']} | PROPOSED {str(record['operation']).upper()}",
                    )
                )
        return {
            "trace_id": trace_id,
            "headline": str(first["headline"]),
            "caption": str(first["caption"]),
            "original_image_url": str(first["original_image_url"]),
            "original": str(self.original_path(trace_id)) if self.original_path(trace_id).is_file() else None,
            "gallery": [(path, caption) for path, caption in gallery if path is not None],
            "override_gallery": override_gallery,
            "override_records": [
                self.overrides[(trace_id, str(row["aspect_ratio"]))]
                for row in rows
                if (trace_id, str(row["aspect_ratio"])) in self.overrides
            ],
            "rows": rows,
        }

    def ensure_original(self, trace_id: str, *, timeout: float = 20.0, maximum_side: int = 1800) -> Path | None:
        from PIL import Image, ImageOps

        output_path = self.original_path(trace_id)
        if output_path.is_file():
            return output_path
        if timeout <= 0 or maximum_side <= 0:
            raise ValueError("invalid original image settings")
        url = str(self._rows_by_trace[trace_id][0]["original_image_url"])
        request = urllib.request.Request(url, headers={"User-Agent": "news-crop-benchmark-viewer/1.0"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
            with Image.open(io.BytesIO(payload)) as image:
                original = ImageOps.exif_transpose(image).convert("RGB")
                original.thumbnail((maximum_side, maximum_side), Image.Resampling.LANCZOS)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                temporary_path = output_path.with_suffix(".tmp.jpg")
                original.save(temporary_path, format="JPEG", quality=92, optimize=True)
                temporary_path.replace(output_path)
            return output_path
        except (OSError, ValueError):
            return None

    def original_path(self, trace_id: str) -> Path:
        return self.cache_dir / "originals" / f"{trace_id}.jpg"

    def _default_cache_dir(self) -> Path:
        local_root = Path(os.getenv("LOCALAPPDATA", Path.home() / ".cache"))
        stat = self.parquet_path.stat()
        fingerprint = hashlib.sha256(
            f"{self.parquet_path}|{stat.st_size}|{stat.st_mtime_ns}|{self.reason_prefix}".encode("utf-8")
        ).hexdigest()[:12]
        return local_root / "news-crop-benchmark" / "cropped-viewer" / fingerprint

    def _manifest(self, maximum_side: int, quality: int) -> dict[str, Any]:
        stat = self.parquet_path.stat()
        return {
            "source": str(self.parquet_path),
            "source_size": stat.st_size,
            "source_mtime_ns": stat.st_mtime_ns,
            "rows": len(self.rows),
            "maximum_side": maximum_side,
            "quality": quality,
            "reason_prefix": self.reason_prefix,
            "row_indices": [int(row["_row_index"]) for row in self.rows],
        }

    def _iter_preview_payloads(self, parquet_file: Any, batch_size: int) -> Any:
        selected_indices = {int(row["_row_index"]) for row in self.rows}
        if len(selected_indices) == parquet_file.metadata.num_rows:
            row_index = 0
            for batch in parquet_file.iter_batches(
                batch_size=batch_size,
                columns=["cropped_image"],
                use_threads=False,
            ):
                for payload in batch.column(0).to_pylist():
                    yield row_index, payload
                    row_index += 1
            return

        offset = 0
        for row_group_index in range(parquet_file.metadata.num_row_groups):
            row_count = parquet_file.metadata.row_group(row_group_index).num_rows
            local_indices = [index - offset for index in selected_indices if offset <= index < offset + row_count]
            if local_indices:
                column = parquet_file.read_row_group(
                    row_group_index,
                    columns=["cropped_image"],
                    use_threads=False,
                ).column("cropped_image")
                for local_index in sorted(local_indices):
                    yield offset + local_index, column[local_index].as_py()
            offset += row_count

    def _load_overrides(self) -> dict[tuple[str, str], dict[str, Any]]:
        if self.override_dir is None:
            return {}
        manifest_path = self.override_dir / "overrides.jsonl"
        if not manifest_path.is_file():
            raise FileNotFoundError(manifest_path)
        records = {}
        for line in manifest_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            key = (str(record["trace_id"]), str(record["aspect_ratio"]))
            if key in records:
                raise ValueError(f"duplicate rendered override: {key}")
            relative_path = Path(str(record["image_path"]))
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(f"invalid override image path: {relative_path}")
            resolved_path = Path(os.path.abspath(self.override_dir / relative_path))
            if not resolved_path.is_file():
                raise FileNotFoundError(resolved_path)
            records[key] = {**record, "resolved_image_path": resolved_path}
        return records


def build_cropped_dataset_app(dataset: CroppedDataset) -> Any:
    import gradio as gr
    from PIL import Image

    initial_ids = dataset.trace_ids
    show_overrides = bool(dataset.overrides)

    def load_image(path: str) -> Any:
        try:
            with Image.open(path) as image:
                return image.copy()
        except OSError:
            return None

    def render(trace_ids: list[str], index: int) -> tuple[Any, ...]:
        if not trace_ids:
            return 0, 1, "/ 0", "### No matching samples", None, [], [], [], [], []
        index = max(0, min(index, len(trace_ids) - 1))
        trace_id = trace_ids[index]
        dataset.ensure_original(trace_id)
        view = dataset.story_view(trace_id)
        heading = (
            f"### {html.escape(view['headline'])}\n"
            f"`{html.escape(view['trace_id'])}`\n\n"
            f"{html.escape(view['caption'])}\n\n"
            f"[Original image URL]({view['original_image_url']})"
        )
        table = [
            [
                row["aspect_ratio"],
                row["render_mode"],
                row["crop_required"],
                row["feasible"],
                f"{row['output_width']} x {row['output_height']}",
                row["confidence"],
            ]
            for row in view["rows"]
        ]
        current_reasons = "\n\n".join(
            f"**{html.escape(str(row['aspect_ratio']))} · {html.escape(str(row['render_mode']).upper())}**  \n"
            f"{html.escape(str(row['reason']))}"
            for row in view["rows"]
        )
        proposed_reasons = "\n\n".join(
            f"**{html.escape(str(record['aspect_ratio']))} · PROPOSED "
            f"{html.escape(str(record['operation']).upper())}**  \n"
            f"{html.escape(str(record['reason']))}"
            for record in view["override_records"]
        )
        reasons = f"#### Current cropped_v2 provenance\n\n{current_reasons}"
        if proposed_reasons:
            reasons += f"\n\n---\n\n#### Proposed editorial reasons\n\n{proposed_reasons}"
        return (
            index,
            index + 1,
            f"/ {len(trace_ids)}",
            heading,
            load_image(view["original"]) if view["original"] else None,
            [(load_image(path), caption) for path, caption in view["gallery"]],
            [(load_image(path), caption) for path, caption in view["override_gallery"]],
            table,
            reasons,
            view["rows"],
        )

    def apply_filters(
        render_mode: str,
        feasible: str,
        confidence: str,
        has_override: str,
        query: str,
    ) -> tuple[Any, ...]:
        trace_ids = dataset.filter_trace_ids(
            render_mode=None if render_mode == ALL_FILTER else render_mode,
            feasible=None if feasible == ALL_FILTER else feasible,
            confidence=None if confidence == ALL_FILTER else confidence,
            has_override=None if has_override == ALL_FILTER else has_override,
            query=query,
        )
        return (trace_ids, *render(trace_ids, 0))

    def move(trace_ids: list[str], index: int, offset: int) -> tuple[Any, ...]:
        return render(trace_ids, index + offset)

    def jump(trace_ids: list[str], requested: Any) -> tuple[Any, ...]:
        return render(trace_ids, sample_index(requested, len(trace_ids)))

    initial = render(initial_ids, 0)
    summary = dataset.summary
    mode_text = " &nbsp; ".join(
        f"**{html.escape(mode.title())}:** {count}"
        for mode, count in sorted(summary["render_modes"].items())
    )
    with gr.Blocks(title="Cropped Dataset Viewer") as app:
        gr.Markdown(
            f"# Cropped Dataset\n`{html.escape(str(dataset.parquet_path))}`\n\n"
            f"**Stories:** {summary['stories']} &nbsp; **Rows:** {summary['rows']} &nbsp; {mode_text}",
            elem_classes="dataset-header",
        )
        trace_ids_state = gr.State(initial_ids)
        index_state = gr.State(initial[0])

        with gr.Row(elem_classes="filter-row"):
            mode_filter = gr.Dropdown([ALL_FILTER, *dataset.render_modes], value=ALL_FILTER, label="Render mode")
            feasible_filter = gr.Dropdown([ALL_FILTER, "True", "False"], value=ALL_FILTER, label="Feasible")
            confidence_filter = gr.Dropdown(
                [ALL_FILTER, *dataset.confidence_values], value=ALL_FILTER, label="Confidence"
            )
            override_filter = gr.Dropdown(
                [ALL_FILTER, "Yes", "No"],
                value=ALL_FILTER,
                label="Has override",
                visible=show_overrides,
            )
            query = gr.Textbox(label="Search", placeholder="Headline, caption, reason, or trace ID")

        with gr.Row(elem_classes="navigation-row"):
            previous_button = gr.Button("Previous", variant="secondary")
            sample_number = gr.Number(initial[1], label="Sample", minimum=1, precision=0, step=1)
            sample_total = gr.Markdown(initial[2])
            jump_button = gr.Button("Jump", variant="secondary")
            next_button = gr.Button("Next", variant="primary")

        sample_heading = gr.Markdown(initial[3], elem_classes="sample-heading")
        with gr.Row(equal_height=True):
            original = gr.Image(
                initial[4],
                label="Original",
                interactive=False,
                elem_classes="original-image",
            )
            gallery = gr.Gallery(
                initial[5],
                label="Final merged renders" if dataset.reason_prefix is not None else "Current renders",
                columns=2,
                rows=2,
                object_fit="contain",
                height=640,
                elem_classes="crop-gallery",
            )
        proposed = gr.Gallery(
            initial[6],
            label="Proposed overrides",
            columns=4,
            rows=1,
            object_fit="contain",
            height=420,
            elem_classes="crop-gallery",
            visible=show_overrides,
        )
        details = gr.Dataframe(
            value=initial[7],
            headers=["Ratio", "Mode", "Crop required", "Feasible", "Output", "Confidence"],
            datatype=["str", "str", "bool", "bool", "str", "str"],
            interactive=False,
            label="Render details",
        )
        with gr.Accordion("Reasons and provenance", open=True):
            reasons = gr.Markdown(initial[8])
        with gr.Accordion("Raw records", open=False):
            raw_records = gr.JSON(initial[9], label=None)

        filter_inputs = [mode_filter, feasible_filter, confidence_filter, override_filter, query]
        outputs = [
            trace_ids_state,
            index_state,
            sample_number,
            sample_total,
            sample_heading,
            original,
            gallery,
            proposed,
            details,
            reasons,
            raw_records,
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