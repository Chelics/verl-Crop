from __future__ import annotations

import html
import io
import os
from pathlib import Path
from typing import Any

from news_crop_benchmark.result_viewer import sample_index


PAIR_VIEWER_CSS = """
.gradio-container { max-width: 1500px !important; }
.pair-header { border-bottom: 1px solid var(--border-color-primary); padding-bottom: 10px; }
.pair-header h1 { font-size: 24px !important; letter-spacing: 0 !important; }
.navigation-row { align-items: end; }
.sample-heading { min-height: 96px; }
.pair-image img { object-fit: contain !important; }
button, .form { border-radius: 6px !important; }
"""


METADATA_COLUMNS = (
    "image_id",
    "crop_image_id",
    "title",
    "ImageCaption",
    "Reason",
    "source_original_url",
    "source_cropped_url",
    "source_event_count",
    "source_title_count",
    "reason_count",
)


class ImageOncePairDataset:
    def __init__(self, parquet_path: Path) -> None:
        import pyarrow.parquet as pq

        self.parquet_path = Path(os.path.abspath(parquet_path.expanduser()))
        if not self.parquet_path.is_file():
            raise FileNotFoundError(self.parquet_path)
        table = pq.read_table(self.parquet_path, columns=list(METADATA_COLUMNS), pre_buffer=False)
        self.rows = table.to_pylist()
        if not self.rows:
            raise ValueError(f"No rows found in {self.parquet_path}")
        self.indices = list(range(len(self.rows)))

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "rows": len(self.rows),
            "empty_reasons": sum(not str(row["Reason"]).strip() for row in self.rows),
            "unique_images": len({str(row["image_id"]) for row in self.rows}),
        }

    def filter_indices(self, query: str | None = None) -> list[int]:
        normalized = (query or "").strip().casefold()
        if not normalized:
            return self.indices.copy()
        return [
            index
            for index, row in enumerate(self.rows)
            if normalized
            in " ".join(
                str(row.get(field, ""))
                for field in ("title", "ImageCaption", "image_id", "crop_image_id", "source_original_url")
            ).casefold()
        ]

    def pair_view(self, row_index: int) -> dict[str, Any]:
        import pyarrow.parquet as pq

        if not 0 <= row_index < len(self.rows):
            raise IndexError(row_index)
        image_table = pq.read_table(
            self.parquet_path,
            columns=["original_image", "cropped_image"],
            pre_buffer=False,
        ).slice(row_index, 1)
        original = self._decode(image_table.column("original_image")[0].as_py())
        crop = self._decode(image_table.column("cropped_image")[0].as_py())
        row = self.rows[row_index]
        return {**row, "row_index": row_index, "original": original, "crop": crop}

    @staticmethod
    def _decode(payload: bytes) -> Any:
        from PIL import Image, ImageOps

        with Image.open(io.BytesIO(payload)) as image:
            return ImageOps.exif_transpose(image).convert("RGB")


def build_image_once_pair_app(dataset: ImageOncePairDataset) -> Any:
    import gradio as gr

    initial_indices = dataset.indices

    def render(indices: list[int], position: int) -> tuple[Any, ...]:
        if not indices:
            return 0, 1, "/ 0", "### No matching samples", None, None, [], []
        position = max(0, min(position, len(indices) - 1))
        view = dataset.pair_view(indices[position])
        original_size = f"{view['original'].width} x {view['original'].height}"
        crop_size = f"{view['crop'].width} x {view['crop'].height}"
        heading = (
            f"### {html.escape(str(view['title']))}\n"
            f"`image_id: {html.escape(str(view['image_id']))}`  \n"
            f"`crop_image_id: {html.escape(str(view['crop_image_id']))}`\n\n"
            f"{html.escape(str(view['ImageCaption']))}"
        )
        details = [
            ["Original size", original_size],
            ["Crop size", crop_size],
            ["Crop ratio", f"{view['crop'].width / view['crop'].height:.4f}"],
            ["Event count", view["source_event_count"]],
            ["Title count", view["source_title_count"]],
            ["Reason count", view["reason_count"]],
        ]
        links = (
            f"[Original source]({view['source_original_url']}) &nbsp; "
            f"[Cropped source]({view['source_cropped_url']})"
        )
        raw = {key: value for key, value in view.items() if key not in {"original", "crop"}}
        return position, position + 1, f"/ {len(indices)}", heading, view["original"], view["crop"], details, [links, raw]

    def search(query: str) -> tuple[Any, ...]:
        indices = dataset.filter_indices(query)
        rendered = render(indices, 0)
        links, raw = rendered[-1]
        return indices, *rendered[:-1], links, raw

    def move(indices: list[int], position: int, offset: int) -> tuple[Any, ...]:
        rendered = render(indices, position + offset)
        links, raw = rendered[-1]
        return *rendered[:-1], links, raw

    def jump(indices: list[int], requested: Any) -> tuple[Any, ...]:
        rendered = render(indices, sample_index(requested, len(indices)))
        links, raw = rendered[-1]
        return *rendered[:-1], links, raw

    initial = render(initial_indices, 0)
    initial_links, initial_raw = initial[-1]
    summary = dataset.summary
    with gr.Blocks(title="Image-Once Pair Viewer") as app:
        gr.Markdown(
            f"# Image-Once Empty-Reason Dataset\n`{html.escape(str(dataset.parquet_path))}`\n\n"
            f"**Rows:** {summary['rows']} &nbsp; **Unique images:** {summary['unique_images']} &nbsp; "
            f"**Empty reasons:** {summary['empty_reasons']}",
            elem_classes="pair-header",
        )
        indices_state = gr.State(initial_indices)
        position_state = gr.State(initial[0])
        query = gr.Textbox(label="Search", placeholder="Title, caption, image ID, crop ID, or URL")
        with gr.Row(elem_classes="navigation-row"):
            previous_button = gr.Button("Previous", variant="secondary")
            sample_number = gr.Number(initial[1], label="Sample", minimum=1, precision=0, step=1)
            sample_total = gr.Markdown(initial[2])
            jump_button = gr.Button("Jump", variant="secondary")
            next_button = gr.Button("Next", variant="primary")
        sample_heading = gr.Markdown(initial[3], elem_classes="sample-heading")
        with gr.Row(equal_height=True):
            original = gr.Image(initial[4], label="Original", interactive=False, elem_classes="pair-image")
            crop = gr.Image(initial[5], label="Reference crop", interactive=False, elem_classes="pair-image")
        details = gr.Dataframe(
            value=initial[6],
            headers=["Field", "Value"],
            datatype=["str", "str"],
            interactive=False,
            label="Pair details",
        )
        links = gr.Markdown(initial_links)
        with gr.Accordion("Raw metadata", open=False):
            raw = gr.JSON(initial_raw, label=None)

        navigation_outputs = [
            position_state,
            sample_number,
            sample_total,
            sample_heading,
            original,
            crop,
            details,
            links,
            raw,
        ]
        query.change(
            search,
            inputs=query,
            outputs=[indices_state, *navigation_outputs],
        )
        previous_button.click(
            lambda indices, position: move(indices, position, -1),
            inputs=[indices_state, position_state],
            outputs=navigation_outputs,
        )
        next_button.click(
            lambda indices, position: move(indices, position, 1),
            inputs=[indices_state, position_state],
            outputs=navigation_outputs,
        )
        jump_inputs = [indices_state, sample_number]
        jump_button.click(jump, inputs=jump_inputs, outputs=navigation_outputs)
        sample_number.submit(jump, inputs=jump_inputs, outputs=navigation_outputs)
    return app