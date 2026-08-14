from __future__ import annotations

import html
import io
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from news_crop_benchmark.result_viewer import ALL_FILTER, sample_index


MANUAL_VIEWER_CSS = """
.gradio-container { max-width: 1500px !important; }
.manual-header { border-bottom: 1px solid var(--border-color-primary); padding-bottom: 10px; }
.manual-header h1 { font-size: 24px !important; letter-spacing: 0 !important; }
.filter-row, .navigation-row { align-items: end; }
.sample-heading { min-height: 96px; }
.manual-gallery img { object-fit: contain !important; }
button, .form { border-radius: 6px !important; }
"""


class ManualCropsDataset:
    def __init__(self, parquet_path: Path) -> None:
        import pyarrow.parquet as pq

        self.parquet_path = Path(os.path.abspath(parquet_path.expanduser()))
        if not self.parquet_path.is_file():
            raise FileNotFoundError(self.parquet_path)
        self.rows = pq.read_table(self.parquet_path).to_pylist()
        if not self.rows:
            raise ValueError(f"No rows found in {self.parquet_path}")

        self._rows_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            self._rows_by_image[str(row["image_id"])].append(row)
        for image_rows in self._rows_by_image.values():
            image_rows.sort(key=lambda row: float(row["ratio"]))
        self.image_ids = sorted(
            self._rows_by_image,
            key=lambda image_id: (
                min(str(row.get("saved_at", "")) for row in self._rows_by_image[image_id]),
                image_id,
            ),
        )

    @property
    def ratios(self) -> list[str]:
        return sorted({str(row["ratio"]) for row in self.rows}, key=float)

    @property
    def summary(self) -> dict[str, Any]:
        return {
            "rows": len(self.rows),
            "images": len(self.image_ids),
            "filled": sum(bool(row["is_filled"]) for row in self.rows),
            "cropped": sum(bool(row["is_cropped"]) for row in self.rows),
            "ratios": dict(Counter(str(row["ratio"]) for row in self.rows)),
        }

    def filter_image_ids(
        self,
        *,
        ratio: str | None = None,
        filled: str | None = None,
        cropped: str | None = None,
        query: str | None = None,
    ) -> list[str]:
        normalized_query = (query or "").strip().casefold()

        def matches(row: dict[str, Any]) -> bool:
            if ratio is not None and str(row["ratio"]) != ratio:
                return False
            if filled is not None and str(row["is_filled"]) != filled:
                return False
            if cropped is not None and str(row["is_cropped"]) != cropped:
                return False
            if normalized_query:
                searchable = " ".join(
                    str(row.get(field, ""))
                    for field in ("title", "ImageCaption", "crop_reason", "image_id", "save_id")
                ).casefold()
                if normalized_query not in searchable:
                    return False
            return True

        return [
            image_id
            for image_id in self.image_ids
            if any(matches(row) for row in self._rows_by_image[image_id])
        ]

    def image_view(self, image_id: str) -> dict[str, Any]:
        rows = self._rows_by_image[image_id]
        first = rows[0]
        gallery = []
        for row in rows:
            image = self._decode(row.get("manual_crop"))
            if image is not None:
                mode = "CROPPED" if row["is_cropped"] else ("FILLED" if row["is_filled"] else "UNCHANGED")
                gallery.append((image, f"{float(row['ratio']):g}:1 | {mode}"))
        return {
            "image_id": image_id,
            "title": str(first.get("title", "")),
            "caption": str(first.get("ImageCaption", "")),
            "gallery": gallery,
            "rows": [{key: value for key, value in row.items() if key != "manual_crop"} for row in rows],
        }

    @staticmethod
    def _decode(payload: bytes | None) -> Any:
        if payload is None:
            return None
        from PIL import Image, ImageOps

        try:
            with Image.open(io.BytesIO(payload)) as image:
                return ImageOps.exif_transpose(image).convert("RGBA")
        except OSError:
            return None


def build_manual_crops_app(dataset: ManualCropsDataset) -> Any:
    import gradio as gr

    initial_ids = dataset.image_ids

    def render(image_ids: list[str], index: int) -> tuple[Any, ...]:
        if not image_ids:
            return 0, 1, "/ 0", "### No matching samples", [], [], [], []
        index = max(0, min(index, len(image_ids) - 1))
        view = dataset.image_view(image_ids[index])
        heading = (
            f"### {html.escape(view['title'])}\n"
            f"`{html.escape(view['image_id'])}`\n\n"
            f"{html.escape(view['caption'])}"
        )
        table = [
            [
                row["ratio"],
                row["is_filled"],
                row["is_cropped"],
                row.get("fill_color_code") or "",
                row.get("theme_color") or "",
                str(row.get("saved_at", "")),
            ]
            for row in view["rows"]
        ]
        reasons = "\n\n".join(
            f"**{float(row['ratio']):g}:1**  \n{html.escape(str(row.get('crop_reason') or 'No reason provided'))}"
            for row in view["rows"]
        )
        return index, index + 1, f"/ {len(image_ids)}", heading, view["gallery"], table, reasons, view["rows"]

    def apply_filters(ratio: str, filled: str, cropped: str, query: str) -> tuple[Any, ...]:
        image_ids = dataset.filter_image_ids(
            ratio=None if ratio == ALL_FILTER else ratio,
            filled=None if filled == ALL_FILTER else filled,
            cropped=None if cropped == ALL_FILTER else cropped,
            query=query,
        )
        return image_ids, *render(image_ids, 0)

    def move(image_ids: list[str], index: int, offset: int) -> tuple[Any, ...]:
        return render(image_ids, index + offset)

    def jump(image_ids: list[str], requested: Any) -> tuple[Any, ...]:
        return render(image_ids, sample_index(requested, len(image_ids)))

    initial = render(initial_ids, 0)
    summary = dataset.summary
    with gr.Blocks(title="Manual Crops Viewer") as app:
        gr.Markdown(
            f"# Manual Crops\n`{html.escape(str(dataset.parquet_path))}`\n\n"
            f"**Images:** {summary['images']} &nbsp; **Rows:** {summary['rows']} &nbsp; "
            f"**Filled:** {summary['filled']} &nbsp; **Cropped:** {summary['cropped']}",
            elem_classes="manual-header",
        )
        image_ids_state = gr.State(initial_ids)
        index_state = gr.State(initial[0])

        with gr.Row(elem_classes="filter-row"):
            ratio_filter = gr.Dropdown([ALL_FILTER, *dataset.ratios], value=ALL_FILTER, label="Ratio")
            filled_filter = gr.Dropdown([ALL_FILTER, "True", "False"], value=ALL_FILTER, label="Filled")
            cropped_filter = gr.Dropdown([ALL_FILTER, "True", "False"], value=ALL_FILTER, label="Cropped")
            query = gr.Textbox(label="Search", placeholder="Title, caption, reason, image ID, or save ID")

        with gr.Row(elem_classes="navigation-row"):
            previous_button = gr.Button("Previous", variant="secondary")
            sample_number = gr.Number(initial[1], label="Sample", minimum=1, precision=0, step=1)
            sample_total = gr.Markdown(initial[2])
            jump_button = gr.Button("Jump", variant="secondary")
            next_button = gr.Button("Next", variant="primary")

        sample_heading = gr.Markdown(initial[3], elem_classes="sample-heading")
        gallery = gr.Gallery(
            initial[4],
            label="Manual crops",
            columns=4,
            rows=1,
            object_fit="contain",
            height=560,
            elem_classes="manual-gallery",
        )
        details = gr.Dataframe(
            value=initial[5],
            headers=["Ratio", "Filled", "Cropped", "Fill color", "Theme color", "Saved at"],
            datatype=["number", "bool", "bool", "str", "str", "str"],
            interactive=False,
            label="Manual crop details",
        )
        with gr.Accordion("Reasons", open=True):
            reasons = gr.Markdown(initial[6])
        with gr.Accordion("Raw records", open=False):
            raw_records = gr.JSON(initial[7], label=None)

        filter_inputs = [ratio_filter, filled_filter, cropped_filter, query]
        outputs = [
            image_ids_state,
            index_state,
            sample_number,
            sample_total,
            sample_heading,
            gallery,
            details,
            reasons,
            raw_records,
        ]
        for component in filter_inputs:
            component.change(apply_filters, inputs=filter_inputs, outputs=outputs)
        navigation_outputs = outputs[1:]
        previous_button.click(
            lambda image_ids, index: move(image_ids, index, -1),
            inputs=[image_ids_state, index_state],
            outputs=navigation_outputs,
        )
        next_button.click(
            lambda image_ids, index: move(image_ids, index, 1),
            inputs=[image_ids_state, index_state],
            outputs=navigation_outputs,
        )
        jump_inputs = [image_ids_state, sample_number]
        jump_button.click(jump, inputs=jump_inputs, outputs=navigation_outputs)
        sample_number.submit(jump, inputs=jump_inputs, outputs=navigation_outputs)
    return app