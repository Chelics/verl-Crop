from __future__ import annotations

import html
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


class ResultDataset:
    def __init__(self, result_dir: Path) -> None:
        self.result_dir = Path(os.path.abspath(result_dir.expanduser()))
        summary_path = self.result_dir / "summary.json"
        details_path = self.result_dir / "details.jsonl"
        if not summary_path.is_file():
            raise FileNotFoundError(summary_path)
        if not details_path.is_file():
            raise FileNotFoundError(details_path)

        self.summary = json.loads(summary_path.read_text(encoding="utf-8"))
        self.rows = [
            json.loads(line)
            for line in details_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not self.rows:
            raise ValueError(f"No result rows found in {details_path}")

        self._rows_by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in self.rows:
            self._rows_by_image[str(row["image_id"])].append(row)
        for image_rows in self._rows_by_image.values():
            image_rows.sort(key=lambda row: float(row.get("target_ratio", 0)))
        self.image_ids = sorted(
            self._rows_by_image,
            key=lambda image_id: (
                min(int(row.get("source_index", 0)) for row in self._rows_by_image[image_id]),
                image_id,
            ),
        )

    @property
    def ratios(self) -> list[str]:
        return self._unique_values(row.get("target_ratio") for row in self.rows)

    @property
    def tiers(self) -> list[str]:
        return self._unique_values(row.get("judge_label") for row in self.rows)

    @property
    def rules(self) -> list[str]:
        return sorted(
            {
                str(rule)
                for row in self.rows
                for rule in self._as_list(row.get("judge_rules"))
                if rule not in (None, "")
            }
        )

    @property
    def modes(self) -> list[str]:
        return self._unique_values(self._mode(row) for row in self.rows)

    @property
    def statuses(self) -> list[str]:
        return self._unique_values(row.get("generation_status") for row in self.rows)

    def filter_image_ids(
        self,
        *,
        ratio: str | None = None,
        tier: str | None = None,
        rule: str | None = None,
        mode: str | None = None,
        status: str | None = None,
    ) -> list[str]:
        filters = {
            "target_ratio": ratio,
            "judge_label": tier,
            "generation_status": status,
        }

        def matches(row: dict[str, Any]) -> bool:
            if mode is not None and self._mode(row) != mode:
                return False
            if any(
                expected is not None and str(row.get(field)) != expected
                for field, expected in filters.items()
            ):
                return False
            return rule is None or rule in {str(value) for value in self._as_list(row.get("judge_rules"))}

        return [
            image_id
            for image_id in self.image_ids
            if any(matches(row) for row in self._rows_by_image[image_id])
        ]

    def image_view(self, image_id: str) -> dict[str, Any]:
        image_rows = self._rows_by_image[image_id]
        first = image_rows[0]
        candidates = []
        for row in image_rows:
            candidate_path = self.resolve_asset(row.get("candidate_path"))
            if candidate_path is not None:
                candidates.append(
                    (
                        candidate_path,
                        self._candidate_caption(row),
                    )
                )
        return {
            "image_id": image_id,
            "title": str(first.get("title", "")),
            "caption": str(first.get("caption", "")),
            "original": self.resolve_asset(first.get("original_render_path")),
            "candidates": candidates,
            "rows": image_rows,
        }

    def resolve_asset(self, value: Any) -> str | None:
        if not value:
            return None
        relative_path = Path(str(value))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            return None
        candidate = Path(os.path.abspath(self.result_dir / relative_path))
        try:
            candidate.relative_to(self.result_dir)
        except ValueError:
            return None
        return str(candidate) if candidate.is_file() else None

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return [part.strip() for part in value.split(",") if part.strip()]
            return parsed if isinstance(parsed, list) else [parsed]
        return [value]

    @staticmethod
    def _unique_values(values: Iterable[Any]) -> list[str]:
        unique = {str(value) for value in values if value not in (None, "")}
        try:
            return sorted(unique, key=float)
        except ValueError:
            return sorted(unique)

    @staticmethod
    def _candidate_caption(row: dict[str, Any]) -> str:
        parts = [f"Ratio {row.get('target_ratio', 'N/A')}"]
        mode = ResultDataset._mode(row)
        if mode:
            parts.append(mode.upper())
        if row.get("judge_label") is not None:
            parts.append(f"Tier {row['judge_label']}")
        rules = ResultDataset._as_list(row.get("judge_rules"))
        if rules:
            parts.append(", ".join(str(rule) for rule in rules))
        if row.get("ratio_compliant") is not None:
            parts.append("RATIO OK" if row["ratio_compliant"] else "RATIO MISMATCH")
        if row.get("output_ratio_error") is not None:
            parts.append(f"Error {float(row['output_ratio_error']) * 100:.3f}%")
        return " | ".join(parts)

    @staticmethod
    def _mode(row: dict[str, Any]) -> str | None:
        value = row.get("predicted_mode") or row.get("layout_operation") or row.get("operation")
        return str(value) if value not in (None, "") else None


class ResultCollection:
    def __init__(self, result_path: Path) -> None:
        result_path = Path(os.path.abspath(result_path.expanduser()))
        preferred_name = result_path.name if self._is_result_dir(result_path) else None
        self.root = result_path.parent if preferred_name else result_path
        self._result_paths = {
            child.name: child
            for child in self.root.iterdir()
            if child.is_dir() and self._is_result_dir(child)
        }
        if preferred_name and preferred_name not in self._result_paths:
            self._result_paths[preferred_name] = result_path
        if not self._result_paths:
            raise ValueError(f"No result directories found under {self.root}")
        self.initial_name = preferred_name or self.names[0]
        self._cache: dict[str, ResultDataset] = {}

    @classmethod
    def from_dataset(cls, dataset: ResultDataset) -> ResultCollection:
        collection = cls.__new__(cls)
        collection.root = dataset.result_dir.parent
        collection._result_paths = {dataset.result_dir.name: dataset.result_dir}
        collection.initial_name = dataset.result_dir.name
        collection._cache = {dataset.result_dir.name: dataset}
        return collection

    @property
    def names(self) -> list[str]:
        return sorted(self._result_paths)

    def get(self, name: str) -> ResultDataset:
        if name not in self._result_paths:
            raise KeyError(name)
        if name not in self._cache:
            self._cache[name] = ResultDataset(self._result_paths[name])
        return self._cache[name]

    @staticmethod
    def _is_result_dir(path: Path) -> bool:
        return (path / "summary.json").is_file() and (path / "details.jsonl").is_file()


ALL_FILTER = "All"
VIEWER_CSS = """
.gradio-container { max-width: 1500px !important; }
.viewer-header { border-bottom: 1px solid var(--border-color-primary); padding-bottom: 10px; }
.viewer-header h1 { font-size: 24px !important; letter-spacing: 0 !important; }
.filter-row, .navigation-row { align-items: end; }
.sample-heading { min-height: 96px; }
.original-image img, .candidate-gallery img { object-fit: contain !important; }
button, .form { border-radius: 6px !important; }
"""


def sample_index(value: Any, count: int) -> int:
    if count <= 0:
        return 0
    try:
        requested = int(float(value))
    except (TypeError, ValueError, OverflowError):
        requested = 1
    return max(0, min(requested - 1, count - 1))


def build_app(source: ResultCollection | ResultDataset) -> Any:
    import gradio as gr
    from PIL import Image

    collection = source if isinstance(source, ResultCollection) else ResultCollection.from_dataset(source)
    initial_run = collection.initial_name
    initial_dataset = collection.get(initial_run)
    initial_ids = initial_dataset.image_ids

    def load_image(path: str | None) -> Any:
        if path is None:
            return None
        try:
            with Image.open(path) as image:
                return image.copy()
        except OSError:
            return None

    def render(run_name: str, image_ids: list[str], index: int) -> tuple[Any, ...]:
        if not image_ids:
            return 0, 1, "/ 0", "### No matching samples", None, [], [], []
        dataset = collection.get(run_name)
        index = max(0, min(index, len(image_ids) - 1))
        view = dataset.image_view(image_ids[index])
        heading = (
            f"### {html.escape(view['title'])}\n"
            f"`{html.escape(view['image_id'])}`\n\n"
            f"{html.escape(view['caption'])}"
        )
        table = []
        for row in view["rows"]:
            quality = row.get("judge_label")
            if quality is None and row.get("ratio_compliant") is not None:
                quality = "yes" if row["ratio_compliant"] else "no"
            notes = ", ".join(str(rule) for rule in dataset._as_list(row.get("judge_rules")))
            if not notes and row.get("output_ratio_error") is not None:
                notes = f"ratio error {float(row['output_ratio_error']) * 100:.3f}%"
            table.append(
                [
                    row.get("target_ratio"),
                    dataset._mode(row) or "",
                    row.get("generation_status") or "",
                    quality if quality is not None else "",
                    notes,
                    row.get("total_attempt_count", row.get("attempt_count", "")),
                ]
            )
        return (
            index,
            index + 1,
            f"/ {len(image_ids)}",
            heading,
            load_image(view["original"]),
            [(load_image(path), caption) for path, caption in view["candidates"]],
            table,
            view["rows"],
        )

    def apply_filters(
        run_name: str,
        ratio: str,
        tier: str,
        rule: str,
        mode: str,
        status: str,
    ) -> tuple[Any, ...]:
        dataset = collection.get(run_name)
        image_ids = dataset.filter_image_ids(
            ratio=None if ratio == ALL_FILTER else ratio,
            tier=None if tier == ALL_FILTER else tier,
            rule=None if rule == ALL_FILTER else rule,
            mode=None if mode == ALL_FILTER else mode,
            status=None if status == ALL_FILTER else status,
        )
        return (image_ids, *render(run_name, image_ids, 0))

    def switch_experiment(run_name: str) -> tuple[Any, ...]:
        dataset = collection.get(run_name)
        image_ids = dataset.image_ids
        filter_updates = [
            gr.Dropdown(choices=[ALL_FILTER, *values], value=ALL_FILTER)
            for values in (dataset.ratios, dataset.tiers, dataset.rules, dataset.modes, dataset.statuses)
        ]
        return (_summary_markdown(dataset.summary), *filter_updates, image_ids, *render(run_name, image_ids, 0))

    def move(run_name: str, image_ids: list[str], index: int, offset: int) -> tuple[Any, ...]:
        return render(run_name, image_ids, index + offset)

    def jump(run_name: str, image_ids: list[str], requested: Any) -> tuple[Any, ...]:
        return render(run_name, image_ids, sample_index(requested, len(image_ids)))

    initial = render(initial_run, initial_ids, 0)
    with gr.Blocks(title="Crop Evaluation Results") as app:
        summary_header = gr.Markdown(_summary_markdown(initial_dataset.summary), elem_classes="viewer-header")
        image_ids_state = gr.State(initial_ids)
        index_state = gr.State(initial[0])

        experiment = gr.Dropdown(
            collection.names,
            value=initial_run,
            label="Experiment",
            filterable=True,
        )

        with gr.Row(elem_classes="filter-row"):
            ratio_filter = gr.Dropdown([ALL_FILTER, *initial_dataset.ratios], value=ALL_FILTER, label="Ratio")
            tier_filter = gr.Dropdown([ALL_FILTER, *initial_dataset.tiers], value=ALL_FILTER, label="Tier")
            rule_filter = gr.Dropdown([ALL_FILTER, *initial_dataset.rules], value=ALL_FILTER, label="Rule")
            mode_filter = gr.Dropdown([ALL_FILTER, *initial_dataset.modes], value=ALL_FILTER, label="Mode")
            status_filter = gr.Dropdown([ALL_FILTER, *initial_dataset.statuses], value=ALL_FILTER, label="Status")

        with gr.Row(elem_classes="navigation-row"):
            previous_button = gr.Button("Previous", variant="secondary")
            sample_number = gr.Number(initial[1], label="Sample", minimum=1, precision=0, step=1)
            sample_total = gr.Markdown(initial[2])
            jump_button = gr.Button("Jump", variant="secondary")
            next_button = gr.Button("Next", variant="primary")

        sample_heading = gr.Markdown(initial[3], elem_classes="sample-heading")
        with gr.Row(equal_height=True):
            original = gr.Image(initial[4], label="Original", interactive=False, elem_classes="original-image")
            candidates = gr.Gallery(
                initial[5],
                label="Candidates",
                columns=2,
                rows=2,
                object_fit="contain",
                height=640,
                elem_classes="candidate-gallery",
            )
        details = gr.Dataframe(
            value=initial[6],
            headers=["Ratio", "Mode", "Generation", "Tier / Ratio OK", "Rules / Ratio Error", "Attempts"],
            datatype=["number", "str", "str", "str", "str", "number"],
            interactive=False,
            label="Task details",
        )
        with gr.Accordion("Raw records", open=False):
            raw_records = gr.JSON(initial[7], label=None)

        filter_inputs = [experiment, ratio_filter, tier_filter, rule_filter, mode_filter, status_filter]
        filter_outputs = [
            image_ids_state,
            index_state,
            sample_number,
            sample_total,
            sample_heading,
            original,
            candidates,
            details,
            raw_records,
        ]
        for component in filter_inputs[1:]:
            component.change(apply_filters, inputs=filter_inputs, outputs=filter_outputs)

        navigation_outputs = [
            index_state,
            sample_number,
            sample_total,
            sample_heading,
            original,
            candidates,
            details,
            raw_records,
        ]
        previous_button.click(
            lambda run_name, image_ids, index: move(run_name, image_ids, index, -1),
            inputs=[experiment, image_ids_state, index_state],
            outputs=navigation_outputs,
        )
        next_button.click(
            lambda run_name, image_ids, index: move(run_name, image_ids, index, 1),
            inputs=[experiment, image_ids_state, index_state],
            outputs=navigation_outputs,
        )
        jump_inputs = [experiment, image_ids_state, sample_number]
        jump_button.click(jump, inputs=jump_inputs, outputs=navigation_outputs)
        sample_number.submit(jump, inputs=jump_inputs, outputs=navigation_outputs)
        experiment.change(
            switch_experiment,
            inputs=experiment,
            outputs=[
                summary_header,
                ratio_filter,
                tier_filter,
                rule_filter,
                mode_filter,
                status_filter,
                image_ids_state,
                *navigation_outputs,
            ],
        )
    return app


def _summary_markdown(summary: dict[str, Any]) -> str:
    overall = summary.get("overall", {})
    preferred = (
        "tasks",
        "generation_success_count",
        "judge_completed_count",
        "mean_reward",
        "mean_label",
        "crop_count",
        "pad_count",
    )
    metrics = [(key, overall[key]) for key in preferred if key in overall]
    if not metrics:
        metrics = [(key, value) for key, value in overall.items() if not isinstance(value, (dict, list))][:7]
    metric_text = " &nbsp; ".join(
        f"**{html.escape(key.replace('_', ' ').title())}:** {html.escape(str(value))}"
        for key, value in metrics
    )
    model_name = html.escape(str(summary.get("model_name") or summary.get("run_id") or "Evaluation Results"))
    run_id = html.escape(str(summary.get("run_id", "")))
    return f"# {model_name}\n`{run_id}`\n\n{metric_text}"