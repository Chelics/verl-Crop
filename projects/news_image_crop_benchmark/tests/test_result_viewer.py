import json
from pathlib import Path

import pytest

from news_crop_benchmark.result_viewer import ResultCollection, ResultDataset, sample_index


def write_result(result_dir: Path) -> None:
    (result_dir / "renders" / "originals").mkdir(parents=True)
    (result_dir / "renders" / "candidates").mkdir(parents=True)
    (result_dir / "summary.json").write_text(
        json.dumps({"model_name": "Test model", "overall": {"tasks": 3}}),
        encoding="utf-8",
    )
    rows = [
        {
            "task_id": "image-a__ratio_1",
            "source_index": 0,
            "image_id": "image-a",
            "title": "Title A",
            "caption": "Caption A",
            "target_ratio": 1.0,
            "original_render_path": "renders/originals/image-a.jpg",
            "candidate_path": "renders/candidates/image-a-1.jpg",
            "generation_status": "valid",
            "judge_label": 0,
            "judge_rules": ["T0.1"],
        },
        {
            "task_id": "image-a__ratio_1.91",
            "source_index": 0,
            "image_id": "image-a",
            "title": "Title A",
            "caption": "Caption A",
            "target_ratio": 1.91,
            "original_render_path": "renders/originals/image-a.jpg",
            "candidate_path": "renders/candidates/image-a-1.91.jpg",
            "generation_status": "valid",
            "judge_label": 4,
            "judge_rules": ["T4.5"],
        },
        {
            "task_id": "image-b__ratio_1",
            "source_index": 1,
            "image_id": "image-b",
            "title": "Title B",
            "caption": "Caption B",
            "target_ratio": 1.0,
            "original_render_path": "../outside.jpg",
            "candidate_path": None,
            "generation_status": "request_failed",
            "judge_label": None,
            "judge_rules": "[]",
        },
    ]
    (result_dir / "details.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )
    for name in ("image-a.jpg",):
        (result_dir / "renders" / "originals" / name).write_bytes(b"original")
    for name in ("image-a-1.jpg", "image-a-1.91.jpg"):
        (result_dir / "renders" / "candidates" / name).write_bytes(b"candidate")


def test_loads_filters_and_groups_results(tmp_path):
    write_result(tmp_path)
    dataset = ResultDataset(tmp_path)

    assert dataset.summary["model_name"] == "Test model"
    assert dataset.image_ids == ["image-a", "image-b"]
    assert dataset.ratios == ["1.0", "1.91"]
    assert dataset.tiers == ["0", "4"]
    assert dataset.rules == ["T0.1", "T4.5"]
    assert dataset.filter_image_ids(tier="4") == ["image-a"]
    assert dataset.filter_image_ids(rule="T0.1", ratio="1.0") == ["image-a"]
    assert dataset.filter_image_ids(status="request_failed") == ["image-b"]

    view = dataset.image_view("image-a")
    assert view["title"] == "Title A"
    assert len(view["candidates"]) == 2
    assert view["candidates"][1][1] == "Ratio 1.91 | Tier 4 | T4.5"


def test_rejects_assets_outside_result_directory(tmp_path):
    write_result(tmp_path)
    (tmp_path.parent / "outside.jpg").write_bytes(b"outside")
    dataset = ResultDataset(tmp_path)

    assert dataset.image_view("image-b")["original"] is None
    assert dataset.resolve_asset("renders/originals/image-a.jpg") is not None


def test_discovers_experiments_lazily_and_clamps_sample_numbers(tmp_path):
    first = tmp_path / "run-a"
    second = tmp_path / "run-b"
    write_result(first)
    write_result(second)

    collection = ResultCollection(first)

    assert collection.names == ["run-a", "run-b"]
    assert collection.initial_name == "run-a"
    assert collection._cache == {}
    assert collection.get("run-b").summary["model_name"] == "Test model"
    assert list(collection._cache) == ["run-b"]
    assert sample_index(1, 10) == 0
    assert sample_index(7, 10) == 6
    assert sample_index(999, 10) == 9
    assert sample_index("invalid", 10) == 0


def test_builds_gradio_app_when_viewer_extra_is_installed(tmp_path):
    pytest.importorskip("gradio")
    from PIL import Image

    from news_crop_benchmark.result_viewer import build_app

    write_result(tmp_path)
    for path in (tmp_path / "renders").rglob("*.jpg"):
        Image.new("RGB", (2, 2), "white").save(path)
    app = build_app(ResultDataset(tmp_path))

    assert app.title == "Crop Evaluation Results"