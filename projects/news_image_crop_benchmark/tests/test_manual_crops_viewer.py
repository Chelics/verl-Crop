import io

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from news_crop_benchmark.manual_crops_viewer import ManualCropsDataset, build_manual_crops_app


def image_bytes(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (12, 8), color).save(output, format="PNG")
    return output.getvalue()


def test_loads_groups_filters_and_builds_app(tmp_path):
    path = tmp_path / "manual.parquet"
    rows = []
    for index, image_id in enumerate(("image-a", "image-b")):
        for ratio in (1.0, 1.91):
            rows.append(
                {
                    "save_id": f"save-{index}-{ratio}",
                    "image_id": image_id,
                    "manual_crop": image_bytes("red" if image_id == "image-a" else "blue"),
                    "ratio": ratio,
                    "theme_color": "#000000",
                    "fill_color_code": "#000000",
                    "is_filled": ratio == 1.0,
                    "is_cropped": ratio == 1.91,
                    "crop_reason": "Keep logo",
                    "title": f"Title {image_id}",
                    "ImageCaption": f"Caption {image_id}",
                    "saved_at": None,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path)

    dataset = ManualCropsDataset(path)
    view = dataset.image_view("image-a")
    app = build_manual_crops_app(dataset)

    assert dataset.summary == {"rows": 4, "images": 2, "filled": 2, "cropped": 2, "ratios": {"1.0": 2, "1.91": 2}}
    assert dataset.filter_image_ids(ratio="1.91", cropped="True") == ["image-a", "image-b"]
    assert dataset.filter_image_ids(query="title image-b") == ["image-b"]
    assert len(view["gallery"]) == 2
    assert "manual_crop" not in view["rows"][0]
    assert app.title == "Manual Crops Viewer"