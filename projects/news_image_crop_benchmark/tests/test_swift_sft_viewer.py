import io
import json

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from news_crop_benchmark.swift_sft_viewer import SwiftSFTDataset, build_swift_sft_app, render_target_action


def image_bytes(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (16, 10), color).save(output, format="JPEG")
    return output.getvalue()


def target_action(ratio: float) -> str:
    if ratio == 1.0:
        payload = {"mode": "crop", "bbox": [187, 0, 813, 1000], "reason": "Square crop"}
    else:
        payload = {
            "mode": "crop_then_pad",
            "bbox": [0, 0, 1000, 1000],
            "padding_colors_rgb": {
                "top": [10, 20, 30],
                "right": [40, 50, 60],
                "bottom": [70, 80, 90],
                "left": [100, 110, 120],
            },
            "padding_weights": {"top": 1.0, "right": 3.0, "bottom": 1.0, "left": 1.0},
            "padding_style": "solid",
            "reason": "Preserve the complete source and pad",
        }
    return f"<crop>{json.dumps(payload, separators=(',', ':'))}</crop>"


def test_renders_crop_and_weighted_side_padding():
    source = Image.new("RGB", (200, 100), "white")
    cropped = render_target_action(source, target_action(1.0), 1.0)
    padded = render_target_action(source, target_action(1.91), 3.0)

    assert cropped.size == (126, 100)
    assert padded.size == (300, 100)
    assert padded.getpixel((0, 50)) == (100, 110, 120)
    assert padded.getpixel((299, 50)) == (40, 50, 60)
    assert padded.getpixel((25, 50)) == (255, 255, 255)


def test_groups_filters_caches_unique_traces_and_builds_app(tmp_path):
    path = tmp_path / "sft.parquet"
    rows = []
    for trace_index, trace_id in enumerate(("trace-a", "trace-b")):
        payload = image_bytes("red" if trace_index == 0 else "blue")
        for ratio in (1.0, 1.59, 1.77, 1.91):
            rows.append(
                {
                    "sample_id": f"{trace_id}:{ratio:g}:1",
                    "data_source": "news_image_crop_sft",
                    "ability": "visual_editorial_cropping",
                    "messages": [
                        {"role": "user", "content": f"<image>\nArticle headline: Title {trace_id}\nImage caption: Caption {trace_id}\nTarget width:height ratio: {ratio:g}:1"},
                        {"role": "assistant", "content": "<crop>{}</crop>"},
                    ],
                    "images": [{"bytes": payload}],
                    "enable_thinking": False,
                    "trace_id": trace_id,
                    "aspect_ratio": f"{ratio:g}:1",
                    "target_ratio": ratio,
                    "render_mode": "cropped" if ratio == 1.0 else "padded",
                    "was_cropped": True,
                    "was_padded": ratio != 1.0,
                    "edge_artifact_trim_pixels": [0, 0, 0, 0],
                    "target_action": target_action(ratio),
                    "reason": f"Reason {trace_id}",
                    "confidence": "high",
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path)
    dataset = SwiftSFTDataset(path, cache_dir=tmp_path / "cache")

    dataset.prepare_previews(batch_size=1)
    view = dataset.trace_view("trace-a")
    app = build_swift_sft_app(dataset)

    assert dataset.summary["rows"] == 8
    assert dataset.summary["traces"] == 2
    assert dataset.summary["ratios"] == {"1.59:1": 2, "1.77:1": 2, "1.91:1": 2, "1:1": 2}
    assert dataset.summary["confidences"] == {"high": 8}
    assert dataset.summary["cropped"] == 8
    assert dataset.summary["padded"] == 6
    assert dataset.filter_trace_ids(mode="padded") == ["trace-a", "trace-b"]
    assert dataset.filter_trace_ids(query="Title trace-b") == ["trace-b"]
    assert len(list((dataset.cache_dir / "previews").glob("*.jpg"))) == 2
    assert len(list(dataset.cache_dir.glob("targets-v1/*/*.jpg"))) == 8
    assert view["title"] == "Title trace-a"
    assert len(view["rows"]) == 4
    assert len(view["targets"]) == 4
    assert app.title == "Swift Crop SFT Viewer"