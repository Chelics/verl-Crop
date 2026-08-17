import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from news_crop_benchmark.cropped_dataset_viewer import CroppedDataset


def image_bytes(color: str) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", (8, 6), color).save(output, format="JPEG")
    return output.getvalue()


def write_dataset(path):
    rows = []
    for source_index, trace_id in enumerate(("trace-a", "trace-b")):
        for ratio, mode, color in ((1.0, "cropped", "red"), (1.91, "padded", "blue")):
            rows.append(
                {
                    "source_index": source_index,
                    "trace_id": trace_id,
                    "headline": f"Headline {trace_id}",
                    "caption": f"Caption {trace_id}",
                    "original_image_url": "https://example.com/image.jpg",
                    "original_width": 8,
                    "original_height": 6,
                    "aspect_ratio": "1:1" if ratio == 1.0 else "1.91:1",
                    "target_ratio": ratio,
                    "crop_required": True,
                    "feasible": mode == "cropped",
                    "render_mode": mode,
                    "was_cropped": True,
                    "was_padded": mode == "padded",
                    "bbox_normalized": [0.0, 0.0, 1.0, 1.0],
                    "bbox_pixels": [0, 0, 8, 6],
                    "edge_artifact_trim_pixels": [0, 0, 0, 0],
                    "background_color_rgb": None,
                    "padding_color_rgb": [0, 0, 255] if mode == "padded" else None,
                    "output_width": 8,
                    "output_height": 8,
                    "reason": f"Reason {mode}",
                    "confidence": "high",
                    "cropped_image_format": "JPEG",
                    "cropped_image": image_bytes(color),
                    "error": None,
                }
            )
    pq.write_table(pa.Table.from_pylist(rows), path)


def test_loads_filters_and_extracts_grouped_previews(tmp_path):
    parquet_path = tmp_path / "cropped.parquet"
    write_dataset(parquet_path)
    dataset = CroppedDataset(parquet_path, cache_dir=tmp_path / "cache")

    assert dataset.summary["rows"] == 4
    assert dataset.summary["stories"] == 2
    assert "cropped_image" not in dataset.rows[0]
    assert dataset.render_modes == ["cropped", "padded"]
    assert dataset.filter_trace_ids(render_mode="padded") == ["trace-a", "trace-b"]
    assert dataset.filter_trace_ids(query="headline trace-b") == ["trace-b"]

    progress = []
    dataset.prepare_previews(batch_size=1, progress=lambda completed, total: progress.append((completed, total)))
    view = dataset.story_view("trace-a")

    assert len(view["gallery"]) == 2
    assert all(dataset.preview_path(row).is_file() for row in dataset.rows)
    assert progress[-1] == (4, 4)


def test_reuses_complete_preview_cache(tmp_path):
    parquet_path = tmp_path / "cropped.parquet"
    write_dataset(parquet_path)
    dataset = CroppedDataset(parquet_path, cache_dir=tmp_path / "cache")
    dataset.prepare_previews()

    dataset.prepare_previews(progress=lambda *_: (_ for _ in ()).throw(AssertionError("cache miss")))

    assert len(list((dataset.cache_dir / "previews").glob("*.jpg"))) == 4


def test_filters_before_selective_preview_extraction(tmp_path):
    parquet_path = tmp_path / "cropped.parquet"
    write_dataset(parquet_path)
    dataset = CroppedDataset(
        parquet_path,
        cache_dir=tmp_path / "filtered-cache",
        reason_prefix="Reason padded",
    )

    dataset.prepare_previews(batch_size=1)

    assert len(dataset.rows) == 2
    assert [row["_row_index"] for row in dataset.rows] == [1, 3]
    assert sorted(path.name for path in (dataset.cache_dir / "previews").glob("*.jpg")) == [
        "00001.jpg",
        "00003.jpg",
    ]
    assert len(dataset.story_view("trace-a")["gallery"]) == 1


def test_downloads_and_reuses_original_image(tmp_path):
    payload = image_bytes("green")
    request_count = 0

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            nonlocal request_count
            request_count += 1
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_):
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        parquet_path = tmp_path / "cropped.parquet"
        write_dataset(parquet_path)
        dataset = CroppedDataset(parquet_path, cache_dir=tmp_path / "cache")
        for row in dataset.rows:
            row["original_image_url"] = f"http://127.0.0.1:{server.server_port}/image.jpg"

        first = dataset.ensure_original("trace-a")
        second = dataset.ensure_original("trace-a")

        assert first is not None and first.is_file()
        assert second == first
        assert request_count == 1
        assert dataset.story_view("trace-a")["original"] == str(first)
    finally:
        server.shutdown()
        thread.join()


def test_builds_gradio_app(tmp_path):
    from news_crop_benchmark.cropped_dataset_viewer import build_cropped_dataset_app

    parquet_path = tmp_path / "cropped.parquet"
    write_dataset(parquet_path)
    dataset = CroppedDataset(parquet_path, cache_dir=tmp_path / "cache")
    dataset.prepare_previews()

    app = build_cropped_dataset_app(dataset)

    assert app.title == "Cropped Dataset Viewer"


def test_loads_and_filters_rendered_overrides(tmp_path):
    parquet_path = tmp_path / "cropped.parquet"
    write_dataset(parquet_path)
    override_dir = tmp_path / "overrides"
    (override_dir / "images").mkdir(parents=True)
    override_image = override_dir / "images" / "trace-a.jpg"
    Image.new("RGB", (12, 8), "green").save(override_image)
    record = {
        "trace_id": "trace-a",
        "aspect_ratio": "1:1",
        "operation": "crop",
        "image_path": "images/trace-a.jpg",
    }
    (override_dir / "overrides.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")

    dataset = CroppedDataset(parquet_path, cache_dir=tmp_path / "cache", override_dir=override_dir)
    view = dataset.story_view("trace-a")

    assert dataset.filter_trace_ids(has_override="Yes") == ["trace-a"]
    assert dataset.filter_trace_ids(has_override="No") == ["trace-b"]
    assert view["override_gallery"] == [(str(override_image), "1:1 | PROPOSED CROP")]
    assert view["override_records"][0]["operation"] == "crop"