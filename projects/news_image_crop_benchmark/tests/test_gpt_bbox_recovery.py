import io
import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from news_crop_benchmark.gpt_bbox_recovery import recover_gpt_layouts


def image_bytes(size=(120, 100), color="blue"):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


class FakeStream:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return None

    def get_final_response(self):
        return SimpleNamespace(
            id="response-1",
            output_text='{"operation":"crop_pad","x1_pct":10,"y1_pct":10,"x2_pct":90,"y2_pct":90}',
        )


class FakeClient:
    def __init__(self):
        self.responses = self
        self.calls = 0

    def stream(self, **_):
        self.calls += 1
        return FakeStream()


def test_reuses_audited_and_recovers_remaining_with_gpt(tmp_path):
    train = tmp_path / "train.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": "image-a",
                    "original_image": image_bytes(),
                    "title": "Title",
                    "ImageCaption": "Caption",
                    "source_original_url": "https://example.com/source",
                }
            ]
        ),
        train,
    )
    manual = tmp_path / "manual.parquet"
    rows = []
    for ratio, size in ((1.0, (100, 100)), (1.2, (120, 100))):
        rows.append(
            {
                "save_id": f"save-{ratio}",
                "image_id": "image-a",
                "manual_crop": image_bytes(size),
                "ratio": ratio,
                "theme_color": "#000000",
                "fill_color_code": "#010203",
                "is_filled": True,
                "is_cropped": True,
                "crop_reason": "",
                "title": "Title",
                "ImageCaption": "Caption",
                "saved_at": None,
            }
        )
    schema = pa.schema(
        [
            ("save_id", pa.string()),
            ("image_id", pa.string()),
            ("manual_crop", pa.binary()),
            ("ratio", pa.float64()),
            ("theme_color", pa.string()),
            ("fill_color_code", pa.string()),
            ("is_filled", pa.bool_()),
            ("is_cropped", pa.bool_()),
            ("crop_reason", pa.string()),
            ("title", pa.string()),
            ("ImageCaption", pa.string()),
            ("saved_at", pa.timestamp("us", tz="UTC")),
        ]
    )
    from datetime import UTC, datetime

    for index, row in enumerate(rows):
        row["saved_at"] = datetime(2026, 1, 1, index, tzinfo=UTC)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), manual)
    audited = tmp_path / "audited.jsonl"
    audited.write_text(
        json.dumps(
            {
                "trace_id": "image-a",
                "aspect_ratio": "1:1",
                "operation": "pad",
                "source_box_normalized": [0, 0, 1, 1],
                "reason": "audited",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    client = FakeClient()

    records = recover_gpt_layouts(
        train,
        manual,
        audited,
        tmp_path / "output",
        client_factory=lambda: (client, "test-model"),
    )

    assert len(records) == 2
    assert client.calls == 1
    assert {record["provenance"] for record in records} == {"audited_override_v1", "gpt_pair_recovery"}
    assert all(record["bbox_normalized"] is not None for record in records)
    assert all((tmp_path / "output" / record["image_path"]).is_file() for record in records)