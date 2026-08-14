import io

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from news_crop_benchmark.image_once_pair_viewer import ImageOncePairDataset, build_image_once_pair_app


def image_bytes(size, color):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="WEBP")
    return output.getvalue()


def test_loads_searches_and_builds_pair_app(tmp_path):
    path = tmp_path / "pairs.parquet"
    rows = []
    for index in range(2):
        rows.append(
            {
                "image_id": f"image-{index}",
                "crop_image_id": f"crop-{index}",
                "original_image": image_bytes((20, 16), "red"),
                "cropped_image": image_bytes((12, 8), "blue"),
                "title": f"Title {index}",
                "ImageCaption": f"Caption {index}",
                "Reason": "",
                "source_original_url": f"https://example.com/original-{index}",
                "source_cropped_url": f"https://example.com/crop-{index}",
                "source_event_count": 1,
                "source_title_count": 2,
                "reason_count": 0,
            }
        )
    pq.write_table(pa.Table.from_pylist(rows), path)

    dataset = ImageOncePairDataset(path)
    view = dataset.pair_view(0)
    app = build_image_once_pair_app(dataset)

    assert dataset.summary == {"rows": 2, "empty_reasons": 2, "unique_images": 2}
    assert dataset.filter_indices("title 1") == [1]
    assert view["original"].size == (20, 16)
    assert view["crop"].size == (12, 8)
    assert app.title == "Image-Once Pair Viewer"