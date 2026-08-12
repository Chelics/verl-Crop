import importlib.util
from io import BytesIO
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image


def load_converter_module():
    script_path = Path(__file__).parents[1] / "scripts" / "convert_image_once_to_verl.py"
    spec = importlib.util.spec_from_file_location("test_convert_image_once_to_verl", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_source(path: Path, color: str, title: str) -> None:
    module = load_converter_module()
    output = BytesIO()
    Image.new("RGB", (80, 60), color=color).save(output, format="WEBP", lossless=True)
    payload = output.getvalue()
    with Image.open(BytesIO(payload)) as image:
        image_id = module.normalized_pixel_hash(image)
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "image_id": image_id,
                    "original_image": payload,
                    "title": title,
                    "ImageCaption": f"Caption for {title}",
                }
            ]
        ),
        path,
    )


def test_converts_raw_train_and_test_to_four_ratio_verl_data(tmp_path):
    module = load_converter_module()
    train_path = tmp_path / "image_once_train.parquet"
    test_path = tmp_path / "image_once_test.parquet"
    make_source(train_path, "red", "Train title")
    make_source(test_path, "blue", "Test title")
    prompt_path = tmp_path / "v1.txt"
    prompt_path.write_text(
        "<image>\nNews title: {title}\nTarget ratio: {target_ratio}\n"
        'Return {"cx_pct": CX_PERCENT, "cy_pct": CY_PERCENT, "area_pct": AREA_PERCENT}',
        encoding="utf-8",
    )

    report = module.convert_image_once_datasets(
        train_path=train_path,
        test_path=test_path,
        output_dir=tmp_path / "converted",
        policy_prompt_path=prompt_path,
    )

    assert report["train"]["output_rows"] == 4
    assert report["test"]["output_rows"] == 4
    train_rows = pq.read_table(report["train"]["output_path"]).to_pylist()
    test_rows = pq.read_table(report["test"]["output_path"]).to_pylist()
    assert {row["extra_info"]["split"] for row in train_rows} == {"train"}
    assert {row["extra_info"]["split"] for row in test_rows} == {"test"}
    assert {row["extra_info"]["target_ratio"] for row in train_rows} == {1.0, 1.91, 1.77, 1.59}
    assert train_rows[0]["extra_info"]["caption"] == "Caption for Train title"
    assert "cx_pct" in train_rows[0]["prompt"][0]["content"]
    assert Path(train_rows[0]["images"][0]).is_file()