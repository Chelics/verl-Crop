import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_module():
    script = Path(__file__).parents[1] / "scripts" / "convert_swift_sft_to_verl.py"
    spec = importlib.util.spec_from_file_location("convert_swift_sft_to_verl", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_only_converts_swift_image_schema(tmp_path):
    module = load_module()
    row = {
        "messages": [
            {"role": "user", "content": "<image>\nKeep this prompt exactly."},
            {"role": "assistant", "content": '{"description":"Keep this target exactly."}'},
        ],
        "images": ["/mnt/blob_output/example.webp"],
        "image_id": "image-a",
        "source_index": 7,
    }
    input_path = tmp_path / "swift.parquet"
    output_path = tmp_path / "verl.parquet"
    pq.write_table(pa.Table.from_pylist([row]), input_path)

    report = module.convert_swift_sft_to_verl(input_path=input_path, output_path=output_path)

    output = pq.read_table(output_path).to_pylist()[0]
    assert {key: output[key] for key in row if key != "images"} == {key: row[key] for key in row if key != "images"}
    assert output["images"] == [
        {
            "image": row["images"][0],
            "min_pixels": 65536,
            "max_pixels": 1048576,
        }
    ]
    assert report["messages_unchanged"] is True
    assert report["only_transformed_column"] == "images"


def test_swift_compatible_split_preserves_all_messages(tmp_path):
    module = load_module()
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"<image>\nprompt-{index}"},
                {"role": "assistant", "content": f"target-{index}"},
            ],
            "images": [f"/mnt/blob_output/{index}.webp"],
            "source_index": index,
        }
        for index in range(100)
    ]
    input_path = tmp_path / "swift.parquet"
    output_dir = tmp_path / "split"
    pq.write_table(pa.Table.from_pylist(rows), input_path)

    report = module.convert_and_split_swift_sft_to_verl(input_path=input_path, output_dir=output_dir)

    train = pq.read_table(output_dir / "train.parquet").to_pylist()
    validation = pq.read_table(output_dir / "validation.parquet").to_pylist()
    combined = train + validation
    assert report["split_rows"] == {"train": 99, "validation": 1}
    assert report["swift_split_seed"] == 1608637542
    assert sorted(row["source_index"] for row in combined) == list(range(100))
    assert {row["messages"][0]["content"] for row in combined} == {
        row["messages"][0]["content"] for row in rows
    }
    assert report["messages_unchanged"] is True