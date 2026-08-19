import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def load_module():
    script = Path(__file__).parents[1] / "scripts" / "strip_detail_sft_descriptions.py"
    spec = importlib.util.spec_from_file_location("test_strip_detail_sft_descriptions", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_moves_description_out_of_assistant_without_changing_actions(tmp_path):
    module = load_module()
    target = {
        "target_ratio": 1.59,
        "is_cropped": True,
        "is_filled": True,
        "crop_box": [0.1, 0.2, 0.9, 0.8],
        "fill_color": [12, 34, 56],
        "description": "Keep the complete visible subject and use matching padding.",
    }
    row = {
        "messages": [
            {"role": "user", "content": f"<image>\nHeadline\n{module.DETAIL_INSTRUCTION}"},
            {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))},
        ],
        "images": ["/mnt/blob_output/image.webp"],
        "image_id": "image-a",
        "source_index": 1,
        "target_ratio": 1.59,
        "is_cropped": True,
        "is_filled": True,
    }
    input_path = tmp_path / "train.parquet"
    output_path = tmp_path / "action" / "train.parquet"
    pq.write_table(pa.Table.from_pylist([row]), input_path)

    report = module.strip_descriptions(input_path=input_path, output_path=output_path)

    output = pq.read_table(output_path).to_pylist()[0]
    assistant = json.loads(output["messages"][1]["content"])
    assert assistant == {key: target[key] for key in module.ACTION_FIELDS}
    assert "description" not in output["messages"][1]["content"]
    assert "Do not output a description" in output["messages"][0]["content"]
    assert output["reference_description"] == target["description"]
    assert output["images"] == [
        {
            "image": row["images"][0],
            "min_pixels": 65536,
            "max_pixels": 1048576,
        }
    ]
    assert report["description_in_assistant"] is False