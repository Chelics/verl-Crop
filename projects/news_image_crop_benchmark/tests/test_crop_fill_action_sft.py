import importlib.util
import json
from pathlib import Path

import pytest


def load_validator():
    script = Path(__file__).parents[1] / "scripts" / "validate_crop_fill_action_sft.py"
    spec = importlib.util.spec_from_file_location("test_validate_crop_fill_action_sft", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def row_for(target):
    return {
        "messages": [
            {
                "role": "user",
                "content": "<image>\nDo not output a description or any additional fields.",
            },
            {"role": "assistant", "content": json.dumps(target, separators=(",", ":"))},
        ],
        "images": [{"image": "/mnt/blob_output/image.webp", "min_pixels": 65536, "max_pixels": 1048576}],
        "target_ratio": target["target_ratio"],
        "is_cropped": target["is_cropped"],
        "is_filled": target["is_filled"],
        "reference_description": "Reference explanation for audit only.",
    }


def test_validates_all_four_actions():
    module = load_validator()
    targets = [
        {"target_ratio": 1.0, "is_cropped": True, "is_filled": False, "crop_box": [0, 0, 1, 1], "fill_color": None},
        {"target_ratio": 1.59, "is_cropped": True, "is_filled": True, "crop_box": [0.1, 0.1, 0.9, 0.9], "fill_color": [1, 2, 3]},
        {"target_ratio": 1.77, "is_cropped": False, "is_filled": True, "crop_box": None, "fill_color": [255, 255, 255]},
        {"target_ratio": 1.0, "is_cropped": False, "is_filled": False, "crop_box": None, "fill_color": None},
    ]

    assert [module.validate_action_row(row_for(target), index) for index, target in enumerate(targets)] == [
        "crop",
        "crop_fill",
        "fill",
        "keep",
    ]


def test_rejects_description_in_assistant_target():
    module = load_validator()
    target = {
        "target_ratio": 1.0,
        "is_cropped": False,
        "is_filled": False,
        "crop_box": None,
        "fill_color": None,
        "description": "Must not be trained.",
    }

    with pytest.raises(ValueError, match="target fields"):
        module.validate_action_row(row_for(target), 0)