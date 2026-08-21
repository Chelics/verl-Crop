import importlib.util
import json
from pathlib import Path


def load_module():
    script = Path(__file__).parents[1] / "scripts" / "validate_swift_format_only_sft.py"
    spec = importlib.util.spec_from_file_location("validate_swift_format_only_sft", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_validates_original_six_field_target():
    module = load_module()
    target = {
        "target_ratio": 1.59,
        "is_cropped": True,
        "is_filled": False,
        "crop_box": [0.1, 0.2, 0.9, 0.8],
        "fill_color": None,
        "description": "Keep the original detailed supervision.",
    }
    row = {
        "messages": [
            {"role": "user", "content": "<image>\nOriginal prompt"},
            {"role": "assistant", "content": json.dumps(target)},
        ],
        "target_ratio": 1.59,
        "is_cropped": True,
        "is_filled": False,
    }

    module.validate_target(row, 0)