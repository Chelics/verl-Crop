import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image


def load_module():
    scripts_dir = Path(__file__).parents[1] / "scripts"
    sys.path.insert(0, str(scripts_dir))
    try:
        script = scripts_dir / "generate_swift_crop_ratio_report.py"
        spec = importlib.util.spec_from_file_location("test_generate_swift_crop_ratio_report", script)
        module = importlib.util.module_from_spec(spec)
        assert spec.loader is not None
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def test_inscribed_reference_box_supports_top_left_and_center_alignment():
    module = load_module()

    assert module.inscribed_reference_box((10, 20, 210, 120), 1.0, alignment="top_left") == (10, 20, 110, 120)
    assert module.inscribed_reference_box((10, 20, 210, 120), 1.0, alignment="center") == (60, 20, 160, 120)
    assert module.inscribed_reference_box((10, 20, 110, 220), 1.0, alignment="top_left") == (10, 20, 110, 120)
    assert module.inscribed_reference_box((10, 20, 110, 220), 1.0, alignment="center") == (10, 70, 110, 170)


def test_generates_crop_only_report_and_structured_outputs(tmp_path):
    module = load_module()
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / "sample.webp"
    Image.new("RGB", (200, 100), color=(230, 230, 225)).save(image_path, format="WEBP", lossless=True)
    prompt = (
        "<image>\nNews headline: A sample headline\nImage caption: A sample caption\n"
        "Target aspect ratio (width/height): 1"
    )
    plans = [
        {
            "target_ratio": 1.0,
            "is_cropped": True,
            "is_filled": False,
            "crop_box": [0.0, 0.0, 1.0, 1.0],
            "fill_color": None,
            "description": "The model keeps the full image.",
        },
        {
            "target_ratio": 1.0,
            "is_cropped": False,
            "is_filled": True,
            "crop_box": None,
            "fill_color": [230, 230, 225],
            "description": "The model pads the image.",
        },
    ]
    results_path = tmp_path / "results.jsonl"
    results_path.write_text(
        "".join(
            json.dumps(
                {
                    "response": json.dumps(plan),
                    "messages": [{"role": "user", "content": prompt}],
                    "images": [{"path": "/cluster/sample.webp"}],
                }
            )
            + "\n"
            for plan in plans
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "report"

    summary = module.generate_report(
        results_path=results_path,
        image_root=image_root,
        output_dir=output_dir,
        max_thumbnail_edge=120,
        overwrite=False,
    )

    detail = json.loads((output_dir / "details.jsonl").read_text(encoding="utf-8"))
    assert summary["source_records"] == 2
    assert summary["crop_only_records"] == 1
    assert summary["operation_counts"] == {"crop": 1, "fill": 1}
    assert detail["actual_ratio"] == 2.0
    assert detail["target_ratio"] == 1.0
    assert detail["prediction_width"] == 200
    assert detail["prediction_height"] == 100
    assert detail["reference_width"] == 100
    assert detail["reference_height"] == 100
    assert detail["remove_pixels"] == 100.0
    assert detail["remove_percent"] == 50.0
    assert detail["final_render_action"] is None
    assert (output_dir / "details.csv").is_file()
    assert (output_dir / "summary.json").is_file()
    assert (output_dir / "report.html").is_file()
    assert (output_dir / "assets" / "thumbnails" / "sample.jpg").is_file()
    report = (output_dir / "report.html").read_text(encoding="utf-8")
    assert "A sample headline" in report
    assert "200 × 100 px" in report
    assert "100.0 px of width" in report
    assert "querySelectorAll('button[data-alignment]')" in report