import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image


def load_judge_module():
    script_path = Path(__file__).parents[1] / "scripts" / "judge_image_once_layout.py"
    spec = importlib.util.spec_from_file_location("test_judge_image_once_layout", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_detail(task_id, ratio, mode, candidate_path):
    return {
        "task_id": task_id,
        "source_index": 0,
        "image_id": "image",
        "title": "Example title",
        "caption": "Example caption",
        "target_ratio": ratio,
        "predicted_mode": mode,
        "render_status": "rendered",
        "candidate_path": candidate_path,
        "original_render_path": "renders/originals/image.jpg",
        "background_hex": "#E6EBF0" if mode == "pad" else None,
        "background_color": [230, 235, 240] if mode == "pad" else None,
        "content_box": [10, 0, 90, 80] if mode == "pad" else None,
        "padding_fraction": 0.2 if mode == "pad" else 0.0,
        "cx_pct": 50 if mode == "crop" else None,
        "cy_pct": 50 if mode == "crop" else None,
        "area_pct": 70 if mode == "crop" else None,
    }


def test_judges_crop_and_pad_and_writes_stratified_reports(tmp_path, monkeypatch):
    module = load_judge_module()
    source_dir = tmp_path / ".source_images"
    candidate_dir = tmp_path / "renders" / "candidates"
    original_preview_dir = tmp_path / "renders" / "originals"
    source_dir.mkdir(parents=True)
    candidate_dir.mkdir(parents=True)
    original_preview_dir.mkdir(parents=True)
    original = Image.new("RGB", (80, 60), color=(230, 235, 240))
    original.save(source_dir / "image.webp", format="WEBP", lossless=True)
    original.save(original_preview_dir / "image.jpg")
    original.save(candidate_dir / "crop.jpg")
    Image.new("RGB", (80, 80), color=(230, 235, 240)).save(candidate_dir / "pad.jpg")
    details = [
        make_detail("image__ratio_1.91", 1.91, "crop", "renders/candidates/crop.jpg"),
        make_detail("image__ratio_1", 1.0, "pad", "renders/candidates/pad.jpg"),
    ]
    (tmp_path / "details.jsonl").write_text(
        "".join(json.dumps(detail) + "\n" for detail in details), encoding="utf-8"
    )
    (tmp_path / "_LAYOUT_PIPELINE_COMPLETE.json").write_text(
        json.dumps({"tasks": 2}), encoding="utf-8"
    )
    contexts = []

    class FakeScorer:
        def __init__(self, _prompt_path):
            pass

        def score_detailed(
            self,
            _original,
            _candidate,
            _caption,
            _headline,
            log_context=None,
            evaluation_context=None,
        ):
            contexts.append(evaluation_context)
            if evaluation_context["selected_mode"] == "crop":
                label = 0.0
                rule = "C0.1"
                appropriateness = "appropriate"
                relationship = "cropped"
            else:
                label = 2.0
                rule = "P2.1"
                appropriateness = "inappropriate"
                relationship = "padded"
            output = json.dumps(
                {
                    "comparison": {"layout_relationship": relationship},
                    "evaluation": {
                        "label": str(int(label)),
                        "tier_name": "Excellent" if label == 0 else "Suboptimal",
                        "rules": [rule],
                        "confidence_score": "high",
                        "mode_appropriateness": appropriateness,
                    },
                }
            )
            return SimpleNamespace(
                status="completed",
                label=label,
                reward=(5.0 - label) / 5.0,
                output_text=output,
                response_id=f"response-{log_context['task_id']}",
                attempt_count=1,
                latency_ms=10.0,
                error_type=None,
            )

    monkeypatch.setattr(module, "CropVLMScorer", FakeScorer)
    prompt_path = tmp_path / "prompt.txt"
    prompt_path.write_text("Judge layouts", encoding="utf-8")

    loaded = module.load_layout_details(tmp_path)
    module.run_judge(loaded, tmp_path, prompt_path, judge_workers=1)
    judged = module.build_judged_details(loaded, tmp_path)
    summary = module.summarize(judged)
    module.write_results(judged, summary, tmp_path)
    module.render_html_report(judged, summary, tmp_path)
    module.render_markdown_report(judged, summary, tmp_path)

    assert {context["selected_mode"] for context in contexts} == {"crop", "pad"}
    assert next(context for context in contexts if context["selected_mode"] == "pad")[
        "background_hex"
    ] == "#E6EBF0"
    assert summary["overall"]["judge_completed_count"] == 2
    assert summary["overall"]["mean_judge_label"] == 1.0
    assert summary["overall"]["tier_0_1_acceptable_rate"] == 0.5
    assert summary["by_mode"]["crop"]["mean_judge_label"] == 0.0
    assert summary["by_mode"]["pad"]["mode_inappropriate_rate"] == 1.0
    assert summary["overall"]["rule_counts"] == {"C0.1": 1, "P2.1": 1}
    assert (tmp_path / "judge_summary.json").is_file()
    assert (tmp_path / "judge_details.parquet").is_file()
    assert (tmp_path / "judge_report.html").is_file()
    assert (tmp_path / "judge_report.md").is_file()