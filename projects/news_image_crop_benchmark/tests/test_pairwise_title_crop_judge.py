import importlib.util
import json
import sys
from pathlib import Path

from PIL import Image


def load_module():
    script_path = Path(__file__).parents[1] / "scripts" / "judge_pairwise_title_crops.py"
    spec = importlib.util.spec_from_file_location("test_pairwise_title_crop_judge_module", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def evaluation(winner, title_a=3, title_b=4, quality_a=3, quality_b=3):
    return {
        "winner": winner,
        "title_relevance": {"A": title_a, "B": title_b},
        "crop_quality": {"A": quality_a, "B": quality_b},
        "title_relevant_elements": ["subject"],
        "A_missing_or_damaged_elements": [],
        "B_missing_or_damaged_elements": [],
        "reason": "Visible comparison.",
        "confidence": 0.9,
    }


def make_task(module):
    return module.PairTask(
        task_id="image__ratio_1",
        image_id="image",
        source_index=0,
        title="Example title",
        caption="Example caption",
        target_ratio=1.0,
        original_path=Path("original.webp"),
        visual_path=Path("visual.png"),
        mllm_path=Path("mllm.png"),
    )


def test_parse_judge_response_validates_schema_and_ranges():
    module = load_module()
    payload = evaluation("B")
    assert module.parse_judge_response(f"prefix\n{json.dumps(payload)}\nsuffix") == payload

    payload["title_relevance"]["A"] = 5
    try:
        module.parse_judge_response(json.dumps(payload))
    except ValueError as error:
        assert "title_relevance.A" in str(error)
    else:
        raise AssertionError("out-of-range scores must fail")


def test_ab_swap_maps_stable_mllm_win_and_source_scores():
    module = load_module()
    task = make_task(module)
    results = {
        "visual_a": {"status": "completed", "order": "visual_a", "evaluation": evaluation("B")},
        "mllm_a": {
            "status": "completed",
            "order": "mllm_a",
            "evaluation": evaluation("A", title_a=4, title_b=3),
        },
    }
    detail = module.combine_task(task, results)
    assert detail["stable"] is True
    assert detail["final_outcome"] == "mllm"
    assert detail["mllm_title_relevance"] == 4.0
    assert detail["visual_title_relevance"] == 3.0
    summary = module.summarize_details([detail])
    assert summary["mllm_win_rate"] == 1.0
    assert summary["mean_mllm_minus_visual_title_relevance"] == 1.0


def test_ab_swap_marks_disagreement_unstable():
    module = load_module()
    task = make_task(module)
    results = {
        "visual_a": {"status": "completed", "order": "visual_a", "evaluation": evaluation("A")},
        "mllm_a": {"status": "completed", "order": "mllm_a", "evaluation": evaluation("A")},
    }
    detail = module.combine_task(task, results)
    assert detail["stable"] is False
    assert detail["final_outcome"] == "unstable"
    summary = module.summarize_details([detail])
    assert summary["unstable_rate"] == 1.0
    assert summary["mllm_win_rate"] is None


def test_writes_visual_review_report_with_swapped_decisions(tmp_path):
    module = load_module()
    original_path = tmp_path / "original.webp"
    visual_path = tmp_path / "visual.png"
    mllm_path = tmp_path / "mllm.png"
    Image.new("RGB", (120, 80), (220, 220, 220)).save(original_path)
    Image.new("RGB", (80, 80), (40, 100, 160)).save(visual_path)
    Image.new("RGB", (80, 80), (160, 80, 40)).save(mllm_path)
    task = module.PairTask(
        task_id="image__ratio_1",
        image_id="image",
        source_index=0,
        title="A visible subject makes news",
        caption="The visible subject in context.",
        target_ratio=1.0,
        original_path=original_path,
        visual_path=visual_path,
        mllm_path=mllm_path,
    )
    results = {
        "visual_a": {"status": "completed", "order": "visual_a", "evaluation": evaluation("B")},
        "mllm_a": {
            "status": "completed",
            "order": "mllm_a",
            "evaluation": evaluation("A", title_a=4, title_b=3),
        },
    }
    detail = module.combine_task(task, results)
    summary = module.write_reports([detail], tmp_path, "fake-judge")

    report = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert summary["overall"]["mllm_win_rate"] == 1.0
    assert "A visible subject makes news" in report
    assert "The visible subject in context." in report
    assert "GAIC" in report and "LLM" in report
    assert "A = GAIC, B = LLM" in report
    assert "A = LLM, B = GAIC" in report
    assert (tmp_path / "assets" / "image" / "original.jpg").is_file()
    assert (tmp_path / "assets" / "image" / "ratio_1_gaic.jpg").is_file()
    assert (tmp_path / "assets" / "image" / "ratio_1_llm.jpg").is_file()
    assert (tmp_path / "details.csv").is_file()