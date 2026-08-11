import importlib.util
from pathlib import Path


def load_module():
    script = Path(__file__).parents[1] / "scripts" / "compare_policy_models.py"
    spec = importlib.util.spec_from_file_location("test_compare_policy_models", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def detail(task_id, label, status="completed"):
    return {
        "task_id": task_id,
        "image_id": task_id.split("__", 1)[0],
        "title": "Title",
        "target_ratio": 1.0,
        "generation_status": "valid",
        "had_invalid_output": False,
        "invalid_attempt_count": 0,
        "judge_status": status,
        "judge_label": label,
        "judge_reward": None if label is None else (5 - label) / 5,
        "candidate_path": "renders/candidate.jpg",
    }


def run(name, labels):
    details = {f"image-{index}__ratio_1": detail(f"image-{index}__ratio_1", label) for index, label in enumerate(labels)}
    return {
        "name": name,
        "path": Path(name),
        "config": {
            "data_sha256": "data",
            "policy_prompt_sha256": "policy",
            "vlm_prompt_sha256": "judge",
            "output_protocol_version": "crop-json-canonicalization-v1",
            "target_ratios": [1.0],
            "max_attempts": 10,
            "seed": 42,
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": 128,
            "max_images": None,
            "max_model_len": 2176,
            "image_max_pixels": 1048576,
            "image_min_pixels": 65536,
            "internvl_max_dynamic_patch": 4,
            "judge_config": {"deployment": "gpt-5.6-sol"},
        },
        "details": details,
    }


def test_paired_comparison_uses_lower_tier_as_win():
    module = load_module()
    reference = run("qwen", [1, 3, 2])
    candidate = run("molmo", [0, 3, 4])
    task_ids = module.validate_compatible_runs([reference, candidate])
    rows = module.build_paired_rows([reference, candidate], task_ids)
    summary = module.summarize_comparison([reference, candidate], task_ids, rows)

    assert [row["molmo_vs_qwen"] for row in rows] == ["win", "tie", "loss"]
    assert summary["overall"]["molmo_vs_qwen"] == {
        "win": 1,
        "tie": 1,
        "loss": 1,
        "unscored": 0,
        "win_rate": 1 / 3,
        "tie_rate": 1 / 3,
        "loss_rate": 1 / 3,
    }


def test_incompatible_prompt_hash_is_rejected():
    module = load_module()
    reference = run("qwen", [1])
    candidate = run("internvl", [1])
    candidate["config"]["policy_prompt_sha256"] = "different"

    try:
        module.validate_compatible_runs([reference, candidate])
    except ValueError as error:
        assert "policy_prompt_sha256" in str(error)
    else:
        raise AssertionError("incompatible runs must be rejected")