import importlib.util
from pathlib import Path


def load_module():
    script = Path(__file__).parents[1] / "scripts" / "check_eval_gate.py"
    spec = importlib.util.spec_from_file_location("test_eval_gate", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def summary(**overrides):
    overall = {
        "tasks": 4,
        "generation_success_count": 4,
        "retry_exhausted_count": 0,
        "canonical_format_count": 4,
        "judge_completed_count": 4,
        "judge_failed_count": 0,
        "judge_parse_fallback_count": 0,
    }
    overall.update(overrides)
    return {"overall": overall}


def test_gate_accepts_complete_preflight():
    load_module().check_gate(summary(), expected_tasks=4)


def test_gate_rejects_technical_pass_with_no_valid_generations():
    module = load_module()
    try:
        module.check_gate(
            summary(generation_success_count=0, retry_exhausted_count=4, judge_completed_count=0),
            expected_tasks=4,
        )
    except RuntimeError as error:
        assert "generation_success_count=0" in str(error)
        assert "retry_exhausted_count=4" in str(error)
        assert "canonical_format_count=4" not in str(error)
        assert "judge_completed_count=0" in str(error)
    else:
        raise AssertionError("incomplete preflight must fail the gate")