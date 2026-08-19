from pathlib import Path


def test_lora_checkpoint_override_uses_hydra_append_syntax():
    launcher = Path(__file__).parents[1] / "run_qwen3_5_9b_crop_fill_sft.sh"
    text = launcher.read_text(encoding="utf-8")

    assert "ARGS+=(+checkpoint.save_lora_only=True)" in text
    assert "ARGS+=(checkpoint.save_lora_only=True)" not in text