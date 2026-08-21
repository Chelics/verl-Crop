from pathlib import Path


def test_lora_checkpoint_override_uses_hydra_append_syntax():
    launcher = Path(__file__).parents[1] / "run_qwen3_5_9b_crop_fill_sft.sh"
    text = launcher.read_text(encoding="utf-8")

    assert "ARGS+=(+checkpoint.save_lora_only=True)" in text
    assert "ARGS+=(checkpoint.save_lora_only=True)" not in text


def test_lora_dropout_is_configurable_end_to_end():
    project_root = Path(__file__).parents[3]
    launcher = (Path(__file__).parents[1] / "run_qwen3_5_9b_crop_fill_sft.sh").read_text(encoding="utf-8")
    model_config = (project_root / "verl/workers/config/model.py").read_text(encoding="utf-8")
    model_yaml = (project_root / "verl/trainer/config/model/hf_model.yaml").read_text(encoding="utf-8")
    fsdp_impl = (project_root / "verl/workers/engine/fsdp/transformer_impl.py").read_text(encoding="utf-8")

    assert "LORA_DROPOUT=${LORA_DROPOUT:-0.0}" in launcher
    assert "model.lora_dropout=${LORA_DROPOUT}" in launcher
    assert '"model.target_modules=\'${LORA_TARGET_MODULES}\'"' in launcher
    assert "lora_dropout: float = 0.0" in model_config
    assert "lora_dropout: 0.0" in model_yaml
    assert '"lora_dropout": self.model_config.lora_dropout' in fsdp_impl