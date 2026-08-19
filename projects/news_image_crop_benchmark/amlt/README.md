# AMLT Configurations

Run these configs from the repository root. Each config uploads the repository root through `code.local_dir`, so commands can continue to reference `projects/news_image_crop_benchmark/...`.

```powershell
amlt run projects/news_image_crop_benchmark/amlt/<config>.yaml <experiment-name>
```

Use `--dump` for a local config-resolution check. This does not submit a job:

```powershell
amlt run projects/news_image_crop_benchmark/amlt/<config>.yaml --dump
```

## Config Groups

| Workflow | Configs |
|---|---|
| Baseline | `amlt_baseline.yaml` |
| Cropped-v3 action SFT and evaluation | `amlt_cropped_v3_action_*.yaml` |
| Image-once VLM evaluation | `amlt_image_once_*_vlm_eval.yaml` |
| Layout mode, pipeline, and judging | `amlt_image_once_qwen35_mode_eval.yaml`, `amlt_image_once_qwen35_layout_pipeline.yaml`, `amlt_image_once_layout_*.yaml` |
| Unified layout baseline | `amlt_unified_layout_base_*.yaml` |
| GRPO smoke tests | `amlt_vlm_grpo_smoke.yaml`, `amlt_image_once_v1_grpo_smoke.yaml` |
| Diagnostics and recovery | `amlt_qwen35_cuda13_probe.yaml`, `amlt_gpt_bbox_recovery.yaml` |

Before a full run, use the smallest available preflight job and follow the repository's AMLT submission discipline.
