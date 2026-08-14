# News Image Crop Benchmark

An isolated benchmark project for training Qwen3.5-9B with verl to crop a news image conditioned on its title and a requested aspect ratio.

## Objective

The policy receives an original image, a news title, and one target aspect ratio from `1.0`, `1.91`, `1.77`, or `1.59`. It emits a normalized crop action:

```text
<crop>{"cx": 500, "cy": 620, "area": 480}</crop>
```

The deterministic geometry layer turns this action into an in-bounds bbox with the exact requested aspect ratio. Only the title and original image are trusted training inputs. `CroppedImageUrl` and `Reason` are unpaired audit fields and are not used for training or reward.

## Verified Inputs

- Model: `/mnt/blob_output/HuggingFace/Models/Qwen/Qwen3.5-9B`
- Source data: `/mnt/blob_output/v-yukunban/imageCroppingDataset_cropped_binary.parquet`
- Source rows: 11,390
- Unique original image URLs: 5,672
- Model architecture: `Qwen3_5ForConditionalGeneration`
- Current node: one NVIDIA A100-SXM4-80GB

The source Parquet is not passed directly to verl. Preprocessing produces train/validation/test Parquets with verl's `prompt`, `images`, `data_source`, `reward_model`, and `extra_info` fields.

## Status

| Area | Status |
|---|---|
| Fixed-ratio crop geometry | Implemented and unit tested |
| Model output parser | Implemented and unit tested |
| Proxy reward aggregation | Implemented and unit tested |
| verl custom reward + real CLIP | Implemented and smoke tested |
| Source data audit | Implemented |
| Lightweight source index | Implemented |
| verl row schema and group split | Implemented and unit tested |
| Four-ratio sample expansion | Generated and fully validated |
| Initial visual proxy adapters | Implemented; product calibration pending |
| verl dataset conversion | Generated; checksum split fully validated |
| Qwen3.5-9B GRPO launcher | Implemented; Hydra config validated |
| Qwen rollout + actor update | Real rollout and one-step plumbing validated; formal multi-GPU run pending |
| Human golden evaluation set | Pending |

Current generated data:

- `/mnt/blob_output/v-yukunban/news_image_crop_content_split/news_image_crop_train.parquet`: 31,884 rows
- `/mnt/blob_output/v-yukunban/news_image_crop_content_split/news_image_crop_validation.parquet`: 3,040 rows
- `/mnt/blob_output/v-yukunban/news_image_crop_content_split/news_image_crop_test.parquet`: 4,156 rows
- `/mnt/blob_output/v-yukunban/news_image_crop_assets/`: 5,672 unique originals, about 1.70 GB

The 5,672 source URLs contain 2,646 exact image contents. All 39,080 sample IDs are unique, all four ratios contain 9,770 rows, and the image SHA-256 intersections between splits are empty. The previous URL-based Parquets remain in the parent directory only for audit comparison and must not be used for formal experiments.

## Layout

```text
config/benchmark.yaml        Versioned benchmark defaults
docs/environment.md          Local and cluster environment contract
docs/data_conversion.md      Staged raw-to-verl conversion design
docs/data_quality_audit.md   URL-alias leakage evidence and corrected split validation
docs/baseline_evaluation.md  Frozen Qwen baseline and pre-training gates
docs/experiment_plan.md      Baselines, stages, metrics, and gates
docs/reward_design.md        Reward definitions and anti-gaming rules
docs/training_integration.md Concrete reward, dry-run, and GRPO smoke commands
scripts/inspect_source.py    Read-only source Parquet audit
scripts/resplit_by_image_content.py  Rebuild splits from an existing asset manifest
src/news_crop_benchmark/     Reusable benchmark code
tests/                       Focused unit tests
```

## Development Check

From the verl repository root:

```bash
PYTHONPATH=projects/news_image_crop_benchmark/src \
  python3 -m unittest discover -s projects/news_image_crop_benchmark/tests -v
```

Inspect the source without reading the 3.1 GB image-bytes column:

```bash
python projects/news_image_crop_benchmark/scripts/inspect_source.py \
  --input /mnt/blob_output/v-yukunban/imageCroppingDataset_cropped_binary.parquet
```

Build a lightweight deterministic source index:

```bash
PYTHONPATH=projects/news_image_crop_benchmark/src \
  python projects/news_image_crop_benchmark/scripts/build_source_index.py \
  --input /mnt/blob_output/v-yukunban/imageCroppingDataset_cropped_binary.parquet \
  --output /mnt/blob_output/v-yukunban/news_image_crop_benchmark/index/source_index.parquet
```

Run a 100-row conversion smoke test into a temporary output directory:

```bash
tmpdir=$(mktemp -d)
PYTHONPATH=projects/news_image_crop_benchmark/src \
  python projects/news_image_crop_benchmark/scripts/convert_to_verl.py \
  --input /mnt/blob_output/v-yukunban/imageCroppingDataset_cropped_binary.parquet \
  --output-dir "$tmpdir" \
  --policy-prompt-path projects/news_image_crop_benchmark/config/policy_prompts/v0_original.txt \
  --limit 100
```

The full conversion writes `news_image_crop_train.parquet`, `news_image_crop_validation.parquet`, `news_image_crop_test.parquet`, and deduplicated original-image assets to the selected output directory.

## Versioned Policy Prompts

Policy prompts live under `config/policy_prompts/`. Each template must start with `<image>` on its
own line and contain both `{title}` and `{target_ratio}`. JSON braces in the output contract are
written normally; they do not need escaping. Copy `v0_original.txt` to a new version instead of
editing it in place.

To apply a new template to existing verl Parquets without mutating the originals:

```bash
PYTHONPATH=projects/news_image_crop_benchmark/src \
  python projects/news_image_crop_benchmark/scripts/rewrite_prompts.py \
  --data /mnt/blob_output/v-yukunban/news_image_crop_content_split/news_image_crop_train.parquet \
  --data /mnt/blob_output/v-yukunban/news_image_crop_content_split/news_image_crop_validation.parquet \
  --output-dir /mnt/blob_output/v-yukunban/news_image_crop_prompt_v1 \
  --policy-prompt-path projects/news_image_crop_benchmark/config/policy_prompts/v1.txt
```

The conversion and rewrite reports include the effective policy prompt SHA-256. Point
`TRAIN_FILE` and `TEST_FILE` at the versioned outputs when launching GRPO; changing a template file
does not alter prompts already stored in Parquet.

## Unified Layout Policy

The unified layout policy makes crop, crop-then-pad, and full-image padding a
single model decision. Its prompt is
`config/policy_prompts/v3_unified_layout.txt`, and the strict action schema is:

```json
{"operation":"crop_pad","x1_pct":3,"y1_pct":12,"x2_pct":97,"y2_pct":88}
```

`operation` is `crop`, `crop_pad`, or `pad`. For `crop`, the renderer places the
largest requested-aspect-ratio rectangle inside the selected source rectangle.
For `crop_pad`, it retains the selected rectangle exactly and then pads it to the
requested ratio. For `pad`, the action must use the full source rectangle. The
renderer never stretches source pixels and chooses padding color from the source
edge median.

The frozen Qwen3.5-9B base uses the existing resumable image-once evaluator with
`--action-protocol layout-json-v1` and the final-layout rubric in
`config/layout_vlm_prompt.txt`. AMLT entrypoints are:

```text
amlt_unified_layout_base_preflight.yaml  4 images / 16 tasks
amlt_unified_layout_base_full.yaml       120 images / 480 tasks
amlt_unified_layout_base_resume.yaml     Judge-only recovery from persisted progress
```

All later SFT and GRPO evaluations must keep this prompt, action protocol,
renderer, test manifest, sampling configuration, and Layout Judge fixed when
comparing against the base.

## Mode-Only Crop-or-Pad Diagnosis

`scripts/evaluate_image_once_mode.py` is a standalone zero-shot diagnostic that runs before any crop
execution. For each source image and target ratio, the model returns exactly `{"mode":"crop"}` or
`{"mode":"pad"}`. The script does not call the crop renderer, the GPT visual judge, or a background
color selector.

Run a small Qwen3.5-9B diagnostic with:

```bash
PYTHONPATH=projects/news_image_crop_benchmark/src \
  python projects/news_image_crop_benchmark/scripts/evaluate_image_once_mode.py \
  --model /mnt/blob_output/HuggingFace/Models/Qwen/Qwen3.5-9B \
  --model-family qwen35 \
  --model-name Qwen3.5-9B \
  --data /mnt/blob_output/v-yukunban/crop-image-dataset/image_once_test.parquet \
  --mode-prompt-path projects/news_image_crop_benchmark/config/mode_prompts/v1_mode_only.txt \
  --output-dir /mnt/blob_output/v-yukunban/news_image_crop_mode_only_qwen35_v1 \
  --run-id qwen35-mode-only-v1 \
  --max-images 200
```

Use `--resume` with the same arguments to continue an interrupted run. The output includes
`report.html` and `report.md` for qualitative inspection, `review_template.csv` with blank
`human_label` and `notes` columns, task-level JSONL/Parquet predictions, raw generation attempts,
and descriptive crop/pad counts by ratio. No classification-quality metric is reported until human
labels are added.

## Mode-First Crop-or-Pad Pipeline

`scripts/evaluate_image_once_layout.py` connects mode selection to final layout rendering. It first
selects `crop` or `pad` for every image-ratio task. Crop tasks are sent to the v1 percentage crop
prompt; pad tasks do not require a second model call. For padding, the renderer takes the
per-channel median of the outer 5% source-image border, expands the smallest enclosing canvas at the
target ratio, and centers the source image without resizing or stretching it.

Run the full pipeline from scratch by omitting `--mode-results-dir`. To reuse a completed mode-only
run, pass its result directory:

```bash
PYTHONPATH=projects/news_image_crop_benchmark/src \
  python projects/news_image_crop_benchmark/scripts/evaluate_image_once_layout.py \
  --model /mnt/blob_output/HuggingFace/Models/Qwen/Qwen3.5-9B \
  --model-family qwen35 \
  --model-name Qwen3.5-9B \
  --data /mnt/blob_output/v-yukunban/crop-image-dataset/image_once_test.parquet \
  --mode-prompt-path projects/news_image_crop_benchmark/config/mode_prompts/v1_mode_only.txt \
  --crop-prompt-path projects/news_image_crop_benchmark/config/policy_prompts/v1_strict_normalized.txt \
  --mode-results-dir /mnt/blob_output/v-yukunban/crop-image-dataset/results/qwen35-mode-only-v1-full-20260812 \
  --output-dir /mnt/blob_output/v-yukunban/crop-image-dataset/results/qwen35-layout-v1-edge-pad \
  --run-id qwen35-layout-v1-edge-pad \
  --resume
```

The unified report shows the original image and all four final layouts. Each crop records the v1
policy response and percentage action; each pad records the extracted RGB/hex background,
content box, and padding fraction. `review_template.csv` provides blank mode and layout-quality
fields for later human annotation. This stage reports descriptive pipeline statistics only and
does not call the GPT crop judge.

## Interactive Result Viewer

Install the isolated viewer dependency:

```powershell
uv venv --seed --python 3.12 .venv-viewer
.venv-viewer\Scripts\python.exe -m pip install -e "projects/news_image_crop_benchmark[viewer]"
```

On Windows, mount the Blob result root from a normal, non-administrator terminal so the viewer can
see the drive in the same user session:

```powershell
rclone mount `
  ":azureblob,account=csnewsandfeeds4150361735,use_az=true:unium/v-yukunban/crop-image-dataset/results" `
  S: --read-only --links --vfs-cache-mode full --dir-cache-time 5m
```

Keep the mount terminal running. Start the viewer in another terminal:

```powershell
.venv-viewer\Scripts\python.exe projects/news_image_crop_benchmark/scripts/serve_results.py `
  --result-dir S:\qwen35-image-once-four-ratios-vlm-crop-first-v2
```

`--result-dir` accepts either the mounted results root (for example, `S:\`) or one result directory.
When given one result directory, the viewer selects it initially and discovers other valid sibling
experiments automatically. Use the Experiment dropdown to switch runs without changing code or
restarting the server. Enter a 1-based sample number and press Enter or Jump to move directly within
the current filtered result set.

Pass `--share` to create a temporary Gradio link. For a shared link, set both
`GRADIO_AUTH_USERNAME` and `GRADIO_AUTH_PASSWORD` in the launch terminal to require basic login.
Add `--single-result` when sharing to expose only the selected result directory rather than its
sibling experiments. The viewer reads metadata once and only resolves images for the current sample.

To browse the rendered rows in `cropped.parquet`, mount the dataset directory and start the dedicated
dataset viewer. The source file has one large image row group, so the first launch streams it once and
creates compressed local previews; later launches reuse that cache.

```powershell
rclone mount `
  ":azureblob,account=csnewsandfeeds4150361735,use_az=true:unium/v-yukunban/crop-image-dataset" `
  T: --read-only --links --vfs-cache-mode full --dir-cache-time 5m

.venv-viewer\Scripts\python.exe projects/news_image_crop_benchmark/scripts/render_cropped_overrides.py `
  --train T:\image_once_train.parquet `
  --manifest projects/news_image_crop_benchmark/config/cropped_overrides/v1.jsonl `
  --output-dir "$env:LOCALAPPDATA\news-crop-benchmark\cropped-overrides\v1"

.venv-viewer\Scripts\python.exe projects/news_image_crop_benchmark/scripts/serve_cropped_dataset.py `
  --data T:\cropped.parquet `
  --override-dir "$env:LOCALAPPDATA\news-crop-benchmark\cropped-overrides\v1"
```

Merge reviewed rows from a `manual_crops.parquet` export into a new dataset without replacing the
baseline. The merger selects the newest ratio-valid save for each `(image_id, ratio)`, preserves the
baseline schema, verifies every untouched row, and writes a JSON audit report next to the output.

```powershell
.venv-viewer\Scripts\python.exe projects/news_image_crop_benchmark/scripts/merge_manual_crops.py `
  --base T:\cropped.parquet `
  --manual "$env:USERPROFILE\Downloads\manual_crops.parquet" `
  --output "$env:USERPROFILE\Downloads\cropped_v2.parquet" `
  --row-group-size 64

amlt storage upload -c amlt_image_once_qwen35_vlm_eval.yaml --storage-id blob_output `
  "$env:USERPROFILE\Downloads\cropped_v2.parquet" `
  v-yukunban/crop-image-dataset/cropped_v2.parquet

.venv-viewer\Scripts\python.exe projects/news_image_crop_benchmark/scripts/serve_cropped_dataset.py `
  --data "$env:USERPROFILE\Downloads\cropped_v2.parquet" `
  --reason-prefix "[manual replacement]" `
  --server-port 7867
```

Do not overwrite `cropped.parquet`; consumers should opt in to the versioned v2 path explicitly.

## Image-Once Zero-Shot Evaluation

`scripts/evaluate_image_once_vlm.py` evaluates a frozen vision-language policy model directly on the raw
`image_once_test.parquet` schema. It reads only `image_id`, `original_image`, `title`, and
`ImageCaption`; reference crops and source reasons are not used.

Each source image is expanded into the four target ratios `1.0`, `1.91`, `1.77`, and `1.59`.
The evaluator keeps one valid model candidate per image-ratio task. An invalid response is sampled
again with a different deterministic seed, up to ten total attempts. Every response and parse error
is persisted. Valid crops are rendered and scored by the GPT visual judge using the source image,
the Qwen crop, the caption, and the headline.

Select the Qwen policy prompt with `--policy-prompt-path`. The evaluator stores its path and
effective SHA-256 in `run_config.yaml`, so `--resume` rejects results produced with another
template. `--vlm-prompt-path` remains the independent GPT judge rubric and should stay fixed during
policy prompt comparisons.

The AMLT config is `amlt_image_once_qwen35_vlm_eval.yaml` at the repository root. It expects:

```text
/mnt/blob_output/v-yukunban/crop-image-dataset/image_once_test.parquet
```

and writes the complete run under:

```text
/mnt/blob_output/v-yukunban/crop-image-dataset/results/<run-id>/
```

Outputs include source and candidate renders, all generation attempts, raw judge responses,
JSONL/Parquet details, JSON/CSV summaries, and a Markdown report. Read the report directly without
downloading image assets:

```bash
amlt storage cat -c amlt_image_once_qwen35_vlm_eval.yaml --storage-id blob_output \
  <result-prefix>/report.md
```

Parse the AMLT config without submitting a job:

```bash
amlt run amlt_image_once_qwen35_vlm_eval.yaml --dump
```

Submit the evaluation with an explicit experiment name:

```bash
amlt run amlt_image_once_qwen35_vlm_eval.yaml qwen35-image-once-vlm-20260810 \
  --description "Frozen Qwen3.5-9B four-ratio crops scored by the GPT visual judge"
```

### Policy Model Families

The evaluator isolates model-specific vLLM input formatting behind `--model-family`:

| Family | Blob model path | Request format | Runtime |
|---|---|---|---|
| `qwen35` | `/mnt/blob_output/HuggingFace/Models/Qwen/Qwen3.5-9B` | Qwen processor chat template | vLLM 0.24 |
| `internvl2` | `/mnt/blob_output/HuggingFace/Models/OpenGVLab/InternVL2-8B` | InternLM2 tokenizer chat template with `<image>` | vLLM 0.24 |
| `molmo` | `/mnt/blob_output/HuggingFace/Models/allenai/Molmo-7B-D-0924` | vLLM Molmo `<|im_start|>...` template | vLLM 0.7.0 |

InternVL2 and Molmo use `trust_remote_code=True`. InternVL2 limits dynamic image tiling to four
patches by default and pins `sentencepiece==0.2.0` because the verified official tokenizer contains
a legacy NUL piece rejected by SentencePiece 0.2.1+. The trusted tokenizer artifact is identified by
SHA-256 `f868398fc4e05ee1e8aeba95ddf18ddcc45b8bce55d5093bead5bbf80429b48b`.
Molmo uses an isolated environment pinned to `torch==2.5.1`,
`transformers==4.48.1`, and `vllm==0.7.0`; this avoids the documented Molmo preprocessing and
quality regressions in newer dependency combinations.

Each model config has a one-image/four-ratio preflight and a separate full job. Always run the
preflight first:

```bash
amlt run amlt_image_once_qwen35_vlm_eval.yaml :qwen35-image-once-preflight \
  qwen35-crop-preflight-<run-id>

amlt run amlt_image_once_internvl2_vlm_eval.yaml :internvl2-8b-preflight \
  internvl2-crop-preflight-<run-id>

amlt run amlt_image_once_molmo_vlm_eval.yaml :molmo-7b-d-preflight \
  molmo-crop-preflight-<run-id>
```

After `_EVAL_COMPLETE.json` and four completed Judge results are present, submit the corresponding
full job explicitly:

```bash
amlt run amlt_image_once_internvl2_vlm_eval.yaml :internvl2-8b-full \
  internvl2-crop-full-<run-id>

amlt run amlt_image_once_molmo_vlm_eval.yaml :molmo-7b-d-full \
  molmo-crop-full-<run-id>
```

Every run records the model name/family, config and weight-index fingerprints, effective policy and
Judge prompt hashes, vLLM/Transformers versions, request format, and rendered policy prompts.

Compare completed full runs by matching the same `task_id`. The first run is the reference; lower
Judge tiers count as wins:

```bash
python projects/news_image_crop_benchmark/scripts/compare_policy_models.py \
  --run qwen=/mnt/blob_output/v-yukunban/crop-image-dataset/results/<qwen-run> \
  --run internvl2=/mnt/blob_output/v-yukunban/crop-image-dataset/results/<internvl-run> \
  --run molmo=/mnt/blob_output/v-yukunban/crop-image-dataset/results/<molmo-run> \
  --output-dir /mnt/blob_output/v-yukunban/crop-image-dataset/results/model-comparison
```

The comparison refuses runs with different dataset, policy prompt, Judge prompt, target ratios, or
sampling/retry settings, or task coverage. It writes paired JSONL/Parquet details plus overall and
per-ratio win/tie/loss reports.

The cross-model comparison prompt is `config/policy_prompts/v1_strict_normalized.txt`. All models
return the same exact JSON object with integer percentage fields `cx_pct`, `cy_pct`, and `area_pct`.
The evaluator validates that public protocol before applying the same deterministic conversion to
internal renderer units (`cx = cx_pct * 10`, `cy = cy_pct * 10`, `area = area_pct * 10`). The prompt
contains no fixed numeric action example. Its editorial guidance favors a moderately wider crop when
tight and wider crops communicate the subject equally well, while still allowing irrelevant background,
secondary subjects, and decorative margins to be removed. Retries use the same percentage-field
requirements for every model.

Every preflight invokes `scripts/check_eval_gate.py`. The AMLT job passes only when all 16 tasks
(four images by four ratios) produce valid actions, no task exhausts its generation retries, all GPT Judge requests
complete, and no Judge request falls back or fails. This prevents a technically successful process
with zero scoreable crops from being mistaken for a usable model integration.

See `docs/environment.md` before installing the training stack. Formal quality claims require the small human golden set described in `docs/experiment_plan.md`.
