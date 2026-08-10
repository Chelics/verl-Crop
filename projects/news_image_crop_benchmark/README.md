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
  --limit 100
```

The full conversion writes `news_image_crop_train.parquet`, `news_image_crop_validation.parquet`, `news_image_crop_test.parquet`, and deduplicated original-image assets to the selected output directory.

## Image-Once Zero-Shot Evaluation

`scripts/evaluate_image_once_vlm.py` evaluates the frozen Qwen3.5-9B model directly on the raw
`image_once_test.parquet` schema. It reads only `image_id`, `original_image`, `title`, and
`ImageCaption`; reference crops and source reasons are not used.

Each source image is expanded into the four target ratios `1.0`, `1.91`, `1.77`, and `1.59`.
The evaluator keeps one valid Qwen candidate per image-ratio task. An invalid response is sampled
again with a different deterministic seed, up to ten total attempts. Every response and parse error
is persisted. Valid crops are rendered and scored by the GPT visual judge using the source image,
the Qwen crop, the caption, and the headline.

The AMLT config is `amlt_image_once_qwen35_vlm_eval.yaml` at the repository root. It expects:

```text
/mnt/blob_output/v-yukunban/crop-image-dataset/image_once_test.parquet
```

and writes the complete run under:

```text
/mnt/blob_output/v-yukunban/crop-image-dataset/results/<run-id>/
```

Outputs include source and candidate renders, all generation attempts, raw judge responses,
JSONL/Parquet details, JSON/CSV summaries, and an HTML report. Parse the AMLT config without
submitting a job:

```bash
amlt run amlt_image_once_qwen35_vlm_eval.yaml --dump
```

Submit the evaluation with an explicit experiment name:

```bash
amlt run amlt_image_once_qwen35_vlm_eval.yaml qwen35-image-once-vlm-20260810 \
  --description "Frozen Qwen3.5-9B four-ratio crops scored by the GPT visual judge"
```

See `docs/environment.md` before installing the training stack. Formal quality claims require the small human golden set described in `docs/experiment_plan.md`.
