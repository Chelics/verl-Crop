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
| verl dataset conversion | Pending |
| Qwen3.5-9B GRPO launcher | Implemented; Hydra config validated |
| Qwen rollout + actor update | Real rollout and one-step plumbing validated; formal multi-GPU run pending |
| Human golden evaluation set | Pending |

Current generated data:

- `/mnt/blob_output/v-yukunban/news_image_crop_train.parquet`: 32,948 rows
- `/mnt/blob_output/v-yukunban/news_image_crop_validation.parquet`: 3,256 rows
- `/mnt/blob_output/v-yukunban/news_image_crop_test.parquet`: 3,536 rows
- `/mnt/blob_output/v-yukunban/news_image_crop_assets/`: 5,672 unique originals, about 1.70 GB

All 39,740 sample IDs are unique, all four ratios contain 9,935 rows, and no original image crosses splits.

## Layout

```text
config/benchmark.yaml        Versioned benchmark defaults
docs/environment.md          Local and cluster environment contract
docs/data_conversion.md      Staged raw-to-verl conversion design
docs/experiment_plan.md      Baselines, stages, metrics, and gates
docs/reward_design.md        Reward definitions and anti-gaming rules
docs/training_integration.md Concrete reward, dry-run, and GRPO smoke commands
scripts/inspect_source.py    Read-only source Parquet audit
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

See `docs/environment.md` before installing the training stack. Formal quality claims require the small human golden set described in `docs/experiment_plan.md`.
