# Cropped V3 Detail SFT

`convert_cropped_v3_to_detail_sft.py` converts the reviewed
`cropped_v3.parquet` annotations and original `image_once_train.parquet` images
into a messages-based multimodal SFT dataset compatible with both Swift and
verl.

## Model Input

Each row gives the model one original image and one user message containing:

- the news headline;
- the image caption;
- one target aspect ratio;
- the crop, fill, coordinate, color, and response contract.

The reference cropped image is not supplied to the model. The converter
materializes lossless original-image WebP files under the output directory and
stores one absolute path in `images`.

## Assistant Target

The assistant target is one compact JSON object with a stable field order:

```json
{"target_ratio":1.59,"is_cropped":true,"is_filled":true,"crop_box":[0.1,0.05,0.9,0.95],"fill_color":[12,34,56],"description":"..."}
```

The mapping from `cropped_v3.parquet` is:

| Detail SFT field | Cropped V3 source |
|---|---|
| `target_ratio` | `target_ratio` |
| `is_cropped` | `was_cropped` |
| `is_filled` | `was_padded` |
| `crop_box` | `bbox_pixels / [original_width, original_height]` when cropped, otherwise `null` |
| `fill_color` | `padding_color_rgb` when padded, otherwise `null` |
| `description` | reviewed `reason` |

`crop_box` is normalized in the original-image coordinate system. It is built
from `bbox_pixels`, not the older `bbox_normalized` column, because the pixel
box includes reviewed edge-artifact trim. A one-pixel containment tolerance is
allowed for integer rounding in one existing annotation. Crop-only boxes must
match the requested aspect ratio within 0.2%.

The full assistant JSON, including `description`, contributes to the normal SFT
token cross-entropy unless the trainer applies a custom field-level loss mask.

For an action-only first run, use `strip_detail_sft_descriptions.py`. It removes
`description` from the assistant JSON and updates the prompt to request exactly
five action fields. The original reason is retained as top-level
`reference_description` metadata, which the standard messages-based trainer
does not tokenize and therefore does not include in loss. The generated rows
continue to reference the same original-image assets.

## Action-Only Training

The action-only training files are:

```text
/mnt/blob_output/v-yukunban/crop-image-dataset/sft/cropped-v3-action-v4/train.parquet
/mnt/blob_output/v-yukunban/crop-image-dataset/sft/cropped-v3-action-v4/validation.parquet
```

The authoritative inference and training prompt is
`config/policy_prompts/v5_crop_fill_action.txt`. The prompt is already embedded
in every Parquet row; the trainer does not rewrite it. The processor preflight
reconstructs all prompts from the standalone template and requires exact
matches.

Action-v4 stores each image as a verl descriptor with `image`, `min_pixels`,
and `max_pixels` fields. The earlier action-v1/v2 Parquets used Swift's
`list<string>` image schema and must not be passed to verl's
`MultiTurnSFTDataset`. It also explicitly defines all four boolean action
combinations in the prompt so rare keep examples do not carry the policy alone.

`run_qwen3_5_9b_crop_fill_sft.sh` configures Qwen3.5-9B with LoRA rank 32,
alpha 64, learning rate `5e-5`, 10% warmup, global batch size 16, five epochs,
FSDP2 over four GPUs, thinking disabled, and visual modules excluded from LoRA.
With 1,716 training rows this is approximately 107 optimizer steps per epoch
and 535 steps total. Validation and LoRA-only checkpointing run after every
epoch.

`projects/news_image_crop_benchmark/amlt/amlt_cropped_v3_action_sft.yaml`
contains one `cropped-v3-action-sft` job. It runs full train and validation
processor checks before printing the resolved Hydra configuration and starting
training. The job is submitted explicitly; preparing or validating the config
does not launch it.

## Output Schema

```text
messages       list<struct<role: string, content: string>>
images         list<string>
image_id       string
source_index   int64
target_ratio   double
is_cropped     bool
is_filled      bool
```

Rows are split by `image_id`, so all four target-ratio examples for one image
remain together. The converter writes `train.parquet`, `validation.parquet`,
`split_manifest.jsonl`, `conversion_report.json`, and the referenced original
image assets.

## Conversion

```bash
PYTHONPATH=projects/news_image_crop_benchmark/src \
python projects/news_image_crop_benchmark/scripts/convert_cropped_v3_to_detail_sft.py \
  --annotations /mnt/blob_output/v-yukunban/crop-image-dataset/cropped_v3.parquet \
  --raw-train /mnt/blob_output/v-yukunban/crop-image-dataset/image_once_train.parquet \
  --output-dir /mnt/blob_output/v-yukunban/crop-image-dataset/sft/cropped-v3-detail-v1 \
  --seed 42 \
  --validation-fraction 0.1
```

When staging on a local machine before uploading, pass
`--serialized-asset-root` with the final cluster-visible asset path. The
converter writes files under the local output directory while storing the
provided POSIX paths in the Parquet `images` column.

Conversion fails on missing images, title/caption mismatches, duplicate
image-ratio keys, invalid conditional nulls, invalid RGB values, empty reasons,
bad crop geometry, or annotation/image dimension mismatches.
