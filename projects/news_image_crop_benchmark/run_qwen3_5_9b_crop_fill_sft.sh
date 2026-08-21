#!/usr/bin/env bash
set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
MODEL_PATH=${MODEL_PATH:-/mnt/blob_output/HuggingFace/Models/Qwen/Qwen3.5-9B}
TRAIN_FILE=${TRAIN_FILE:-/mnt/blob_output/v-yukunban/crop-image-dataset/sft/cropped-v3-action-v4/train.parquet}
VAL_FILE=${VAL_FILE:-/mnt/blob_output/v-yukunban/crop-image-dataset/sft/cropped-v3-action-v4/validation.parquet}

NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-4}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-16}
MICRO_BATCH_SIZE_PER_GPU=${MICRO_BATCH_SIZE_PER_GPU:-1}
USE_DYNAMIC_BSZ=${USE_DYNAMIC_BSZ:-True}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_TOKEN_LEN_PER_GPU=${MAX_TOKEN_LEN_PER_GPU:-8192}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-5}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
LEARNING_RATE=${LEARNING_RATE:-5e-5}
WARMUP_RATIO=${WARMUP_RATIO:-0.10}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
ADAM_BETA1=${ADAM_BETA1:-0.9}
ADAM_BETA2=${ADAM_BETA2:-0.999}
LORA_RANK=${LORA_RANK:-32}
LORA_ALPHA=${LORA_ALPHA:-64}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
LORA_EXCLUDE_MODULES=${LORA_EXCLUDE_MODULES:-'.*visual.*'}
SAVE_FREQ=${SAVE_FREQ:-after_each_epoch}
TEST_FREQ=${TEST_FREQ:-after_each_epoch}
LOGGER=${LOGGER:-console}
PROJECT_NAME=${PROJECT_NAME:-news_image_crop_fill_sft}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen35_cropped_v3_action_v4}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/blob_output/v-yukunban/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
PRINT_CONFIG=${PRINT_CONFIG:-0}

for path in "${MODEL_PATH}" "${TRAIN_FILE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done
if [[ "${VAL_FILE}" != null && ! -e "${VAL_FILE}" ]]; then
    echo "Required path does not exist: ${VAL_FILE}" >&2
    exit 1
fi

ARGS=(
    data.train_files="${TRAIN_FILE}"
    data.val_files="${VAL_FILE}"
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.micro_batch_size_per_gpu=${MICRO_BATCH_SIZE_PER_GPU}
    data.max_length=${MAX_LENGTH}
    data.max_token_len_per_gpu=${MAX_TOKEN_LEN_PER_GPU}
    data.pad_mode=no_padding
    data.truncation=error
    data.use_dynamic_bsz=${USE_DYNAMIC_BSZ}
    data.balance_dp_token=False
    data.num_workers=4
    +data.image_key=images
    +data.image_patch_size=16
    data.enable_thinking_default=False
    +data.apply_chat_template_kwargs.enable_thinking=False
    model.path="${MODEL_PATH}"
    model.enable_gradient_checkpointing=True
    model.use_remove_padding=False
    +model.override_config.attn_implementation=sdpa
    model.lora_rank=${LORA_RANK}
    model.lora_alpha=${LORA_ALPHA}
    model.lora_dropout=${LORA_DROPOUT}
    "model.target_modules='${LORA_TARGET_MODULES}'"
    model.exclude_modules="${LORA_EXCLUDE_MODULES}"
    engine=fsdp
    engine.strategy=fsdp2
    engine.fsdp_size=${NDEVICES_PER_NODE}
    engine.use_torch_compile=False
    optim=fsdp
    optim.lr=${LEARNING_RATE}
    optim.lr_warmup_steps_ratio=${WARMUP_RATIO}
    optim.lr_scheduler_type=cosine
    optim.weight_decay=${WEIGHT_DECAY}
    optim.betas="[${ADAM_BETA1},${ADAM_BETA2}]"
    trainer.seed=42
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=1
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.logger="['${LOGGER}']"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.default_local_dir="${OUTPUT_DIR}"
    trainer.resume_mode=auto
    trainer.max_ckpt_to_keep=5
)
if (( LORA_RANK > 0 )); then
    ARGS+=(+checkpoint.save_lora_only=True)
fi

cd "${REPO_ROOT}"
if [[ "${PRINT_CONFIG}" = 1 ]]; then
    "${PYTHON_BIN}" -m verl.trainer.sft_trainer --cfg job "${ARGS[@]}" "$@"
else
    "${PYTHON_BIN}" -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NDEVICES_PER_NODE}" \
        -m verl.trainer.sft_trainer "${ARGS[@]}" "$@"
fi