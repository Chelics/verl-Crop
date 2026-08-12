#!/usr/bin/env bash
set -xeuo pipefail

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
PROJECT_ROOT="${REPO_ROOT}/projects/news_image_crop_benchmark"
export PYTHONPATH="${PROJECT_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
PYTHON_BIN=${PYTHON_BIN:-python3}

MODEL_PATH=${MODEL_PATH:-/mnt/blob_output/HuggingFace/Models/Qwen/Qwen3.5-9B}
TRAIN_FILE=${TRAIN_FILE:-/mnt/blob_output/v-yukunban/news_image_crop_content_split/news_image_crop_train.parquet}
TEST_FILE=${TEST_FILE:-/mnt/blob_output/v-yukunban/news_image_crop_content_split/news_image_crop_validation.parquet}
REWARD_FILE=${REWARD_FILE:-${PROJECT_ROOT}/rewards/crop_reward.py}
CLIP_MODEL_PATH=${CLIP_MODEL_PATH:-/mnt/blob_output/HuggingFace/Models/clip-vit-large-patch14}
VLM_PROMPT_PATH=${VLM_PROMPT_PATH:-${PROJECT_ROOT}/config/crop_vlm_prompt.txt}
ACTION_PROTOCOL=${ACTION_PROTOCOL:-legacy-crop-json}

NNODES=${NNODES:-1}
NDEVICES_PER_NODE=${NDEVICES_PER_NODE:-8}
FSDP_SIZE=${FSDP_SIZE:-${NDEVICES_PER_NODE}}
ROLLOUT_TP=${ROLLOUT_TP:-4}
ROLLOUT_N=${ROLLOUT_N:-8}
ROLLOUT_GPU_MEM_UTIL=${ROLLOUT_GPU_MEM_UTIL:-0.25}
ROLLOUT_MAX_NUM_SEQS=${ROLLOUT_MAX_NUM_SEQS:-16}
ROLLOUT_ENFORCE_EAGER=${ROLLOUT_ENFORCE_EAGER:-True}
ROLLOUT_TEMPERATURE=${ROLLOUT_TEMPERATURE:-0.7}
ROLLOUT_TOP_P=${ROLLOUT_TOP_P:-0.95}

TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-64}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-16}
MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-2048}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-128}
ROLLOUT_MAX_MODEL_LEN=${ROLLOUT_MAX_MODEL_LEN:-$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH))}
IMAGE_MAX_PIXELS=${IMAGE_MAX_PIXELS:-1048576}
IMAGE_MIN_PIXELS=${IMAGE_MIN_PIXELS:-65536}
ACTOR_LR=${ACTOR_LR:-5e-7}
KL_LOSS_COEF=${KL_LOSS_COEF:-0.02}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-sdpa}
LORA_RANK=${LORA_RANK:-0}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_TARGET_MODULES=${LORA_TARGET_MODULES:-all-linear}
LORA_EXCLUDE_MODULES=${LORA_EXCLUDE_MODULES:-'.*visual.*'}

REWARD_MODE=${REWARD_MODE:-proxy}
REWARD_NUM_WORKERS=${REWARD_NUM_WORKERS:-4}
TRANSFER_QUEUE_STORAGE_UNITS=${TRANSFER_QUEUE_STORAGE_UNITS:-1}
CLIP_DEVICE=${CLIP_DEVICE:-cpu}
TRAIN_MAX_SAMPLES=${TRAIN_MAX_SAMPLES:--1}
VAL_MAX_SAMPLES=${VAL_MAX_SAMPLES:--1}

TOTAL_EPOCHS=${TOTAL_EPOCHS:-1}
TOTAL_TRAINING_STEPS=${TOTAL_TRAINING_STEPS:-null}
SAVE_FREQ=${SAVE_FREQ:--1}
TEST_FREQ=${TEST_FREQ:-10}
LOGGER=${LOGGER:-console}
PROJECT_NAME=${PROJECT_NAME:-news_image_crop_grpo}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen3_5_9b_${REWARD_MODE}_$(date +%Y%m%d_%H%M)}
OUTPUT_DIR=${OUTPUT_DIR:-/mnt/blob_output/v-yukunban/checkpoints/${PROJECT_NAME}/${EXPERIMENT_NAME}}
DRY_RUN=${DRY_RUN:-0}

for path in "${MODEL_PATH}" "${TRAIN_FILE}" "${TEST_FILE}" "${REWARD_FILE}"; do
    if [[ ! -e "${path}" ]]; then
        echo "Required path does not exist: ${path}" >&2
        exit 1
    fi
done
if [[ "${REWARD_MODE}" = proxy && ! -d "${CLIP_MODEL_PATH}" ]]; then
    echo "CLIP_MODEL_PATH is required for proxy reward: ${CLIP_MODEL_PATH}" >&2
    exit 1
fi
if [[ "${REWARD_MODE}" = vlm && ! -f "${VLM_PROMPT_PATH}" ]]; then
    echo "VLM_PROMPT_PATH is required for VLM reward: ${VLM_PROMPT_PATH}" >&2
    exit 1
fi
if [[ "${REWARD_MODE}" != proxy && "${REWARD_MODE}" != smoke && "${REWARD_MODE}" != vlm ]]; then
    echo "REWARD_MODE must be proxy, smoke, or vlm" >&2
    exit 1
fi
if [[ "${ACTION_PROTOCOL}" != legacy-crop-json && "${ACTION_PROTOCOL}" != percent-json-v1 ]]; then
    echo "ACTION_PROTOCOL must be legacy-crop-json or percent-json-v1" >&2
    exit 1
fi

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="${TRAIN_FILE}"
    data.val_files="${TEST_FILE}"
    data.train_max_samples=${TRAIN_MAX_SAMPLES}
    data.val_max_samples=${VAL_MAX_SAMPLES}
    data.image_key=images
    data.image_patch_size=16
    +data.mm_processor_kwargs.size.longest_edge=${IMAGE_MAX_PIXELS}
    +data.mm_processor_kwargs.size.shortest_edge=${IMAGE_MIN_PIXELS}
    +data.apply_chat_template_kwargs.enable_thinking=False
    data.train_batch_size=${TRAIN_BATCH_SIZE}
    data.max_prompt_length=${MAX_PROMPT_LENGTH}
    data.max_response_length=${MAX_RESPONSE_LENGTH}
    data.filter_overlong_prompts=True
    data.filter_overlong_prompts_workers=8
    data.truncation=error
    data.shuffle=True
)

REWARD=(
    reward.num_workers=${REWARD_NUM_WORKERS}
    reward.custom_reward_function.path="${REWARD_FILE}"
    reward.custom_reward_function.name=compute_score
    +reward.custom_reward_function.reward_kwargs.reward_mode=${REWARD_MODE}
    +reward.custom_reward_function.reward_kwargs.clip_model_path="${CLIP_MODEL_PATH}"
    +reward.custom_reward_function.reward_kwargs.clip_device=${CLIP_DEVICE}
    +reward.custom_reward_function.reward_kwargs.vlm_prompt_path="${VLM_PROMPT_PATH}"
    +reward.custom_reward_function.reward_kwargs.action_protocol=${ACTION_PROTOCOL}
)

MODEL=(
    actor_rollout_ref.model.path="${MODEL_PATH}"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=${ATTN_IMPLEMENTATION}
)
if (( LORA_RANK > 0 )); then
    MODEL+=(
        actor_rollout_ref.model.lora_rank=${LORA_RANK}
        actor_rollout_ref.model.lora_alpha=${LORA_ALPHA}
        actor_rollout_ref.model.target_modules=${LORA_TARGET_MODULES}
        actor_rollout_ref.model.exclude_modules="${LORA_EXCLUDE_MODULES}"
    )
    actor_offload=False
    optimizer_offload=False
else
    actor_offload=True
    optimizer_offload=True
fi

ACTOR=(
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.optim.lr=${ACTOR_LR}
    actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE}
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_kl_loss=True
    actor_rollout_ref.actor.kl_loss_coef=${KL_LOSS_COEF}
    actor_rollout_ref.actor.kl_loss_type=low_var_kl
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${FSDP_SIZE}
    actor_rollout_ref.actor.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.actor.fsdp_config.entropy_checkpointing=True
    actor_rollout_ref.actor.entropy_from_logits_with_chunking=True
    actor_rollout_ref.actor.fsdp_config.offload_policy=True
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload}
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${optimizer_offload}
)

REF=(
    actor_rollout_ref.ref.strategy=fsdp2
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1
    actor_rollout_ref.ref.fsdp_config.reshard_after_forward=True
    actor_rollout_ref.ref.fsdp_config.param_offload=True
    actor_rollout_ref.ref.use_torch_compile=False
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP}
    actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEM_UTIL}
    actor_rollout_ref.rollout.max_model_len=${ROLLOUT_MAX_MODEL_LEN}
    actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS}
    actor_rollout_ref.rollout.n=${ROLLOUT_N}
    actor_rollout_ref.rollout.temperature=${ROLLOUT_TEMPERATURE}
    actor_rollout_ref.rollout.top_p=${ROLLOUT_TOP_P}
    actor_rollout_ref.rollout.enable_chunked_prefill=False
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.enforce_eager=${ROLLOUT_ENFORCE_EAGER}
    actor_rollout_ref.rollout.free_cache_engine=True
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
)

TRAINER=(
    transfer_queue.backend.SimpleStorage.num_data_storage_units=${TRANSFER_QUEUE_STORAGE_UNITS}
    trainer.use_v1=True
    trainer.balance_batch=False
    trainer.logger="${LOGGER}"
    trainer.project_name="${PROJECT_NAME}"
    trainer.experiment_name="${EXPERIMENT_NAME}"
    trainer.n_gpus_per_node=${NDEVICES_PER_NODE}
    trainer.nnodes=${NNODES}
    trainer.default_local_dir="${OUTPUT_DIR}"
    trainer.val_before_train=False
    trainer.save_freq=${SAVE_FREQ}
    trainer.test_freq=${TEST_FREQ}
    trainer.total_epochs=${TOTAL_EPOCHS}
    trainer.total_training_steps=${TOTAL_TRAINING_STEPS}
)

if [[ "${DRY_RUN}" = 1 ]]; then
    "${PYTHON_BIN}" "${REPO_ROOT}/scripts/print_cfg.py" --cfg job \
        "${DATA[@]}" "${REWARD[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${REF[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}" "$@"
else
    "${PYTHON_BIN}" -m verl.trainer.main_ppo \
        "${DATA[@]}" "${REWARD[@]}" "${MODEL[@]}" "${ACTOR[@]}" "${REF[@]}" "${ROLLOUT[@]}" "${TRAINER[@]}" "$@"
fi