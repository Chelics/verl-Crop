# Reward 与训练集成

## 已实现内容

- verl 自定义 Reward 入口：`rewards/crop_reward.py::compute_score`
- 确定性动作解析和严格比例裁剪渲染
- 基于缓存 CLIP-L/14、以原图为基线的标题相关性
- 梯度显著性、显著性中心构图、裁剪边界完整性和最小面积得分
- Azure OpenAI 多模态裁剪质量 Reward，可选 API key 或 managed identity 认证
- Qwen3.5-9B FSDP2/vLLM 启动脚本：`run_qwen3_5_9b_grpo.sh`
- 持久化联调数据：`/mnt/blob_output/v-yukunban/news_image_crop_smoke_*.parquet`

当前 proxy reward 可用于工程联调和早期 GRPO 实验，但尚未成为经过验证的产品质量指标。主体 grounding、人脸/OCR 完整性和人工评测仍是后续验收项。

## 当前 Prompt

训练数据和原始模型 baseline 使用同一个 Prompt：

```text
<image>
News title: {title}
Target aspect ratio (width / height): {target_ratio}
Select the crop that best illustrates the news title.
Return exactly one line: <crop>{"cx": CX, "cy": CY, "area": AREA}</crop>
CX is the horizontal crop-center coordinate: 0 is the left edge and 1000 is the right edge.
CY is the vertical crop-center coordinate: 0 is the top edge and 1000 is the bottom edge.
AREA is the crop area as thousandths of the original image area: 1 is 0.1% and 1000 is the full image.
Use integers only. Do not include explanations or any other text.
```

模型不直接预测四个 bbox 边界，而是预测归一化中心 `(cx, cy)` 和面积 `area`。程序根据目标比例确定性地计算 bbox，因此生成结果天然满足比例约束。Prompt 使用英文是因为模型和新闻标题以英文为主，同时保持训练与线上推理模板一致。

## 训练前顺序

不要直接开始 GRPO。推荐顺序如下：

1. 固定 test split 和当前 Prompt。
2. 运行未训练 Qwen3.5-9B 的 vLLM zero-shot baseline。
3. 保存每个候选动作、格式合法率、面积/中心分布和 proxy reward。
4. 与中心裁剪和启发式裁剪比较。
5. 人工检查一小批可视化结果，确认 proxy reward 排序方向基本合理。
6. 固定 baseline 和 Reward 版本后再启动 GRPO。

训练后必须用相同 test 样本、Prompt、采样参数和 Reward 重新评测，报告相对 zero-shot baseline 的增益。

Baseline 的详细设计和验收门槛见 [baseline_evaluation.md](baseline_evaluation.md)。正式训练不得绕过该门禁。

## Reward 模式

`smoke`:

- 合法裁剪协议得 `1`，非法输出得 `-1`；
- 不读取图片，也不加载 CLIP；
- 仅用于一步链路联调，因为它只能训练格式，不能训练裁剪质量。

`proxy`:

- 渲染模型提出的候选裁剪；
- 将候选图/标题 CLIP 相似度与原图/标题相似度进行比较；
- 计算基于图片的 proxy 指标；
- CLIP 无法加载或推理时直接让任务失败，而不是静默返回错误 Reward。

`vlm`:

- 保留与其他模式相同的输出协议、图片边界和目标宽高比硬校验；
- 将原图和实际渲染的候选裁剪图发送给 Azure OpenAI Responses API；
- 使用 `config/crop_vlm_prompt.txt` 的固定 rubric，将 Tier 0–5 映射为 `1.0–0.0` Reward；
- 缺少 caption 时显式传入 `[not provided]`，不会使用 `CroppedImageUrl` 或 `Reason`；
- 使用 Responses API 流式读取完整评价，避免双图长 rubric 在等待首个同步响应时超时；
- 请求默认重试两次，全部失败后返回可配置的 Tier 5（Reward `0`）；
- verl 的 `compute_score` 是逐样本同步回调，因此请求并发由 `REWARD_NUM_WORKERS` 控制，每个 worker 缓存一个客户端。

返回字典中的每个字段都会由 verl 作为 Reward 指标继续传递。

## VLM Reward 配置

先在训练环境安装可选依赖：

```bash
uv pip install --python .venv-qwen35/bin/python \
  -e 'projects/news_image_crop_benchmark[vlm]'
```

认证优先读取 API key；未设置 API key 时使用 managed identity：

```bash
export GPT5_AZURE_OPENAI_API_KEY=...  # 使用 managed identity 时不要设置
export GPT5_AZURE_OPENAI_ENDPOINT=https://csnf-singularity-aoai-eastus2.openai.azure.com/
export GPT5_AZURE_OPENAI_DEPLOYMENT=gpt-5.6-sol
export GPT5_AZURE_OPENAI_API_VERSION=2025-04-01-preview
export GPT5_AZURE_MANAGED_IDENTITY_CLIENT_ID=e6162a0d-e540-4454-995f-30bcb97f35b4
```

也可以把这些变量放在仓库根目录或项目目录的 `.env` 中；不要提交包含密钥的 `.env`。常用调优变量包括：

- `CROP_VLM_TIMEOUT=45`
- `CROP_VLM_MAX_RETRIES=2`
- `CROP_VLM_FALLBACK_LABEL=5.0`
- `CROP_VLM_IMAGE_SIZE=384`
- `CROP_VLM_IMAGE_FORMAT=JPEG`
- `CROP_VLM_JPEG_QUALITY=70`
- `CROP_VLM_MAX_OUTPUT_TOKENS=1024`
- `CROP_VLM_REASONING_EFFORT=low`
- `CROP_VLM_OUTPUT_VERBOSITY=low`
- `CROP_VLM_LOG_PATH=/shared/output/vlm_responses.jsonl`

`CROP_VLM_LOG_PATH` 未设置时不保存原始响应。设置后，每个请求会追加一条 JSONL，包含 deployment、response ID、标题、样本/裁切动作上下文、解析后的 Tier/Reward 和完整模型输出。日志不包含 API key、managed identity token 或图片 base64；但标题和模型分析仍可能是敏感数据，应写入受控目录并按实验产物管理。

启动一步 VLM Reward 联调：

```bash
PYTHON_BIN=$PWD/.venv-qwen35/bin/python \
REWARD_MODE=vlm \
REWARD_NUM_WORKERS=1 \
TRAIN_FILE=/mnt/blob_output/v-yukunban/news_image_crop_smoke_train.parquet \
TEST_FILE=/mnt/blob_output/v-yukunban/news_image_crop_smoke_validation.parquet \
TRAIN_MAX_SAMPLES=2 \
VAL_MAX_SAMPLES=2 \
TRAIN_BATCH_SIZE=2 \
PPO_MINI_BATCH_SIZE=2 \
ROLLOUT_N=2 \
TOTAL_TRAINING_STEPS=1 \
TEST_FREQ=-1 \
projects/news_image_crop_benchmark/run_qwen3_5_9b_grpo.sh
```

先保持 `REWARD_NUM_WORKERS=1` 验证 Azure 限流和端到端延迟，再逐步增加 worker。VLM 模式会返回 `vlm_label` 和 `vlm_enabled`；proxy 分项保持为 `0`，避免将两套 Reward 混为一谈。

## 本地 Reward 联调

该步骤不需要 verl 训练环境：

```bash
export PYTHONPATH=$PWD/projects/news_image_crop_benchmark/src:$PWD

/opt/conda/envs/ptca/bin/python \
  projects/news_image_crop_benchmark/scripts/smoke_reward.py \
  --data /mnt/blob_output/v-yukunban/news_image_crop_smoke_train.parquet \
  --reward-file $PWD/projects/news_image_crop_benchmark/rewards/crop_reward.py \
  --mode proxy \
  --clip-model-path /mnt/blob_output/HuggingFace/Models/clip-vit-large-patch14 \
  --count 2
```

## Hydra Dry Run

启动脚本支持 `DRY_RUN=1`，但仍需要安装 Hydra 和 verl 配置依赖的 Python 环境：

```bash
PYTHON_BIN=$PWD/.venv-qwen35/bin/python \
DRY_RUN=1 \
REWARD_MODE=smoke \
TRAIN_FILE=/mnt/blob_output/v-yukunban/news_image_crop_smoke_train.parquet \
TEST_FILE=/mnt/blob_output/v-yukunban/news_image_crop_smoke_validation.parquet \
NDEVICES_PER_NODE=8 \
projects/news_image_crop_benchmark/run_qwen3_5_9b_grpo.sh
```

该步骤已在 `.venv-qwen35` Python 3.12 环境中验证。解析后的 YAML 已通过断言，确认包含自定义 Reward 路径/kwargs、FSDP2 actor/reference 策略、vLLM rollout、V1 trainer 和联调数据路径。

当前 `ptca` 环境本身无法执行该步骤，因为它缺少 Hydra 和 tensordict，并且使用 vLLM 0.10.2。模型执行应使用已经按照 `docs/environment.md` 配置完成的 `.venv-qwen35` 环境。

## 一步集群联调

在 Ray 训练前，先验证一次真实多模态 vLLM rollout：

```bash
source .venv-qwen35/bin/activate
python projects/news_image_crop_benchmark/scripts/smoke_vllm.py \
  --model /mnt/blob_output/HuggingFace/Models/Qwen/Qwen3.5-9B \
  --data /mnt/blob_output/v-yukunban/news_image_crop_smoke_train.parquet
```

在单个 8 GPU 节点上准备好文档规定的 verl 环境后执行：

```bash
PYTHON_BIN=$PWD/.venv-qwen35/bin/python \
REWARD_MODE=smoke \
TRAIN_FILE=/mnt/blob_output/v-yukunban/news_image_crop_smoke_train.parquet \
TEST_FILE=/mnt/blob_output/v-yukunban/news_image_crop_smoke_validation.parquet \
TRAIN_MAX_SAMPLES=32 \
VAL_MAX_SAMPLES=16 \
TRAIN_BATCH_SIZE=8 \
PPO_MINI_BATCH_SIZE=8 \
ROLLOUT_N=2 \
TOTAL_TRAINING_STEPS=1 \
TEST_FREQ=-1 \
projects/news_image_crop_benchmark/run_qwen3_5_9b_grpo.sh
```

验收条件：模型/processor 成功加载、完成多模态 rollout、调用 Reward、完成一次 actor update，并正常退出。

在当前单 GPU 节点上，全参数更新已到达 `update_actor`，但单个 worker 使用约 173.75GB 主机内存，超过 Ray 的 95% 阈值。本地更新联调可使用 `LORA_RANK=8 LORA_ALPHA=16`。启动脚本会冻结 `.*visual.*`，只适配 LLM linear 层，与仓库中的多模态 LoRA 模式一致。在资源充足的多 GPU 集群上，默认仍使用全参数训练（`LORA_RANK=0`）。

启动脚本默认使用 `ATTN_IMPLEMENTATION=sdpa`，因此正确性联调不要求本地编译 FlashAttention。只有在每个集群节点都安装了与 PyTorch CUDA runtime 兼容的 wheel/build 后，才设置为 `flash_attention_2`。

启动脚本还将 Qwen 图片 processor 限制为最多 1,048,576 像素。未经限制时，一个真实的 2900x2900 样本会产生 8,370 个 prompt token；配置该预算后，在不修改已存储原图的情况下减少到 1,113 个 token。

该任务关闭 Qwen thinking。第一次真实 zero-shot rollout 把有限的响应长度用于分析文字；`enable_thinking=False` 会为模型提供严格输出形式的 assistant 前缀。

本 Benchmark 中 TransferQueue 默认只使用一个 storage unit。上游默认值 8 会占用本地 10 个 CPU 中的 8 个，导致 actor 的 strict placement group 无法调度。只有在节点具有足够空闲 CPU 时，才提高 `TRANSFER_QUEUE_STORAGE_UNITS`。

rollout server 默认将上下文限制为 prompt 长度加 response 长度、最多 16 个 sequence，并使用 eager 模式。如果保留 Qwen3.5 原生 262k context 和完整 cudagraph 初始化，一步联调会花费数分钟配置根本不会用到的容量。只有在正确性运行通过后才调整这些参数。

## 一步 Proxy Reward 联调

使用 `REWARD_MODE=proxy`、`REWARD_NUM_WORKERS=1` 和 `TRAIN_BATCH_SIZE=2` 重复上述测试。CLIP 默认在 CPU 上运行，此处有意保持小规模。在增加 worker 数量前，先测量 Reward 延迟和主机内存占用。

## 正式 GRPO

完成全量数据转换和一步验收后，先运行原始模型 baseline：

```bash
source .venv-qwen35/bin/activate
python projects/news_image_crop_benchmark/scripts/evaluate_vllm_baseline.py \
  --model /mnt/blob_output/HuggingFace/Models/Qwen/Qwen3.5-9B \
  --data /mnt/blob_output/v-yukunban/news_image_crop_content_split/news_image_crop_test.parquet \
  --reward-file $PWD/projects/news_image_crop_benchmark/rewards/crop_reward.py \
  --clip-model-path /mnt/blob_output/HuggingFace/Models/clip-vit-large-patch14 \
  --clip-device cuda \
  --output-dir /mnt/blob_output/v-yukunban/news_image_crop_baselines/qwen3_5_9b_zero_shot \
  --groups 4 \
  --n 4
```

该命令会输出 `details.jsonl` 和 `summary.json`，包含格式合法率、分比例得分、best-of-N 得分以及相对中心裁剪的胜率。保存该结果后再执行正式训练：

```bash
PYTHON_BIN=$PWD/.venv-qwen35/bin/python \
REWARD_MODE=proxy \
NDEVICES_PER_NODE=8 \
NNODES=1 \
REWARD_NUM_WORKERS=4 \
projects/news_image_crop_benchmark/run_qwen3_5_9b_grpo.sh
```

长时间训练前，如果 Reward 延迟成为 rollout 的主要瓶颈，应将 CLIP 评分迁移到批处理服务。扩展到多节点时，只需要调整 Ray 集群启动方式以及 `NNODES`/资源配置。

## 已验证的本地结果

- Qwen3.5-9B 已通过 vLLM 0.24 在单张 A100 上加载，并根据真实转换图片生成合法裁剪动作。
- 关闭 thinking，并设置 `temperature=0.7, top_p=0.95` 后，8/8 个采样动作都成功通过解析。
- 真实生成动作已完成裁剪渲染和 CLIP proxy 评分。
- 全参数 V1 训练已完成 rollout、Reward、old/ref log-prob 并到达 `update_actor`；随后单 worker 超过节点 205GB 的 Ray 内存限制。这是本地资源边界，不是数据或 Reward 接线错误。
- Rank-8 LoRA 已完成一个完整 trainer step，包括 rollout、Reward、actor update 调用和 adapter 权重同步。但 Qwen3.5 LoRA 存在严重的 rollout/training 概率不一致，并在 trainer 路径产生无效短轨迹。在验证 vLLM adapter 映射前，LoRA 只能用于链路联调。

因此正式 Benchmark 保持 `LORA_RANK=0`，目标环境至少为一个 8 GPU 节点。不得将本地 LoRA 运行结果作为质量实验结论。