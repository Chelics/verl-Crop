# 环境约定

## 训练软件栈

本 Benchmark 固定使用当前 verl 仓库提交及其安装方式，不复用无关的 `/opt/conda/envs/ptca` 环境。

建议在 verl 仓库根目录执行以下本地安装步骤：

```bash
uv venv --seed --python 3.12 .venv-qwen35
source .venv-qwen35/bin/activate

python -m pip install "vllm==0.24.0"
python -m pip install -e .
python -m pip install -e "projects/news_image_crop_benchmark[data,train,test]"
```

始终使用 `python -m pip`，不要直接使用裸 `pip`，并确认 `python` 和 `pip` 都解析到 `.venv-qwen35`。V1 trainer 还需要 `TransferQueue==0.1.8`；当前 CI 和 Docker 镜像会在 `setup.py` 之外单独安装该依赖。

Qwen3.5 rollout 不能使用 `ptca` 中旧的 vLLM 0.10.2，因为其模型注册表不包含 `Qwen3_5ForConditionalGeneration`。本地已验证的软件栈为 Python 3.12、PyTorch 2.11.0+cu130、Transformers 5.10.4、vLLM 0.24.0、Ray 2.56.1 和 TransferQueue 0.1.8。

启动脚本默认在 FSDP 训练中使用 PyTorch SDPA，因此正确性联调不要求本地编译 FlashAttention。vLLM 自带兼容的推理 kernel。

## 本地验证范围

当前单张 A100 80GB 节点用于：

- 源数据转换和 Reward 开发；
- Qwen3.5-9B processor/模型加载联调；
- 单图片 vLLM rollout 测试；
- 小规模 Reward 和策略训练集成测试。

完整的 9B GRPO 属于集群任务，不作为单机本地验收的必要条件。

## 集群约定

每个 Ray 节点必须具备：

- 相同的 verl commit 和 Python 环境；
- 相互兼容的 NVIDIA 驱动、CUDA、NCCL 和 GPU 类型；
- 以相同路径访问模型、处理后的数据和输出目录；
- 节点间开放 Ray 和 NCCL 所需端口；
- 足够的本地缓存和共享内存，用于暂存模型与数据集。

Ray 拓扑通过以下配置提供：

```text
trainer.n_gpus_per_node
trainer.nnodes
actor_rollout_ref.actor.fsdp_config.fsdp_size
actor_rollout_ref.rollout.tensor_model_parallel_size
```

先从单个 8 GPU 节点开始。只有在单节点通过正确性和 Reward 质量验证后，才将多节点扩展作为独立 Benchmark 执行。

## 可复现性记录

每次运行必须记录：

- verl git SHA 和工作区 dirty 状态；
- Python、PyTorch、Transformers、vLLM、CUDA 和驱动版本；
- GPU 型号/数量和 Ray 集群资源；
- 模型 checkpoint 路径和索引哈希；
- 处理后数据集 manifest 的哈希；
- Benchmark 配置和 Reward adapter checkpoint；
- 随机种子和完整 Hydra overrides。
