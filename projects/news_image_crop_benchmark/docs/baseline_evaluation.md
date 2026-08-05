# 原始模型 Baseline 评测设计

## 目的

正式训练前必须先评测冻结的原始 Qwen3.5-9B。Baseline 需要回答：

1. 模型能否稳定输出合法裁剪动作；
2. 模型是否理解新闻标题与图片内容之间的关系；
3. 输出是否退化为全图、极小框或固定中心；
4. 四种目标宽高比下的表现是否一致；
5. 原始模型相对最大中心裁剪是否已经具有优势；
6. Proxy Reward 的排序方向是否与人工视觉判断基本一致。

如果这些问题没有结论，训练后即使 Reward 上升，也不能证明真实裁剪质量提高。

Baseline 只使用冻结的原始模型，不加载训练 adapter，不执行梯度更新，也不修改模型权重。

## 固定输入与 Prompt

Baseline、GRPO 训练和训练后评测必须使用同一个 test split、图片预处理参数和 Prompt：

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

动作含义：

- `cx`：归一化裁剪中心横坐标；
- `cy`：归一化裁剪中心纵坐标；
- `area`：裁剪面积占原图面积的千分比。

程序根据目标比例将 `(cx, cy, area)` 确定性转换为不越界 bbox，因此模型不直接预测四条边，也不负责自行满足宽高比。

固定推理配置：

```text
enable_thinking=False
temperature=0.7
top_p=0.95
max_tokens=128
image_max_pixels=1048576
image_min_pixels=65536
```

模型、Prompt、采样参数、随机种子或图片预算发生变化时，应建立新的 Baseline 版本，不能覆盖旧结果。

## 两阶段评测规模

### 阶段 A：快速诊断

- 从 test split 固定抽取 20 个不同的“原图+标题”组合；
- 每个组合包含 `1.0`、`1.91`、`1.77` 和 `1.59` 四种比例；
- 共 80 个 prompt；
- 每个 prompt 采样 8 次；
- 共 640 个候选结果。

该阶段用于发现 Prompt、格式、动作分布和 Reward 方向上的明显问题，不用于形成最终质量结论。

建议人工检查其中 50～100 个“原图+标题+比例”组，优先覆盖不同图片尺寸、人物图、文字/logo 图、信息图和普通新闻照片。

### 阶段 B：固定 Baseline

阶段 A 通过后：

- 从 test split 固定抽取 100～300 个不同的“原图+标题”组合；
- 每个组合覆盖四种比例；
- 每个 prompt 固定采样 8 次；
- 固定随机种子和样本 ID 清单；
- 保存所有原始响应、解析后的动作、Reward 分项和候选裁剪图。

该样本清单作为训练前后统一评测集。训练后不得重新抽样以获得更有利的结果。

## 对照组

至少包含以下三组：

1. **最大中心裁剪**：在目标比例下，以图片中心为中心，取能够放入原图的最大 bbox；
2. **Qwen zero-shot pass@1**：原始 Qwen3.5-9B 的第一个输出，不使用 Reward 筛选；
3. **Qwen zero-shot best@8**：原始模型生成 8 个候选，由 Proxy Reward 选择最高分候选。

后续可以增加基于显著性、主体 grounding 或人脸检测的启发式裁剪，但新增基线不能替换上述固定基线。

## 报告方式

每个模型或策略同时报告：

- `pass@1`：第一个候选结果；
- `mean@8`：8 个候选的平均表现；
- `best@8`：由 Proxy Reward 选择的最高分候选。

不能只报告 `best@8`。从更多候选中取最大值会天然提高同一个 Reward 的得分，即使真实质量没有提高。

训练前后必须保持候选数一致。不能训练前使用 `pass@1`、训练后使用 `best@8` 进行不对等比较。

## 指标

### 工程指标

- `<crop>` 格式合法率；
- 数值和坐标范围合法率；
- 推理耗时、输入/输出吞吐量；
- GPU 峰值显存和 CPU 峰值内存。

### 动作分布

- `cx`、`cy` 的均值、方差和直方图；
- `area` 的均值、方差和直方图；
- `area >= 950` 的近全图率；
- `area <= 50` 的极小裁剪率；
- 相同原图在四种比例下的动作变化；
- 是否对不同图片持续输出相近中心或面积。

### Proxy 质量指标

- 标题相关性；
- 显著区域保留；
- 构图得分；
- 裁剪边界完整性；
- 面积先验得分；
- 总 Proxy Reward；
- 相对最大中心裁剪的胜率；
- 按 `1.0`、`1.91`、`1.77` 和 `1.59` 分比例统计。

## 人工抽查

Proxy Reward 不能同时作为唯一候选选择器和最终裁判。人工评测至少采用以下选项：

```text
Qwen 更好 / 中心裁剪更好 / 两者都可以 / 两者都不好
```

重点检查：

- 新闻主体是否完整保留；
- 人脸、文字和 logo 是否被切断；
- 与标题相关的内容是否位于裁剪范围内；
- 是否存在 Proxy Reward 高但肉眼明显较差的候选；
- `best@8` 是否只是利用 Reward 偏差，而非产生更好的裁剪。

阶段 A 的人工抽查用于校准方向。正式质量结论需要独立、固定的人工 golden set，并报告人工胜率和置信区间。

## Proxy Reward 的选择偏差

如果同一个 Proxy Reward 同时用于：

1. 从 8 个候选中选择最高分结果；
2. 评价该结果是否优于基线；

则会形成选择偏差。即使候选质量没有提高，候选数增加也通常会使最高 Reward 上升。

因此：

- `R_train` 可以用于 GRPO 和 best-of-N 候选选择；
- 最终评价必须同时包含 `pass@1`、独立自动指标和人工判断；
- 训练用 Reward 的 checkpoint、权重和实现应在训练前冻结；
- test split 不用于反复调 Reward；
- 如果引入独立 `R_eval`，它不得将分数反馈给 GRPO。

## 决策门槛

| Baseline 结果 | 后续动作 |
|---|---|
| 格式合法率低于 95% | 先修改 Prompt、使用约束解码或执行少量格式 SFT |
| 输出大量接近全图或极小框 | 先调整面积先验和 Prompt，不启动长训练 |
| 候选缺少多样性 | 调整采样温度、top-p 或 Prompt |
| Proxy Reward 排序与人工判断明显不一致 | 先修复 Reward，禁止启动正式 GRPO |
| 原始模型没有优于中心裁剪，但候选有多样性且 Reward 排序合理 | 适合进入 GRPO |
| 原始模型已经优于中心裁剪 | 仍可训练，但目标应是相对 zero-shot 的增量提升 |
| 原始模型已经很好且 GRPO 增益空间很小 | 考虑只使用 best-of-N，而不是训练 |

进入正式 GRPO 前，至少应满足：

- 格式合法率不低于 95%；
- 动作分布不存在明显全图、极小框或固定中心退化；
- Proxy Reward 对人工偏好的排序方向基本一致；
- Baseline 样本清单、配置和结果文件已经冻结并归档。

## 产物

Baseline 目录至少保存：

```text
baseline_config.yaml
sample_manifest.jsonl
details.jsonl
summary.json
renders/
human_review.csv
```

其中 `details.jsonl` 保存每个原始候选；`summary.json` 保存整体和分比例统计；`renders/` 保存人工可视化图片；`human_review.csv` 保存人工比较结果。

当前实现入口为 `scripts/evaluate_vllm_baseline.py`。在完成上述设计评审前，不要求立即运行完整 Baseline。

预诊断完成后，优先打开输出目录中的 `report.html`。页面按标题和目标比例展示原图、最大中心裁剪、全部 Qwen 候选、best 标记、动作参数和 Reward 分项。机器汇总见 `summary.json`，逐候选原始数据见 `details.jsonl`，人工记录可填写 `human_review.csv`。
