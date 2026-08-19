# Qwen3.5-9B 四比例新闻图片裁剪零样本评测报告

评测日期：2026-08-10  
实验：`qwen35-image-once-vlm-feeds-20260810-v2`  
Run ID：`qwen35-image-once-four-ratios-vlm-20260810-v2`  
Job ID：`sleepy_tail_jf2ktqp1lt`

## 1. 结论摘要

本次实验链路完整通过，但原生 Qwen3.5-9B 的零样本裁剪质量不满足直接上线要求。

- 120 张原图按 `1.0`、`1.91`、`1.77`、`1.59` 四种比例展开为 480 个任务。
- 480 个任务全部生成有效裁剪，最终生成成功率为 100%。
- GPT 视觉 Judge 完成 480/480 次评测，无 API 失败、无解析 fallback。
- Tier 0-1 可接受裁剪为 126/480（26.25%）。
- Tier 3-5 严重问题为 339/480（70.63%）。
- 平均 Judge label 为 2.80，平均 reward 为 0.44。
- 最主要失败模式是信息图内容丢失：`T4.5` 出现 224 次，涉及 63/120 张原图。
- 四种比例均表现较差；方形 `1.0` 相对最好，`1.91` 最差，但差异不足以改变总体结论。

需要区分两类成功：模型输出协议和实验基础设施表现稳定，但视觉裁剪质量明显不足。当前结果支持继续做 prompt、训练或约束解码改进，不支持把原生模型作为可直接使用的裁剪策略。

## 2. 实验设置

### 2.1 输入与任务

数据：

```text
/mnt/blob_output/v-yukunban/crop-image-dataset/image_once_test.parquet
```

仅使用：

- `image_id`
- `original_image`
- `title`
- `ImageCaption`

未使用数据集自带的 `cropped_image`、`crop_image_id` 和 `Reason`。因此本实验不是对原有裁剪结果的复现或对比，而是原生 Qwen3.5-9B 的独立零样本绝对评测。

每张原图生成四个固定比例任务：

```text
1.0, 1.91, 1.77, 1.59
```

每个任务要求模型输出：

```text
<crop>{"cx": CX, "cy": CY, "area": AREA}</crop>
```

每个任务只保留第一个可解析候选；格式无效时使用不同的确定性 seed 重试，最多 10 次。

### 2.2 模型和评分

策略模型：

```text
Qwen3.5-9B
/mnt/blob_output/HuggingFace/Models/Qwen/Qwen3.5-9B
```

推理环境：

```text
Feeds / 8 x A100-SXM4-80GB
vLLM 0.24.0
PyTorch 2.11.0+cu130
CUDA 13.0
```

Judge：Azure OpenAI deployment `gpt-5.6-sol`。Judge 接收原图、Qwen 裁剪图、ImageCaption 和标题，使用固定 Tier 0-5 photo-editor rubric。

### 2.3 运行完整性

| 项目 | 结果 |
|---|---:|
| AMLT 状态 | Pass |
| 总时长 | 49 分钟 |
| 原图 | 120 |
| 评测任务 | 480 |
| 生成成功 | 480 / 480 |
| Judge 成功 | 480 / 480 |
| Judge API 失败 | 0 |
| Judge 解析 fallback | 0 |
| 未执行 Judge | 0 |

## 3. 核心结果

### 3.1 Tier 分布

| Tier | 含义 | 数量 | 占比 |
|---:|---|---:|---:|
| 0 | 无裁剪问题 | 107 | 22.29% |
| 1 | 可忽略问题 | 19 | 3.96% |
| 2 | 轻微问题 | 15 | 3.13% |
| 3 | 明显问题 | 65 | 13.54% |
| 4 | 严重裁剪失败 | 270 | 56.25% |
| 5 | 旋转问题 | 4 | 0.83% |

聚合指标：

| 指标 | 结果 |
|---|---:|
| Tier 0-1 可接受率 | 26.25% |
| Tier 3-5 严重问题率 | 70.63% |
| 平均 label | 2.80 |
| 平均 reward | 0.44 |
| Judge high confidence | 471 / 480（98.13%） |

### 3.2 四比例对比

| 目标比例 | 可接受率 Tier 0-1 | 严重率 Tier 3-5 | 平均 label | 平均 reward | 平均 AREA |
|---:|---:|---:|---:|---:|---:|
| 1.0 | **29.17%** | **67.50%** | **2.70** | **0.46** | 449.01 |
| 1.59 | 26.67% | 70.00% | 2.80 | 0.44 | 346.15 |
| 1.77 | 25.83% | 71.67% | 2.85 | 0.43 | 328.20 |
| 1.91 | 23.33% | 73.33% | 2.85 | 0.43 | 332.15 |

方形比例相对最好，但最好结果仍只有 29.17% 可接受率。随着比例变宽，模型平均选择的裁剪面积从 449 降至约 328-346，内容损失随之增加。比例本身会影响结果，但不是当前低质量的唯一原因。

### 3.3 图片级稳定性

将同一张原图的四个比例放在一起统计：

| 图片级指标 | 图片数 | 占比 |
|---|---:|---:|
| 四个比例全部可接受（Tier 0-1） | 18 / 120 | 15.00% |
| 至少一个比例可接受 | 50 / 120 | 41.67% |
| 四个比例均不可接受 | 70 / 120 | 58.33% |
| 四个比例全部严重（Tier 3-5） | 67 / 120 | 55.83% |
| 四比例 label 完全一致 | 75 / 120 | 62.50% |
| 最大 label 差至少 3 级 | 24 / 120 | 20.00% |

这说明失败既有稳定的图像类别效应，也有明显的比例敏感性。部分图片在方形比例为 Tier 0，但在宽比例变为 Tier 4；不能用某一个比例的表现代表其他比例。

## 4. 输出协议与重试

| 指标 | 结果 |
|---|---:|
| 首次生成有效 | 475 / 480（98.96%） |
| 曾出现无效输出的任务 | 5 / 480（1.04%） |
| 无效输出总次数 | 6 |
| 重试后恢复成功 | 5 / 5 |
| 10 次后仍失败 | 0 |
| 严格格式输出 | 464 / 480（96.67%） |
| 可恢复的非严格格式 | 16 / 480（3.33%） |
| 平均尝试次数 | 1.0125 |

重试机制工作正常。5 个任务发生格式无效，其中 4 个在第二次成功，1 个在第三次成功。需要注意：格式恢复解决的是协议稳定性，不改善裁剪质量；5 个重试任务中有 4 个最终仍被 Judge 判为 Tier 4。

## 5. 主要失败模式

Judge rule 可同时命中多个规则，因此下面计数不可相加。

| 规则 | 任务数 | 涉及原图 | 解释 |
|---|---:|---:|---|
| `T4.5` | 224 | 63 | 信息图、图表或文字内容被大量裁掉 |
| `T3.2` | 63 | 30 | 主体部分丢失 |
| `T4.3` | 46 | 19 | 标题对应的主要主体缺失或不可识别 |
| `T4.1` | 26 | 14 | 人脸关键部位被切断 |
| `T5.1` | 4 | 1 | 90 度旋转异常 |

### 5.1 信息图是首要瓶颈

`T4.5` 占 224/480（46.67%）任务，涉及 63/120（52.50%）原图。模型常输出约 200-300 的 AREA，只保留信息图局部，丢失标题、轴标签、图例、说明文字或成组内容。

代表性标题包括：

- `Centre for Indigenous Science papers urge Indigenous-centered conservation genomics`
- `Highly sensitive blood test spots pancreatic cancer before scans`
- `Why more people choose a childfree life`
- `Vintage Jell-O oddities resurface in nostalgic and bizarre look back`

这类图片不适合仅靠“标题相关主体”进行紧裁。策略需要先识别信息图类型，再提高最小面积或保留所有内容承载区域。

### 5.2 主体和人物关键部位丢失

`T4.3`、`T4.1` 和 `T3.2` 表明模型也会在普通照片上选错焦点或裁得过紧。例如：

- `Polls show Democrats gain but still trail in Senate fight`：方形裁剪被判定为主体垂直切断和主体缺失。
- `US Airline Stocks Rise on Cheaper Fuel and Strong Demand`：方形裁剪为 Tier 4，但 1.77/1.91 为 Tier 0，说明同图比例敏感。
- `FIFA confirms Balogun’s one-match ban for Belgium tie`：方形为 Tier 0，三个宽比例均为 Tier 4。

### 5.3 Tier 5 是源数据方向异常

全部 4 个 Tier 5 都来自同一张 `Collector Motorcycles: Market Growth and Trends for 2026` 图片。人工抽查确认原图本身横倒 90 度，四个 Qwen 候选没有额外引入旋转。

因此这 4 条应标记为源数据方向异常，而不是裁剪动作错误。去除这张图后：

- Tier 3-5 严重率约为 70.38%（335/476）；
- Tier 0-1 可接受率约为 26.47%（126/476）。

结论基本不变。

## 6. 动作分布

总体平均动作：

```text
CX = 561.62
CY = 457.94
AREA = 363.88
```

AREA 分布：

| 范围 | 数量 | 占比 |
|---|---:|---:|
| `AREA <= 50` | 4 | 0.83% |
| `50 < AREA <= 200` | 53 | 11.04% |
| `200 < AREA <= 500` | 358 | 74.58% |
| `500 < AREA < 950` | 3 | 0.63% |
| `AREA >= 950` | 62 | 12.92% |

模型明显偏好中小面积裁剪，并存在轻微向右偏移：30.00% 的动作中心位于右侧区域，而左侧仅 5.42%。对信息图而言，这种中小面积先验是主要风险之一。

## 7. 结果解释限制

1. 数据集由 `IsCroppedImage=True` 事件构建，可能偏向已知存在裁剪需求或裁剪风险的困难样本，不能直接外推到所有新闻图片。
2. 本次只使用一个 GPT Judge deployment 和一次确定性评分，没有人工复核或多 Judge 一致性估计。
3. 本实验按需求未使用原有 cropped image，也未运行中心裁剪或其他模型 baseline，因此只能做绝对质量判断，不能声明相对提升或退化。
4. `image_once` 每张原图只保留一个标题，不能充分测量标题变化对同图裁剪的影响。
5. Judge 虽有 98.13% high confidence，仍应对 Tier 4 样本分层人工抽检，尤其是照片/信息图边界案例。

## 8. 建议

### P0：训练或推理前增加图片类型路由

先区分普通照片与信息图。对于信息图：

- 提高最小 `AREA`；
- 优先保留标题、图例、轴、标注和完整集合；
- 可将“信息图尽量保留全图”直接加入策略 prompt 或规则后处理。

### P0：建立可比较 baseline

下一轮至少加入：

- 最大中心裁剪；
- 数据集已有 cropped image；
- 改进 prompt 的 Qwen；
- 后续微调模型。

所有候选使用同一 Judge 和同一数据 manifest，报告成对胜率，而不只看绝对平均分。

### P1：优先修复宽比例策略

`1.91` 的可接受率最低（23.33%）。建议对宽比例增加最小面积或限制纵向主体截断，并单独评测人像、信息图和场景照片。

### P1：人工复核分层样本

建议抽取：

- 所有 4 个 Tier 5；
- 每个高频规则 20 个 high-confidence 样本；
- 24 张比例敏感图片；
- 自动 Tier 0 和 Tier 4 各随机 30 条。

## 9. 产物索引

共享结果根目录：

```text
/mnt/blob_output/v-yukunban/crop-image-dataset/results/
qwen35-image-once-four-ratios-vlm-20260810-v2/
```

主要文件：

| 文件 | 用途 |
|---|---|
| `summary.json` | 总体和分比例指标 |
| `summary.csv` | 表格化汇总 |
| `details.parquet` | 480 条任务级结构化明细 |
| `details.jsonl` | 含完整 Judge 文本的任务明细 |
| `generation_attempts.jsonl` | 每次 Qwen 生成和重试记录 |
| `judge_responses.jsonl` | GPT Judge 原始响应和延迟 |
| `report.md` | 可直接通过 AMLT/Blob 读取的文本报告 |
| `report.html` | 该历史运行保留的可视化报告；需要同时获取 `renders/` |
| `renders/originals/` | 120 张原图预览 |
| `renders/candidates/` | 480 张最终候选裁剪 |
| `progress/` | 可恢复的逐任务生成和评分记录 |
| `_EVAL_COMPLETE.json` | 完成标记 |

直接查看 Markdown：

```powershell
amlt storage cat -c projects/news_image_crop_benchmark/amlt/amlt_image_once_qwen35_vlm_eval.yaml --storage-id blob_output `
	v-yukunban/crop-image-dataset/results/qwen35-image-once-four-ratios-vlm-20260810-v2/report.md
```

Markdown 不内嵌图片。需要视觉复核时，根据报告中的 `task_id` 或 render 相对路径只下载对应样本即可。
