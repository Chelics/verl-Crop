# 新闻图片裁剪实验数据质量审计报告

审计日期：2026-08-05

## 当前修复状态

2026-08-05 已基于原图 SHA-256 重新去重和划分，并保留旧文件用于审计对照。正式实验应使用以下新产物：

```text
/mnt/blob_output/v-yukunban/news_image_crop_content_split/news_image_crop_train.parquet
/mnt/blob_output/v-yukunban/news_image_crop_content_split/news_image_crop_validation.parquet
/mnt/blob_output/v-yukunban/news_image_crop_content_split/news_image_crop_test.parquet
```

新产物从旧数据的 39,740 条任务中移除了 660 条由 URL 别名造成的内容/标题/比例重复任务，保留 39,080 条。2,646 个图片内容组严格按 SHA-256 分配，train/validation/test 两两内容交集均为 0。本文后续第 1 至第 8 节保留的是修复前数据问题及其证据，用于说明为什么必须重划分。

| Split | 图片内容组 | 内容组占比 | 展开任务 | 任务占比 |
|---|---:|---:|---:|---:|
| train | 2,143 | 80.99% | 31,884 | 81.59% |
| validation | 236 | 8.92% | 3,040 | 7.78% |
| test | 267 | 10.09% | 4,156 | 10.63% |
| 合计 | 2,646 | 100% | 39,080 | 100% |

任务行比例不会严格等于 80/10/10，因为同一图片内容关联的所有标题都必须留在一个 split，且不同图片的标题数量高度不均衡。四种目标比例仍完全平衡，各 9,770 条。

## 1. 结论摘要

当前全量数据已经成功转换为 verl 可读取的多模态 Parquet。字段结构、样本 ID、目标比例、图片路径以及按 `OriginalImageUrl` 进行的 split 隔离均通过检查，可以用于工程链路联调。

但是，当前 train/validation/test **不能作为相互独立的正式评测集合**。主要原因是数据只按原图 URL 划分，而不同 URL 大量指向字节完全相同的图片：

- 5,672 个原图 URL 只对应 2,646 个不同的 SHA-256 图片内容；
- 314 个图片内容哈希同时出现在多个 split，占全部不同图片内容的 11.87%；
- 83 个图片内容哈希同时出现在 train、validation 和 test；
- 受跨 split 内容重复影响的 URL 资产为 2,863 个，占全部 URL 资产的 50.48%；
- 受影响的展开任务为 18,748 条，占全部 39,740 条任务的 47.18%。

因此，当前数据满足“同一个 URL 不跨 split”，但不满足“同一张实际图片不跨 split”。在按图片内容重新划分并冻结 test 集之前，不应使用当前 validation/test 分数声明模型泛化能力或正式质量增益。

## 2. 审计范围与证据来源

本次审计只读取数据，没有修改原始 Parquet、转换后 Parquet 或图片资产。

| 证据 | 路径或实现 | 用途 |
|---|---|---|
| 原始数据 | `/mnt/blob_output/v-yukunban/imageCroppingDataset_cropped_binary.parquet` | 核对源行数、列、空值和字段基数 |
| 转换报告 | `/mnt/blob_output/v-yukunban/news_image_crop_conversion_report.json` | 核对去重、拒绝原因和展开数量 |
| 图片 manifest | `/mnt/blob_output/v-yukunban/news_image_crop_assets.parquet` | 核对 URL 资产、路径、尺寸和 SHA-256 |
| verl 数据 | `/mnt/blob_output/v-yukunban/news_image_crop_{train,validation,test}.parquet` | 全量扫描 schema、ID、比例、Prompt、路径和 split |
| 设计文档 | [data_conversion.md](data_conversion.md) | 核对设计目标和验收条件 |
| 转换实现 | [convert_to_verl.py](../scripts/convert_to_verl.py) | 确认资产校验、去重和导出逻辑 |
| split 实现 | [data.py](../src/news_crop_benchmark/data.py) | 确认当前按 `seed + OriginalImageUrl` 哈希划分 |

这里的“相同图片内容”指 manifest 中 SHA-256 完全相同的原始图片字节。这是严格且保守的重复判定：它能证明图片完全相同，但无法识别经过重新压缩、缩放或轻微编辑的视觉近重复。因此，本报告给出的泄漏数量是下界，而不是上界。

## 3. 原始数据现状

源 Parquet 审计结果如下：

| 指标 | 数值 |
|---|---:|
| 文件逻辑大小 | 3,289,142,107 bytes，约 3.06 GiB |
| 行数 | 11,390 |
| 列数 | 14 |
| row group 数 | 1 |
| 唯一 `OriginalImageUrl` | 5,672 |
| 唯一 `CroppedImageUrl` | 5,672 |
| 唯一 `GemId` | 6,533 |
| 唯一 `GemSnapshotId` | 10,208 |
| 唯一 `Reason` | 6,365 |
| `IsCroppedImage=true` | 11,390 |

`GemTitle`、`OriginalImageUrl`、`CroppedImageUrl`、`Reason` 和 `IsCroppedImage` 均无空值。源文件只有一个约 3.1 GB 的 row group，因此转换脚本使用小 batch 扫描 `OriginalImageBytes` 是必要的；一次性读取二进制列会造成明显的 driver 内存风险。

源数据中有 484 个原图 URL 对应多个不同的 `Reason`。这进一步说明 `Reason` 不是稳定的图片级标签。当前转换没有将 `CroppedImageUrl` 或 `Reason` 写入训练数据，这一点符合设计约束。

## 4. 转换与结构验收

### 4.1 去重和四比例展开

11,390 条源数据按规范化后的 `(OriginalImageUrl, GemTitle)` 去重：

| 指标 | 数值 |
|---|---:|
| 唯一可信 URL/标题组合 | 9,935 |
| 重复 URL/标题行 | 1,455 |
| 空标题拒绝 | 0 |
| 无效图片拒绝 | 0 |
| 冲突图片 payload | 0 |
| 缺失资产的可信组合 | 0 |
| 四比例展开后任务 | 39,740 |

四种目标比例 `1.0`、`1.59`、`1.77`、`1.91` 各有 9,935 条，比例完全平衡。

### 4.2 split 分布

| Split | URL 资产 | URL/标题组合 | 展开任务 | 任务占比 |
|---|---:|---:|---:|---:|
| train | 4,572 | 8,237 | 32,948 | 82.91% |
| validation | 551 | 814 | 3,256 | 8.19% |
| test | 549 | 884 | 3,536 | 8.90% |
| 合计 | 5,672 | 9,935 | 39,740 | 100% |

任务数没有严格呈现 80/10/10，是因为 split 按 URL 分组，而不同图片携带的标题数差异很大。这不是随机误差，而是样本集中度造成的组大小不均衡。

### 4.3 已通过的全量检查

- 39,740 个 `sample_id` 全部唯一；
- `extra_info.index` 完整覆盖 0 至 39,739；
- 三个 split 的 URL/图片路径集合两两不相交；
- 所有图片路径都是绝对路径并且文件存在；
- 每行恰好包含一个 `<image>` 占位符和一个图片条目；
- `reward_model.ground_truth` 与 `extra_info` 中的宽、高和目标比例一致；
- 存储的 39,740 个 Prompt 与当前 `build_prompt` 实现完全一致；
- 输出 schema 不包含 `CroppedImageUrl` 或 `Reason`；
- 转换报告记录 5,672 张图片均通过 PIL 格式、尺寸和解码校验；本次审计另外抽样解码了实际资产，结果通过。

这些检查说明数据在工程格式上可用，但不能排除下面的数据独立性和分布问题。

## 5. 关键问题与证据

### P0：图片内容跨 split 泄漏

当前 `assign_group_split` 的 group key 是 `OriginalImageUrl`。manifest 全量比较显示，5,672 个 URL 资产只有 2,646 个不同 SHA-256 内容，说明多个 URL 经常保存同一份图片字节。

| 内容级检查 | 数值 |
|---|---:|
| 不同 SHA-256 内容 | 2,646 |
| 出现在多个 split 的内容 | 314 |
| 同时出现在三个 split 的内容 | 83 |
| train/validation 内容交集 | 199 |
| train/test 内容交集 | 195 |
| validation/test 内容交集 | 86 |
| 涉及的 URL 资产 | 2,863 / 5,672，50.48% |
| 涉及的展开任务 | 18,748 / 39,740，47.18% |

两两交集包含同时出现在三个 split 的内容，因此三项不能直接相加。

最严重的一个重复内容通过 107 个 URL 出现，并形成 2,936 条展开任务，占全量任务的 7.39%。该内容同时存在于 train、validation 和 test。另一个内容通过 130 个 URL 出现，也同时横跨三个 split。

**影响：** 模型可能在训练阶段见过与 validation/test 完全相同的图片，只是 URL 或标题不同。此时测试结果同时衡量了图片记忆、主题模板复用和真正的未见图片泛化，三者无法区分。正式 baseline、GRPO 增益和人工 test 抽样都会受到污染。

### P1：少数图片绑定过多标题

按 URL 统计标题数量：

- 5,081 张图片只有一个规范化标题；
- 591 张图片有多个标题；
- 单个 URL 最多绑定 628 个不同标题，四比例展开后可产生 2,512 条任务；
- 还存在单 URL 分别绑定 245、125、110、107 个标题的样本。

人工抽查最高频样本后发现，它们不是损坏文件，而是被大量相关新闻复用的信息图或通用主题图片，例如国际冲突信息图、401(k) 通用配图和星座主题图。

**影响：** 当前按任务行均匀采样会显著放大少数图片和主题。模型梯度、自动 Reward 均值以及按行抽取的人工样本都可能被这些高频内容主导。39,740 条任务不能解释为 39,740 个近似独立的训练例子。

### P1：图片内容多样性明显低于 URL 数量

图片 manifest 的基本统计：

| 指标 | 数值 |
|---|---:|
| URL 资产 | 5,672 |
| 不同 SHA-256 内容 | 2,646 |
| 只出现于一个 URL 的内容 | 2,058 |
| 被至少两个 URL 复用的内容 | 588 |
| 图片逻辑总大小 | 1,700,075,490 bytes，约 1.58 GiB |
| 格式 | 5,672 张均为 WebP |

不同内容数量只有 URL 资产数量的 46.65%。此外，精确哈希无法识别视觉近重复，真实视觉多样性可能更低。

### P2：图片方向分布高度偏向横图

| 方向 | 数量 | 占比 |
|---|---:|---:|
| 横图 | 5,127 | 90.39% |
| 方图 | 537 | 9.47% |
| 竖图 | 8 | 0.14% |

图片宽度的最小值/中位数/最大值为 143/2,403/9,840，高度为 100/1,484/5,750。该分布符合新闻横图常见形态，但几乎不能支持对竖图输入的泛化结论。如果目标产品包含明显比例的竖图，需要单独补充数据或限制适用范围。

### P2：产物布局和状态文档不一致

[data_conversion.md](data_conversion.md) 规划了 `index/`、`manifests/`、`verl/` 和 `reports/` 分阶段产物。当前实际全量文件平铺在 `/mnt/blob_output/v-yukunban/`，配置中的 `/mnt/blob_output/v-yukunban/news_image_crop_benchmark` 目录不存在，也没有全量 `source_index.parquet` 或 `samples.parquet`。

这不影响当前 Parquet 读取，但会削弱阶段级审计和断点恢复能力。项目 README 中“verl dataset conversion Pending”的状态也与已经生成并验证的全量数据不一致。

## 6. 对当前实验结论的约束

当前数据可以用于：

- 验证 verl 多模态数据读取；
- 验证 Qwen rollout、动作解析、裁剪渲染和 Reward 接线；
- 小规模性能测试和训练稳定性调试。

当前数据不宜直接用于：

- 声明 validation/test 是未见图片评测；
- 声明 GRPO 相对 zero-shot 的正式泛化增益；
- 从按行均值推断对真实新闻图片分布的整体质量；
- 构建最终 300 条人工 golden set。

如果在修复前运行实验，结果必须标注为“受已知内容泄漏影响的工程预实验”，不能作为正式 benchmark 结果。

## 7. 修复建议与重新验收条件

### 7.1 必须修复

1. 先物化并校验图片，得到每个 URL 对应的 SHA-256。
2. 使用图片内容哈希作为 split group key；同一内容的所有 URL、标题和比例必须进入同一个 split。
3. 去重键从 `(OriginalImageUrl, normalized_title)` 调整为 `(image_checksum, normalized_title)`，避免同图同标题仅因 URL 不同而重复。
4. 重新生成 train/validation/test，并冻结新的 test 清单。
5. 在转换报告中增加内容哈希数量、内容复用分布和跨 split 内容交集。

重新验收必须满足：

- train/validation/test 的 SHA-256 集合两两交集为 0；
- URL 集合两两交集为 0；
- `(image_checksum, normalized_title, target_ratio)` 全局唯一；
- 每个内容哈希只对应一个 split；
- 所有图片可解码，Prompt、比例、路径和 sample ID 验收继续通过。

### 7.2 建议修复

- 对每个内容哈希设置标题数上限，或按内容哈希进行均衡采样；
- 分别报告“按任务平均”和“按图片内容等权平均”的评测结果；
- 对精确去重后的图片运行感知哈希或视觉 embedding 近重复聚类；
- 人工 golden set 按内容哈希、主题和目标比例分层抽样；
- 明确产品输入是否包含竖图，并据此补充竖图覆盖。

## 8. 具体数据实例

### 8.1 URL 展示和核验规则

源数据中的 5,672 个 `OriginalImageUrl` 和 5,672 个 `CroppedImageUrl` 全部来自 `th.bing.com`，查询参数均只有 `id` 和 `pid`。下面保留这两个非签名参数，便于定位实际记录。

每个 URL 另外提供一个 16 位指纹，计算方式为：

```python
hashlib.sha256(raw_url.encode()).hexdigest()[:16]
```

图片内容 SHA-256 来自 `OriginalImageBytes`。相同内容 SHA-256 可以证明原图字节完全一致。源 Parquet 没有 `CroppedImageBytes`，因此下面只能列出 `CroppedImageUrl`，不能仅凭当前数据证明不同裁剪 URL 返回的图片字节是否相同。

### 8.2 相同原图内容、不同 URL、跨 split 实例

#### 实例 A：耳部解剖图跨 train/validation

两条记录的原图 URL 不同，但原图内容 SHA-256 都是：

```text
00c94deb9e6f84b91e9cecdd3baf8cf243f901fb37130db97b0bade04a455fcd
```

train 记录：

- `TraceId`：`a7008348-e96f-4fd5-b326-0f6c6f322b0a`
- 标题：`Experts warn early hearing loss may harm brain health`
- 原图 URL 指纹：`93453dd7f00b220a`
- `OriginalImageUrl`：`https://th.bing.com/th?id=OMSN.AA26BTJL.webp&pid=wdpv2`
- 裁剪图 URL 指纹：`faddf53f602f6338`
- `CroppedImageUrl`：`https://th.bing.com/th?id=OMSN.AA26BTJL.webp-Gem&pid=wdpv2`

validation 记录：

- `TraceId`：`9927f1dd-3dd6-444c-870c-6dbf14e8b525`
- 标题：`Hearing Loss in Diabetes: Prevalence, Mechanisms, and Screening Value`
- 原图 URL 指纹：`65112ec04eccd9f6`
- `OriginalImageUrl`：`https://th.bing.com/th?id=OMSN.AA27cjTz.webp&pid=wdpv2`
- 裁剪图 URL 指纹：`b1f17c3dcb193f96`
- `CroppedImageUrl`：`https://th.bing.com/th?id=OMSN.AA27cjTz.webp-Gem&pid=wdpv2`

两条 Reason 都描述“完整耳部解剖图被裁到只剩局部 inset”，语义一致。这是典型的同一图片以不同 Bing ID 进入不同 split。

#### 实例 B：六行星排列图跨 train/test

共同的原图内容 SHA-256：

```text
064e00665e1c3aac78a6beaacf480abf2096522829c870d9d842882f9d189deb
```

train 记录：

- `TraceId`：`b120c0b4-d27a-4f2c-ac91-e77225b23f0b`
- 标题：`New models show twilight zones on tidally locked exoplanets may support life`
- 原图 URL 指纹：`f50e74a6ed76e436`
- `OriginalImageUrl`：`https://th.bing.com/th?id=OMSN.AA27K6MO.webp&pid=wdpv2`
- 裁剪图 URL 指纹：`80b4338dc240e2ac`
- `CroppedImageUrl`：`https://th.bing.com/th?id=OMSN.AA27K6MO.webp-Gem&pid=wdpv2`
- Reason 摘要：原图有六颗行星，裁剪图只保留前四颗。

test 记录：

- `TraceId`：`20c69522-957b-4347-a00b-26582569ae29`
- 标题：`Cornell team narrows 6,000 exoplanets to 45 prime life candidates`
- 原图 URL 指纹：`762116d4a7d853d5`
- `OriginalImageUrl`：`https://th.bing.com/th?id=OMSN.AA27b49a.webp&pid=wdpv2`
- 裁剪图 URL 指纹：`c8f2d1f4cb19b82e`
- `CroppedImageUrl`：`https://th.bing.com/th?id=OMSN.AA27b49a.webp-Gem&pid=wdpv2`
- Reason 摘要：原图有六颗行星且以地球结尾，裁剪图只保留前三颗。

这组样本不仅共享完全相同的原图，标题主题和 Reason 也高度接近，因此 test 记录不能视为独立的未见视觉样本。

#### 实例 C：热带风暴地图跨 train/test

共同的原图内容 SHA-256：

```text
093985fe0836ee27c6d3a8bc91f2c3177ed674406a152f782c40899ab039c4f2
```

train 记录：

- `TraceId`：`8d006bbb-a110-4f82-9789-ba52bc569099`
- 标题：`Bertha strengthens to 50 mph as Gulf Coast braces`
- 原图 URL 指纹：`7b669bc567b67b5f`
- `OriginalImageUrl`：`https://th.bing.com/th?id=OMSN.AA28nBXg.webp&pid=wdpv2`
- 裁剪图 URL 指纹：`1e004e586a91ca25`
- `CroppedImageUrl`：`https://th.bing.com/th?id=OMSN.AA28nBXg.webp-Gem&pid=wdpv2`

test 记录：

- `TraceId`：`00e40071-a037-48db-abf1-3c60675125d4`
- 标题：`Florida Gulf Coast rain and tee time tips during Tropical Depression 2`
- 原图 URL 指纹：`68f40c397a68a8da`
- `OriginalImageUrl`：`https://th.bing.com/th?id=OMSN.AA28sPyW.webp&pid=wdpv2`
- 裁剪图 URL 指纹：`be5d1ee4ecd74b10`
- `CroppedImageUrl`：`https://th.bing.com/th?id=OMSN.AA28sPyW.webp-Gem&pid=wdpv2`

两条 Reason 都指出裁剪移除了风暴信息图的标题、图例和解释性文本。该组再次说明 URL 级隔离没有实现图片内容级隔离。

### 8.3 同一个 OriginalImageUrl 对应不同 Reason 的实例

全量统计中有 484 个 `OriginalImageUrl` 对应多个不同 Reason。同时，每个 `OriginalImageUrl` 都只对应一个 `CroppedImageUrl`，即“同原图不同 Reason”并不是因为一个原图在表中连接了多个裁剪 URL。

#### 实例 D：同一海湾地图、同一裁剪 URL、两个 Reason

- 原图 URL 指纹：`0009d4827ddded17`
- 原图内容 SHA-256：`2c0498cb052cd1b348ddf2c70e0cc3fac4708df2193175e4985dd3c1dcbe1444`
- `OriginalImageUrl`：`https://th.bing.com/th?id=OMSN.AA26OM09.webp&pid=wdpv2`
- 裁剪图 URL 指纹：`7e0318a551d336c3`
- `CroppedImageUrl`：`https://th.bing.com/th?id=OMSN.AA26OM09.webp-Gem&pid=wdpv2`

记录 1：

- `TraceId`：`3b9c1944-bd71-49d4-aec8-c7c20b0493f5`
- 标题：`US strikes Iran's Chabahar port, raising India trade fears`
- Reason：`The original infographic in Image A contains a title, legend, explanatory text, and other key elements that provide context and meaning to the map. The cropped version in Image B retains only the central map area, removing the legend that explains the color coding, the title that identifies the subject, and the text that describes the events. This loss of content significantly impairs the viewer's ability to understand the graphic.`

记录 2：

- `TraceId`：`5788d300-9f49-4da7-ad7e-b1f9bd8b86bd`
- 标题：`Oman to jointly manage Hormuz with Iran amid US-Iran clashes`
- Reason：`The original infographic in Image A contains a title, legend, date, and multiple labeled locations that explain the nature and scope of the attacks. In Image B, these are cropped out, leaving only part of the map without the explanatory context. This loss of content-bearing elements severely impairs the viewer’s ability to interpret the graphic.`

这两个 Reason 语义大体一致，但文本和强调的缺失元素不同，说明 Reason 是行级生成文本，而不是稳定的原图或裁剪图标签。

#### 实例 E：同一 URL 的 Reason 出现明显内容矛盾

- 原图 URL 指纹：`0171a8be0befb949`
- 原图内容 SHA-256：`fe666872ad2732e9815d21fc022d34ad6d51ff9809fdf7ea4447f588ab9cfd34`
- `OriginalImageUrl`：`https://th.bing.com/th?id=OMSN.AA26JX6e.webp&pid=wdpv2`
- 裁剪图 URL 指纹：`1233c70a05156b63`
- `CroppedImageUrl`：`https://th.bing.com/th?id=OMSN.AA26JX6e.webp-Gem&pid=wdpv2`

记录 1：

- `TraceId`：`a51ff5e0-2660-4975-9fcc-d0fe465a9a60`
- 标题：`Trump urges 12% GDP growth, accuses Fed of stifling economy`
- Reason：`The original graphic in Image A contains multiple sections: a title, explanatory text, portraits, quotes, financial data, and a pie chart. Image B retains only the lower portion with the pie chart and family members, omitting the top section that provides essential context and data. Since this is an informational graphic, the missing elements are part of the primary subject, and their removal significantly impairs understanding.`

记录 2：

- `TraceId`：`afe2e123-4009-4a9c-b15c-ff9c77b5b5ad`
- 标题：`Poll shows Trump approval at highest since spring`
- Reason：`The original infographic in Image A contains a title, legend, and explanatory text that are integral to interpreting the map's colored markers and understanding the depicted events. In Image B, these elements are cropped out, leaving only the map and markers without context, which severely impairs comprehension.`

同一个原图和同一个裁剪 URL，一条 Reason 描述“人物、财务数据、饼图和家庭成员”，另一条却描述“地图和彩色标记”。两者不可能同时准确描述同一图像对。这是 `Reason` 与当前行标题/图片配对不可靠的直接实例，也支持“不将 Reason 用作训练或 Reward 信号”的现有设计。

#### 实例 F：同一法国债务图的两个近义 Reason

- 原图 URL 指纹：`01ca54f0a5052fde`
- 原图内容 SHA-256：`3a2a3ed122e3cb927b81f0d528801a9c5cf60a7d769a73903e4b52665d75d5aa`
- `OriginalImageUrl`：`https://th.bing.com/th?id=OMSN.AA27qwZv.webp&pid=wdpv2`
- 裁剪图 URL 指纹：`bb47bdfc231829e5`
- `CroppedImageUrl`：`https://th.bing.com/th?id=OMSN.AA27qwZv.webp-Gem&pid=wdpv2`

记录 1：

- `TraceId`：`be8e7f2e-71db-4a0c-9a09-4bb429ff3a14`
- 标题：`OECD warns France's debt could hit 203% of GDP by 2050`
- Reason：裁剪移除了 `French government debt and deficit` 标题及 `In % of GDP` 副标题，影响图表解释。

记录 2：

- `TraceId`：`f5c51982-7653-478d-a29b-26111a2c28d5`
- 标题：`France warned of rising debt without swift reforms`
- Reason：裁剪移除了相同主标题和 y 轴单位，影响图表的信息完整性。

这组 Reason 内容基本一致，说明“同原图多 Reason”不一定都是错误，也可能只是针对不同标题生成的近义解释。因此不能仅按 Reason 字符串不同判断标签冲突；需要结合图片内容人工或模型审计。实例 E 则属于无需依赖措辞差异即可识别的明显矛盾。

## 9. 复现命令

源数据轻量审计，不读取图片二进制列：

```bash
/opt/conda/envs/ptca/bin/python \
  projects/news_image_crop_benchmark/scripts/inspect_source.py \
  --input /mnt/blob_output/v-yukunban/imageCroppingDataset_cropped_binary.parquet
```

复核内容哈希跨 split。下面的 `affected_rows` 定义为：其图片内容同时存在于至少两个 split 的所有展开任务数。

```bash
/opt/conda/envs/ptca/bin/python - <<'PY'
from collections import Counter, defaultdict
from pathlib import Path

import pyarrow.parquet as pq

root = Path("/mnt/blob_output/v-yukunban")
manifest = pq.read_table(root / "news_image_crop_assets.parquet").to_pylist()
path_to_checksum = {row["path"]: row["checksum"] for row in manifest}
checksum_paths = defaultdict(set)
for row in manifest:
    checksum_paths[row["checksum"]].add(row["path"])

checksum_splits = defaultdict(set)
checksum_rows = Counter()
for split in ("train", "validation", "test"):
    rows = pq.read_table(root / f"news_image_crop_{split}.parquet").to_pylist()
    for row in rows:
        checksum = path_to_checksum[row["images"][0]]
        checksum_splits[checksum].add(split)
        checksum_rows[checksum] += 1

leaked = {checksum for checksum, splits in checksum_splits.items() if len(splits) > 1}
affected_paths = {path for checksum in leaked for path in checksum_paths[checksum]}
print("url_assets", len(manifest))
print("unique_checksums", len(checksum_splits))
print("cross_split_checksums", len(leaked))
print("all_three_splits", sum(len(splits) == 3 for splits in checksum_splits.values()))
print("affected_paths", len(affected_paths))
print("affected_rows", sum(checksum_rows[checksum] for checksum in leaked))
for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
    overlap = sum({first, second}.issubset(splits) for splits in checksum_splits.values())
    print(f"{first}_{second}_overlap", overlap)
PY
```

预期关键输出：

```text
url_assets 5672
unique_checksums 2646
cross_split_checksums 314
all_three_splits 83
affected_paths 2863
affected_rows 18748
train_validation_overlap 199
train_test_overlap 195
validation_test_overlap 86
```

## 10. 最终判定

| 维度 | 旧 URL split | 新 SHA-256 split |
|---|---|---|
| verl schema 与可读性 | 通过 | 通过 |
| 样本 ID、Prompt、比例和路径 | 通过 | 通过 |
| 图片内容级 split 隔离 | **不通过** | **通过，交集为 0** |
| 内容/标题/比例键唯一性 | 未保证 | **通过，39,080/39,080** |
| 样本权重与主题均衡性 | **有风险** | **仍需降权或分组报告** |
| 正式 validation/test 可用性 | **不通过** | 可用于后续 baseline；仍需近重复审计和人工 golden set |
| 工程联调可用性 | 通过 | 通过 |

综合判定：旧 URL split 只保留用于问题复现和历史结果溯源。新 SHA-256 split 已修复精确图片内容泄漏，应作为后续 baseline、训练和 test 抽样的唯一数据入口。SHA-256 无法识别重新压缩或轻微编辑的视觉近重复，高频图片权重和近重复聚类仍是正式质量结论前的剩余风险。