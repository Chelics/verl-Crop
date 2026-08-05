# 数据转换设计

## 目标

- 将不可变的原始业务 Parquet 转换为 verl 可用的 train/validation/test Parquet。
- 原图不直接写入输出 Parquet，避免复制 3.1 GB 的二进制列或将其加载到 driver 内存。
- 保留同一图片对应的多个标题样本，同时防止原图跨数据划分泄漏。
- 仅将 `GemTitle`、`OriginalImageUrl` 和 `OriginalImageBytes` 视为可信训练数据。
- 将 `CroppedImageUrl` 和 `Reason` 仅作为未配对的源数据审计字段，绝不导出到训练数据或 Reward 中。
- 为配置的四种目标宽高比分别创建任务，不从未配对的裁剪图推断比例。
- 所有高开销阶段都支持断点恢复，并且可以独立审计。

## 产物目录

```text
<work_dir>/
  index/source_index.parquet
  index/source_index.report.json
  assets/original/<asset_id>.<ext>
  manifests/original_assets.parquet
  manifests/samples.parquet
  verl/train.parquet
  verl/validation.parquet
  verl/test.parquet
  reports/conversion_report.json
```

写入 verl Parquet 的所有路径均为共享存储上的绝对路径。每个 Ray 节点都必须在相同位置挂载 `<work_dir>`。

## 转换阶段

### 1. 建立索引

只读取轻量级源数据列，并生成：

- `sample_id`：唯一 `TraceId` 的 SHA-256 命名空间哈希；
- `original_asset_id`：原图 URL 的哈希；
- 确定性数据划分：对 `seed + OriginalImageUrl` 进行哈希。

原始字段会被重命名为 `unpaired_cropped_url` 和 `unpaired_reason`，明确表示它们与当前行不存在可信的配对关系。这两个字段只保留在源数据索引中。

该阶段由 `scripts/build_source_index.py` 实现。已确认 11,390 条源数据中的 `TraceId` 全部唯一。

### 2. 物化原图

以较小的 record batch 扫描 `OriginalImageUrl` 和 `OriginalImageBytes`。对于每个 asset ID 的首次出现：

1. 解码并验证图片；
2. 确定图片格式、宽度和高度；
3. 写入临时文件后执行原子重命名；
4. 将状态和校验和记录到 `original_assets.parquet`。

重复行复用同一个图片资产。源文件只有一个 3.1 GB 的 row group，因此转换器不得使用包含 `OriginalImageBytes` 的 `read_table` 一次性读取整表。

### 3. 去重并展开比例

按照完全相同的 `(OriginalImageUrl, normalized GemTitle)` 组合去重。11,390 条源数据中共有 9,935 个唯一可信组合。每个组合分别生成目标比例为 `1.0`、`1.91`、`1.77` 和 `1.59` 的四个独立任务，过滤前共得到 39,740 条数据。

仅根据 `OriginalImageUrl` 分配一次数据划分；同一原图对应的所有标题和比例都必须位于同一个 split。最终训练 ID 由原图 URL、规范化标题和目标比例共同生成。

仅在以下情况下丢弃可信组合：

- 原图缺失或无法解码；
- 标题为空；
- sample ID 重复；
- 缺少共享存储上的绝对图片路径。

两个未配对字段都不得进入源数据索引之后的处理阶段。

### 4. 导出 verl Parquet

每条数据采用以下逻辑结构：

```python
{
    "data_source": "news_image_crop",
    "prompt": [{"role": "user", "content": "<image>\nNews title: ..."}],
    "images": ["/shared/absolute/path/to/original.webp"],
    "ability": "news_image_cropping",
    "reward_model": {
        "style": "proxy",
        "ground_truth": "{\"image_height\":...,\"image_width\":...,\"target_ratio\":...}",
    },
    "extra_info": {
        "index": 123,
        "sample_id": "...",
        "split": "train",
        "title": "...",
        "target_ratio": 1.59,
        "image_width": 2900,
        "image_height": 2900,
        "original_image_path": "/shared/...",
    },
}
```

`prompt` 和 `images` 是策略模型的输入。`reward_model.ground_truth` 和 `extra_info` 向自定义 Reward 提供图片尺寸和目标比例。

## 验收条件

- 每个 sample ID 都唯一。
- train/validation/test 中的 `OriginalImageUrl` 集合两两不相交。
- 每个图片路径都是绝对路径、文件存在且可以解码。
- 每个目标比例都是 `1.0`、`1.91`、`1.77` 或 `1.59` 之一。
- Prompt 中恰好包含一个 `<image>` 占位符和一个图片条目。
- 输出训练数据中不包含 `CroppedImageUrl` 或 `Reason`。
- 小规模输出 Parquet 可以通过 Hugging Face Datasets 和 `RLHFDataset` 完成读写往返。
- 转换报告包含源数据行数、唯一可信组合数、展开后行数、各 split 数量和所有拒绝原因。

## 从小规模联调到全量转换

1. 使用 `--limit 100` 构建全部产物，并检查生成的 prompt 和图片。
2. 构建全量索引和图片资产，验证数量及断点恢复行为。
3. 每个 split 导出 32 条数据，实例化 Qwen3.5 processor 和 verl dataset。
4. 仅在全部验收条件通过后执行全量导出。
