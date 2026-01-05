# IndexTTS2 数据模块

## 概述

数据模块负责生成和处理训练所需的数据。由于网络受限无法下载大规模数据集，本模块提供了伪数据生成器(Mock Data Factory)用于测试和开发。

## 模块组成

### mock_factory.py - 伪数据生成器

**功能**：生成符合真实数据分布但内容随机的张量，用于测试模型的形状变换、Loss计算和反向传播。

**主要类**：
- `MockBatch`: 伪数据批次生成器

**生成的数据字段**：

| 字段名 | 形状 | 说明 |
|--------|------|------|
| `text_ids` | (B, Text_Len) | 文本Token ID序列 |
| `text_lengths` | (B,) | 每个样本的实际文本长度 |
| `semantic_ids` | (B, Semantic_Len) | 语义Token ID序列 |
| `semantic_lengths` | (B,) | 每个样本的实际语义序列长度 |
| `mel_gt` | (B, MEL_BINS, Frame_Len) | 梅尔频谱Ground Truth |
| `mel_lengths` | (B,) | 每个样本的实际帧长度 |
| `speaker_embed` | (B, Speaker_Dim) | 说话人嵌入向量 |
| `emotion_embed` | (B, Emotion_Dim) | 情感嵌入向量 |
| `target_len` | (B,) | 目标语义序列长度（用于时长控制） |
| `speaker_ids` | (B,) | 说话人ID（用于分类任务） |
| `emotion_ids` | (B,) | 情感ID（用于分类任务） |

## 使用示例

```python
from src.data.mock_factory import create_mock_batch

# 创建一个批次的伪数据
batch = create_mock_batch(
    batch_size=16,
    text_len=30,
    semantic_len=150,
    mel_frame_len=300,
    device='cuda'
)

# 访问数据字段
text_ids = batch.text_ids  # (16, 30)
semantic_ids = batch.semantic_ids  # (16, 150)
mel_gt = batch.mel_gt  # (16, 80, 300)
```

## 设计原理

伪数据生成器的核心思想是：
1. **形状匹配**：生成的张量形状完全符合真实数据的规范
2. **变长支持**：通过`*_lengths`字段支持变长序列
3. **随机内容**：内容完全随机，但不影响模型结构测试
4. **完整性**：包含训练所需的所有字段

这样可以在没有真实数据的情况下，完整测试整个模型流水线。
