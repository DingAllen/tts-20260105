# T2S (Text-to-Semantic) 模块

## 概述

T2S模块负责将文本转换为语义Token序列。这是一个基于自回归Transformer的生成模型，具有时长控制和情感解耦功能。

## 核心特性

### 1. 时长控制 (Duration Control)

通过**共享权重**机制实现精确的时长控制：

- 语义Token的位置编码 $W_{sem}$ 和时长嵌入 $W_{num}$ 使用相同的权重矩阵
- 时长向量计算：$p = W_{num} \cdot \text{OneHot}(T)$
- 这种设计使得模型能够根据指定的目标长度生成对应数量的语义Token

### 2. 情感解耦 (Emotion Disentanglement)

通过**梯度反转层(GRL)**实现说话人信息的解耦：

- Stage 1训练：只训练主任务（语义预测）
- Stage 2训练：加入GRL，对抗性地消除说话人信息
- Loss公式：$\mathcal{L}_{total} = \mathcal{L}_{AR} - \lambda \cdot \mathcal{L}_{SpeakerClassifier}$

### 3. 自回归生成

采用标准的自回归Transformer Decoder架构：

- 使用Causal Mask确保生成时只能看到历史信息
- 支持多种采样策略（Temperature, Top-K, Nucleus）
- 生成高质量的语义Token序列

## 模块组成

### embedding.py - 共享嵌入层

**主要类**：
- `SharedEmbedding`: 实现语义Token嵌入和时长编码的权重共享
- `T2SInputBuilder`: 构造完整的输入序列

**输入序列格式**：
```
[Speaker+Emotion, Duration, <BT>, Text, <BA>, Semantic_Tokens]
```

其中：
- Speaker+Emotion: 说话人和情感的融合向量
- Duration: 时长嵌入向量
- <BT>: Begin Token
- Text: 文本Token序列
- <BA>: Begin Answer（分隔符）
- Semantic_Tokens: 语义Token序列（自回归生成目标）

### model.py - T2S主模型

**主要类**：
- `T2SModel`: 完整的T2S自回归Transformer模型

**输入参数**：

| 参数 | 形状 | 说明 |
|------|------|------|
| text_ids | (B, Text_Len) | 文本Token IDs |
| semantic_ids | (B, Semantic_Len) | 语义Token IDs（训练时） |
| speaker_embed | (B, Speaker_Dim) | 说话人嵌入 |
| emotion_embed | (B, Emotion_Dim) | 情感嵌入 |
| target_lengths | (B,) | 目标序列长度（时长控制） |

**输出**：

| 输出 | 形状 | 说明 |
|------|------|------|
| logits | (B, Semantic_Len, Vocab_Size) | 语义Token预测logits |
| hidden_states | (B, Total_Len, Hidden_Dim) | 隐藏状态（用于S2M的GPT融合） |
| speaker_logits | (B, Num_Speakers) | 说话人分类logits（如果use_grl=True） |

### trainer.py - 训练逻辑

**主要类**：
- `T2STrainer`: T2S训练器

**Loss组件**：
1. `loss_semantic`: 语义Token预测Loss（主Loss）
2. `loss_text`: 文本预测Loss（可选）
3. `loss_grl`: 情感解耦Loss（Stage 2）

**训练阶段**：
- **Stage 1**: 只训练主任务，$\lambda_{GRL} = 0$
- **Stage 2**: 加入对抗训练，$\lambda_{GRL} = 1.0$

## 使用示例

### 训练模式

```python
from src.modules.t2s.model import T2SModel
from src.modules.t2s.trainer import T2STrainer
from src.data.mock_factory import create_mock_batch

# 创建模型
model = T2SModel(use_grl=True)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 创建训练器
trainer = T2STrainer(
    model=model,
    optimizer=optimizer,
    lambda_semantic=1.0
)

# 准备数据
batch = create_mock_batch(batch_size=16)

# Stage 1训练
trainer.set_stage(1)
loss_dict = trainer.train_step(batch, stage=1)

# Stage 2训练
trainer.set_stage(2)
loss_dict = trainer.train_step(batch, stage=2)
```

### 推理模式

```python
# 生成语义Token
generated_tokens = model.generate(
    text_ids=text_ids,
    speaker_embed=speaker_embed,
    emotion_embed=emotion_embed,
    target_length=target_length,
    temperature=0.8,
    top_k=50
)
```

## 设计原理

### 共享权重的数学意义

位置编码和时长编码共享权重的设计基于以下观察：

1. **位置编码**告诉模型"当前在序列的哪个位置"
2. **时长编码**告诉模型"整个序列有多长"

两者都是对"位置"概念的编码，因此使用相同的嵌入空间是合理的。

### GRL的工作原理

```
特征 → GRL → 说话人分类器
  ↓      ↓
正梯度  反梯度
```

- 前向传播：恒等映射，特征不变
- 反向传播：梯度取反，使主网络学习**对说话人分类不利**的特征
- 结果：模型学习到与说话人无关的表示

## 关键技术细节

1. **Causal Mask**: 确保自回归生成时不会泄露未来信息
2. **位置编码**: 使用可学习的位置嵌入而非固定的正弦编码
3. **梯度裁剪**: 防止梯度爆炸，稳定训练
4. **共享嵌入**: 语义Token和时长控制共享权重，实现精确控制
