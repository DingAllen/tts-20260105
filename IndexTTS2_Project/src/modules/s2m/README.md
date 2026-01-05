# S2M (Semantic-to-Mel) 模块

## 概述

S2M模块负责将语义Token序列转换为梅尔频谱。这是一个基于条件流匹配(Conditional Flow Matching, CFM)的生成模型，具有GPT潜变量融合功能，可以生成高质量的声学特征。

## 核心特性

### 1. 条件流匹配 (Conditional Flow Matching)

CFM是一种新型的生成建模方法，相比传统的扩散模型：
- **更高效**：直接回归速度向量场，无需多步去噪
- **更稳定**：训练目标更简单，收敛更快
- **更灵活**：支持多种ODE求解器

**数学原理**：

给定起点 $x_0$（噪声）和终点 $x_1$（真实Mel），定义插值路径：
$$x_t = (1-t) x_0 + t x_1$$

目标速度向量（真实的流）：
$$v_t^* = x_1 - x_0$$

模型训练Loss：
$$\mathcal{L}_{CFM} = \mathbb{E}_{t, x_0, x_1} || v_\theta(x_t, t, c) - (x_1 - x_0) ||^2$$

推理时，从 $x_0$ 出发，沿着学习到的向量场积分到 $x_1$。

### 2. GPT潜变量融合 (GPT Fusion)

利用T2S模型的隐藏状态（GPT特征）增强S2M的生成能力：

**核心挑战**：
- T2S输出长度（语义Token数）≠ Mel帧数
- 需要对齐和融合两种特征

**解决方案**：
1. **长度调节器(Length Regulator)**：将GPT特征插值到Mel长度
2. **融合层**：`Fused = MLP(Concat(Q_sem, H_GPT))`
3. **Dropout**：训练时随机丢弃GPT特征，避免过度依赖

### 3. ODE求解器

提供三种数值求解方法：

| 求解器 | 精度 | 速度 | 适用场景 |
|--------|------|------|----------|
| Euler | 一阶 | 最快 | 快速推理 |
| Heun | 二阶 | 中等 | 平衡质量和速度 |
| RK4 | 四阶 | 较慢 | 高质量生成 |

## 模块组成

### diffusion.py - CFM模型

**主要类**：

#### SinusoidalPositionEmbedding
正弦位置编码，用于编码时间步 $t \in [0, 1]$

#### VectorFieldEstimator
向量场估计网络，预测速度向量 $v_t$

**架构**：
- Mel投影层
- 时间步嵌入
- 条件投影
- Transformer编码器
- 输出投影

#### ConditionalFlowMatching
完整的CFM模型

**方法**：
- `compute_loss()`: 训练Loss计算
- `forward()`: 推理时预测速度向量

### solver.py - ODE求解器

**主要类**：
- `EulerSolver`: 欧拉方法
- `HeunSolver`: Heun方法
- `RK4Solver`: 四阶龙格-库塔方法

**便捷函数**：
```python
euler_solve(model, z, condition, steps=50)
rk4_solve(model, z, condition, steps=50)
heun_solve(model, z, condition, steps=50)
```

### model.py - 完整S2M模型

**主要类**：

#### LengthRegulator
长度调节器，支持两种方法：
- `repeat`: 简单重复
- `interpolate`: 线性插值（默认）

#### GPTFusionLayer
GPT潜变量融合层

**输入**：
- `semantic_features`: S2M的语义特征 (B, Mel_Len, Dim)
- `gpt_hidden_states`: T2S的隐藏状态 (B, Seq_Len, Dim)

**输出**：
- `fused_features`: 融合后的条件特征 (B, Mel_Len, Dim)

#### S2MModel
完整的S2M模型

**训练模式**：
```python
outputs = model(
    semantic_ids=semantic_ids,
    gpt_hidden_states=gpt_hidden,
    mel_gt=mel_gt,
    compute_loss=True
)
loss = outputs['loss']
```

**推理模式**：
```python
mel = model.generate(
    semantic_ids=semantic_ids,
    gpt_hidden_states=gpt_hidden,
    target_len=200,
    num_steps=50,
    solver='euler'
)
```

## 使用示例

### 训练

```python
from src.modules.s2m.model import S2MModel
from src.data.mock_factory import create_mock_batch

# 创建模型
model = S2MModel()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# 准备数据
batch = create_mock_batch(batch_size=16)

# 前向传播
outputs = model(
    semantic_ids=batch.semantic_ids,
    gpt_hidden_states=batch.t2s_hidden_states,
    mel_gt=batch.mel_gt,
    compute_loss=True
)

# 反向传播
loss = outputs['loss']
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

### 推理

```python
# 生成Mel频谱
mel = model.generate(
    semantic_ids=semantic_ids,
    gpt_hidden_states=gpt_hidden,
    target_len=300,
    num_steps=50,
    solver='rk4',  # 使用高精度求解器
    verbose=True   # 显示进度
)
```

## 设计原理

### 为什么使用流匹配？

传统扩散模型（DDPM）的问题：
1. 训练和推理不一致（训练用加噪，推理用去噪）
2. 需要大量采样步数
3. 训练目标复杂（预测噪声或数据）

流匹配的优势：
1. 训练和推理一致（都是沿着流积分）
2. 可以用少量步数高质量生成
3. 训练目标简单（直接回归速度向量）

### GPT融合的必要性

T2S模型的隐藏状态包含了丰富的韵律和发音信息：
- 停顿位置
- 重音模式
- 情感色彩

通过GPT融合，S2M可以：
- 生成更准确的韵律
- 保持与文本的一致性
- 提升整体音质

### ODE vs SDE

CFM使用ODE（常微分方程）而非SDE（随机微分方程）：

**ODE**：
- 确定性路径
- 更稳定
- 适合条件生成

**SDE**：
- 随机性更强
- 多样性更高
- 适合无条件生成

对于TTS任务，确定性的ODE更合适。

## 关键技术细节

1. **时间步编码**：使用正弦编码确保模型能感知生成进度
2. **条件注入**：通过加法融合Mel、时间步和条件特征
3. **Transformer架构**：利用自注意力捕捉长程依赖
4. **梯度稳定**：Layer Norm和残差连接确保训练稳定
