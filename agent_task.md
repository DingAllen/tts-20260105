# IndexTTS2 全栈工程开发终极任务书 (The Ultimate Bible)

**适用对象**: 全栈 AI 工程师 Agent
**环境假设**: 网络受限（无法下载大规模数据集），但**完全支持**通过 `pip` 安装必要的 Python 库。
**最高指令**:

1. **持续执行**: 你必须按顺序完成阶段一到阶段四的所有任务。除非遇到无法修复的致命错误，否则**严禁**在中间阶段停止工作。
2. **文档驱动**: 对于你编写的每一个功能模块（Modules），都必须在对应目录下生成一个详细的 `README.md`，说明输入输出、算法原理和使用方法。
3. **积极安装依赖**: 遇到缺少的库时，必须积极尝试安装（`pip install`）。只有在确认为无法获取该库时，才允许寻找替代方案。
4. **中文注释**: 所有代码必须包含详尽的中文注释（Docstrings）。

## 目录 (Table of Contents)

1. **项目协议与规范 (Protocols)**
    - 1.1 依赖管理与环境配置
    - 1.2 目录结构规范
    - 1.3 文档与测试规范
2. **核心理论知识库 (Knowledge Base)**
    - 2.1 系统架构概览
    - 2.2 关键数学定义 (时长控制、流匹配、GRL)
3. **阶段一：基础设施搭建 (Infrastructure)**
    - 3.1 依赖安装与环境检查
    - 3.2 伪数据生成器 (Mock Data Factory)
    - 3.3 基础组件接口 (Tokenizer, Codec)
4. **阶段二：文本到语义模块 (T2S Module)**
    - 4.1 共享嵌入层与时长编码
    - 4.2 梯度反转层 (GRL) 与情感分类器
    - 4.3 自回归 Transformer 主体
    - 4.4 训练循环逻辑
5. **阶段三：语义到梅尔模块 (S2M Module)**
    - 5.1 GPT 潜变量融合层
    - 5.2 条件流匹配模型 (CFM)
    - 5.3 ODE 求解器
6. **阶段四：集成与推理 (Integration)**
    - 6.1 完整推理流水线
    - 6.2 最终验收测试

## 1. 项目协议与规范 (Protocols)

### 1.1 依赖管理与环境配置

Agent 必须首先尝试安装以下核心库。如果环境中有更高级的替代品（如 `flash-attn`），请优先使用。

**Target `requirements.txt`**:

```
torch>=2.0.0
numpy>=1.24.0
scipy>=1.10.0
einops>=0.6.0       # 用于简化张量操作
tqdm                # 进度条
matplotlib          # 用于测试时的可视化
tensorboard         # 日志记录
sentencepiece       # Tokenizer (可选，若无则使用 Char-level)

```

### 1.2 目录结构规范

项目根目录为 `IndexTTS2_Project/`。你必须在初始化时构建完整的树状结构，并为每个子目录创建 `README.md`。

```
IndexTTS2_Project/
├── README.md           # 项目总览
├── requirements.txt    # 依赖列表
├── docs/               # 全局文档
├── tests/              # 单元测试 (必须覆盖所有模块)
│   ├── test_infrastructure.py
│   ├── test_t2s.py
│   ├── test_s2m.py
│   └── test_integration.py
├── src/
│   ├── __init__.py
│   ├── common/         # 通用工具
│   │   ├── __init__.py
│   │   ├── utils.py    # 包含 Mask 生成等
│   │   └── config.py   # 全局超参数配置
│   ├── data/           # 数据处理
│   │   ├── __init__.py
│   │   ├── mock_factory.py  # 伪数据生成器 (核心)
│   │   └── README.md   # 数据模块说明
│   ├── modules/
│   │   ├── t2s/        # T2S 模型
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── embedding.py
│   │   │   └── README.md
│   │   ├── s2m/        # S2M 模型
│   │   │   ├── __init__.py
│   │   │   ├── model.py
│   │   │   ├── diffusion.py
│   │   │   └── README.md
│   │   └── layers/     # 基础层
│   │       ├── __init__.py
│   │       ├── grl.py
│   │       └── README.md
│   └── inference.py    # 推理脚本
└── main.py             # 训练入口 (Mock Training Loop)

```

### 1.3 文档与测试规范

1. **文档第一**：在编写代码前，先在对应的 `README.md` 中写下该模块的设计思路。
2. **测试驱动**：每个阶段的代码完成后，必须运行对应的 `tests/` 脚本。
    - **测试标准**：使用 `mock_factory` 生成的随机张量，确保 `Shape` 变换正确，`Loss` 可计算，`Backward` 不报错。
3. **闭环确认**：必须在日志中输出 "Phase X Complete: All tests passed" 后，才能开始下一阶段。

## 2. 核心理论知识库 (Knowledge Base)

### 2.1 系统架构概览

IndexTTS2 是一个级联系统：

1. **T2S (Text-to-Semantic)**: 自回归 Transformer。输入文本和提示，输出语义 Token 序列。
    - *核心特性*: **时长控制** (通过共享权重的时长嵌入) 和 **情感解耦** (通过 GRL)。
2. **S2M (Semantic-to-Mel)**: 基于流匹配 (Flow Matching) 的生成模型。
    - *核心特性*: **GPT 潜变量融合** (利用 T2S 的隐层特征增强发音)。
3. **Vocoder**: (本项目仅需接口) 将 Mel 转为波形。

### 2.2 关键数学定义

### A. 时长控制 (Duration Control)

为了实现精确的时长控制，必须强制语义位置编码 ($W_{sem}$) 和时长嵌入 ($W_{num}$) 共享权重：
$$ W_{sem} \equiv W_{num} $$
输入时长向量 $p$ 计算如下：
$$ p = W_{num} \cdot \text{OneHot}(T) $$
其中 $T$ 是目标语义序列的长度。

### B. 梯度反转层 (GRL)

用于 Stage 2 训练，消除 Speaker 信息：
$$ \mathcal{L}*{total} = \mathcal{L}*{AR} - \lambda \cdot \mathcal{L}_{SpeakerClassifier} $$
在代码实现中，GRL 层在前向传播时是恒等映射，反向传播时将梯度乘以 $-\lambda$。

### C. 条件流匹配 (Conditional Flow Matching)

S2M 模块的目标是回归向量场 $v_t$：
$$ \mathcal{L}*{CFM} = \mathbb{E}*{t, x_0, x_1} || v_t(x_t, \text{cond}) - (x_1 - x_0) ||^2 $$
其中 $x_t = (1-t)x_0 + t x_1$。

## 3. 阶段一：基础设施搭建 (Infrastructure)

**目标**: 建立稳固的地基，确保数据流转通畅。

### 3.1 伪数据生成器 (Mock Data Factory)

在 `src/data/mock_factory.py` 中实现 `MockBatch` 类。

- **功能**: 生成符合真实数据分布（但内容随机）的张量。
- **必需字段**:
    - `text_ids`: (B, Text_Len)
    - `semantic_ids`: (B, Semantic_Len)
    - `mel_gt`: (B, Mel_Bins, Frame_Len)
    - `speaker_embed`: (B, Spk_Dim)
    - `emotion_embed`: (B, Emo_Dim)
    - `target_len`: (B,) - 整数，表示 $T$

### 3.2 基础组件接口

- **Config**: 在 `src/common/config.py` 定义：
    - `TEXT_VOCAB=5000`, `SEMANTIC_VOCAB=1024`
    - `MEL_BINS=80`, `HIDDEN_DIM=512`
- **Utils**: 实现 `get_mask_from_lengths`，用于处理变长序列的 Padding Mask。

> 阶段验收: 运行 tests/test_infrastructure.py，确认能生成 Batch，且 Mask 生成正确。
> 

## 4. 阶段二：文本到语义模块 (T2S Module)

**目标**: 实现复杂的自回归控制逻辑。

### 4.1 共享嵌入层与时长编码

文件: `src/modules/t2s/embedding.py`

- 定义 `SharedEmbedding` 类。
- **关键逻辑**: 实例化一个 `nn.Embedding` 对象，同时用于 `Semantic Tokens` 的 Embedding 和 `Duration (Length)` 的 Embedding。
- **输入构造**: 实现将 `[Speaker+Emotion, Duration, <BT>, Text, <BA>]` 拼接的逻辑。

### 4.2 梯度反转层 (GRL)

文件: `src/modules/layers/grl.py`

- 继承 `torch.autograd.Function`。
- 实现 `forward` (return input) 和 `backward` (return -lambda * grad)。
- 编写一个简单的 `SpeakerClassifier` (MLP)，接在 GRL 后面。

### 4.3 自回归 Transformer 主体

文件: `src/modules/t2s/model.py`

- 使用 `torch.nn.TransformerDecoder` 或手动实现 Stack of Decoder Layers。
- **重要输出**: `forward` 函数必须返回两个值：
    1. `logits`: 用于计算交叉熵 Loss。
    2. `hidden_states`: **最后一层** Transformer 的输出 (用于 S2M 的 GPT Fusion)。
- **Causal Mask**: 确保预测 $t$ 时刻只能看到 $0...t-1$ 时刻。

### 4.4 训练循环逻辑

文件: `src/modules/t2s/trainer.py` (仅逻辑)

- 实现 Step 函数，包含三个 Loss：
    1. `loss_text`: 文本预测 (可选)
    2. `loss_semantic`: 语义预测 (主 Loss)
    3. `loss_grl`: 情感解耦 (仅在 Stage 2 开启)

> 阶段验收: 运行 tests/test_t2s.py。
> 
> 1. 输入 Mock Batch。
> 2. 验证输出 Logits 形状为 (B, Seq_Len, Semantic_Vocab)。
> 3. 验证 Hidden States 形状为 (B, Seq_Len, Hidden_Dim)。
> 4. 验证 Backward 过程无报错。

## 5. 阶段三：语义到梅尔模块 (S2M Module)

**目标**: 实现高质量声学特征生成。

### 5.1 GPT 潜变量融合层

文件: `src/modules/s2m/model.py` (内部类)

- **输入**: $Q_{sem}$ (S2M 自己的语义 Embedding) 和 $H_{GPT}$ (来自 T2S)。
- **对齐**: 由于 T2S 输出长度和 Mel 帧数可能不一致 (通常 Semantic Token 需要上采样)，这里实现简单的 `Length Regulator` (重复) 或插值。
- **融合**: `Output = MLP(Concat(Q_sem, H_GPT))`。
- **Dropout**: 训练时随机将 $H_{GPT}$ 置为 0，迫使模型不完全依赖 T2S。

### 5.2 条件流匹配模型 (CFM)

文件: `src/modules/s2m/diffusion.py`

- **Backbone**: 实现一个基于 Transformer (如 DiT) 或 ResNet (如 WaveNet结构) 的 `VectorFieldEstimator`。
- **输入**: $x_t$ (Noisy Mel), $t$ (时间步), $Condition$ (融合后的特征 + Speaker)。
- **输出**: $v_t$ (速度向量，形状同 Mel)。

### 5.3 ODE 求解器

文件: `src/modules/s2m/solver.py`

- 实现 `euler_solve(model, z, cond, steps=50)`。
- 循环：
$$ x_{next} = x_{curr} + model(x_{curr}, t, cond) \cdot dt $$

> 阶段验收: 运行 tests/test_s2m.py。
> 
> 1. 验证 GPT 融合逻辑是否支持形状不匹配的输入。
> 2. 验证 CFM 的 Loss 计算。
> 3. 运行 Solver，检查生成的 Mel 形状是否正确。

## 6. 阶段四：集成与推理 (Integration)

**目标**: 串联所有模块，实现最终调用。

### 6.1 完整推理流水线

文件: `src/inference.py`

- 实现 `class IndexTTS2Inference`:
    - `load_models()`: 加载 T2S 和 S2M。
    - `predict(text, prompt_audio, target_duration=None)`:
        1. 预处理 Text 和 Prompt。
        2. **T2S**: 生成 Semantic Tokens 和 Hidden States。
            - 若 `target_duration` 存在，构建 $p$ 向量；否则 $p=0$。
        3. **S2M**: 利用 Tokens 和 Hidden States 生成 Mel。
        4. **Vocoder**: (Mock) Mel -> Waveform。

### 6.2 最终验收测试

文件: `tests/test_integration.py`

- 模拟一次完整的 Zero-shot 推理。
- **断言**:
    - 最终输出是否为音频张量。
    - 如果指定了时长 $T=100$，生成的 Semantic Token 数量是否接近 100。

**致 Agent**:
现在，请从 **阶段一** 开始工作，直到完成所有阶段的内容。务必记住，每完成一个文件，就要检查是否需要安装库，是否编写了文档，以及是否可以通过测试。**Good Luck.**
