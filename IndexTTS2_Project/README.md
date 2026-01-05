# IndexTTS2 Project

> **IndexTTS2 全栈TTS系统** - 基于流匹配和自回归Transformer的高质量文本到语音合成系统

## 项目概述

IndexTTS2是一个完整的级联式TTS(Text-to-Speech)系统，包含两个主要模块：

1. **T2S (Text-to-Semantic)**: 文本到语义Token的自回归生成模型
2. **S2M (Semantic-to-Mel)**: 语义Token到梅尔频谱的流匹配生成模型

### 核心特性

- ✅ **时长控制**: 通过共享嵌入机制实现精确的语义序列长度控制
- ✅ **情感解耦**: 使用梯度反转层(GRL)实现说话人信息的解耦
- ✅ **高质量生成**: 基于条件流匹配(CFM)的声学特征生成
- ✅ **GPT融合**: 利用T2S的隐藏状态增强S2M的生成能力
- ✅ **Zero-shot**: 支持新说话人的零样本推理
- ✅ **批量推理**: 高效的批量处理能力

## 项目结构

```
IndexTTS2_Project/
├── README.md              # 项目总览（本文件）
├── requirements.txt       # 依赖列表
├── main.py               # 训练入口
├── docs/                 # 文档目录
├── tests/                # 单元测试
│   ├── test_infrastructure.py
│   ├── test_t2s.py
│   ├── test_s2m.py
│   └── test_integration.py
└── src/
    ├── common/           # 通用工具
    │   ├── config.py     # 全局配置
    │   └── utils.py      # 工具函数
    ├── data/             # 数据处理
    │   └── mock_factory.py  # 伪数据生成器
    ├── modules/          # 核心模块
    │   ├── layers/       # 基础层（GRL等）
    │   ├── t2s/          # T2S模块
    │   │   ├── embedding.py
    │   │   ├── model.py
    │   │   └── trainer.py
    │   └── s2m/          # S2M模块
    │       ├── diffusion.py
    │       ├── solver.py
    │       └── model.py
    └── inference.py      # 推理脚本
```

## 快速开始

### 1. 环境配置

```bash
# 克隆仓库
cd IndexTTS2_Project

# 安装依赖
pip install -r requirements.txt
```

### 2. 运行测试

```bash
# 测试基础设施
python tests/test_infrastructure.py

# 测试T2S模块
python tests/test_t2s.py

# 测试S2M模块
python tests/test_s2m.py

# 测试完整集成
python tests/test_integration.py
```

### 3. 训练模型

```bash
# 运行Mock训练循环
python main.py
```

### 4. 推理

```python
from src.inference import IndexTTS2Inference
import torch

# 创建推理引擎
inference = IndexTTS2Inference(device='cpu')

# 准备输入
text_ids = torch.randint(4, 5000, (20,))  # 文本Token IDs
speaker_embed = torch.randn(256)          # 说话人嵌入
emotion_embed = torch.randn(128)          # 情感嵌入

# 生成语音
outputs = inference.predict(
    text_ids=text_ids,
    speaker_embed=speaker_embed,
    emotion_embed=emotion_embed,
    target_duration=100,  # 目标长度
    s2m_steps=50,         # ODE求解步数
    verbose=True
)

# 获取结果
semantic_tokens = outputs['semantic_tokens']  # 语义Token
mel_spectrogram = outputs['mel_spectrogram']  # 梅尔频谱
waveform = outputs['waveform']                # 波形（Mock）
```

## 系统架构

### 整体流程

```
文本 → [T2S] → 语义Token + 隐藏状态 → [S2M] → 梅尔频谱 → [Vocoder] → 波形
```

### T2S模块

**输入**:
- 文本Token IDs
- 说话人嵌入
- 情感嵌入
- 目标长度（可选）

**输出**:
- 语义Token序列
- 隐藏状态（用于S2M的GPT融合）

**关键技术**:
- 共享嵌入: 位置编码和时长编码共享权重
- GRL: 梯度反转层实现情感解耦
- 自回归生成: Transformer Decoder架构

### S2M模块

**输入**:
- 语义Token IDs
- T2S的隐藏状态（GPT特征）

**输出**:
- 梅尔频谱

**关键技术**:
- CFM: 条件流匹配，直接回归速度向量场
- GPT融合: 利用T2S的隐藏状态增强生成
- ODE求解: 支持Euler、RK4、Heun多种方法

## 核心算法

### 1. 时长控制

通过共享权重实现：
```
W_sem ≡ W_num
p = W_num · OneHot(T)
```

### 2. 梯度反转层

对抗训练Loss：
```
L_total = L_AR - λ * L_SpeakerClassifier
```

### 3. 条件流匹配

训练目标：
```
L_CFM = E[||v_θ(x_t, t, c) - (x_1 - x_0)||²]
```

其中：
- x_t = (1-t)x_0 + tx_1 (线性插值)
- v_θ: 学习的速度向量场
- c: 条件特征（来自GPT融合）

## 训练策略

### T2S训练

**Stage 1**: 主任务训练
- Loss: 语义Token预测的交叉熵
- 目标: 学习文本到语义的映射

**Stage 2**: 对抗训练
- Loss: 主任务Loss - λ × 说话人分类Loss
- 目标: 解耦说话人信息

### S2M训练

- Loss: CFM的MSE Loss
- 目标: 学习从语义到Mel的条件生成

## 配置说明

主要超参数在 `src/common/config.py` 中定义：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| TEXT_VOCAB | 5000 | 文本词汇表大小 |
| SEMANTIC_VOCAB | 1024 | 语义Token词汇表大小 |
| MEL_BINS | 80 | 梅尔频谱bins数量 |
| HIDDEN_DIM | 512 | 模型隐藏层维度 |
| T2S_NUM_LAYERS | 6 | T2S Transformer层数 |
| S2M_NUM_LAYERS | 8 | S2M模型层数 |
| CFM_TIME_STEPS | 50 | ODE求解步数 |
| GRL_LAMBDA | 1.0 | GRL反转强度 |

## 测试结果

所有测试均已通过 ✅

```
Phase 1: 基础设施测试 ✅
  ✓ 配置加载
  ✓ Mask生成
  ✓ Mock数据生成

Phase 2: T2S模块测试 ✅
  ✓ GRL梯度反转
  ✓ 共享嵌入层
  ✓ T2S模型前向传播
  ✓ T2S反向传播
  ✓ T2S训练器

Phase 3: S2M模块测试 ✅
  ✓ 正弦位置编码
  ✓ 向量场估计器
  ✓ CFM Loss计算
  ✓ ODE求解器（Euler, RK4, Heun）
  ✓ GPT融合层
  ✓ S2M完整模型

Phase 4: 集成测试 ✅
  ✓ T2S-S2M集成
  ✓ 端到端生成
  ✓ 批量推理
  ✓ 时长控制
  ✓ Mock训练循环
  ✓ Zero-shot推理
```

## 局限性与未来工作

### 当前局限

1. **数据**: 使用Mock数据，未在真实数据集上训练
2. **Vocoder**: 使用Mock实现，需要集成真实Vocoder（如HiFi-GAN）
3. **文本处理**: 缺少完整的文本前端（分词、发音预测等）
4. **优化**: 未进行性能优化和量化

### 未来改进

1. ✨ 集成真实数据集（如LJSpeech, VCTK）
2. ✨ 添加真实Vocoder
3. ✨ 实现完整的文本前端
4. ✨ 支持多语言
5. ✨ 添加韵律控制
6. ✨ 实现实时推理优化

## 技术栈

- **深度学习框架**: PyTorch 2.0+
- **核心组件**: 
  - Transformer (自回归和编码器)
  - 条件流匹配 (CFM)
  - 梯度反转层 (GRL)
  - ODE求解器 (Euler, RK4, Heun)
- **工具库**: 
  - NumPy
  - SciPy
  - einops
  - tqdm
  - matplotlib
  - tensorboard

## 引用

如果这个项目对您有帮助，请考虑引用：

```bibtex
@misc{indextts2,
  title={IndexTTS2: A Flow Matching Based Text-to-Speech System},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/your-repo/IndexTTS2}}
}
```

## 参与贡献

欢迎贡献代码、报告问题或提出建议！

## 许可证

本项目仅用于学习和研究目的。

## 致谢

感谢以下工作的启发：
- Flow Matching论文及其在生成模型中的应用
- Gradient Reversal Layer的域适应技术
- Transformer架构在序列建模中的成功

---

**注意**: 这是一个教学性质的完整实现，包含了TTS系统的所有核心组件。在实际生产环境中使用前，需要在真实数据集上进行训练和充分测试。
