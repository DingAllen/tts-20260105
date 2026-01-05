"""
IndexTTS2 全局配置文件
定义了系统的所有超参数和常量
"""

# ===== 词汇表大小 =====
TEXT_VOCAB = 5000      # 文本词汇表大小
SEMANTIC_VOCAB = 1024  # 语义Token词汇表大小

# ===== 音频特征参数 =====
MEL_BINS = 80          # 梅尔频谱的频率bins数量
SAMPLE_RATE = 22050    # 音频采样率

# ===== 模型维度参数 =====
HIDDEN_DIM = 512       # Transformer隐藏层维度
TEXT_EMBED_DIM = 256   # 文本嵌入维度
SEMANTIC_EMBED_DIM = 512  # 语义Token嵌入维度（必须等于HIDDEN_DIM，因为共享权重）

# ===== 说话人和情感参数 =====
SPEAKER_EMBED_DIM = 256   # 说话人嵌入维度
EMOTION_EMBED_DIM = 128   # 情感嵌入维度

# ===== T2S 模型参数 =====
T2S_NUM_LAYERS = 6        # T2S Transformer层数
T2S_NUM_HEADS = 8         # 多头注意力头数
T2S_FFN_DIM = 2048        # 前馈网络维度
T2S_DROPOUT = 0.1         # Dropout比率

# ===== S2M 模型参数 =====
S2M_NUM_LAYERS = 8        # S2M Diffusion模型层数
S2M_NUM_HEADS = 8         # 多头注意力头数
S2M_HIDDEN_DIM = 512      # S2M隐藏层维度
S2M_DROPOUT = 0.1         # Dropout比率

# ===== 流匹配 (Flow Matching) 参数 =====
CFM_TIME_STEPS = 50       # ODE求解步数
CFM_SIGMA = 1e-4          # 噪声标准差

# ===== 梯度反转层 (GRL) 参数 =====
GRL_LAMBDA = 1.0          # GRL反转强度

# ===== 训练参数 =====
BATCH_SIZE = 16           # 批次大小
LEARNING_RATE = 1e-4      # 学习率
MAX_GRAD_NORM = 1.0       # 梯度裁剪阈值

# ===== 特殊Token ID =====
PAD_TOKEN_ID = 0          # Padding token
BOS_TOKEN_ID = 1          # Begin of sequence token (BT - Begin Token)
EOS_TOKEN_ID = 2          # End of sequence token (BA - Begin Answer/End Token)
UNK_TOKEN_ID = 3          # Unknown token

# ===== 设备配置 =====
DEVICE = "cuda"           # 默认使用CUDA，如果不可用会降级到CPU
