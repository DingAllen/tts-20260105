"""
T2S模块的共享嵌入层
实现语义Token嵌入和时长编码的权重共享机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ...common import config


class SharedEmbedding(nn.Module):
    """
    共享嵌入层
    
    核心思想：语义Token的位置编码和时长嵌入共享同一个权重矩阵。
    这样可以实现精确的时长控制：
        - W_sem ≡ W_num（相同的嵌入矩阵）
        - 时长向量 p = W_num · OneHot(T)
    
    其中T是目标语义序列的长度。
    """
    
    def __init__(
        self,
        vocab_size=config.SEMANTIC_VOCAB,
        embed_dim=config.SEMANTIC_EMBED_DIM,
        max_position=2048,
        padding_idx=config.PAD_TOKEN_ID
    ):
        """
        初始化共享嵌入层
        
        Args:
            vocab_size: 词汇表大小（语义Token数量）
            embed_dim: 嵌入维度
            max_position: 最大位置数（用于位置编码）
            padding_idx: Padding token的ID
        """
        super(SharedEmbedding, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_position = max_position
        self.padding_idx = padding_idx
        
        # Token嵌入（语义Token -> 向量）
        self.token_embedding = nn.Embedding(
            vocab_size, 
            embed_dim, 
            padding_idx=padding_idx
        )
        
        # 位置编码嵌入（与时长编码共享）
        # 这是核心：同一个嵌入层既用于位置编码，也用于时长控制
        self.position_embedding = nn.Embedding(max_position, embed_dim)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化嵌入权重"""
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)
        
        # Padding位置置零
        if self.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.padding_idx].fill_(0)
    
    def forward(self, token_ids, positions=None):
        """
        前向传播
        
        Args:
            token_ids: Token ID序列 (B, Seq_Len)
            positions: 位置索引 (B, Seq_Len)，如果为None则自动生成
            
        Returns:
            embeddings: Token嵌入 + 位置嵌入 (B, Seq_Len, Embed_Dim)
        """
        batch_size, seq_len = token_ids.size()
        
        # Token嵌入
        token_embed = self.token_embedding(token_ids)  # (B, Seq_Len, Embed_Dim)
        
        # 位置嵌入
        if positions is None:
            # 自动生成位置索引 [0, 1, 2, ..., seq_len-1]
            positions = torch.arange(seq_len, device=token_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        position_embed = self.position_embedding(positions)  # (B, Seq_Len, Embed_Dim)
        
        # 相加得到最终嵌入
        embeddings = token_embed + position_embed
        
        return embeddings
    
    def get_duration_embedding(self, target_lengths):
        """
        获取时长嵌入向量（用于时长控制）
        
        根据公式：p = W_num · OneHot(T)
        其中W_num就是self.position_embedding
        
        Args:
            target_lengths: 目标长度 (B,)，每个值是目标语义序列的长度T
            
        Returns:
            duration_embed: 时长嵌入向量 (B, Embed_Dim)
        """
        # 直接使用位置嵌入的权重
        # target_lengths是索引，对应OneHot(T)中为1的位置
        duration_embed = self.position_embedding(target_lengths)  # (B, Embed_Dim)
        
        return duration_embed


class T2SInputBuilder(nn.Module):
    """
    T2S模型的输入构造器
    
    将各种输入（说话人、情感、时长、文本、语义Token）组合成完整的输入序列。
    
    输入序列格式：
        [Speaker+Emotion, Duration, <BT>, Text, <BA>, Semantic_Tokens]
    
    其中：
        - Speaker+Emotion: 说话人和情感的融合向量（投影到embed_dim）
        - Duration: 时长嵌入向量（通过SharedEmbedding获取）
        - <BT>: Begin Token
        - Text: 文本Token序列
        - <BA>: Begin Answer（实际上是分隔符）
        - Semantic_Tokens: 语义Token序列（自回归生成目标）
    """
    
    def __init__(
        self,
        embed_dim=config.HIDDEN_DIM,
        speaker_dim=config.SPEAKER_EMBED_DIM,
        emotion_dim=config.EMOTION_EMBED_DIM,
        text_vocab=config.TEXT_VOCAB
    ):
        """
        初始化输入构造器
        
        Args:
            embed_dim: 嵌入维度
            speaker_dim: 说话人嵌入维度
            emotion_dim: 情感嵌入维度
            text_vocab: 文本词汇表大小
        """
        super(T2SInputBuilder, self).__init__()
        
        self.embed_dim = embed_dim
        
        # 说话人+情感投影层（将两者拼接后投影到embed_dim）
        self.speaker_emotion_proj = nn.Linear(speaker_dim + emotion_dim, embed_dim)
        
        # 文本嵌入层
        self.text_embedding = nn.Embedding(text_vocab, embed_dim, padding_idx=config.PAD_TOKEN_ID)
        
        # 特殊Token的可学习嵌入
        self.bt_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Begin Token
        self.ba_token = nn.Parameter(torch.randn(1, 1, embed_dim))  # Begin Answer
        
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        nn.init.normal_(self.text_embedding.weight, mean=0.0, std=0.02)
        if self.text_embedding.padding_idx is not None:
            with torch.no_grad():
                self.text_embedding.weight[self.text_embedding.padding_idx].fill_(0)
        
        nn.init.normal_(self.bt_token, mean=0.0, std=0.02)
        nn.init.normal_(self.ba_token, mean=0.0, std=0.02)
    
    def forward(
        self,
        text_ids,
        semantic_ids,
        speaker_embed,
        emotion_embed,
        duration_embed
    ):
        """
        构造完整的T2S输入序列
        
        Args:
            text_ids: 文本Token IDs (B, Text_Len)
            semantic_ids: 语义Token IDs (B, Semantic_Len)
            speaker_embed: 说话人嵌入 (B, Speaker_Dim)
            emotion_embed: 情感嵌入 (B, Emotion_Dim)
            duration_embed: 时长嵌入 (B, Embed_Dim)
            
        Returns:
            input_sequence: 完整输入序列 (B, Total_Len, Embed_Dim)
            其中 Total_Len = 1 + 1 + 1 + Text_Len + 1 + Semantic_Len
                           = 4 + Text_Len + Semantic_Len
        """
        batch_size = text_ids.size(0)
        
        # 1. 处理说话人+情感
        spk_emo = torch.cat([speaker_embed, emotion_embed], dim=-1)  # (B, Speaker_Dim + Emotion_Dim)
        spk_emo_embed = self.speaker_emotion_proj(spk_emo).unsqueeze(1)  # (B, 1, Embed_Dim)
        
        # 2. 时长嵌入（已经是(B, Embed_Dim)）
        duration_embed = duration_embed.unsqueeze(1)  # (B, 1, Embed_Dim)
        
        # 3. 文本嵌入
        text_embed = self.text_embedding(text_ids)  # (B, Text_Len, Embed_Dim)
        
        # 4. 语义Token嵌入（需要从外部的SharedEmbedding获取）
        # 这里假设semantic_ids已经通过SharedEmbedding处理过了
        # 在实际使用中，这部分会在T2S模型中处理
        
        # 5. 特殊Token
        bt = self.bt_token.expand(batch_size, -1, -1)  # (B, 1, Embed_Dim)
        ba = self.ba_token.expand(batch_size, -1, -1)  # (B, 1, Embed_Dim)
        
        # 6. 拼接所有部分（不包括semantic_ids，因为那是自回归的目标）
        # 顺序: [Speaker+Emotion, Duration, <BT>, Text, <BA>]
        input_prefix = torch.cat([
            spk_emo_embed,   # (B, 1, Embed_Dim)
            duration_embed,  # (B, 1, Embed_Dim)
            bt,              # (B, 1, Embed_Dim)
            text_embed,      # (B, Text_Len, Embed_Dim)
            ba               # (B, 1, Embed_Dim)
        ], dim=1)  # (B, 4 + Text_Len, Embed_Dim)
        
        return input_prefix


# 测试代码
if __name__ == "__main__":
    print("=== 测试共享嵌入层 ===")
    
    # 创建共享嵌入层
    shared_emb = SharedEmbedding()
    
    # 测试Token嵌入
    token_ids = torch.randint(4, config.SEMANTIC_VOCAB, (4, 100))
    embeddings = shared_emb(token_ids)
    
    print(f"Token IDs shape: {token_ids.shape}")
    print(f"Embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (4, 100, config.SEMANTIC_EMBED_DIM)
    
    # 测试时长嵌入
    target_lengths = torch.tensor([50, 80, 120, 100])
    duration_embed = shared_emb.get_duration_embedding(target_lengths)
    
    print(f"Target lengths: {target_lengths}")
    print(f"Duration embedding shape: {duration_embed.shape}")
    assert duration_embed.shape == (4, config.SEMANTIC_EMBED_DIM)
    
    print("\n=== 测试输入构造器 ===")
    
    input_builder = T2SInputBuilder()
    
    # 准备输入
    text_ids = torch.randint(4, config.TEXT_VOCAB, (4, 20))
    semantic_ids = torch.randint(4, config.SEMANTIC_VOCAB, (4, 100))
    speaker_embed = torch.randn(4, config.SPEAKER_EMBED_DIM)
    emotion_embed = torch.randn(4, config.EMOTION_EMBED_DIM)
    duration_embed = shared_emb.get_duration_embedding(target_lengths)
    
    # 构造输入
    input_prefix = input_builder(
        text_ids, semantic_ids, speaker_embed, emotion_embed, duration_embed
    )
    
    print(f"Input prefix shape: {input_prefix.shape}")
    # 4 + Text_Len = 4 + 20 = 24
    assert input_prefix.shape == (4, 24, config.HIDDEN_DIM)
    
    print("\n✅ 共享嵌入层和输入构造器测试通过！")
